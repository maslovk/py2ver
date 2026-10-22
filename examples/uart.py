# A UART, from how its frames are specified to the FPGA: its transmitter and its
# receiver written as the protocol says, a frame at a time, as Python generators
# in which a yield is a clock (see docs/processes.md, the guide). py2ver
# builds each as a state machine, clock for clock what the
# generator does, and the generator itself is the reference.
#
# A frame (8N1): the line idle high; a start bit, low; eight data bits, least
# significant first; a stop bit, high; each bit BIT clocks.
#
#   python examples/uart.py              # loopback: bytes sent, the line, bytes received
#   python examples/uart.py --explain    # the receiver's states, each a line of uart_rx
#   python examples/uart.py --vcd        # the receiver's pins as waveforms: uart_rx.vcd
#   python examples/uart.py --sim        # both simulated (Verilog), the same as the Python
#   python examples/uart.py --syn        # both in Quartus: 82 and 81 LUTs, 198 and 250 MHz
#   python examples/uart.py --pins       # the echo on the DE0-Nano's serial pins (GPIO_0_RX,
#                                        # GPIO_0_TX), the byte on the LEDs (compiled only)
import os
import sys

CLOCK = 50_000_000
BAUD = 115_200
BIT = CLOCK // BAUD                    # 434 clocks a bit


def wait_until(port, name, value=1):
    """Clocks pass until the pin is value."""
    while getattr(port, name) != value:
        yield


def read_byte(port, bit):
    """Eight bits of the line, bit clocks apart, least significant first."""
    data = 0
    for i in range(8):
        yield bit                      # to the next bit's middle
        data = data | (port.rx << i)
    return data


def uart_tx(port, bit=BIT):
    """port.start (a clock high): send port.data; port.tx the line, port.busy high
    while a frame goes out."""
    port.tx = 1
    while True:
        port.busy = 0
        yield from wait_until(port, "start")     # idle, the line high
        data = port.data
        port.busy = 1
        port.tx = 0                    # start bit
        yield bit
        for i in range(8):
            port.tx = (data >> i) & 1  # least significant bit first
            yield bit
        port.tx = 1                    # stop bit
        yield bit


def uart_rx(port, bit=BIT):
    """port.rx the line; a byte received: port.data, and port.valid high a clock
    (port.error too if its stop bit was low)."""
    while True:
        port.valid = 0
        yield from wait_until(port, "rx", 0)     # idle: wait for the start bit's edge
        yield bit // 2                 # to the start bit's middle
        if port.rx:                    # (a glitch, not a start bit)
            continue
        data = yield from read_byte(port, bit)
        yield bit                      # the stop bit's middle
        port.data = data
        port.error = 1 - port.rx
        port.valid = 1
        yield


def echo():
    """The receiver and the transmitter together: each byte received sent back."""
    from py2ver import connect
    return connect(rx=(uart_rx, RX), tx=(uart_tx, TX), name="uart_echo",
                   wires={"tx.data": "rx.data", "tx.start": "rx.valid"})


# The inputs' widths (the outputs': followed from what's written; the line idle high)
TX = {"start": 1, "data": 8, "tx": {"init": 1}}
RX = {"rx": 1}
# (the echo on the board: the line in on GPIO_0_RX, back out on GPIO_0_TX, the board's
# serial pins, the byte on the LEDs)
ECHO_PINS = {"rx": "GPIO_0_RX", "tx": "GPIO_0_TX", "data": "LED"}


def frames(data, gap=3):
    """The transmitter's inputs, a clock each: each byte started, then held off
    until its frame is out (and gap clocks more)."""
    clocks = []
    for d in data:
        clocks.append({"start": 1, "data": d})
        clocks += [{"start": 0, "data": 0}] * (10 * BIT + gap)
    return clocks


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from py2ver import Py2ver, process

    message = b"py2ver"
    clocks = frames(message)
    line = [{"rx": r["tx"]} for r in process.simulate(uart_tx, clocks, TX)]
    got = [r["data"] for r in process.simulate(uart_rx, line, RX) if r["valid"]]
    print(f"the generators: sent {message!r} ({len(line):,} clocks of the line), "
          f"received {bytes(got)!r}")
    tx, rx = Py2ver(uart_tx, TX), Py2ver(uart_rx, RX)
    print(f"the machines: {tx.process.states} and {rx.process.states} states; their models the "
          f"generators clock for clock: {tx.PY_process()(clocks) == tx.GEN_process()(clocks)}, "
          f"{rx.PY_process()(line) == rx.GEN_process()(line)}")
    if "--explain" in sys.argv[1:]:
        print(rx.explain())
    if "--vcd" in sys.argv[1:]:
        rx.GEN_process(vcd="uart_rx.vcd")(line)
        print("the receiver's pins, a clock at a time: uart_rx.vcd (GTKWave, Surfer)")
    if "--sim" in sys.argv[1:]:
        sent = tx.TB_process()(clocks)
        back = rx.TB_process()([{"rx": r["tx"]} for r in sent])
        print(f"simulated (Verilog): the same as their models: "
              f"{sent == tx.PY_process()(clocks)}, received {bytes(r['data'] for r in back if r['valid'])!r}")
    quartus = {"QUARTUS_DIR": os.environ.get("QUARTUS_DIR", "/opt/intelFPGA/22.1std/quartus/bin")}
    if "--syn" in sys.argv[1:]:
        for name, p in (("transmitter", tx), ("receiver", rx)):
            print(name, p.SYN(quartus))
    sent = process.simulate(echo(), line + [{"rx": 1}] * (12 * BIT))
    back = process.simulate(uart_rx, [{"rx": r["tx"]} for r in sent], RX)
    print(f"the echo (the receiver and the transmitter connected): sent back "
          f"{bytes(r['data'] for r in back if r['valid'])!r}")
    if "--pins" in sys.argv[1:]:
        board = Py2ver(echo(), ECHO_PINS)
        print("the echo on the board's pins (syn/uart_echo_pins/DE0_Nano.sof):",
              board.PINS(quartus))
