# A UART, from how its frames are specified to the FPGA: its transmitter and its
# receiver written as the protocol says, a frame at a time, as Python generators
# in which a yield is a clock (see py2ver/process.py and "Processes" in the
# README). py2ver builds each as a state machine, clock for clock what the
# generator does, and the generator itself is the reference.
#
# A frame (8N1): the line idle high; a start bit, low; eight data bits, least
# significant first; a stop bit, high; each bit BIT clocks.
#
#   python examples/uart.py              # loopback: bytes sent, the line, bytes received
#   python examples/uart.py --sim        # both simulated (Verilog), the same as the Python
#   python examples/uart.py --syn        # both in Quartus: 82 and 81 LUTs, 198 and 250 MHz
#   python examples/uart.py --pins       # the receiver on the DE0-Nano's pins: rx on
#                                        # GPIO_0[0], the byte on the LEDs (compiled only)
import os
import sys

CLOCK = 50_000_000
BAUD = 115_200
BIT = CLOCK // BAUD                    # 434 clocks a bit


def uart_tx(port):
    """port.start (a clock high): send port.data; port.tx the line, port.busy high
    while a frame goes out."""
    port.tx = 1
    while True:
        port.busy = 0
        while not port.start:          # idle, the line high
            yield
        data = port.data
        port.busy = 1
        port.tx = 0                    # start bit
        yield BIT
        for i in range(8):
            port.tx = (data >> i) & 1  # least significant bit first
            yield BIT
        port.tx = 1                    # stop bit
        yield BIT


def uart_rx(port):
    """port.rx the line; a byte received: port.data, and port.valid high a clock
    (port.error too if its stop bit was low)."""
    while True:
        port.valid = 0
        while port.rx:                 # idle: wait for the start bit's edge
            yield
        yield BIT // 2                 # to the start bit's middle
        if port.rx:                    # (a glitch, not a start bit)
            continue
        data = 0
        for i in range(8):
            yield BIT                  # to the next bit's middle
            data = data | (port.rx << i)
        yield BIT                      # the stop bit's middle
        port.data = data
        port.error = 1 - port.rx
        port.valid = 1
        yield


TX = {"start": {"width": 1}, "data": {"width": 8}, "tx": {"width": 1, "init": 1},
      "busy": {"width": 1}}
RX = {"rx": {"width": 1}, "data": {"width": 8}, "valid": {"width": 1}, "error": {"width": 1}}


def frames(data, gap=3):
    """The transmitter's inputs, a clock each: each byte started, then held off
    until its frame is out (and gap clocks more)."""
    clocks = []
    for d in data:
        clocks.append({"start": 1, "data": d})
        clocks += [{"start": 0, "data": 0}] * (10 * BIT + gap)
    return clocks


if __name__ == "__main__":
    import contextlib
    import io
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from py2ver import Py2ver, process

    message = b"py2ver"
    clocks = frames(message)
    line = [{"rx": r["tx"]} for r in process.simulate(uart_tx, clocks, TX)]
    got = [r["data"] for r in process.simulate(uart_rx, line, RX) if r["valid"]]
    print(f"the generators: sent {message!r} ({len(line):,} clocks of the line), "
          f"received {bytes(got)!r}")
    with contextlib.redirect_stdout(io.StringIO()):
        tx, rx = Py2ver(uart_tx, TX), Py2ver(uart_rx, RX)
    print(f"the machines: {tx.process.states} and {rx.process.states} states; their models the "
          f"generators clock for clock: {tx.PY_process()(clocks) == tx.GEN_process()(clocks)}, "
          f"{rx.PY_process()(line) == rx.GEN_process()(line)}")
    if "--sim" in sys.argv[1:]:
        sent = tx.TB_process()(clocks)
        back = rx.TB_process()([{"rx": r["tx"]} for r in sent])
        print(f"simulated (Verilog): the same as their models: "
              f"{sent == tx.PY_process()(clocks)}, received {bytes(r['data'] for r in back if r['valid'])!r}")
    quartus = {"QUARTUS_DIR": os.environ.get("QUARTUS_DIR", "/opt/intelFPGA/22.1std/quartus/bin")}
    if "--syn" in sys.argv[1:]:
        for name, p in (("transmitter", tx), ("receiver", rx)):
            print(name, p.SYN(quartus))
    if "--pins" in sys.argv[1:]:
        with contextlib.redirect_stdout(io.StringIO()):
            board = Py2ver(uart_rx, dict(RX, rx=dict(RX["rx"], pin="GPIO_0[0]"),
                                         data=dict(RX["data"], pin="LED[7:0]")))
        print("the receiver on the board's pins (syn/uart_rx_pins/DE0_Nano.sof):",
              board.PINS(quartus))
