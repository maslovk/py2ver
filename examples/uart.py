# A UART transmitter and receiver, written the way the protocol is described.
#
# A UART frame (8N1) looks like this on the line, each bit BIT clock cycles long:
#
#   idle (high) | start bit (low) | 8 data bits, least significant first | stop bit (high)
#
# Each side below is a Python generator in which `yield` means "one clock cycle
# passes" (and `yield n`, n cycles). py2ver turns each into a state machine that
# does, clock cycle for clock cycle, exactly what the generator does, so the
# generator itself is the reference to test against. See docs/processes.md.
#
#   python examples/uart.py              # send a message from one to the other
#   python examples/uart.py --explain    # show the receiver's states, each a line of uart_rx
#   python examples/uart.py --vcd        # save the receiver's pins as waveforms (uart_rx.vcd)
#   python examples/uart.py --sim        # also simulate the generated Verilog
#   python examples/uart.py --syn        # also build both in Quartus (size and speed)
#   python examples/uart.py --pins       # build an echo for the DE0-Nano's GPIO_0_RX/TX pins
#   python examples/uart.py --pins --signaltap   # ... with SignalTap recording it on the board
import os
import sys

CLOCK = 50_000_000                     # the board's clock, in Hz
BAUD = 115_200
BIT = CLOCK // BAUD                    # one bit lasts 434 clock cycles


def wait_until(port, name, value=1):
    """Wait until the pin has the given value."""
    while getattr(port, name) != value:
        yield


def read_byte(port, bit):
    """Read 8 bits from the line, one every `bit` cycles, least significant first."""
    data = 0
    for i in range(8):
        yield bit                      # move to the middle of the next bit
        data = data | (port.rx << i)
    return data


def uart_tx(port, bit=BIT):
    """The transmitter.

    Inputs:  start (high for one cycle to send), data (the byte to send).
    Outputs: tx (the line), busy (high while a frame goes out).
    """
    port.tx = 1
    while True:
        port.busy = 0
        yield from wait_until(port, "start")
        data = port.data
        port.busy = 1
        port.tx = 0                    # start bit
        yield bit
        for i in range(8):
            port.tx = (data >> i) & 1  # data bits, least significant first
            yield bit
        port.tx = 1                    # stop bit
        yield bit


def uart_rx(port, bit=BIT):
    """The receiver.

    Input:   rx (the line).
    Outputs: data (the byte received), valid (high for one cycle per byte),
             error (high with valid if the stop bit was missing).
    """
    while True:
        port.valid = 0
        yield from wait_until(port, "rx", 0)     # wait for a start bit
        yield bit // 2                 # move to the middle of the start bit
        if port.rx:                    # high again: only a glitch, not a start bit
            continue
        data = yield from read_byte(port, bit)
        yield bit                      # the middle of the stop bit
        port.data = data
        port.error = 1 - port.rx       # the stop bit should be high
        port.valid = 1
        yield


def echo():
    """The receiver and the transmitter connected: every byte received is sent back."""
    from py2ver import connect
    return connect(rx=(uart_rx, RX), tx=(uart_tx, TX), name="uart_echo",
                   wires={"tx.data": "rx.data", "tx.start": "rx.valid"})


# The widths of the inputs; py2ver works out the outputs' widths itself. (tx is
# high from reset, because uart_tx sets it to 1 before its first yield.)
TX = {"start": 1, "data": 8}
RX = {"rx": 1}

# For the board: the line comes in on GPIO_0_RX and goes back out on GPIO_0_TX,
# and the last byte received shows on the LEDs.
ECHO_PINS = {"rx": "GPIO_0_RX", "tx": "GPIO_0_TX", "data": "LED"}


def frames(data, gap=3):
    """The transmitter's inputs, one dict per clock cycle, to send each byte of data.

    Each byte is started, then the next waits until its frame is out.
    """
    clocks = []
    for byte in data:
        clocks.append({"start": 1, "data": byte})
        clocks += [{"start": 0, "data": 0}] * (10 * BIT + gap)
    return clocks


def received(rows):
    """The bytes a receiver's outputs say it received."""
    return bytes(row["data"] for row in rows if row["valid"])


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from py2ver import Py2ver, process

    args = sys.argv[1:]
    quartus = {"QUARTUS_DIR": os.environ.get("QUARTUS_DIR", "/opt/intelFPGA/22.1std/quartus/bin")}
    message = b"py2ver"

    # 1. Run the generators as plain Python: the transmitter's line feeds the receiver.
    clocks = frames(message)
    line = [{"rx": row["tx"]} for row in process.simulate(uart_tx, clocks, TX)]
    print(f"Python:   sent {message!r}, received {received(process.simulate(uart_rx, line, RX))!r}"
          f" ({len(line):,} clock cycles)")

    # 2. Build the hardware, and check its model against the generators.
    tx, rx = Py2ver(uart_tx, TX), Py2ver(uart_rx, RX)
    same = (tx.PY_process()(clocks) == tx.GEN_process()(clocks)
            and rx.PY_process()(line) == rx.GEN_process()(line))
    print(f"Hardware: {tx.process.states} and {rx.process.states} states, "
          f"{'the same as' if same else 'DIFFERENT FROM'} the generators")

    if "--explain" in args:
        print(rx.explain())
    if "--vcd" in args:
        rx.GEN_process(vcd="uart_rx.vcd")(line)
        print("Waveforms: uart_rx.vcd (open it in GTKWave or Surfer)")
    if "--sim" in args:
        sent = tx.TB_process()(clocks)
        back = rx.TB_process()([{"rx": row["tx"]} for row in sent])
        print(f"Verilog:  received {received(back)!r}, "
              f"{'the same as' if sent == tx.PY_process()(clocks) else 'DIFFERENT FROM'} Python")
    if "--syn" in args:
        for name, p in (("transmitter", tx), ("receiver", rx)):
            print(name, p.SYN(quartus))

    # 3. The echo: what goes in comes back out.
    sent = process.simulate(echo(), line + [{"rx": 1}] * (12 * BIT))
    back = process.simulate(uart_rx, [{"rx": row["tx"]} for row in sent], RX)
    print(f"Echo:     sent back {received(back)!r}")

    if "--pins" in args:
        signaltap = "--signaltap" in args
        board = Py2ver(echo(), pins=ECHO_PINS, probes=signaltap)
        print(board.PINS(quartus, signaltap=signaltap))
        print("Program syn/uart_echo_pins/DE0_Nano.sof to put it on the board"
              + (" (SignalTap's bits are listed in syn/uart_echo_pins/signaltap.txt)"
                 if signaltap else "") + ".")
