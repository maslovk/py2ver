# An echo over the JTAG UART: each byte typed at the PC (juart-terminal, over the
# same USB-Blaster cable that programs the board: no serial adapter) sent back, the
# last one on the LEDs. A process (see docs/processes.md, the guide) on the JTAG
# UART's pins: JTAG_RX_* the bytes from the PC, JTAG_TX_* the bytes to it.
#
#   python examples/jtag_echo.py                # the testbench: the generator, the model
#   python examples/jtag_echo.py --pins         # compiled for the board (not programmed)
#   python examples/jtag_echo.py --pins --signaltap   # and SignalTap recording it
#
# On the board: program syn/jtag_echo_pins/DE0_Nano.sof, then
#   juart-terminal          (in Quartus's bin directory) and type.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py2ver.protocol import wait_until  # noqa: E402


def jtag_echo(port):
    """Each byte from the PC (port.byte, port.valid high the clock it's given,
    while port.ready) sent back (port.out, port.send high a clock, when there's
    room: port.room), and shown on port.leds."""
    port.ready = 1
    port.send = 0
    while True:
        yield from wait_until(port, "valid")    # a byte, taken this clock
        b = port.byte
        port.ready = 0                          # (no more until it's sent back)
        port.out = b
        port.leds = b
        yield from wait_until(port, "room")     # room toward the PC
        port.send = 1
        yield
        port.send = 0
        port.ready = 1


PINS = {"byte": "JTAG_RX_DATA", "valid": "JTAG_RX_VALID", "ready": "JTAG_RX_READY",
        "out": "JTAG_TX_DATA", "send": "JTAG_TX_VALID", "room": "JTAG_TX_READY",
        "leds": "LED"}


def pc(text):
    """A testbench playing the JTAG UART's side: each byte given while the design
    is ready, what it sends back collected; the bytes back."""
    def tb(dut):
        dut.room = 1
        got, k = [], 0
        while len(got) < len(text):
            dut.valid = 0
            if k < len(text) and dut.ready:
                dut.byte, dut.valid = text[k], 1
                k += 1
            if dut.send:
                got.append(dut.out)
            yield
        return bytes(got)
    return tb


if __name__ == "__main__":
    from py2ver import Py2ver

    tap = "--signaltap" in sys.argv[1:]
    p = Py2ver(jtag_echo, PINS, probes=tap)
    run = p.check(pc(b"hello"), verilog="--sim" in sys.argv[1:])
    print(f"{len(run.inputs)} clocks: sent back {run.value!r}")
    if "--pins" in sys.argv[1:]:
        print(p.PINS({"QUARTUS_DIR": os.environ.get(
            "QUARTUS_DIR", "/opt/intelFPGA/22.1std/quartus/bin")}, signaltap=tap))
