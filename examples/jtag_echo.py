# An echo over the JTAG UART: every byte typed on the PC comes back, and the
# last one shows on the board's LEDs.
#
# The JTAG UART runs over the same USB-Blaster cable that programs the board,
# so no serial adapter is needed. The design talks to it through the board's
# JTAG UART pins: JTAG_RX_* bring bytes from the PC, JTAG_TX_* send bytes to it.
# See docs/processes.md for how a process like this one works.
#
#   python examples/jtag_echo.py                      # test it against the generator
#   python examples/jtag_echo.py --pins               # build it for the board
#   python examples/jtag_echo.py --pins --signaltap   # ... with SignalTap recording it
#
# Then program syn/jtag_echo_pins/DE0_Nano.sof, run juart-terminal (in Quartus's
# bin directory) and type.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py2ver.protocol import wait_until  # noqa: E402


def jtag_echo(port):
    """Send every byte from the PC back to it, and show it on the LEDs."""
    port.ready = 1                              # ready to take a byte
    port.send = 0
    while True:
        yield from wait_until(port, "valid")    # a byte arrives (taken this cycle)
        received = port.byte
        port.ready = 0                          # take no more until it's sent back
        port.out = received
        port.leds = received
        yield from wait_until(port, "room")     # wait for room to send it
        port.send = 1
        yield
        port.send = 0
        port.ready = 1


# Which board pin each port is on
PINS = {"byte": "JTAG_RX_DATA", "valid": "JTAG_RX_VALID", "ready": "JTAG_RX_READY",
        "out": "JTAG_TX_DATA", "send": "JTAG_TX_VALID", "room": "JTAG_TX_READY",
        "leds": "LED"}


def pc(text):
    """A testbench that plays the PC: it types `text`, one byte whenever the
    design is ready, and returns the bytes the design sends back."""
    def tb(dut):
        dut.room = 1
        answer, typed = [], 0
        while len(answer) < len(text):
            dut.valid = 0
            if typed < len(text) and dut.ready:
                dut.byte, dut.valid = text[typed], 1
                typed += 1
            if dut.send:
                answer.append(dut.out)
            yield
        return bytes(answer)
    return tb


if __name__ == "__main__":
    from py2ver import Py2ver

    args = sys.argv[1:]
    signaltap = "--signaltap" in args
    p = Py2ver(jtag_echo, PINS, probes=signaltap)
    run = p.check(pc(b"hello"), verilog="--sim" in args)
    print(f"Typed b'hello', got back {run.value!r} ({len(run.inputs)} clock cycles)")
    if "--pins" in args:
        print(p.PINS({"QUARTUS_DIR": os.environ.get(
            "QUARTUS_DIR", "/opt/intelFPGA/22.1std/quartus/bin")}, signaltap=signaltap))
