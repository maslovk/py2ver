# A debug console over the JTAG UART: a design that tells the PC what it's doing.
# A banner at reset; each press of KEY[1] (debounced) counted and printed; typing s
# at the PC prints the switches. juart-terminal shows it, over the USB-Blaster cable
# that programs the board (see docs/processes.md, the guide).
#
#   python examples/jtag_console.py              # the testbench: the generator, the model
#   python examples/jtag_console.py --pins       # compiled for the board (not programmed)
#
# On the board: program syn/jtag_console_pins/DE0_Nano.sof, run juart-terminal (in
# Quartus's bin directory), press KEY1, type s.
import functools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py2ver import connect  # noqa: E402
from py2ver.protocol import debounced, jtag_print, jtag_print_hex  # noqa: E402

BANNER = b"py2ver console: press KEY1, or type s\n"
KEY_S = ord("s")
BOUNCE = 500_000                         # 10 ms at 50 MHz: a key's bounces settled


def console(port):
    """Prints over the JTAG UART (port.tx_*): the banner, each press of port.key
    (0 while pressed, as the board's keys read), and the switches (port.sw) when
    an s comes from the PC (port.rx_*)."""
    port.rx_ready = 0
    yield from jtag_print(port, BANNER)
    count = 0
    was = 0
    while True:
        port.rx_ready = 1                # (a byte from the PC taken when it comes)
        key = port.key
        if was == 1 and key == 0:        # pressed
            port.rx_ready = 0
            count = (count + 1) & 255
            yield from jtag_print(port, b"presses: 0x")
            yield from jtag_print_hex(port, count, 2)
            yield from jtag_print(port, b"\n")
        elif port.rx_valid and port.rx_data == KEY_S:
            port.rx_ready = 0
            yield from jtag_print(port, b"switches: 0x")
            yield from jtag_print_hex(port, port.sw, 1)
            yield from jtag_print(port, b"\n")
        was = key
        yield


def board_console(bounce=BOUNCE):
    """The console, its key through a debouncer."""
    return connect(db=(functools.partial(debounced, clocks=bounce), {"raw": 1}),
                   con=(console, {"sw": 4, "rx_data": 8, "rx_valid": 1, "tx_room": 1}),
                   name="jtag_console", wires={"con.key": "db.clean"})


PINS = {"raw": "KEY[1]", "sw": "SW",
        "rx_data": "JTAG_RX_DATA", "rx_valid": "JTAG_RX_VALID", "rx_ready": "JTAG_RX_READY",
        "tx_data": "JTAG_TX_DATA", "tx_send": "JTAG_TX_VALID", "tx_room": "JTAG_TX_READY"}


def session(presses=2, switches=0b1010, bounce=4):
    """A testbench: the key released, pressed twice (bouncing), an s typed; what
    the console printed."""
    def tb(dut):
        dut.raw, dut.sw, dut.tx_room = 1, switches, 1
        text = []

        def clock():
            if dut.tx_send:
                text.append(dut.tx_data)
            dut.rx_valid = 0
            yield
        for _ in range(len(BANNER) * 3 + 10 * bounce):
            yield from clock()
        for _ in range(presses):
            for v in [0, 1, 0, 1, 0] + [0] * (4 * bounce) + [1, 0, 1] + [1] * (4 * bounce):
                dut.raw = v
                yield from clock()
            for _ in range(100):
                yield from clock()
        while not dut.rx_ready:
            yield from clock()
        dut.rx_data, dut.rx_valid = KEY_S, 1
        yield
        for _ in range(100):
            yield from clock()
        return bytes(text)
    return tb


if __name__ == "__main__":
    from py2ver import Py2ver

    p = Py2ver(board_console(bounce=4), PINS)
    print(p.check(session(), verilog="--sim" in sys.argv[1:]).value.decode())
    if "--pins" in sys.argv[1:]:
        board = Py2ver(board_console(), PINS)
        print(board.PINS({"QUARTUS_DIR": os.environ.get(
            "QUARTUS_DIR", "/opt/intelFPGA/22.1std/quartus/bin")}))
