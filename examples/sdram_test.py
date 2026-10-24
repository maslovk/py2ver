# A memory test of the DE0-Nano's 32 MB SDRAM, through py2ver.devices.sdram's
# controller (a process): each pass writes every word, then changes one byte of
# each, then reads them all back, and prints on the JTAG console how many were
# wrong. The LEDs count the passes; LED 7 lights at the first error.
#
#   python examples/sdram_test.py             # the testbench (a few words, the chip's model)
#   python examples/sdram_test.py --pins      # compiled for the board (not programmed)
#
# On the board: program syn/sdram_test_pins/DE0_Nano.sof, run juart-terminal.
import functools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py2ver import connect  # noqa: E402
from py2ver.devices.sdram import (SDRAM_ATTR, SDRAM_PINS, SdramModel, SdramTiming,  # noqa: E402
                                  WORDS, sdram_controller, sdram_read, sdram_write)
from py2ver.protocol import jtag_print, jtag_print_hex  # noqa: E402

MASK = 0xFFFFFFFF


def memory_test(port, words=WORDS, stride=1):
    """Test `words` words of the SDRAM, `stride` words apart (an odd stride
    reaches every word once), pass after pass."""
    port.req = 0
    port.req_write = 0
    port.req_address = 0
    port.req_data = 0
    port.req_mask = 15
    port.req_line = 0
    port.leds = 0
    yield from jtag_print(port, b"sdram test\n")
    passes = 0
    seed = 1
    failed = 0
    while True:
        # every word written with the sequence from this pass's seed
        x = seed
        address = 0
        k = 0
        while k < words:
            x = x ^ ((x << 13) & MASK)                   # (xorshift: the next random number)
            x = x ^ (x >> 17)
            x = x ^ ((x << 5) & MASK)
            yield from sdram_write(port, address, x)
            address = (address + stride) & (WORDS - 1)
            k = k + 1
        # one byte of each word changed: byte k % 4 of word k, inverted
        x = seed
        address = 0
        k = 0
        while k < words:
            x = x ^ ((x << 13) & MASK)                   # (xorshift: the next random number)
            x = x ^ (x >> 17)
            x = x ^ ((x << 5) & MASK)
            byte = k & 3
            yield from sdram_write(port, address, x ^ MASK, 1 << byte)
            address = (address + stride) & (WORDS - 1)
            k = k + 1
        # and read back
        x = seed
        address = 0
        k = 0
        errors = 0
        while k < words:
            x = x ^ ((x << 13) & MASK)                   # (xorshift: the next random number)
            x = x ^ (x >> 17)
            x = x ^ ((x << 5) & MASK)
            byte = k & 3
            got = yield from sdram_read(port, address)
            if got != x ^ (0xFF << (byte << 3)):
                errors = errors + 1
            address = (address + stride) & (WORDS - 1)
            k = k + 1
        passes = passes + 1
        seed = x
        if errors:
            failed = 1
        port.leds = (failed << 7) | (passes & 127)
        yield from jtag_print(port, b"pass ")
        yield from jtag_print_hex(port, passes, 4)
        yield from jtag_print(port, b": ")
        yield from jtag_print_hex(port, errors, 6)
        yield from jtag_print(port, b" words wrong\n")


TEST_ATTR = {"x": 32, "seed": 32, "errors": 24, "passes": 16, "tx_room": 1, "req_ready": 1, "resp_valid": 1, "resp_data": 32}

# On the board: the SDRAM, the JTAG UART for the console, the LEDs
PINS = dict(SDRAM_PINS, tx_data="JTAG_TX_DATA", tx_send="JTAG_TX_VALID",
            tx_room="JTAG_TX_READY", leds="LED")


def sdram_test(words=WORDS, stride=1, timing=SdramTiming()):
    """The controller and the test, connected."""
    test = functools.partial(memory_test, words=words, stride=stride)
    controller = functools.partial(sdram_controller, timing=timing)
    return connect(sdram=(controller, SDRAM_ATTR), test=(test, TEST_ATTR), name="sdram_test",
                   wires={f"sdram.{n}": f"test.{n}" for n in ("req", "req_write", "req_address",
                                                               "req_data", "req_mask", "req_line")}
                   | {f"test.{n}": f"sdram.{n}" for n in ("req_ready", "resp_valid", "resp_data")})


def console(passes=1, chip=None):
    """A testbench: the design on the chip's model, until it has printed
    `passes` passes. Returns what it printed."""
    from py2ver.protocol import alongside
    chip = chip or SdramModel()

    def printed(dut):
        dut.tx_room = 1
        text = bytearray()
        while text.count(b"\n") < passes + 1:
            if dut.tx_send:
                text.append(dut.tx_data)
            yield
        return bytes(text)

    def tb(dut):
        text = yield from alongside(printed(dut), chip.run(dut))
        return text
    return tb


# A short start (the power-up wait, the refreshes) for testbenches
QUICK = SdramTiming(power_up=20, refresh=60)

if __name__ == "__main__":
    from py2ver import Py2ver

    p = Py2ver(sdram_test(words=40, stride=211_001, timing=QUICK))
    run = p.check(console(), verilog="--sim" in sys.argv[1:])
    print(run.value.decode(), f"({len(run.inputs):,} clock cycles)")
    if "--pins" in sys.argv[1:]:
        board = Py2ver(sdram_test(), pins=PINS)
        print(board.PINS({"QUARTUS_DIR": os.environ.get(
            "QUARTUS_DIR", "/opt/intelFPGA/22.1std/quartus/bin")}))
