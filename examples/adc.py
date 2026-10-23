# The DE0-Nano's ADC (an ADC128S022: 8 inputs, 12 bits) over SPI, its channels
# printed on the JTAG console (see docs/processes.md, the guide).
#
# The ADC128S022's frame: CS low, then 16 clocks of SCLK (idle high: SPI mode 3,
# data out on the falling edge, taken on the rising). The ADC takes the next
# channel's number in bits 13..11 of the word it's sent, and sends back 4 zeros
# and the 12-bit result of the channel the frame before chose.
#
#   python examples/adc.py              # the testbench (a model of the ADC)
#   python examples/adc.py --pins       # compiled for the board (not programmed)
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py2ver.protocol import SPI, jtag_print, jtag_print_hex, spi_transfer  # noqa: E402

HALF = 10                                # 2.5 MHz SCLK at 50 MHz (the ADC: 0.8 to 3.2)
PERIOD = 10_000_000                      # 0.2 s at 50 MHz between sweeps


def adc_frame(port, channel, half):
    """One frame: channel chosen for the next one; the 16 bits read back (4 zeros,
    the result of the channel chosen before)."""
    port.cs_n = 0
    yield half
    got = yield from spi_transfer(port, SPI(half, mode=3), channel << 11, 16)   # (its pins: sclk, mosi, miso)
    port.cs_n = 1
    yield half
    return got


def adc_reader(port, half=HALF, period=PERIOD):
    """Every period, each of the 8 inputs read and printed (as 4 hex digits:
    the first, the frame's 4 leading zeros, always 0)."""
    port.cs_n = 1
    port.sclk = 1                        # (SPI mode 3: the clock idle high)
    yield 1000
    yield from jtag_print(port, b"adc128s022\n")
    while True:
        for ch in range(8):
            yield from adc_frame(port, ch, half)               # (ch chosen)
            value = yield from adc_frame(port, ch, half)       # (and its result)
            yield from jtag_print_hex(port, ch, 1)
            yield from jtag_print(port, b":")
            yield from jtag_print_hex(port, value, 4)
            yield from jtag_print(port, b" ")
        yield from jtag_print(port, b"\n")
        yield period


ATTR = {"miso": 1, "tx_room": 1, "cs_n": {"init": 1}, "sclk": {"init": 1}}   # (idle from reset)
PINS = dict(ATTR, cs_n={"init": 1, "pin": "ADC_CS_N"}, sclk={"init": 1, "pin": "ADC_SCLK"},
            mosi="ADC_SADDR", miso="ADC_SDAT", tx_data="JTAG_TX_DATA", tx_send="JTAG_TX_VALID",
            tx_room="JTAG_TX_READY")


def adc_chip(values, frames):
    """A testbench step playing an ADC128S022 with these 8 inputs' values: in each
    frame, the result of the channel chosen the frame before out on miso (4 zeros
    first, a bit at each falling edge, the first as CS falls), the next channel
    taken from mosi at the rising edges."""
    from py2ver.protocol import wait_until

    def chip(dut):
        chosen = 0
        dut.miso = 0
        for _ in range(frames):
            yield from wait_until(dut, "cs_n", 0)
            word = values[chosen] & 0xFFF
            dut.miso = (word >> 15) & 1                            # (the first, as CS falls)
            got = 0
            for i in range(16):
                yield from wait_until(dut, "sclk", 0)              # falling: the next bit
                if i:
                    dut.miso = (word >> (15 - i)) & 1
                yield from wait_until(dut, "sclk", 1)              # rising: taken
                got = (got << 1) | dut.mosi
            chosen = (got >> 11) & 7
            yield from wait_until(dut, "cs_n", 1)
    return chip


def sweep(values):
    """A testbench: the reader with a model of the ADC; what it printed for a sweep."""
    from py2ver.protocol import together

    def printed(dut):
        dut.tx_room = 1
        text = []
        while text.count(ord("\n")) < 2:
            if dut.tx_send:
                text.append(dut.tx_data)
            yield
        return bytes(text)

    def tb(dut):
        text, _ = yield from together(printed(dut), adc_chip(values, 16)(dut))
        return text
    return tb


def readings(text):
    """{channel: value} of each sweep line the reader printed."""
    out = []
    for line in text.decode(errors="replace").splitlines():
        if ":" in line:
            out.append({int(c, 16): int(v, 16) for c, v in (p.split(":") for p in line.split())})
    return out


if __name__ == "__main__":
    import functools
    from py2ver import Py2ver

    values = [0x000, 0x123, 0x456, 0x789, 0xABC, 0xDEF, 0x800, 0xFFF]
    p = Py2ver(functools.partial(adc_reader, half=2, period=100), ATTR)
    print(p.check(sweep(values), verilog="--sim" in sys.argv[1:]).value.decode())
    if "--pins" in sys.argv[1:]:
        print(Py2ver(adc_reader, PINS).PINS({"QUARTUS_DIR": os.environ.get(
            "QUARTUS_DIR", "/opt/intelFPGA/22.1std/quartus/bin")}))
