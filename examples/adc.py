# The DE0-Nano's ADC (an ADC128S022: 8 inputs, 12 bits) over SPI, read with
# py2ver.devices.adc128s022's driver, its channels printed on the JTAG console
# (see docs/processes.md, the guide).
#
#   python examples/adc.py              # the testbench (a model of the ADC)
#   python examples/adc.py --pins       # compiled for the board (not programmed)
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py2ver.devices.adc128s022 import (ADC128S022, adc128s022_idle,  # noqa: E402
                                       adc128s022_model, adc128s022_read)
from py2ver.protocol import jtag_print, jtag_print_hex  # noqa: E402

ADC = ADC128S022()                       # 2.5 MHz SCLK at 50 MHz (the ADC: 0.8 to 3.2)
PERIOD = 10_000_000                      # 0.2 s at 50 MHz between sweeps


def adc_reader(port, adc=ADC, period=PERIOD):
    """Every period, each of the 8 inputs read and printed (as 4 hex digits:
    the first, the frame's 4 leading zeros, always 0)."""
    yield from adc128s022_idle(port, adc)
    yield 1000
    yield from jtag_print(port, b"adc128s022\n")
    while True:
        for ch in range(8):
            value = yield from adc128s022_read(port, adc, ch)
            yield from jtag_print_hex(port, ch, 1)
            yield from jtag_print(port, b":")
            yield from jtag_print_hex(port, value, 4)
            yield from jtag_print(port, b" ")
        yield from jtag_print(port, b"\n")
        yield period


ATTR = {"miso": 1, "tx_room": 1}         # the inputs' widths

# On the board: the ADC's own pins, and the JTAG UART's for the console
PINS = {"cs_n": "ADC_CS_N", "sclk": "ADC_SCLK", "mosi": "ADC_SADDR", "miso": "ADC_SDAT",
        "tx_data": "JTAG_TX_DATA", "tx_send": "JTAG_TX_VALID", "tx_room": "JTAG_TX_READY"}


def sweep(values, adc=ADC):
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
        text, _ = yield from together(printed(dut), adc128s022_model(values, 16, adc)(dut))
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
    from py2ver.protocol import SPI

    values = [0x000, 0x123, 0x456, 0x789, 0xABC, 0xDEF, 0x800, 0xFFF]
    fast = ADC128S022(SPI(2, mode=3))
    p = Py2ver(functools.partial(adc_reader, adc=fast, period=100), ATTR)
    print(p.check(sweep(values, fast), verilog="--sim" in sys.argv[1:]).value.decode())
    if "--pins" in sys.argv[1:]:
        print(Py2ver(adc_reader, ATTR, pins=PINS).PINS({"QUARTUS_DIR": os.environ.get(
            "QUARTUS_DIR", "/opt/intelFPGA/22.1std/quartus/bin")}))
