"""
The ADC128S022 analog-to-digital converter: 8 inputs, 12 bits, on SPI.

It uses SPI mode 3 (SCLK idles high) at 0.8 to 3.2 MHz. Each frame is 16 SCLK
pulses with CS low. In a frame the chip is told which input to convert next,
and sends back the result for the input chosen in the frame before. So reading
one input takes two frames, which adc128s022_read() does for you:

    ADC = ADC128S022()                                  # 2.5 MHz SCLK at 50 MHz
    value = yield from adc128s022_read(port, ADC, 5)    # input 5

Give its pins their idle levels from reset in attr:
{'cs_n': {'init': 1}, 'sclk': {'init': 1}}.
"""
from typing import NamedTuple

from ..protocol import SPI, spi_transfer, wait_until


class ADC128S022(NamedTuple):
    """The ADC's settings: its SPI bus and the name of its chip-select pin.

    The default bus runs SCLK at 2.5 MHz from a 50 MHz clock (half=10).
    """
    spi: SPI = SPI(10, mode=3)
    cs_n: str = "cs_n"


def adc128s022_frame(port, adc, channel):
    """Run one frame: choose `channel` for the next frame, and return what came back.

    What comes back is 16 bits: 4 zeros, then the 12-bit result for the
    channel chosen in the previous frame. If the top 4 bits aren't zero, the
    frame was out of step.
    """
    setattr(port, adc.cs_n, 0)
    yield adc.spi.half
    received = yield from spi_transfer(port, adc.spi, channel << 11, 16)
    setattr(port, adc.cs_n, 1)
    yield adc.spi.half
    return received


def adc128s022_read(port, adc, channel):
    """Read one input: choose it in one frame, and take its result from the next."""
    yield from adc128s022_frame(port, adc, channel)
    value = yield from adc128s022_frame(port, adc, channel)
    return value


# ---------------------------------------------------------------------------
# A model of the chip, for testbenches
# ---------------------------------------------------------------------------

def adc128s022_model(values, frames, adc=ADC128S022()):
    """A testbench step that plays the ADC, with `values` on its 8 inputs.

    Like the real chip, it answers each frame with the result for the
    channel chosen in the frame before, and it stops after `frames` frames.
    """
    sclk, mosi, miso = adc.spi.sclk, adc.spi.mosi, adc.spi.miso

    def chip(dut):
        chosen = 0
        setattr(dut, miso, 0)
        for _ in range(frames):
            yield from wait_until(dut, adc.cs_n, 0)
            word = values[chosen] & 0xFFF
            setattr(dut, miso, (word >> 15) & 1)              # the first bit goes out as CS falls
            command = 0
            for i in range(16):
                yield from wait_until(dut, sclk, 0)           # on each falling edge: the next bit out
                if i:
                    setattr(dut, miso, (word >> (15 - i)) & 1)
                yield from wait_until(dut, sclk, 1)           # on each rising edge: a bit in
                command = (command << 1) | getattr(dut, mosi)
            chosen = (command >> 11) & 7
            yield from wait_until(dut, adc.cs_n, 1)
    return chip
