"""
The ADC128S022: 8 inputs of 12 bits, on SPI (mode 3: SCLK idle high, data out
on its falling edge, taken on its rising), 0.8 to 3.2 MHz. In a frame (CS low,
16 clocks of SCLK) it takes the next channel's number in bits 13..11 of the
word it's sent, and sends back 4 zeros and the 12-bit result of the channel
the frame before chose.

    ADC = ADC128S022()                                # 2.5 MHz at 50 MHz; sclk, mosi, miso, cs_n
    value = yield from adc128s022_read(port, ADC, 5)  # input 5

Its pins idle from reset (attr: {'cs_n': {'init': 1}, 'sclk': {'init': 1}}).
"""
from typing import NamedTuple

from ..protocol import SPI, spi_transfer, wait_until


class ADC128S022(NamedTuple):
    """The ADC's settings: its SPI bus (mode 3; half: 10 clocks, 2.5 MHz at 50
    MHz), and its chip select's pin."""
    spi: SPI = SPI(10, mode=3)
    cs_n: str = "cs_n"


def adc128s022_frame(port, adc, channel):
    """One frame: channel chosen for the next; the 16 bits that came back (4
    zeros, then the result of the channel chosen the frame before: the zeros
    say the frame was in step)."""
    setattr(port, adc.cs_n, 0)
    yield adc.spi.half
    got = yield from spi_transfer(port, adc.spi, channel << 11, 16)
    setattr(port, adc.cs_n, 1)
    yield adc.spi.half
    return got


def adc128s022_read(port, adc, channel):
    """The channel read: chosen in a frame, its result the next's."""
    yield from adc128s022_frame(port, adc, channel)
    value = yield from adc128s022_frame(port, adc, channel)
    return value


def adc128s022_model(values, frames, adc=ADC128S022()):
    """A testbench step playing the ADC with these 8 inputs' values, for frames
    frames: in each, the result of the channel chosen the frame before out on
    MISO (4 zeros first, a bit at each falling edge, the first as CS falls), the
    next channel taken from MOSI at the rising edges."""
    sclk, mosi, miso = adc.spi.sclk, adc.spi.mosi, adc.spi.miso

    def chip(dut):
        chosen = 0
        setattr(dut, miso, 0)
        for _ in range(frames):
            yield from wait_until(dut, adc.cs_n, 0)
            word = values[chosen] & 0xFFF
            setattr(dut, miso, (word >> 15) & 1)                # (the first, as CS falls)
            got = 0
            for i in range(16):
                yield from wait_until(dut, sclk, 0)             # falling: the next bit
                if i:
                    setattr(dut, miso, (word >> (15 - i)) & 1)
                yield from wait_until(dut, sclk, 1)             # rising: taken
                got = (got << 1) | getattr(dut, mosi)
            chosen = (got >> 11) & 7
            yield from wait_until(dut, adc.cs_n, 1)
    return chip
