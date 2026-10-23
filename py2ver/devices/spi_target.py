"""
Helpers for models of SPI chips, for testbenches, and for looking at a picture.

spi_listen() does the bit-level work of a chip that listens on SPI (a display,
a DAC, an LED driver): it watches the bus one clock cycle at a time and hands
each received word to the model.
"""
import struct
import zlib


def spi_listen(dut, bus, cs_n, on_word, bits=8, sample=()):
    """A testbench step that listens on the bus forever, and calls on_word for each word.

    bus:     the SPI record (its mode says which clock edge to read on).
    cs_n:    the name of the chip-select pin; while it's high, nothing is read,
             and a word cut short is dropped.
    on_word: called as on_word(word, pins) after each `bits` bits, with pins
             the values of the pins named in `sample` at that moment (a
             display's D/C pin, say).
    """
    read_on_rising = bus.mode in (0, 3)          # mode 0 and 3: rising edge; 1 and 2: falling
    word, count = 0, 0
    before = getattr(dut, bus.sclk)
    while True:
        yield
        now = getattr(dut, bus.sclk)
        if getattr(dut, cs_n) == 1:
            word, count = 0, 0
        elif now != before and now == (1 if read_on_rising else 0):
            word = (word << 1) | getattr(dut, bus.mosi)
            count += 1
            if count == bits:
                on_word(word, {name: getattr(dut, name) for name in sample})
                word, count = 0, 0
        before = now


def save_png(path, pixels, to_rgb):
    """Save rows of pixels as a PNG image; to_rgb turns a pixel into (red, green, blue), 0..255."""
    height, width = len(pixels), len(pixels[0]) if pixels else 0
    raw = b"".join(b"\x00" + bytes(c for p in row for c in to_rgb(p)) for row in pixels)

    def chunk(kind, body):
        return (struct.pack(">I", len(body)) + kind + body
                + struct.pack(">I", zlib.crc32(kind + body)))
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n"
                + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
