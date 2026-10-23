"""
The ILI9341 display controller: a 240 x 320 color TFT, written to over SPI.

Besides the SPI clock and data, the chip has a chip select (CS), a reset pin,
and a D/C pin that says whether a byte is a command (low) or data (high).
Colors are 16-bit RGB565: 5 bits of red, 6 of green, 5 of blue (rgb565()).

    LCD = ILI9341()                                        # pins sclk, mosi, cs_n, dc, reset_n
    yield from ili9341_init(port, LCD)                     # reset, wake up, switch on
    yield from ili9341_fill(port, LCD, 0, 0, 239, 319, rgb565(0, 0, 255))   # all blue

To draw pixel by pixel, open a window and send its pixels, left to right and
top to bottom:

    yield from ili9341_window(port, LCD, x0, y0, x1, y1)
    for ...:
        yield from ili9341_pixel(port, LCD, color)

Called first thing, ili9341_init() also gives the pins their levels from reset:
the chip not selected, and held in reset until it starts.
"""
from typing import NamedTuple

from ..protocol import SPI, spi_transfer, stream_receive
from .spi_target import save_png, spi_listen

WIDTH, HEIGHT = 240, 320

# Commands
SWRESET = 0x01          # software reset
SLPOUT = 0x11           # sleep out: wake up (then wait 120 ms)
DISPON = 0x29           # display on
CASET = 0x2A            # column address set: the window's x0 and x1
PASET = 0x2B            # page address set: the window's y0 and y1
RAMWR = 0x2C            # memory write: the pixels that follow fill the window
MADCTL = 0x36           # memory access control: orientation and color order
COLMOD = 0x3A           # pixel format

# The panel's power, voltage and gamma settings, sent before waking it up (the
# sequence the vendors' and Adafruit's drivers use; without it some modules
# show nothing but their backlight). Each entry: a command, how many parameter
# bytes follow, then those bytes.
SETUP = (
    0xEF, 3, 0x03, 0x80, 0x02,
    0xCF, 3, 0x00, 0xC1, 0x30,              # power control B
    0xED, 4, 0x64, 0x03, 0x12, 0x81,        # power-on sequence control
    0xE8, 3, 0x85, 0x00, 0x78,              # driver timing control A
    0xCB, 5, 0x39, 0x2C, 0x00, 0x34, 0x02,  # power control A
    0xF7, 1, 0x20,                          # pump ratio control
    0xEA, 2, 0x00, 0x00,                    # driver timing control B
    0xC0, 1, 0x23,                          # power control 1: GVDD 4.6 V
    0xC1, 1, 0x10,                          # power control 2
    0xC5, 2, 0x3E, 0x28,                    # VCOM control 1
    0xC7, 1, 0x86,                          # VCOM control 2
    0x37, 1, 0x00,                          # vertical scroll start: 0
    0xB1, 2, 0x00, 0x18,                    # frame rate: 79 Hz
    0xB6, 3, 0x08, 0x82, 0x27,              # display function control
    0xF2, 1, 0x00,                          # 3-gamma off
    0x26, 1, 0x01,                          # gamma curve 1
    0xE0, 15, 0x0F, 0x31, 0x2B, 0x0C, 0x0E, 0x08, 0x4E, 0xF1, 0x37, 0x07, 0x10, 0x03,
    0x0E, 0x09, 0x00,                       # positive gamma
    0xE1, 15, 0x00, 0x0E, 0x14, 0x03, 0x11, 0x07, 0x31, 0xC1, 0x48, 0x08, 0x0F, 0x0C,
    0x31, 0x36, 0x0F,                       # negative gamma
)


class ILI9341(NamedTuple):
    """The display's settings.

    spi:          its SPI bus. It only listens (no MISO), at up to 10 MHz; the
                  default runs SCLK at 8.3 MHz from a 50 MHz clock.
    cs_n, dc, reset_n:  the names of its other pins.
    madctl:       orientation and color order (0x48: portrait, BGR panels as most are).
    wake_clocks:  how long the reset pulse lasts, and how long to wait after it, after
                  waking up and after switching on: 150 ms at 50 MHz (as Adafruit's
                  driver waits; the datasheet asks for 120 ms, some panels need more).
    """
    spi: SPI = SPI(3, mode=0, miso="")
    cs_n: str = "cs_n"
    dc: str = "dc"
    reset_n: str = "reset_n"
    madctl: int = 0x48
    wake_clocks: int = 7_500_000


def rgb565(red, green, blue):
    """A color from 8-bit red, green and blue, as the display takes it."""
    return ((red >> 3) << 11) | ((green >> 2) << 5) | (blue >> 3)


def ili9341_command(port, lcd, command):
    """Send a command byte (D/C low)."""
    setattr(port, lcd.dc, 0)
    setattr(port, lcd.cs_n, 0)
    yield from spi_transfer(port, lcd.spi, command, 8)
    setattr(port, lcd.cs_n, 1)
    yield lcd.spi.half


def ili9341_data(port, lcd, value, bits=8):
    """Send `bits` bits of data (D/C high): a command's parameters, or pixels."""
    setattr(port, lcd.dc, 1)
    setattr(port, lcd.cs_n, 0)
    yield from spi_transfer(port, lcd.spi, value, bits)
    setattr(port, lcd.cs_n, 1)
    yield lcd.spi.half


def ili9341_init(port, lcd):
    """Reset the display, set up its panel, wake it up, set 16-bit color, and switch it on."""
    setattr(port, lcd.cs_n, 1)
    setattr(port, lcd.spi.sclk, lcd.spi.mode >> 1)
    setattr(port, lcd.reset_n, 0)                   # a hardware reset
    yield lcd.wake_clocks
    setattr(port, lcd.reset_n, 1)
    yield lcd.wake_clocks
    k = 0
    while k < len(SETUP):                           # the panel's settings, from the table
        yield from ili9341_command(port, lcd, SETUP[k])
        count = SETUP[k + 1]
        k = k + 2
        while count > 0 and k < len(SETUP):
            yield from ili9341_data(port, lcd, SETUP[k])
            k = k + 1
            count = count - 1
    yield from ili9341_command(port, lcd, MADCTL)
    yield from ili9341_data(port, lcd, lcd.madctl)
    yield from ili9341_command(port, lcd, COLMOD)
    yield from ili9341_data(port, lcd, 0x55)        # 16 bits a pixel
    yield from ili9341_command(port, lcd, SLPOUT)
    yield lcd.wake_clocks
    yield from ili9341_command(port, lcd, DISPON)
    yield lcd.wake_clocks


def ili9341_window(port, lcd, x0, y0, x1, y1):
    """Choose the rectangle the next pixels go to (corners included), and start writing.

    The pixels then fill it left to right, top to bottom.
    """
    yield from ili9341_command(port, lcd, CASET)
    yield from ili9341_data(port, lcd, (x0 << 16) | x1, 32)
    yield from ili9341_command(port, lcd, PASET)
    yield from ili9341_data(port, lcd, (y0 << 16) | y1, 32)
    yield from ili9341_command(port, lcd, RAMWR)


def ili9341_pixel(port, lcd, color):
    """Send the next pixel of the window."""
    yield from ili9341_data(port, lcd, color, 16)


def ili9341_fill(port, lcd, x0, y0, x1, y1, color):
    """Fill a rectangle (corners included) with one color."""
    yield from ili9341_window(port, lcd, x0, y0, x1, y1)
    remaining = (x1 - x0 + 1) * (y1 - y0 + 1)
    setattr(port, lcd.dc, 1)
    setattr(port, lcd.cs_n, 0)                      # one long write: every pixel in a row
    while remaining > 0:
        yield from spi_transfer(port, lcd.spi, color, 16)
        remaining = remaining - 1
    setattr(port, lcd.cs_n, 1)
    yield lcd.spi.half


def ili9341_display(port, lcd=ILI9341(), width=WIDTH, height=HEIGHT):
    """A process that shows a stream of pixels: connect() it after a block that
    sends them (see protocol.stream_send).

    It wakes the display, then takes width x height colors from its input
    stream (color, valid; it answers on ready), row by row from the top left,
    and sends them to the screen; then the next picture, and so on.
    """
    port.ready = 0
    yield from ili9341_init(port, lcd)
    while True:
        yield from ili9341_window(port, lcd, 0, 0, width - 1, height - 1)
        setattr(port, lcd.dc, 1)
        setattr(port, lcd.cs_n, 0)                  # one long write: the whole picture
        remaining = width * height
        while remaining > 0:
            color = yield from stream_receive(port, data="color")
            yield from spi_transfer(port, lcd.spi, color, 16)
            remaining = remaining - 1
        setattr(port, lcd.cs_n, 1)
        yield lcd.spi.half


# ---------------------------------------------------------------------------
# A model of the display, for testbenches
# ---------------------------------------------------------------------------

class ILI9341Model:
    """Plays the display: decodes what the design sends, and keeps the picture.

        screen = ILI9341Model()
        def tb(dut):
            yield from alongside(wait_until(dut, "done"), screen.run(dut))
        ...
        screen.pixels[y][x]          # RGB565 colors
        screen.save("screen.png")    # the picture, to look at

    commands lists each command with its parameter bytes. awake and on say
    whether the design woke the display up and switched it on, and written
    counts the pixels written so far.
    """

    def __init__(self, lcd=ILI9341(), width=WIDTH, height=HEIGHT):
        self.lcd = lcd
        self.width, self.height = width, height
        self.pixels = [[0] * width for _ in range(height)]
        self.commands = []
        self.awake = False
        self.on = False
        self.written = 0
        self._window = (0, 0, width - 1, height - 1)
        self._x = self._y = 0
        self._half = None                           # the first byte of a pixel

    def run(self, dut):
        """A testbench step that listens to the bus forever (use it with alongside())."""
        yield from spi_listen(dut, self.lcd.spi, self.lcd.cs_n, self._byte, 8,
                              sample=(self.lcd.dc, self.lcd.reset_n))

    def _byte(self, byte, pins):
        if pins[self.lcd.reset_n] == 0:
            return                                  # held in reset: it hears nothing
        if pins[self.lcd.dc] == 0:
            self.commands.append((byte, []))
            self._half = None
            if byte == SLPOUT:
                self.awake = True
            elif byte == DISPON:
                self.on = True
            elif byte == RAMWR:
                self._x, self._y = self._window[0], self._window[1]
            return
        if not self.commands:
            return                                  # data before any command: ignored
        command, params = self.commands[-1]
        if command != RAMWR:
            params.append(byte)
            if len(params) == 4 and command in (CASET, PASET):
                lo, hi = (params[0] << 8) | params[1], (params[2] << 8) | params[3]
                x0, y0, x1, y1 = self._window
                self._window = (lo, y0, hi, y1) if command == CASET else (x0, lo, x1, hi)
            return
        if self._half is None:
            self._half = byte
            return
        color, self._half = (self._half << 8) | byte, None
        self.written += 1
        x0, y0, x1, y1 = self._window
        if 0 <= self._y < self.height and 0 <= self._x < self.width and self._y <= y1:
            self.pixels[self._y][self._x] = color
        self._x += 1
        if self._x > x1:
            self._x, self._y = x0, self._y + 1

    def save(self, path, scale=1):
        """Save the picture as a PNG image, each pixel `scale` times bigger."""
        rows = [[c for c in row for _ in range(scale)] for row in self.pixels for _ in range(scale)]
        save_png(path, rows, rgb888)


def rgb888(color):
    """An RGB565 color as 8-bit (red, green, blue)."""
    return (((color >> 11) & 31) * 255 // 31, ((color >> 5) & 63) * 255 // 63,
            (color & 31) * 255 // 31)
