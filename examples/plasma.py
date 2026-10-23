# A plasma on an ILI9341 display: colors that flow like liquid, from four waves
# added up at each pixel.
#
# Each pixel's color comes from four sine waves: one across the screen, one
# down it, one along the diagonal, and rings around a center that wanders
# round the screen. Every frame the waves move on and the colors shift along
# the palette, so that the picture flows.
#
#   plasma --color/valid/ready--> ili9341_display
#
# The sine is a table of 256 entries and the palette another; both are read
# by memory blocks (each entry is read into a local, then used after a yield).
# With SPI at 25 MHz the display takes a frame in 50 ms: 20 frames a second.
#
#   python examples/plasma.py          # a few small frames: test them, save plasma.png
#   python examples/plasma.py --sim    # also simulate the generated Verilog
#   python examples/plasma.py --pins   # build it for the board (wiring: mandelbrot_lcd.py)
import functools
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mandelbrot_lcd import PINS  # noqa: E402
from py2ver import connect  # noqa: E402
from py2ver.devices.ili9341 import HEIGHT, ILI9341, WIDTH, ili9341_display, rgb565  # noqa: E402
from py2ver.protocol import SPI, stream_send  # noqa: E402

# A sine wave, a whole turn in 256 steps, from 0 to 63
SINE = tuple(round(31.5 + 31.5 * math.sin(2 * math.pi * i / 256)) for i in range(256))


def rainbow(size=256):
    """A palette that goes round the rainbow and back to where it began, so
    that shifting the colors along shows no seam."""
    colors = []
    for i in range(size):
        turn = 2 * math.pi * i / size
        r, g, b = (round(127.5 + 127.5 * math.cos(turn - shift))
                   for shift in (0, 2 * math.pi / 3, 4 * math.pi / 3))
        colors.append(rgb565(r, g, b))
    return tuple(colors)


RAINBOW = rainbow()


def sine(port, phase):
    """SINE at phase (any whole number: a turn is 256), a clock later. The entry
    is used after the yield, so the table is read by a memory block."""
    value = SINE[phase & 255]
    yield
    return value


def plasma(port, width=WIDTH, height=HEIGHT, speed=3, colors=RAINBOW):
    """The plasma, frame after frame, sending each pixel's color.

    speed: how far the waves move each frame (256 is a whole turn).
    colors: a palette of 256 colors (RGB565).
    Output stream: color, valid; input: ready (see protocol.stream_send).
    """
    port.valid = 0
    t = 0                                       # the time: how far the waves have moved
    while True:
        # The rings' center for this frame, wandering round the screen
        sx = yield from sine(port, t)
        sy = yield from sine(port, 2 * t + 64)
        cx = (sx * width) >> 6
        cy = (sy * height) >> 6
        for row in range(height):
            for col in range(width):
                across = yield from sine(port, 2 * col + t)
                down = yield from sine(port, 3 * row - 2 * t)
                diagonal = yield from sine(port, col + row + 3 * t)
                dx = col - cx
                dy = row - cy
                rings = yield from sine(port, ((dx * dx + dy * dy) >> 7) - 4 * t)
                color = colors[(across + down + diagonal + rings + t) & 255]
                yield
                yield from stream_send(port, color, data="color")
        t = (t + speed) & 255


ATTR = {"ready": 1}

# SCK at 25 MHz, as in mandelbrot_zoom.py (if the picture is noisy: SPI(2, ...))
LCD = ILI9341(spi=SPI(1, mode=0, miso=""))


def show_on_screen(lcd=LCD, width=WIDTH, height=HEIGHT, speed=3):
    """The plasma feeding the display."""
    source = functools.partial(plasma, width=width, height=height, speed=speed)
    display = functools.partial(ili9341_display, lcd=lcd, width=width, height=height)
    return connect(plasma=(source, ATTR), display=(display, {"color": 16, "valid": 1}),
                   name="plasma_show",
                   wires={"display.color": "plasma.color", "display.valid": "plasma.valid",
                          "plasma.ready": "display.ready"})


def picture(t, width, height, colors=RAINBOW):
    """The frame at time t, worked out directly (for checking the stream)."""
    cx = (SINE[t & 255] * width) >> 6
    cy = (SINE[(2 * t + 64) & 255] * height) >> 6
    rows = []
    for row in range(height):
        line = []
        for col in range(width):
            rings = (((col - cx) ** 2 + (row - cy) ** 2) >> 7) - 4 * t
            v = (SINE[(2 * col + t) & 255] + SINE[(3 * row - 2 * t) & 255]
                 + SINE[(col + row + 3 * t) & 255] + SINE[rings & 255] + t)
            line.append(colors[v & 255])
        rows.append(line)
    return rows


def frames(screen, count, width, height):
    """A testbench: run until the display has received `count` frames; a copy of
    the picture after each."""
    from py2ver.protocol import alongside
    shots = []

    def watch(dut):
        while len(shots) < count:
            if screen.written >= (len(shots) + 1) * width * height:
                shots.append([list(row) for row in screen.pixels])
            yield

    def tb(dut):
        yield from alongside(watch(dut), screen.run(dut))
        return shots
    return tb


if __name__ == "__main__":
    from mandelbrot_zoom import film_strip
    from py2ver import Py2ver
    from py2ver.devices.ili9341 import ILI9341Model, rgb888
    from py2ver.devices.spi_target import save_png

    args = sys.argv[1:]
    # A small screen for the simulation: 32 x 24 pixels, the waves moving fast
    w, h, count, speed = 32, 24, 4, 24
    quick = ILI9341(wake_clocks=50)
    screen = ILI9341Model(quick, width=w, height=h)
    p = Py2ver(show_on_screen(quick, w, h, speed))
    run = p.check(frames(screen, count, w, h), verilog="--sim" in args)
    assert run.value == [picture(k * speed, w, h) for k in range(count)]
    save_png("plasma.png", film_strip(run.value), rgb888)
    print(f"{count} frames of {w} x {h} in {len(run.inputs):,} clock cycles, as worked "
          f"out directly: plasma.png")
    if "--pins" in args:
        board = Py2ver(show_on_screen(), pins=PINS)
        print(board.PINS({"QUARTUS_DIR": os.environ.get(
            "QUARTUS_DIR", "/opt/intelFPGA/22.1std/quartus/bin")}))
