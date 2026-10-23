# Doom's fire on an ILI9341 display: flames that rise from the bottom of the
# screen, flicker and blow about. KEY1 puts the fire out, and lights it again.
#
# The fire is a list of heats, one for each 4 x 4 block of the screen, from 0
# (black) to 36 (white): 60 x 80 blocks, in a memory block. Each frame, every
# block takes the heat of the one below it, a little cooler and blown a little
# sideways at random, so the heat rises from the hot bottom row and fades on
# its way up. Then each pixel gets its block's color, from the palette of the
# fire in Doom's PlayStation version (as Fabien Sanglard wrote it up).
#
#   doom_fire --color/valid/ready--> ili9341_display
#
# A list in a memory block is read a clock at a time: `x = heat[i]`, then a
# yield, then x is used (see docs/processes.md, Lists).
#
#   python examples/fire.py          # a few small frames: test them, save fire.png
#   python examples/fire.py --sim    # also simulate the generated Verilog
#   python examples/fire.py --pins   # build it for the board (wiring: mandelbrot_lcd.py)
import functools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mandelbrot_lcd import PINS as LCD_PINS  # noqa: E402
from py2ver import connect  # noqa: E402
from py2ver.devices.ili9341 import HEIGHT, ILI9341, WIDTH, ili9341_display, rgb565  # noqa: E402
from py2ver.protocol import SPI, stream_send  # noqa: E402

# The fire's colors, from black through red, orange and yellow to white
DOOM_FIRE = ((7, 7, 7), (31, 7, 7), (47, 15, 7), (71, 15, 7), (87, 23, 7), (103, 31, 7),
             (119, 31, 7), (143, 39, 7), (159, 47, 7), (175, 63, 7), (191, 71, 7), (199, 71, 7),
             (223, 79, 7), (223, 87, 7), (223, 87, 7), (215, 95, 7), (215, 95, 7), (215, 103, 15),
             (207, 111, 15), (207, 119, 15), (207, 127, 15), (207, 135, 23), (199, 135, 23),
             (199, 143, 23), (199, 151, 31), (191, 159, 31), (191, 159, 31), (191, 167, 39),
             (191, 167, 39), (191, 175, 47), (183, 175, 47), (183, 183, 47), (183, 183, 55),
             (207, 207, 111), (223, 223, 159), (239, 239, 199), (255, 255, 255))
FIRE = tuple(rgb565(*c) for c in DOOM_FIRE)
HOT = len(FIRE) - 1                      # the hottest heat: white


def doom_fire(port, across=WIDTH >> 2, down=HEIGHT >> 2, zoom=2, colors=FIRE):
    """The fire, frame after frame, sending each pixel's color.

    across, down: the fire's blocks across the screen and down it; each block
    2**zoom pixels square (the screen across << zoom by down << zoom pixels).
    Input: key (0 while pressed: the fire out, or lit again); ready.
    Output stream: color (RGB565), valid (see protocol.stream_send).
    """
    port.valid = 0
    heat = [0] * (across * down)                 # each block's heat, row by row
    seed = 1                                     # (the random numbers' state)
    burning = 1
    was = 1
    while True:
        key = port.key                           # pressed (just now): out, or lit again
        if was == 1 and key == 0:
            burning = 1 - burning
        was = key
        # The fire's source: the bottom row, hot while the fire burns
        for x in range(across):
            heat[(down - 1) * across + x] = HOT * burning
            yield
        # Each block passes its heat to the one above it: a little cooler (one
        # step, half the time) and blown up to a block sideways
        for y in range(1, down):
            for x in range(across):
                block = y * across + x
                here = heat[block]
                yield
                seed = seed ^ ((seed << 7) & 0xFFFF)     # (a step of xorshift)
                seed = seed ^ (seed >> 9)
                seed = seed ^ ((seed << 8) & 0xFFFF)
                wind = seed & 3
                if here == 0:
                    heat[block - across] = 0
                else:
                    heat[block - across - wind + 1] = here - (wind & 1)
                yield
        # The picture: each pixel its block's color
        for row in range(down << zoom):
            for col in range(across << zoom):
                level = heat[(row >> zoom) * across + (col >> zoom)]
                yield
                yield from stream_send(port, colors[level], data="color")


ATTR = {"ready": 1, "key": 1}

# SCK at 25 MHz, as in mandelbrot_zoom.py (if the picture is noisy: SPI(2, ...))
LCD = ILI9341(spi=SPI(1, mode=0, miso=""))

# On the board: the display's pins, and KEY1 to put the fire out and light it
PINS = dict(LCD_PINS, key="KEY[1]")


def show_on_screen(lcd=LCD, width=WIDTH, height=HEIGHT, zoom=2):
    """The fire feeding the display."""
    source = functools.partial(doom_fire, across=width >> zoom, down=height >> zoom, zoom=zoom)
    display = functools.partial(ili9341_display, lcd=lcd, width=width, height=height)
    return connect(fire=(source, ATTR), display=(display, {"color": 16, "valid": 1}),
                   name="doom_fire",
                   wires={"display.color": "fire.color", "display.valid": "fire.valid",
                          "fire.ready": "display.ready"})


def frames(screen, count, width, height, press_at=None):
    """A testbench: run until the display has received `count` frames; a copy of
    the picture after each. The key is pressed while frame `press_at` is drawn."""
    from py2ver.protocol import alongside
    shots = []

    def watch(dut):
        while len(shots) < count:
            dut.key = 0 if len(shots) == press_at else 1      # (the board's keys: 0 pressed)
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
    # A small screen for the simulation: 32 x 32 pixels, 8 x 8 blocks of fire;
    # KEY1 pressed in the 7th frame (the fire goes out)
    w, h, count = 32, 32, 10
    quick = ILI9341(wake_clocks=50)
    screen = ILI9341Model(quick, width=w, height=h)
    p = Py2ver(show_on_screen(quick, w, h))
    run = p.check(frames(screen, count, w, h, press_at=6), verilog="--sim" in args)
    save_png("fire.png", film_strip(run.value, scale=4), rgb888)
    print(f"{count} frames of {w} x {h} in {len(run.inputs):,} clock cycles: fire.png")
    if "--pins" in args:
        board = Py2ver(show_on_screen(), pins=PINS)
        print(board.PINS({"QUARTUS_DIR": os.environ.get(
            "QUARTUS_DIR", "/opt/intelFPGA/22.1std/quartus/bin")}))
