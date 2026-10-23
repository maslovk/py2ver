# A fractal show on an ILI9341 display, in two modes; KEY1 switches between them:
#
# - the Mandelbrot set, zooming in on a point and back out, again and again;
# - Julia sets morphing: the set for a c that moves round a circle, a step a frame.
#
# Both share one escape loop (z from 0 and c the pixel for the Mandelbrot set; z
# the pixel and c the frame's for a Julia set), and the colors cycle along the
# palette, so that the bands seem to flow.
#
#   fractal_show (examples/mandelbrot.py) --color/valid/ready--> ili9341_display
#
# The zoom changes 3% a frame, in to 2,000x (about 240 frames) and back; the
# Julia path has 360 frames. With SPI at 25 MHz: 10 to 17 frames a second.
#
#   python examples/mandelbrot_zoom.py          # a few frames of each mode, small: mandelbrot_zoom.png
#   python examples/mandelbrot_zoom.py --sim    # also simulate the generated Verilog
#   python examples/mandelbrot_zoom.py --pins   # build it for the board (wiring: mandelbrot_lcd.py)
import functools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mandelbrot as M  # noqa: E402
from mandelbrot_lcd import PINS as LCD_PINS  # noqa: E402
from py2ver import connect  # noqa: E402
from py2ver.devices.ili9341 import HEIGHT, ILI9341, WIDTH, ili9341_display  # noqa: E402
from py2ver.protocol import SPI  # noqa: E402

# SCK at 25 MHz: beyond the datasheet's 10 MHz for writes, as Adafruit's driver
# runs these modules (24 MHz); checked on the board with short jumper wires. If
# the picture is noisy, use SPI(2, ...): 12.5 MHz.
LCD = ILI9341(spi=SPI(1, mode=0, miso=""))

# On the board: the display's pins, and KEY1 to switch modes
PINS = dict(LCD_PINS, key="KEY[1]")


def show_on_screen(lcd=LCD, area=M.ZOOM, morph=M.JULIA, width=WIDTH, height=HEIGHT,
                   max_iter=M.MAX_ITER, cycle=1):
    """The fractal show feeding the display; its colors cycling `cycle` places a frame."""
    fractal = functools.partial(M.fractal_show, area=area, morph=morph, width=width,
                                height=height, max_iter=max_iter,
                                colors=M.cyclic_palette(max_iter), cycle=cycle)
    display = functools.partial(ili9341_display, lcd=lcd, width=width, height=height)
    return connect(fractal=(fractal, M.ZOOM_ATTR), display=(display, {"color": 16, "valid": 1}),
                   name="fractal_show",
                   wires={"display.color": "fractal.color", "display.valid": "fractal.valid",
                          "fractal.ready": "display.ready"})


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


def film_strip(shots, scale=8, gap=4):
    """Frames side by side, each pixel scale x scale, a white gap between them."""
    rows = []
    for row in range(len(shots[0])):
        line = []
        for shot in shots:
            line += [color for color in shot[row] for _ in range(scale)] + [0xFFFF] * gap
        rows += [line] * scale
    return rows


if __name__ == "__main__":
    from py2ver import Py2ver
    from py2ver.devices.ili9341 import ILI9341Model, rgb888
    from py2ver.devices.spi_target import save_png

    args = sys.argv[1:]
    # A small screen for the simulation: 24 x 18 pixels, a fast zoom (half each
    # frame) and a short Julia path (12 frames round); KEY1 pressed in frame 3
    w, h, iters, count = 24, 18, 32, 7
    quick = ILI9341(wake_clocks=50)
    area = M.zoom(start_width=3.2, end_width=0.01, width=w)._replace(shrink=1)
    morph = M.julia(frames=12, width=w, height=h)
    screen = ILI9341Model(quick, width=w, height=h)
    p = Py2ver(show_on_screen(quick, area, morph, w, h, iters))
    run = p.check(frames(screen, count, w, h, press_at=3), verilog="--sim" in args)
    save_png("mandelbrot_zoom.png", film_strip(run.value), rgb888)
    print(f"{count} frames of {w} x {h} in {len(run.inputs):,} clock cycles (KEY1 pressed "
          f"in the 4th): mandelbrot_zoom.png")
    if "--pins" in args:
        board = Py2ver(show_on_screen(), pins=PINS)
        print(board.PINS({"QUARTUS_DIR": os.environ.get(
            "QUARTUS_DIR", "/opt/intelFPGA/22.1std/quartus/bin")}))
