# A Mandelbrot animation on an ILI9341 display: zooming in on a point, frame
# after frame, and starting over.
#
#   mandelbrot_zoom (examples/mandelbrot.py) --color/valid/ready--> ili9341_display
#
# Each frame the distance between pixels shrinks by 1/16 (a 6% zoom), about 7
# frames a second at 8.3 MHz SPI, and 24 fraction bits allow a 30,000x zoom
# before starting again (about 160 frames).
#
#   python examples/mandelbrot_zoom.py          # a few frames on a small screen: mandelbrot_zoom.png
#   python examples/mandelbrot_zoom.py --sim    # also simulate the generated Verilog
#   python examples/mandelbrot_zoom.py --pins   # build it for the board (wiring: mandelbrot_lcd.py)
import functools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mandelbrot as M  # noqa: E402
from mandelbrot_lcd import PINS  # noqa: E402
from py2ver import connect  # noqa: E402
from py2ver.devices.ili9341 import HEIGHT, ILI9341, WIDTH, ili9341_display  # noqa: E402


def zoom_on_screen(lcd=ILI9341(), area=M.ZOOM, width=WIDTH, height=HEIGHT, max_iter=M.MAX_ITER):
    """The zooming generator feeding the display."""
    fractal = functools.partial(M.mandelbrot_zoom, area=area, width=width, height=height,
                                max_iter=max_iter, colors=M.palette(max_iter))
    display = functools.partial(ili9341_display, lcd=lcd, width=width, height=height)
    return connect(fractal=(fractal, M.ZOOM_ATTR), display=(display, {"color": 16, "valid": 1}),
                   name="mandelbrot_zoom",
                   wires={"display.color": "fractal.color", "display.valid": "fractal.valid",
                          "fractal.ready": "display.ready"})


def frames(screen, count, width, height):
    """A testbench: run until the display has received `count` frames; a copy of
    the picture after each."""
    from py2ver.protocol import alongside
    shots = []

    def watch():
        while len(shots) < count:
            if screen.written >= (len(shots) + 1) * width * height:
                shots.append([list(row) for row in screen.pixels])
            yield

    def tb(dut):
        yield from alongside(watch(), screen.run(dut))
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
    # A small screen for the simulation: 24 x 18 pixels, a fast zoom (half each frame)
    w, h, iters, count = 24, 18, 32, 6
    quick = ILI9341(wake_clocks=50)
    area = M.zoom(start_width=3.2, end_width=0.01, width=w)._replace(shrink=1)
    screen = ILI9341Model(quick, width=w, height=h)
    p = Py2ver(zoom_on_screen(quick, area, w, h, iters))
    run = p.check(frames(screen, count, w, h), verilog="--sim" in args)
    save_png("mandelbrot_zoom.png", film_strip(run.value), rgb888)
    print(f"{count} frames of {w} x {h} in {len(run.inputs):,} clock cycles: mandelbrot_zoom.png")
    if "--pins" in args:
        board = Py2ver(zoom_on_screen(), pins=PINS)
        print(board.PINS({"QUARTUS_DIR": os.environ.get(
            "QUARTUS_DIR", "/opt/intelFPGA/22.1std/quartus/bin")}))
