# The Mandelbrot set on an ILI9341 display: two blocks connected by a stream.
#
#   mandelbrot (examples/mandelbrot.py) --color/valid/ready--> ili9341_display (py2ver.devices)
#
# The generator computes a pixel while the display sends the one before over
# SPI, so the picture takes as long as the SPI link: about 0.15 s a screen.
#
#   python examples/mandelbrot_lcd.py          # test it on a small screen, save mandelbrot_lcd.png
#   python examples/mandelbrot_lcd.py --sim    # also simulate the generated Verilog
#   python examples/mandelbrot_lcd.py --pins   # build it for the board
#
# Wiring for --pins: the DE0-Nano's GPIO_1 header (JP2), 3.3 V logic. Checked on
# the board with a 2.2" MSP2202 module (its pins numbered from VCC):
#
#   display        signal       JP2 pin
#   7 SCK          GPIO_1[0]    2
#   6 SDI/MOSI     GPIO_1[1]    4
#   3 CS           GPIO_1[2]    5
#   5 DC/RS        GPIO_1[3]    6
#   4 RESET        GPIO_1[4]    7
#   1 VCC, 8 LED   3.3 V        29
#   2 GND          GND          30
#   9 SDO/MISO     not connected
import functools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mandelbrot as M  # noqa: E402
from py2ver import connect  # noqa: E402
from py2ver.devices.ili9341 import HEIGHT, ILI9341, WIDTH, ili9341_display  # noqa: E402


def fractal_on_screen(lcd=ILI9341(), width=WIDTH, height=HEIGHT, max_iter=M.MAX_ITER):
    """The Mandelbrot generator feeding the display."""
    fractal = functools.partial(M.mandelbrot, area=M.view(width=width, height=height),
                                width=width, height=height, max_iter=max_iter,
                                colors=M.palette(max_iter))
    display = functools.partial(ili9341_display, lcd=lcd, width=width, height=height)
    return connect(fractal=(fractal, M.ATTR), display=(display, {"color": 16, "valid": 1}),
                   name="mandelbrot_lcd",
                   wires={"display.color": "fractal.color", "display.valid": "fractal.valid",
                          "fractal.ready": "display.ready"})


# On the board: the display's pins
PINS = {"sclk": "GPIO_1[0]", "mosi": "GPIO_1[1]", "cs_n": "GPIO_1[2]", "dc": "GPIO_1[3]",
        "reset_n": "GPIO_1[4]"}


def one_screen(screen, pixels):
    """A testbench: run until the display's model has received a whole screen."""
    from py2ver.protocol import alongside

    def full():
        while screen.written < pixels:
            yield

    def tb(dut):
        yield from alongside(full(), screen.run(dut))
    return tb


if __name__ == "__main__":
    from py2ver import Py2ver
    from py2ver.devices.ili9341 import ILI9341Model

    args = sys.argv[1:]
    # A small screen for the simulation: 32 x 24 pixels, 16 steps, short waits
    w, h, iters = 32, 24, 16
    quick = ILI9341(wake_clocks=50)
    screen = ILI9341Model(quick, width=w, height=h)
    p = Py2ver(fractal_on_screen(quick, w, h, iters))
    run = p.check(one_screen(screen, w * h), verilog="--sim" in args)
    screen.save("mandelbrot_lcd.png", scale=8)
    print(f"A {w} x {h} screen in {len(run.inputs):,} clock cycles: mandelbrot_lcd.png")
    if "--pins" in args:
        board = Py2ver(fractal_on_screen(), pins=PINS)
        print(board.PINS({"QUARTUS_DIR": os.environ.get(
            "QUARTUS_DIR", "/opt/intelFPGA/22.1std/quartus/bin")}))
