# A test picture on an ILI9341 display (240 x 320, SPI), using py2ver.devices.ili9341.
#
# The design resets and wakes the display, paints the screen in eight vertical
# color bars, and puts a white square in the middle. A model of the display
# watches the SPI bus in the testbench and keeps the picture, so you can look
# at it: lcd.png.
#
#   python examples/lcd.py              # test it on a small screen, save lcd.png
#   python examples/lcd.py --sim        # also simulate the generated Verilog
#   python examples/lcd.py --pins       # build it for the board
#
# Wiring for --pins (the DE0-Nano's GPIO_1 header, 3.3 V logic):
#   GPIO_1[0] SCK, GPIO_1[1] SDI (MOSI), GPIO_1[2] CS, GPIO_1[3] D/C, GPIO_1[4] RESET
import functools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py2ver.devices.ili9341 import (HEIGHT, ILI9341, WIDTH, ili9341_fill,  # noqa: E402
                                    ili9341_init, rgb565)

LCD = ILI9341()
BARS = (rgb565(255, 255, 255), rgb565(255, 255, 0), rgb565(0, 255, 255), rgb565(0, 255, 0),
        rgb565(255, 0, 255), rgb565(255, 0, 0), rgb565(0, 0, 255), rgb565(0, 0, 0))
WHITE = rgb565(255, 255, 255)


def test_picture(port, lcd=LCD, width=WIDTH, height=HEIGHT):
    """Draw eight color bars and a white square in the middle, then raise port.done."""
    port.done = 0
    yield from ili9341_init(port, lcd)
    bar = width // 8
    for k in range(8):
        yield from ili9341_fill(port, lcd, k * bar, 0, k * bar + bar - 1, height - 1, BARS[k])
    side = width // 4
    x, y = (width - side) // 2, (height - side) // 2
    yield from ili9341_fill(port, lcd, x, y, x + side - 1, y + side - 1, WHITE)
    port.done = 1
    while True:
        yield


# The pins start idle: chip not selected, not in reset
ATTR = {"cs_n": {"init": 1}, "reset_n": {"init": 1}}
PINS = dict(ATTR, sclk="GPIO_1[0]", mosi="GPIO_1[1]", cs_n={"init": 1, "pin": "GPIO_1[2]"},
            dc="GPIO_1[3]", reset_n={"init": 1, "pin": "GPIO_1[4]"}, done="LED[0]")


def until_done(screen):
    """A testbench: run the design, with the display's model on the bus, until it's done."""
    from py2ver.protocol import alongside, wait_until

    def tb(dut):
        yield from alongside(wait_until(dut, "done"), screen.run(dut))
    return tb


if __name__ == "__main__":
    from py2ver import Py2ver
    from py2ver.devices.ili9341 import ILI9341Model

    # A small, fast screen for the simulation: 32 x 24 pixels, short waits.
    quick = LCD._replace(wake_clocks=50)
    screen = ILI9341Model(quick, width=32, height=24)
    p = Py2ver(functools.partial(test_picture, lcd=quick, width=32, height=24), ATTR)
    run = p.check(until_done(screen), verilog="--sim" in sys.argv[1:])
    screen.save("lcd.png", scale=10)
    print(f"Drew the test picture in {len(run.inputs):,} clock cycles; "
          f"the display is {'on' if screen.on else 'OFF'}. The picture: lcd.png")
    if "--pins" in sys.argv[1:]:
        board = Py2ver(test_picture, PINS)
        print(board.PINS({"QUARTUS_DIR": os.environ.get(
            "QUARTUS_DIR", "/opt/intelFPGA/22.1std/quartus/bin")}))
