# A tunnel on an ILI9341 display: flying down an endless checkered tube that
# fades into the dark far away, and slowly turns.
#
# Each pixel shows a point of the tube's wall. Which point is worked out before
# building, in a table: for each 2 x 2 block of pixels, its angle around the
# screen's center, its depth (how far down the tube, from 1 / the distance to
# the center) and its shade (darker near the center: further away). The tube
# is symmetric, so the table covers a quarter of the screen and the other
# quarters mirror its angle.
#
# Each frame adds the time to the depth (the checkers come towards you) and a
# turn to the angle (the tube twists), and a checker's color comes from where
# it is on the wall.
#
#   tunnel --color/valid/ready--> ili9341_display
#
#   python examples/tunnel.py          # a few small frames: test them, save tunnel.png
#   python examples/tunnel.py --sim    # also simulate the generated Verilog
#   python examples/tunnel.py --pins   # build it for the board (wiring: mandelbrot_lcd.py)
import colorsys
import functools
import math
import os
import sys
from typing import NamedTuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mandelbrot_lcd import PINS  # noqa: E402
from py2ver import connect  # noqa: E402
from py2ver.devices.ili9341 import HEIGHT, ILI9341, WIDTH, ili9341_display, rgb565  # noqa: E402
from py2ver.protocol import SPI, stream_send  # noqa: E402

SHADES = 16                              # how many steps from black to full brightness


class Tube(NamedTuple):
    """The table of a tube's wall for a screen of width x height pixels: an
    entry for each 2 x 2 block of the screen's lower-right quarter, row by row.
    An entry holds the block's angle (bits 0-5: a quarter turn is 64), its
    depth (bits 6-13: the checkers' rings are 16 apart) and its shade (bits 14-17)."""
    cells: tuple


def tube(width=WIDTH, height=HEIGHT, depth_scale=16):
    """The Tube for a screen of width x height pixels (both multiples of 4).
    depth_scale: how deep the tube looks (the rings at distances of
    depth_scale * width / 16, half that, a third, ...)."""
    cells = []
    for qy in range(height // 4):
        for qx in range(width // 4):
            x, y = 2 * qx + 1, 2 * qy + 1          # the block's center, from the screen's
            distance = math.hypot(x, y)
            angle = min(63, int(math.atan2(y, x) / (math.pi / 2) * 64))
            depth = int(depth_scale * width / distance) & 255
            shade = min(SHADES - 1, int(SHADES * distance / (0.8 * width / 2)))
            cells.append(angle | depth << 6 | shade << 14)
    return Tube(tuple(cells))


def wall_colors(shades=SHADES):
    """The colors of the checkers: for each shade (0 black), 8 bands down the
    tube, each a hue of the rainbow, its checkers light and dark."""
    colors = []
    for shade in range(shades):
        for band in range(8):
            for light in (0.45, 1.0):
                r, g, b = colorsys.hsv_to_rgb(band / 8, 0.75, light * shade / (shades - 1))
                colors.append(rgb565(round(255 * r), round(255 * g), round(255 * b)))
    return tuple(colors)


TUBE = tube()
WALL = wall_colors()


def tunnel(port, view=TUBE, width=WIDTH, height=HEIGHT, speed=4, twist=1, colors=WALL):
    """Flying down the tube, frame after frame, sending each pixel's color.

    speed: how far down the tube each frame goes (a ring of checkers is 16).
    twist: how far the tube turns each frame (a whole turn is 256).
    Output stream: color (RGB565), valid; input: ready (see protocol.stream_send).
    """
    port.valid = 0
    t = 0                                        # how far down the tube
    turn = 0                                     # how far it has turned
    while True:
        for row in range(height):
            if row < height // 2:                # the block's row, from the center
                qy = (height // 2 - 1 - row) >> 1
            else:
                qy = (row - height // 2) >> 1
            for col in range(width):
                if col < width // 2:
                    qx = (width // 2 - 1 - col) >> 1
                else:
                    qx = (col - width // 2) >> 1
                cell = view.cells[qy * (width // 4) + qx]
                yield                            # (the entry used after a clock: a memory block)
                angle = cell & 63
                depth = (cell >> 6) & 255
                shade = cell >> 14
                # The quarter's angle, mirrored into the other quarters (0 is to
                # the right of the center, 64 below it, 128 left, 192 above)
                if col < width // 2:
                    if row < height // 2:
                        angle = 128 + angle
                    else:
                        angle = 128 - angle
                elif row < height // 2:
                    angle = 256 - angle
                u = (angle + turn) & 255         # where on the wall: round the tube,
                v = (depth + t) & 255            # and down it
                checker = ((u >> 4) ^ (v >> 4)) & 1
                color = colors[shade * 16 + (v >> 5) * 2 + checker]
                yield
                yield from stream_send(port, color, data="color")
        t = (t + speed) & 255
        turn = (turn + twist) & 255


ATTR = {"ready": 1}

# SCK at 25 MHz, as in mandelbrot_zoom.py (if the picture is noisy: SPI(2, ...))
LCD = ILI9341(spi=SPI(1, mode=0, miso=""))


def show_on_screen(lcd=LCD, view=TUBE, width=WIDTH, height=HEIGHT, speed=4, twist=1):
    """The tunnel feeding the display."""
    source = functools.partial(tunnel, view=view, width=width, height=height,
                               speed=speed, twist=twist)
    display = functools.partial(ili9341_display, lcd=lcd, width=width, height=height)
    return connect(tunnel=(source, ATTR), display=(display, {"color": 16, "valid": 1}),
                   name="tunnel_show",
                   wires={"display.color": "tunnel.color", "display.valid": "tunnel.valid",
                          "tunnel.ready": "display.ready"})


def picture(frame, view, width, height, speed=4, twist=1, colors=WALL):
    """Frame number `frame` worked out directly (for checking the stream)."""
    t, turn = (frame * speed) & 255, (frame * twist) & 255
    rows = []
    for row in range(height):
        up = row < height // 2
        qy = (height // 2 - 1 - row if up else row - height // 2) >> 1
        line = []
        for col in range(width):
            left = col < width // 2
            qx = (width // 2 - 1 - col if left else col - width // 2) >> 1
            cell = view.cells[qy * (width // 4) + qx]
            angle, depth, shade = cell & 63, (cell >> 6) & 255, cell >> 14
            angle = {(False, False): angle, (True, False): 128 - angle,
                     (True, True): 128 + angle, (False, True): 256 - angle}[(left, up)]
            u, v = (angle + turn) & 255, (depth + t) & 255
            line.append(colors[shade * 16 + (v >> 5) * 2 + (((u >> 4) ^ (v >> 4)) & 1)])
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
    # A small screen for the simulation: 32 x 40 pixels, flying fast
    w, h, count, speed, twist = 32, 40, 3, 12, 8
    quick = ILI9341(wake_clocks=50)
    view = tube(w, h)
    screen = ILI9341Model(quick, width=w, height=h)
    p = Py2ver(show_on_screen(quick, view, w, h, speed, twist))
    run = p.check(frames(screen, count, w, h), verilog="--sim" in args)
    assert run.value == [picture(k, view, w, h, speed, twist) for k in range(count)]
    save_png("tunnel.png", film_strip(run.value, scale=4), rgb888)
    print(f"{count} frames of {w} x {h} in {len(run.inputs):,} clock cycles, as worked "
          f"out directly: tunnel.png")
    if "--pins" in args:
        board = Py2ver(show_on_screen(), pins=PINS)
        print(board.PINS({"QUARTUS_DIR": os.environ.get(
            "QUARTUS_DIR", "/opt/intelFPGA/22.1std/quartus/bin")}))
