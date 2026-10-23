# A Mandelbrot set generator: a block that computes a picture and sends it out
# pixel by pixel, as a stream of RGB565 colors, ready for a display.
#
# For each pixel, c is a point of the complex plane, and z = z*z + c is
# repeated from z = 0 until z leaves the circle of radius 2 or MAX_ITER steps
# have passed. The number of steps picks the pixel's color. The hardware does
# one step per clock cycle, in fixed point (FRAC fraction bits).
#
#   python examples/mandelbrot.py           # a small picture: test it, save mandelbrot.png
#   python examples/mandelbrot.py --sim     # also simulate the generated Verilog
#   python examples/mandelbrot.py --syn     # also build it in Quartus (size and speed)
import os
import sys
from typing import NamedTuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py2ver.devices.ili9341 import rgb565  # noqa: E402
from py2ver.protocol import stream_send  # noqa: E402

FRAC = 13                                # fraction bits: numbers are value * 2**13
FOUR = 4 << FRAC                         # |z|**2 > 4: z has left the circle of radius 2
MAX_ITER = 64
WIDTH, HEIGHT = 240, 320                 # the ILI9341's screen


class View(NamedTuple):
    """Which part of the plane the picture shows, in fixed point: the top-left
    pixel's point (x0, y0), and the distance between neighboring pixels."""
    x0: int
    y0: int
    step: int


def view(x_min=-2.2, x_max=1.0, y_center=0.0, width=WIDTH, height=HEIGHT):
    """The View that fits x_min .. x_max across the picture, centered on y_center."""
    step = (x_max - x_min) / width
    y_top = y_center + step * height / 2
    scale = 1 << FRAC
    return View(round(x_min * scale), round(y_top * scale), round(step * scale))


def palette(max_iter=MAX_ITER):
    """A color for each count: blue through white to orange; black for points
    that never escaped (the set itself)."""
    colors = []
    for n in range(max_iter):
        t = n / max_iter
        if t < 0.5:
            r, g, b = int(510 * t * 0.4), int(510 * t * 0.7), 128 + int(254 * t)
        else:
            u = (t - 0.5) * 2
            r, g, b = 102 + int(153 * u), 178 - int(60 * u), 255 - int(255 * u)
        colors.append(rgb565(r, g, b))
    return tuple(colors) + (0,)


VIEW = view()
PALETTE = palette()


def escape_count(port, cx, cy, max_iter, frac=FRAC):
    """How many steps of z = z*z + c it takes z to leave the circle (max_iter if it never does).

    cx, cy and z are fixed point with `frac` fraction bits. The products of z
    (xx, yy and xy) are kept in registers of their own, so that within a clock
    cycle the adds come first and the multiplies after: one multiplier deep,
    not two (which would be too slow at 50 MHz with many fraction bits).
    """
    zx = 0
    zy = 0
    xx = 0                               # zx * zx
    yy = 0                               # zy * zy
    xy = 0                               # 2 * zx * zy
    n = 0
    while n < max_iter and xx + yy <= (4 << frac):
        zx = xx - yy + cx                # z = z*z + c, from the products
        zy = xy + cy
        xx = (zx * zx) >> frac           # and the new z's products
        yy = (zy * zy) >> frac
        xy = (zx * zy) >> (frac - 1)
        n = n + 1
        yield
    return n


def escape_widths(frac=FRAC):
    """The escape loop's widths in attr, for `frac` fraction bits.

    py2ver can't work them out (they depend on |z| <= 2, which links zx and
    zy). Inside the loop |z| <= 2, so after the last step |zx|, |zy| <= 6.2:
    under 8, so frac + 3 bits and a sign. Their squares stay under 64: frac + 6
    bits and a sign; 2 * zx * zy under 128: frac + 7 bits and a sign. One more
    bit each, to spare.
    """
    z = {"signed": 1, "width": frac + 5}
    square = {"signed": 1, "width": frac + 8}
    return {"zx": z, "zy": z, "xx": square, "yy": square,
            "xy": {"signed": 1, "width": frac + 9}}


def mandelbrot(port, area=VIEW, width=WIDTH, height=HEIGHT, max_iter=MAX_ITER, colors=PALETTE):
    """Compute the picture again and again, row by row, sending each pixel's color.

    Output stream: color (RGB565), valid; input: ready (see protocol.stream_send).
    colors: a color for each count, 0 to max_iter (palette(max_iter)).
    """
    port.valid = 0
    while True:
        for row in range(height):
            cy = area.y0 - row * area.step
            for col in range(width):
                cx = area.x0 + col * area.step
                n = yield from escape_count(port, cx, cy, max_iter)
                yield from stream_send(port, colors[n], data="color")


# The one input is ready (the next block can take a pixel); and the escape
# loop's widths, which py2ver can't work out (see escape_widths).
ATTR = {"ready": 1, **escape_widths(FRAC)}


# ----------------------------- the animation: zooming in -----------------------------

ZOOM_FRAC = 24                           # more fraction bits: zooming in needs finer steps


class Zoom(NamedTuple):
    """An animation zooming in on a point, in fixed point with `frac` fraction bits.

    cx, cy:    the point the picture stays centered on
    start:     the first frame's distance between neighboring pixels
    smallest:  the distance at which it starts over
    shrink:    each frame the distance shrinks by 1/2**shrink (4: by 1/16)
    """
    cx: int
    cy: int
    start: int
    smallest: int
    shrink: int = 4
    frac: int = ZOOM_FRAC


def zoom(x=-0.743643887, y=0.131825904, start_width=3.2, end_width=0.0001, width=WIDTH,
         frac=ZOOM_FRAC):
    """A Zoom into the point (x, y): the picture first start_width across, then
    narrower each frame until it is end_width across. (The default point is in
    the "seahorse valley".)"""
    scale = 1 << frac
    return Zoom(round(x * scale), round(y * scale), round(start_width / width * scale),
                max(1, round(end_width / width * scale)), frac=frac)


ZOOM = zoom()


def mandelbrot_zoom(port, area=ZOOM, width=WIDTH, height=HEIGHT, max_iter=MAX_ITER,
                    colors=PALETTE):
    """An animation: a picture after another, each zoomed in a little more on area's point.

    Output stream: color (RGB565), valid; input: ready, as mandelbrot()'s.
    """
    port.valid = 0
    step = area.start
    while True:
        x0 = area.cx - step * (width // 2)       # the top-left pixel, from the center
        y0 = area.cy + step * (height // 2)
        for row in range(height):
            cy = y0 - row * step
            for col in range(width):
                cx = x0 + col * step
                yield                    # a clock for the coordinates' multiply, before
                                         # the escape loop's (one multiplier per clock)
                n = yield from escape_count(port, cx, cy, max_iter, area.frac)
                yield from stream_send(port, colors[n], data="color")
        step = step - (step >> area.shrink) - 1  # zoom in (by 1 at least: once step is
                                                 # under 2**shrink, step >> shrink is 0)
        if step < area.smallest:
            step = area.start                    # and start over


ZOOM_ATTR = {"ready": 1, **escape_widths(ZOOM_FRAC)}


def picture(width, height):
    """A testbench: take one picture's pixels from the stream, as rows of colors."""
    from py2ver.protocol import stream_receive

    def tb(dut):
        pixels = []
        for _ in range(width * height):
            color = yield from stream_receive(dut, data="color")
            pixels.append(color)
        return [pixels[r * width:(r + 1) * width] for r in range(height)]
    return tb


def reference(area, width, height, max_iter):
    """The same picture computed in floating point: each pixel's escape count."""
    scale = 1 << FRAC
    rows = []
    for row in range(height):
        out = []
        for col in range(width):
            c = complex((area.x0 + col * area.step) / scale, (area.y0 - row * area.step) / scale)
            z, n = 0, 0
            while n < max_iter and abs(z) <= 2:
                z = z * z + c
                n += 1
            out.append(n)
        rows.append(out)
    return rows


if __name__ == "__main__":
    import functools
    from py2ver import Py2ver
    from py2ver.devices.ili9341 import rgb888
    from py2ver.devices.spi_target import save_png

    args = sys.argv[1:]
    w, h, iters = 48, 64, 32                    # a small picture, quick to simulate
    small = view(width=w, height=h)
    colors = palette(iters)
    p = Py2ver(functools.partial(mandelbrot, area=small, width=w, height=h, max_iter=iters,
                                 colors=colors), ATTR)
    run = p.check(picture(w, h), verilog="--sim" in args)
    save_png("mandelbrot.png", [[c for c in row for _ in range(4)]
                                for row in run.value for _ in range(4)], rgb888)
    expected = [[colors[n] for n in row] for row in reference(small, w, h, iters)]
    same = sum(a == b for got, want in zip(run.value, expected) for a, b in zip(got, want))
    print(f"A {w} x {h} picture in {len(run.inputs):,} clock cycles: mandelbrot.png. "
          f"{same} of {w * h} pixels as floating point computes them.")
    if "--syn" in args:
        print(Py2ver(mandelbrot, ATTR).SYN({"QUARTUS_DIR": os.environ.get(
            "QUARTUS_DIR", "/opt/intelFPGA/22.1std/quartus/bin")}))
