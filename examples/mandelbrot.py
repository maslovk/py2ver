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


def escape_count(port, cx, cy, max_iter):
    """How many steps of z = z*z + c it takes z to leave the circle (max_iter if it never does)."""
    zx = 0
    zy = 0
    xx = 0                               # zx * zx
    yy = 0                               # zy * zy
    n = 0
    while n < max_iter and xx + yy <= FOUR:
        zy = ((zx * zy) >> (FRAC - 1)) + cy      # 2 * zx * zy + cy
        zx = xx - yy + cx
        xx = (zx * zx) >> FRAC
        yy = (zy * zy) >> FRAC
        n = n + 1
        yield
    return n


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


# The one input is ready (the next block can take a pixel). py2ver can't work
# out how large z gets (that depends on |z| <= 2, which links zx and zy), so
# the escape loop's widths are given: inside the loop |z| <= 2, so after the
# last step |zx|, |zy| <= 6.2 (50,800 in fixed point: 18 signed bits), and
# their squares need 21.
ATTR = {"ready": 1,
        "zx": {"signed": 1, "width": 18}, "zy": {"signed": 1, "width": 18},
        "xx": {"signed": 1, "width": 21}, "yy": {"signed": 1, "width": 21}}


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
