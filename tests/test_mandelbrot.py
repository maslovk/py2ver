import contextlib
import functools
import io
import sys
import time
from pathlib import Path

import pytest

from py2ver import Py2ver, connect, process
from py2ver.protocol import stream_receive, stream_send

# examples/mandelbrot.py: a block that computes a picture and streams it out,
# and the stream steps that hand values from one block to the next.

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
import mandelbrot as M  # noqa: E402


def _quiet(f, attr):
    with contextlib.redirect_stdout(io.StringIO()):
        return Py2ver(f, attr)


def _small(w=12, h=16, iters=16):
    area = M.view(width=w, height=h)
    colors = M.palette(iters)
    f = functools.partial(M.mandelbrot, area=area, width=w, height=h, max_iter=iters,
                          colors=colors)
    return _quiet(f, M.ATTR), area, colors


@pytest.mark.parametrize("sim", [False, pytest.param(True, marks=pytest.mark.sim)])
def test_a_mandelbrot_picture(sim, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    w, h, iters = 12, 16, 16
    p, area, colors = _small(w, h, iters)
    got = p.check(M.picture(w, h), verilog=sim).value
    want = [[colors[n] for n in row] for row in M.reference(area, w, h, iters)]
    same = sum(a == b for g, e in zip(got, want) for a, b in zip(g, e))
    assert same >= 0.95 * w * h                      # (fixed point: a few edge pixels differ)
    assert got[h // 2][w * 3 // 4] == 0              # (inside the set: black)


@pytest.mark.parametrize("sim", [False, pytest.param(True, marks=pytest.mark.sim)])
def test_the_picture_on_a_display(sim, monkeypatch, tmp_path):
    # The generator and the display connected: the screen shows exactly what
    # the generator sends on its own
    import mandelbrot_lcd as L
    from py2ver.devices.ili9341 import ILI9341, ILI9341Model
    monkeypatch.chdir(tmp_path)
    w, h, iters = 12, 8, 8
    alone, _, _ = _small(w, h, iters)
    sent = alone.check(M.picture(w, h), verilog=False).value
    quick = ILI9341(wake_clocks=20)
    screen = ILI9341Model(quick, width=w, height=h)
    both = _quiet(L.fractal_on_screen(quick, w, h, iters), None)
    both.check(L.one_screen(screen, w * h), verilog=sim)
    assert screen.on and screen.pixels == sent


def test_a_stream_hands_on_every_value_once():
    # A counter streaming 0, 1, 2, ... into a summing block that is sometimes slow
    def counter(port):
        port.valid = 0
        n = 0
        while True:
            yield from stream_send(port, n)
            n = (n + 1) & 255

    def summer(port):
        total = 0
        port.ready = 0
        while True:
            value = yield from stream_receive(port)
            total = (total + value) & 0xFFFF
            port.total = total
            if value & 1:
                yield 3                               # (slow now and then)

    design = connect(src=(counter, {"ready": 1}), dst=(summer, {"data": 8, "valid": 1}),
                     name="stream_pair",
                     wires={"dst.data": "src.data", "dst.valid": "src.valid",
                            "src.ready": "dst.ready"})
    rows = process.simulate(design, [{}] * 200)
    totals = sorted({r["total"] for r in rows})
    sums = [k * (k + 1) // 2 for k in range(256)]
    assert totals and all(t in sums for t in totals)  # (0+1+2+...: none lost, none twice)
    assert max(totals) >= sums[20]
    p = _quiet(design, None)
    assert p.PY_process()([{}] * 200) == p.GEN_process()([{}] * 200)


def test_a_step_s_local_given_by_its_own_name():
    p, _, _ = _small()
    assert p.process.attr["escape_count_zx"]["width"] == 18     # ('zx' in attr)

    def two(port):
        a = yield from M.escape_count(port, 1, 2, 4)
        b = yield from other(port)
        port.out = a + b

    def other(port):
        zx = port.x
        yield
        return zx
    with pytest.raises(process.ProcessError, match="'zx' in attr is a local of several steps"):
        process.compile_process(two, {"x": 4, "zx": 8})


def test_an_unbounded_square_is_named_at_once():
    # (squaring doubles a range's bits each time round: it used to take minutes to give up)
    t = time.time()
    with pytest.raises(process.ProcessError, match="can't tell how many bits 'xx'"):
        _quiet(functools.partial(M.mandelbrot, width=4, height=4, max_iter=4), {"ready": 1})
    assert time.time() - t < 10
