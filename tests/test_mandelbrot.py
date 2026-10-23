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


def _frames(width, height, count, press_at=None):
    """A testbench: `count` pictures in a row, from the stream; the key released
    (1, as the board's keys read), but pressed while frame `press_at` comes."""
    def tb(dut):
        shots = []
        for k in range(count):
            dut.key = 0 if k == press_at else 1
            shot = yield from M.picture(width, height)(dut)
            shots.append(shot)
        return shots
    return tb


def _julia_counts(morph, k, width, height, max_iter, frac=M.ZOOM_FRAC):
    """Frame k of the Julia morph, in floating point: each pixel's escape count."""
    scale = 1 << frac
    c = complex(morph.path_x[k], morph.path_y[k]) / scale
    rows = []
    for row in range(height):
        out = []
        for col in range(width):
            z = complex(morph.x0 + col * morph.step, morph.y0 - row * morph.step) / scale
            n = 0
            while n < max_iter and abs(z) <= 2:
                z = z * z + c
                n += 1
            out.append(n)
        rows.append(out)
    return rows


@pytest.mark.parametrize("sim", [False, pytest.param(True, marks=pytest.mark.sim)])
def test_a_key_switches_to_julia_sets_morphing(sim, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    w, h, iters = 12, 8, 16
    area = M.zoom(width=w)
    morph = M.julia(frames=6, width=w, height=h)
    f = functools.partial(M.fractal_show, area=area, morph=morph, width=w, height=h,
                          max_iter=iters, colors=tuple(range(iters + 1)), cycle=0)
    p = _quiet(f, M.ZOOM_ATTR)
    shots = p.check(_frames(w, h, 6, press_at=2), verilog=sim).value

    def like(shot, counts):
        return sum(a == b for x, y in zip(shot, counts) for a, b in zip(x, y)) >= 0.9 * w * h
    first = next(k for k in range(6) if like(shots[k], _julia_counts(morph, 0, w, h, iters)))
    assert first in (2, 3)                                      # (from the press on)
    assert not like(shots[0], _julia_counts(morph, 0, w, h, iters))
    for j in range(first, 6):                                   # (c a step further each frame)
        assert like(shots[j], _julia_counts(morph, j - first, w, h, iters))


@pytest.mark.parametrize("sim", [False, pytest.param(True, marks=pytest.mark.sim)])
def test_a_zoom_frame_after_frame(sim, monkeypatch, tmp_path):
    # Zooming in by half each frame, then back out
    monkeypatch.chdir(tmp_path)
    w, h, iters = 8, 6, 8
    area = M.zoom(start_width=3.2, end_width=0.5, width=w)._replace(shrink=1)
    f = functools.partial(M.fractal_show, area=area, width=w, height=h, max_iter=iters,
                          colors=M.palette(iters), cycle=0)
    p = _quiet(f, M.ZOOM_ATTR)
    shots = p.check(_frames(w, h, 9), verilog=sim).value
    assert shots[0] != shots[1] != shots[2]                     # (each zoomed in further)
    assert shots[8] == shots[0]                                 # (and back out to the start)


@pytest.mark.parametrize("sim", [False, pytest.param(True, marks=pytest.mark.sim)])
def test_the_colors_cycle_frame_after_frame(sim, monkeypatch, tmp_path):
    # The view held still (its start its smallest), the colors moving 3 places a
    # frame; with colors 0, 1, 2, ... each pixel's color is its count, moved along
    monkeypatch.chdir(tmp_path)
    w, h, iters = 8, 6, 8
    area = M.zoom(start_width=3.2, end_width=3.2, width=w)
    f = functools.partial(M.fractal_show, area=area, width=w, height=h, max_iter=iters,
                          colors=tuple(range(iters + 1)), cycle=3)
    p = _quiet(f, M.ZOOM_ATTR)
    shots = p.check(_frames(w, h, 4), verilog=sim).value
    counts = shots[0]                                           # (frame 0: not moved yet)
    for k, shot in enumerate(shots):
        assert shot == [[n if n == iters else (n + 3 * k) % iters for n in row]
                        for row in counts]                      # (the set: black, still)
    assert any(n == iters for row in counts for n in row)       # (some of it in the set)


@pytest.mark.parametrize("sim", [False, pytest.param(True, marks=pytest.mark.sim)])
def test_a_zoom_turns_around_at_both_ends(sim, monkeypatch, tmp_path):
    # In to the smallest step, out to the start, in again; even with steps under
    # 2**shrink, where step >> shrink is 0 (the zoom once stopped shrinking at
    # 15, short of smallest, 7: stuck on one frame for ever). 4 fraction bits:
    # pixels far apart, so each step's frame looks different.
    monkeypatch.chdir(tmp_path)
    w, h, iters = 4, 3, 4
    area = M.Zoom(cx=-8, cy=0, start=40, smallest=7, shrink=4, frac=4)
    f = functools.partial(M.fractal_show, area=area, width=w, height=h, max_iter=iters,
                          colors=tuple(range(iters + 1)), cycle=0)
    p = _quiet(f, {"ready": 1, "key": 1, **M.escape_widths(4)})
    shots = p.check(_frames(w, h, 60), verilog=sim).value
    deepest = [[iters] * w for _ in range(h)]                       # (all in the set)
    assert shots[19] == deepest and shots[20] == shots[18]          # (in, and turning back)
    assert shots[39] == shots[0]                                    # (out to the start)
    assert shots[58] == deepest                                     # (and in again)


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
