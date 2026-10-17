import random

import pytest

from py2ver import Py2ver
from py2ver.core import front_end
from py2ver.fixed import build_model
from py2ver.hwtypes import _namespace
from py2ver.sequencer import NotSequenceable, sequence
from py2ver.share import simulate

# Loops as a state machine: block RAM for long lists, ROMs for tables, a block a
# few clocks, with Python's results.

_r = random.Random(1)
W = tuple(_r.randint(-50, 50) for _ in range(16 * 8))
B = tuple(_r.randint(-100, 100) for _ in range(8))
I8 = {"signed": 1, "width": 8}
X16 = {"x": dict(I8, length=16)}


def mlp(x):
    h = [0] * 8
    for j in range(8):
        acc = B[j]
        for i in range(16):
            acc = acc + x[i] * W[i * 8 + j]
        if acc < 0:
            acc = 0
        h[j] = acc
    best = -1
    bi = 0
    for j in range(8):
        if h[j] > best:
            best = h[j]
            bi = j
    return bi, best


def ram(x):
    h = [0] * 40                          # block RAM, every element written first
    for j in range(40):
        h[j] = x[j % 16] * W[j] + j
    s = 0
    for j in range(40):
        s = s + h[39 - j] * h[j]          # two reads of it in a step
    filled = [3] * 36                     # read before it is written: filled first
    filled[5] = s
    t = filled[5] + filled[6]
    return s, t


def divide(x):
    h = [0] * 8
    tot = 1
    for j in range(8):
        h[j] = x[j] * x[j + 8] - 300
        tot = tot + abs(x[j])
    out = [0] * 8
    for j in range(8):
        out[j] = (h[j] << 6) / tot        # (lowered fixed point divides like this)
    return out


def loop_in_if(x):
    s = 0
    if x[0] > 0:
        for i in range(4):
            s = s + x[i]
    return s


def _vec():
    return [_r.randint(-128, 127) for _ in range(16)]


@pytest.mark.parametrize("f", [mlp, ram, divide])
def test_the_machine_computes_what_python_does(f):
    fe = front_end(f, X16)
    tree, info = sequence(fe.tree, fe.attr, _namespace(f))
    assert info["phases"] > 1
    model = build_model(tree, dict(fe.attr, **info["attr"]), _namespace(f))
    for _ in range(40):
        x = _vec()
        want = f(x)
        if f is divide:                   # (truncating, as the hardware's division)
            want = [int(v) for v in want]
        got = simulate(info, model, (x,))
        assert (list(got) if f is divide else got) == want


def test_long_lists_and_tables_go_to_block_ram():
    fe = front_end(ram, X16)
    _, info = sequence(fe.tree, fe.attr, _namespace(ram))
    assert len(info["rfs"]) == 4 and len(info["roms"]) == 1    # h, filled: two reads each


@pytest.mark.sim
@pytest.mark.parametrize("f", [mlp, ram])
def test_sequenced_hardware_matches(monkeypatch, tmp_path, f):
    monkeypatch.chdir(tmp_path)
    p = Py2ver(f, X16, sequential=True)
    vectors = [(_vec(),) for _ in range(6)]
    assert p.TB_batch()(vectors) == p.PY_batch()(vectors)


def test_what_cant_be_sequenced_says_so():
    fe = front_end(loop_in_if, X16)
    with pytest.raises(NotSequenceable, match="loop inside an if"):
        sequence(fe.tree, fe.attr, _namespace(loop_in_if))


# ----------------------------- pipelined loops: an iteration a clock -----------------------------

def in_place(x):
    m = [0] * 40
    for i in range(40):
        m[i] = x[i % 16] * 3
    for i in range(40):
        m[i] = m[i] * 7 - x[i % 16]         # read and written at the same element: pipelined
    s = 0
    for i in range(40):
        if m[i] > 0:
            s = s + m[i]                    # carried, under a condition
    t = m[7]
    return s, t


def running(x):
    a = [0] * 40
    a[0] = x[0]
    for i in range(1, 40):
        a[i] = a[i - 1] + x[i % 16]         # reads what the iteration before wrote: not pipelined
    j = 0
    for k in range(10):
        j = a[(j + k) % 40] % 40            # carried through a read: not pipelined
    return a[39], j


@pytest.mark.parametrize("f", [mlp, in_place, running])
def test_pipelined_loops_compute_what_python_does(f):
    fe = front_end(f, X16)
    tree, info = sequence(fe.tree, fe.attr, _namespace(f))
    _, plain = sequence(fe.tree, fe.attr, _namespace(f), pipelined=False)
    model = build_model(tree, dict(fe.attr, **info["attr"]), _namespace(f))
    for _ in range(20):
        x = _vec()
        got = simulate(info, model, (x,))
        assert got == f(x)
    expected = {mlp: 1, in_place: 3, running: 0}[f]
    assert info["pipelined"] == expected
    if expected:
        assert info["phases"] < 0.7 * plain["phases"]
    else:
        assert info["phases"] == plain["phases"]


@pytest.mark.sim
def test_pipelined_hardware_matches(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    p = Py2ver(in_place, X16, sequential=True)
    vectors = [(_vec(),) for _ in range(6)]
    assert p.TB_batch()(vectors) == p.PY_batch()(vectors)


def deep(x):
    h = [0] * 40
    for j in range(40):
        h[j] = abs(x[j % 16]) * 5
    s = 0
    for i in range(40):
        a = h[i] * 3 + i                    # several stages: delays of delays
        b = h[h[i] % 40] - a                 # (a read at an address read first)
        s = s + (b if b > a else a) * i
    return s


@pytest.mark.sim
def test_a_deep_pipeline_in_hardware(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    fe = front_end(deep, X16)
    _, info = sequence(fe.tree, fe.attr, _namespace(deep))
    assert info["pipelined"] == 2
    p = Py2ver(deep, X16, sequential=True)
    vectors = [(_vec(),) for _ in range(6)]
    assert p.PY_batch()(vectors) == [(deep(*v),) for v in vectors]
    assert p.TB_batch()(vectors) == p.PY_batch()(vectors)


import functools                                      # noqa: E402
from py2ver.fixed import isqrt                        # noqa: E402

__py2ver_isqrt = functools.partial(isqrt)             # (what fixed point lowers np.sqrt to;
                                                      # not a function, so not inlined)


def roots(x):
    h = [0] * 40
    for j in range(40):
        h[j] = x[j % 16] * x[(j + 3) % 16] + 20000
    s = 0
    for j in range(40):
        r = __py2ver_isqrt(h[j] << 4)                    # a root bit a stage
        s = s + int((h[j] << 6) / (r - x[j % 16]))      # a quotient bit a stage
    return s


def test_loops_with_division_and_square_roots_are_pipelined_step_by_step():
    fe = front_end(roots, X16)
    ns = dict(_namespace(roots), int_=int)                 # (int, renamed from the keyword)
    tree, info = sequence(fe.tree, fe.attr, ns)
    _, plain = sequence(fe.tree, fe.attr, ns, pipelined=False)
    assert info["pipelined"] == 2 and info["phases"] < plain["phases"] / 10
    model = build_model(tree, dict(fe.attr, **info["attr"]), ns)
    reference = build_model(fe.tree, fe.attr, ns)          # (the loops as written)
    for _ in range(20):
        x = _vec()
        assert simulate(info, model, (x,)) == reference(x)


@pytest.mark.sim
def test_step_by_step_units_in_hardware(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    p = Py2ver(roots, X16, sequential=True)
    vectors = [(_vec(),) for _ in range(5)]
    assert p.TB_batch()(vectors) == p.PY_batch()(vectors)


# ----------------------------- lanes: several multiply-accumulates a clock -----------------------------

def test_lanes_run_iterations_side_by_side():
    fe = front_end(mlp, X16)
    ns = _namespace(mlp)
    clocks = {}
    for lanes in (1, 2, 3, 4):
        tree, info = sequence(fe.tree, fe.attr, ns, lanes=lanes)
        clocks[lanes] = info["phases"]
        model = build_model(tree, dict(fe.attr, **info["attr"]), ns)
        for _ in range(10):
            x = _vec()
            assert simulate(info, model, (x,)) == mlp(x)
    # 16 products a sum: 2 lanes halve it, 3 run as 2 (16 doesn't split in 3)
    assert clocks[1] > clocks[2] == clocks[3] > clocks[4]


@pytest.mark.sim
def test_lanes_in_hardware_with_stalls(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    p = Py2ver(mlp, X16, sequential=True, multipliers=2)
    assert "read by" in (tmp_path / "hdl" / "main.v").read_text()   # (one dual-port ROM)
    vectors = [(_vec(),) for _ in range(6)]
    expected = p.PY_batch()(vectors)
    assert p.TB_batch()(vectors) == expected
    assert p.TB_handshake()(vectors, valid=0.6, ready=0.5, seed=3) == expected


def test_a_rate_chooses_the_lanes(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()) as out:
        p = Py2ver(mlp, X16, sequential=True, rate=50e6 / 120)
    assert p.choice.multipliers == 2 and p.ram_phases <= 120
    assert "multipliers=2" in out.getvalue()
    with pytest.raises(ValueError, match="at least"):
        Py2ver(mlp, X16, sequential=True, rate=50e6 / 10)
