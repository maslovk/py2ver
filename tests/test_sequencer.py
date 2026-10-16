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
