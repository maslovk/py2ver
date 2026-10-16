import random

import pytest

from py2ver import Py2ver
from py2ver.choose import choose
from py2ver.core import front_end
from test_fold import FIR, fir

# rate=: the smallest structure that keeps up, chosen and printed.

I16 = {"signed": 1, "width": 16}
_rng = random.Random(9)


def mix(a, b, c, d):
    p = a * b + c * d
    q = (a - b) * (c + d)
    return p * q, a * c * d


MIX = {"a": I16, "b": I16, "c": I16, "d": I16}


def by_index(xs, i, g):
    ys = list(xs)
    ys[i] = g * 3
    return ys


@pytest.mark.parametrize("rate, kind, unroll", [
    (50e6, "unrolled", 1),          # one clock per sample
    (10e6, "fold", 4),              # 5 clocks: 16 taps on 4 copies
    (5e6, "fold", 2),
    (1e6, "fold", 1),               # plenty: one copy
])
def test_a_fir_folds_as_far_as_the_rate_allows(monkeypatch, tmp_path, capsys, rate, kind, unroll):
    monkeypatch.chdir(tmp_path)
    p = Py2ver(fir, FIR, rate=rate)
    assert p.choice.kind == kind and p.choice.unroll == unroll
    assert p.choice.clocks <= p.choice.budget == int(50e6 // rate)
    assert "samples/s at 50 MHz" in capsys.readouterr().out


def test_straight_line_code_gets_the_fewest_multipliers_that_fit():
    fe = front_end(mix, MIX)
    loose, tight = choose(fe.tree, fe.attr, 20), choose(fe.tree, fe.attr, 5)
    assert loose.kind == tight.kind == "multipliers"
    assert loose.multipliers <= tight.multipliers and tight.clocks <= 5
    fe = front_end(by_index, {"xs": dict(I16, length=4), "i": I16, "g": I16})
    assert choose(fe.tree, fe.attr, 10).kind == "unrolled"     # (can't be scheduled)


@pytest.mark.sim
def test_the_chosen_design_matches(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    p = Py2ver(mix, MIX, rate=1e6, clock=6e6)                  # 6 clocks per sample
    assert p.choice.kind == "multipliers" and p.ram_phases <= 6
    vectors = [tuple(_rng.randint(-99, 99) for _ in range(4)) for _ in range(10)]
    assert p.TB_batch()(vectors) == p.PY_batch()(vectors)
    p = Py2ver(fir, FIR, rate=10e6)
    xs = [_rng.randint(-30000, 30000) for _ in range(10)]
    assert p.TB_stream()(xs) == p.PY_stream()(xs)


def test_rate_errors(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="more than one sample per clock"):
        Py2ver(fir, FIR, rate=60e6)
    with pytest.raises(ValueError, match="give rate= or those"):
        Py2ver(fir, FIR, rate=1e6, fold="k")
    with pytest.raises(ValueError, match="drop pipeline="):
        Py2ver(mix, MIX, rate=1e6, pipeline=2)
