import math
import os
import random
import re
import shutil

import pytest

from py2ver import Py2ver

# Folded loops: the same results, over several clocks per sample.

N = 16
TAPS = tuple(round(1000 * math.sin(math.pi * (k + 1) / (N + 1))) for k in range(N))
I16 = {"signed": 1, "width": 16}


def fir(x, delay):
    window = [x] + delay
    acc = 0
    for k in range(N):
        acc += TAPS[k] * window[k]
    y = acc >> 10
    new_delay = window[:-1]
    return y, new_delay


FIR = {"x": I16, "delay": dict(I16, length=N - 1, next="new_delay")}

FTAPS = tuple(math.sin(math.pi * (k + 1) / (N + 1)) / 10 for k in range(N))


def fir_float(x, delay):
    window = [x] + delay
    acc = 0.0
    for k in range(N):
        acc += FTAPS[k] * window[k]
    new_delay = window[:-1]
    return acc, new_delay


Q15 = {"signed": 1, "width": 16, "frac": 15}
FIR_FLOAT = {"x": Q15, "delay": dict(Q15, length=N - 1, next="new_delay"),
             "acc": {"signed": 1, "width": 20, "frac": 17}}


def dot_max(xs, ys):
    s = 0
    m = -1000
    for i in range(1, 9):
        p = xs[i - 1] * ys[i - 1]
        s += p
        if p > m:
            m = p
    return s, m * 2


DOT = {"xs": dict(I16, length=8), "ys": dict(I16, length=8)}


def two_loops(xs):
    a = 0
    for i in range(2):
        a += xs[i]
    b = 0
    for j in range(6):
        b += xs[j] * j
    return a, b


def writes_list(xs):
    ys = [0] * 4
    for i in range(4):
        ys[i] = xs[i] * 2
    return ys


_rng = random.Random(0)


def _multiplies(p):
    text = re.sub(r"//.*", "", (p.outdir / "hdl" / "main.v").read_text())   # (not the comments)
    return len(re.findall(r" \* (?!\(?-?\d+\b)", text))      # not by a literal number


def test_folding_shares_the_loop_body(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    p = Py2ver(fir, FIR)
    unfolded = _multiplies(p)
    p = Py2ver(fir, FIR, fold="k")
    assert p.ram_phases == N and _multiplies(p) == 1 < unfolded
    assert p.input_args_list == ["x"]                         # ports are unchanged
    assert p.output_args_list == ["y"] + [f"new_delay_{i}" for i in range(N - 1)]
    p = Py2ver(fir, FIR, fold=True, unroll=4)
    assert p.ram_phases == 4 and _multiplies(p) == 4
    assert Py2ver(fir, FIR, fold="k", unroll=N).ram_phases == 1    # nothing left to fold


def test_fold_picks_the_longest_loop(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    attr = {"xs": dict(I16, length=6)}
    assert Py2ver(two_loops, attr, fold=True).schedule.loop == "j"
    assert Py2ver(two_loops, attr, fold="i").schedule.phases == 2


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"fold": "k", "unroll": 3}, "must divide the 16 iterations"),
        ({"fold": "z"}, "No top-level 'for z in range"),
        ({"fold": "k", "pipeline": 2}, "can't be combined with pipeline"),
    ],
)
def test_bad_folds_raise_clearly(monkeypatch, tmp_path, kwargs, match):
    monkeypatch.chdir(tmp_path)
    with pytest.raises((ValueError, NotImplementedError), match=match):
        Py2ver(fir, FIR, **kwargs)


def test_a_folded_loop_can_write_list_elements(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    p = Py2ver(writes_list, {"xs": dict(I16, length=4)}, fold="i")
    assert p.ram_phases == 4 and len(p.schedule.regs) == 4     # one register per element
    assert p.output_args_list == [f"ys_{i}" for i in range(4)]


@pytest.mark.sim
@pytest.mark.parametrize("fold, unroll", [("k", 1), ("k", 4), (True, 8)])
def test_folded_fir_matches(monkeypatch, tmp_path, fold, unroll):
    monkeypatch.chdir(tmp_path)
    p = Py2ver(fir, FIR, fold=fold, unroll=unroll)
    xs = [_rng.randint(-30000, 30000) for _ in range(12)]
    assert p.TB_stream()(xs) == p.PY_stream()(xs)


@pytest.mark.sim
def test_folded_fixed_point_and_stateless_designs(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    p = Py2ver(fir_float, FIR_FLOAT, fold="k", unroll=2)
    xs = [_rng.uniform(-1, 1) for _ in range(12)]
    assert p.TB_stream()(xs) == p.PY_stream()(xs)

    p = Py2ver(dot_max, DOT, fold="i")
    vectors = [([_rng.randint(-300, 300) for _ in range(8)],
                [_rng.randint(-300, 300) for _ in range(8)]) for _ in range(10)]
    expected = p.PY_batch()(vectors)
    assert p.TB_batch()(vectors) == expected
    assert p.TB_handshake()(vectors, valid=0.6, ready=0.5, seed=2) == expected


@pytest.mark.sim
@pytest.mark.board
def test_a_folded_design_on_the_virtual_board(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    p = Py2ver(dot_max, DOT, fold="i")
    vectors = [([_rng.randint(-300, 300) for _ in range(8)],
                [_rng.randint(-300, 300) for _ in range(8)]) for _ in range(3)]
    assert p.HW_sim()(vectors) == p.PY_batch()(vectors)


def _quartus_dir():
    env = os.environ.get("QUARTUS_DIR")
    if env and os.path.exists(os.path.join(env, "quartus_sh")):
        return env
    found = shutil.which("quartus_sh")
    return os.path.dirname(found) if found else None


@pytest.mark.quartus
@pytest.mark.skipif(_quartus_dir() is None, reason="Quartus not found")
def test_folding_saves_area_in_quartus(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    q = {"QUARTUS_DIR": _quartus_dir()}
    full = Py2ver(fir, FIR).SYN(q)
    folded = Py2ver(fir, FIR, fold="k").SYN(q)
    assert folded.combinational < full.combinational / 2
