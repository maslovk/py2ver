import cmath
import math
import random

import pytest

from py2ver import Py2ver, auto_fixed
from py2ver.stages import NotSplittable, plan

# A chain built as separate stages: the same Python, the same results.

ROT = cmath.exp(-2j * math.pi * 0.125)
TAPS = (0.02, 0.06, 0.15, 0.27, 0.27, 0.15, 0.06, 0.02)


def downconvert(x, z):
    y = x * z
    zn = z * ROT
    return y, zn


def decimate(xs, delay):
    window = xs[::-1] + delay
    acc = 0
    for k in range(len(TAPS)):
        acc += TAPS[k] * window[k]
    new_delay = window[:len(TAPS) - 1]
    return acc, new_delay


def power(v):
    return v.real * v.real + v.imag * v.imag


def receiver(xs, z, delay):
    ys = []
    for x in xs:
        y, z = downconvert(x, z)
        ys.append(y)
    acc, delay = decimate(ys, delay)
    p = power(acc)
    return p, z, delay


STATE = {"xs": {"serial": True}, "z": {"next": "z", "init": 1, "complex": True},
         "delay": {"length": len(TAPS) - 1, "next": "delay", "complex": True}}

_rng = random.Random(0)
_xs = [(0.5 + 0.3 * math.sin(n / 10)) * math.cos(2 * math.pi * 0.125 * n) for n in range(256)]
BLOCKS = [(_xs[i:i + 4],) for i in range(0, len(_xs), 4)]


@pytest.fixture(scope="module")
def formats():
    return auto_fixed(receiver, BLOCKS, STATE, width=16, max_error=1e-3, round=True)


def test_the_chain_splits_into_stages(monkeypatch, tmp_path, formats):
    monkeypatch.chdir(tmp_path)
    p = Py2ver(receiver, formats, stages=True)
    kinds = [(st.kind, st.callee) for st in p.stages]
    assert kinds == [("sample", "downconvert"), ("block", "decimate"), ("block", "power")]
    assert p.stages[0].design.input_args_list == ["x"]               # one sample per call
    assert p.stages[1].design._serial["in"] and p.stages[1].design._serial["m"] == 4
    assert p.stages[1].design.ram_phases == 4                         # 8 taps folded on 4 clocks
    top = (tmp_path / "hdl" / "receiver_stages.v").read_text()
    assert "module receiver_stages" in top and top.count("_stream u") == 3
    assert (tmp_path / "hdl" / "main.v").read_text().startswith("\n") or "module receiver" in \
        (tmp_path / "hdl" / "main.v").read_text()                     # the fused core is kept


def test_options_override_the_folding(monkeypatch, tmp_path, formats):
    monkeypatch.chdir(tmp_path)
    p = Py2ver(receiver, formats, stages=True,
               stage_options={"decimate": {"fold": None}})
    assert getattr(p.stages[1].design, "ram_phases", 1) == 1


def _module(tmp_path, src, name):
    path = tmp_path / f"{name}.py"
    path.write_text(src)
    ns = {"__name__": name}
    exec(compile(src, str(path), "exec"), ns)
    return ns


HELPERS = '''
def f(a):
    return a + 1

def g(a, b):
    return a * b
'''


@pytest.mark.parametrize(
    "body, match",
    [
        ("    y = f(xs[0])\n    return y", "passes names only"),
        ("    y = f(xs)\n    w = y + 1\n    return w", "holds only stage calls"),
        ("    y = f(xs)\n    w = g(y, xs)\n    return w", "from one stage"),
        ("    y = f(xs)\n    w = f(xs)\n    return w", "from the stage before it"),
        ("    y = f(xs)\n    w = f(y)\n    return y", "must come from its last stage"),
        ("    ys = []\n    for x in xs:\n        y = f(x)\n        y = y + 1\n        ys.append(y)\n"
         "    return ys", "one call and appends"),
    ],
)
def test_chains_that_cant_be_split_say_why(monkeypatch, tmp_path, body, match):
    monkeypatch.chdir(tmp_path)
    ns = _module(tmp_path, HELPERS + "\ndef chain(xs):\n" + body + "\n", "badchain")
    with pytest.raises(NotSplittable, match=match):
        plan(ns["chain"], {"xs": {"signed": 1, "width": 8, "length": 4}})


# An interpolator: a block stage making 4 samples, then a sample stage on each
P0 = (0.0, -0.1, -0.1, 0.0)
P1 = (1.0, 0.8, 0.5, 0.3)
P2 = (0.0, 0.3, 0.6, 0.8)


def interpolate(x, d1, d2):
    ys = [0.0] * 4
    for i in range(4):
        ys[i] = P0[i] * d2 + P1[i] * d1 + P2[i] * x
    return ys, x, d1


def shape(y):
    return y * 0.75 + 0.125


def upsampler(x, d1, d2):
    ys, d1, d2 = interpolate(x, d1, d2)
    out = []
    for y in ys:
        o = shape(y)
        out.append(o)
    return out, d1, d2


Q = {"signed": 1, "width": 16, "frac": 13}
UP = {"x": Q, "d1": dict(Q, next="d1"), "d2": dict(Q, next="d2"), "ys": dict(Q, length=4),
      "o": Q}


@pytest.mark.sim
def test_the_staged_receiver_matches_the_chain(monkeypatch, tmp_path, formats):
    monkeypatch.chdir(tmp_path)
    p = Py2ver(receiver, formats, stages=True)
    blocks = BLOCKS[:16]
    assert p.TB_stages()(blocks) == p.PY_stages()(blocks)
    assert p.stage_cycles <= 4 * len(blocks) + 12                   # about a sample per clock
    assert p.TB_stages()(blocks, valid=0.6, ready=0.5, seed=1) == p.PY_stages()(blocks)


@pytest.mark.sim
def test_a_staged_interpolator_gives_one_sample_per_transfer(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    p = Py2ver(upsampler, UP, stages=True)
    assert [st.kind for st in p.stages] == ["block", "sample"]
    xs = [_rng.uniform(-1, 1) for _ in range(10)]
    assert p.TB_stages()(xs) == p.PY_stages()(xs)


def test_stages_are_smaller_in_quartus(quartus, formats):
    p = Py2ver(receiver, formats, stages=True)
    assert p.SYN_stages(quartus).combinational < p.SYN(quartus).combinational / 2
