import contextlib
import io
import math
import random

import pytest

from py2ver import Py2ver, auto_fixed
from py2ver.core import front_end
from py2ver.fixed import build_model
from py2ver.hwtypes import _namespace
from py2ver.share import NotShareable, share, simulate

# Straight-line code on a few shared multipliers, with its tables in block RAM:
# the same results over several clocks per sample.

_rng = random.Random(7)


def neuron(x0, x1, w0, w1, b):
    h = math.tanh(x0 * w0 + x1 * w1 + b)
    return h * w0 - x0 * x1 * b, h


def poly(x, y):
    a = x * y + 0.5 * x
    b = a * a - y * 3
    return b * x - a * y, a


def dot4(xs, ws):
    acc = xs[0] * ws[0] + xs[1] * ws[1] - xs[2] * ws[2] + xs[3] * ws[3]
    top = max(acc, 0.25)
    scaled = [acc * w for w in ws]
    return scaled, top


I16 = {"signed": 1, "width": 16}


def counter(x, s, t):
    y = x * s + t * 3
    s2 = (y * y >> 6) + x
    t2 = s * t - y
    return y, s2, t2


COUNTER = {"x": I16, "s": dict(I16, next="s2", init=5), "t": dict(I16, next="t2", init=-3)}


def _samples(f):
    if f is neuron:
        return [tuple(_rng.uniform(-1, 1) for _ in range(5)) for _ in range(60)]
    if f is poly:
        return [(_rng.uniform(-2, 2), _rng.uniform(-2, 2)) for _ in range(60)]
    return [([_rng.uniform(-1, 1) for _ in range(4)], [_rng.uniform(-1, 1) for _ in range(4)])
            for _ in range(60)]


def _raw_args(fe, args):
    """Float arguments -> the raw integers the model takes."""
    from py2ver.fixed import quantize
    out = []
    for name, v in zip(fe.py_params, args):
        m = fe.attr[name]
        q = lambda x: quantize(x, int(m["frac"]), int(m["width"]), bool(m["signed"]))  # noqa: E731
        out.append([q(x) for x in v] if isinstance(v, list) else q(v))
    return tuple(out)


@pytest.mark.parametrize("f", [neuron, poly, dot4])
@pytest.mark.parametrize("k", [1, 2, 3])
def test_shared_schedule_computes_what_the_model_does(f, k):
    samples = _samples(f)
    with contextlib.redirect_stdout(io.StringIO()):
        attr = auto_fixed(f, samples, width=18)
    fe = front_end(f, attr)
    tree, info = share(fe.tree, fe.attr, k)
    assert info["phases"] > 1
    model = build_model(tree, dict(fe.attr, **info["attr"]), _namespace(f))
    for args in samples[:25]:
        raw = _raw_args(fe, args)
        assert simulate(info, model, raw) == fe.model(*raw)


def test_products_run_on_the_shared_multipliers():
    with contextlib.redirect_stdout(io.StringIO()):
        attr = auto_fixed(dot4, _samples(dot4), width=18)
    fe = front_end(dot4, attr)
    for k in (1, 2, 4):
        tree, info = share(fe.tree, fe.attr, k)
        assert len(info["mul_widths"]) == k
        # A sum of products accumulates in the multipliers: no adders of its own
        src = __import__("ast").unparse(tree)
        assert src.count(" * ") == k


def test_tables_go_to_block_ram():
    with contextlib.redirect_stdout(io.StringIO()):
        attr = auto_fixed(neuron, _samples(neuron), width=18)
    fe = front_end(neuron, attr)
    _, info = share(fe.tree, fe.attr, 2)
    (rom,) = info["roms"].values()
    assert len(rom["entries"]) >= 64 and rom["address_bits"] == (len(rom["entries"]) - 1).bit_length()


def looped(xs):
    acc = 0
    for k in range(4):
        acc += xs[k] * xs[k]
    return acc


def test_loops_are_refused():
    fe = front_end(looped, {"xs": dict(I16, length=4)})
    with pytest.raises(NotShareable, match="straight-line"):
        share(fe.tree, fe.attr, 2)
    with pytest.raises(NotImplementedError, match="can't be combined"):
        Py2ver(counter, COUNTER, multipliers=2, pipeline=2)


@pytest.mark.sim
def test_shared_design_matches_in_simulation(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    samples = _samples(neuron)
    with contextlib.redirect_stdout(io.StringIO()):
        p = Py2ver(neuron, samples=samples, width=18, multipliers=2)
    assert p.ram_phases > 1 and "Table in block RAM" in (tmp_path / "hdl" / "main.v").read_text()
    expected = p.PY_batch()(samples[:12])
    assert p.TB_batch()(samples[:12]) == expected
    assert p.TB_handshake()(samples[:12], valid=0.6, ready=0.5, seed=3) == expected


@pytest.mark.sim
def test_shared_design_with_state(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    p = Py2ver(counter, COUNTER, multipliers=1)
    assert p.ram_phases > 1
    xs = [_rng.randint(-200, 200) for _ in range(15)]
    assert p.TB_stream()(xs) == p.PY_stream()(xs)


@pytest.mark.sim
def test_pipelined_tables_in_block_ram(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    samples = _samples(neuron)
    with contextlib.redirect_stdout(io.StringIO()):
        p = Py2ver(neuron, samples=samples, width=18, pipeline=2)
    assert "Table in block RAM" in (tmp_path / "hdl" / "main.v").read_text()
    assert p.TB_batch()(samples[:20]) == p.PY_batch()(samples[:20])


def chained(x, w):
    return math.tanh(math.sin(x * w) * w)


def test_each_table_read_takes_a_pipeline_stage(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    samples = [(_rng.uniform(-2, 2), _rng.uniform(-2, 2)) for _ in range(50)]
    with contextlib.redirect_stdout(io.StringIO()):
        Py2ver(chained, samples=samples, width=16, pipeline=2)
        with pytest.raises(NotImplementedError, match="at least 2 pipeline stages"):
            Py2ver(chained, samples=samples, width=16, pipeline=1)
