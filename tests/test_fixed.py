import math
import random
from fractions import Fraction

import pytest

from py2ver import Py2ver
from py2ver.fixed import const_raw, quantize

# Plain float models: the fixed-point formats are all in attr.


def fx(width, frac, signed=1, **kw):
    return dict(signed=signed, width=width, frac=frac, **kw)


# ----------------------------- quantization -----------------------------

def test_quantize_floors_or_rounds_and_wraps_or_saturates():
    assert quantize(0.3, 4, 8, True) == 4
    assert quantize(0.3, 4, 8, True, round=True) == 5
    assert quantize(-0.03125, 4, 8, True, round=True) == 0          # half rounds up
    assert quantize(-0.03125, 4, 8, True) == -1                     # floor
    assert quantize(8, 4, 8, True) == -128                          # wraps
    assert quantize(100, 4, 8, True, saturate=True) == 127
    assert quantize(-100, 4, 8, True, saturate=True) == -128
    assert quantize(1.0, 4, 4, False, saturate=True) == 15


def test_constants_are_exact_or_rounded_to_significant_bits():
    assert const_raw(0.75, 18) == (3, 2)
    assert const_raw(-2.5, 18) == (-5, 1)
    assert const_raw(1000.0, 18) == (1000, 0)
    raw, frac = const_raw(0.1, 18)
    assert raw.bit_length() <= 18 and abs(Fraction(raw, 1 << frac) - Fraction(1, 10)) < 2.0 ** -21
    raw, frac = const_raw(0.001, 8)
    assert raw.bit_length() <= 8 and abs(raw / (1 << frac) - 0.001) < 0.001 / 2 ** 7


# ----------------------------- designs (ordinary float code) -----------------------------

def mac(x, c, a):
    return a + x * c


MAC = {"x": fx(12, 10), "c": fx(12, 11), "a": fx(16, 8), "ret": fx(16, 8, round=True, saturate=True)}


def lpf(x, y):
    yn = y + 0.1 * (x - y)
    return yn


LPF = {"x": fx(12, 10), "y": fx(16, 14, next="yn", init=0.5, round=True)}

TAPS = (0.05, 0.25, 0.4, 0.25, 0.05)


def fir(x, delay):
    window = [x] + delay
    acc = 0.0
    for k in range(len(TAPS)):
        acc += TAPS[k] * window[k]
    y = acc
    new_delay = window[:-1]
    return y, new_delay


FIR = {"x": fx(16, 15), "delay": fx(16, 15, length=4, next="new_delay"),
       "y": fx(16, 15, round=True, saturate=True)}


def mixed(a, b):
    y = 0                                  # int here, fixed point below: one format for y
    if a > 0.5:
        y = a + b / 4
    elif a < -1.25:
        y = b / 3
    m = max(a, b, 0.75)
    z = abs(a) - b * 0.3
    w = a if a < b else b * 2
    return y, m, z, int(a), math.floor(a), math.ceil(-a), float(a) * 3 - 1, w


MIXED = {"a": fx(10, 4), "b": fx(8, 6, signed=0), "z": fx(12, 6, saturate=True)}


def _scale(v):
    return v * 0.75 + 0.125


def lists_and_helpers(xs, g):
    ys = [0] * 3
    for i in range(3):
        ys[i] = _scale(xs[i]) * g
    total = sum(ys) / 2
    ys.append(xs[0])
    return total, ys


LISTS = {"xs": fx(8, 6, length=3), "g": fx(6, 5, signed=0), "total": fx(12, 8, saturate=True)}


def accumulate(i, v, mem):
    s = mem[i] + v
    mem[i] = s
    return s


ACC = {"i": {"signed": 0, "width": 2}, "v": fx(8, 4),
       "mem": fx(12, 4, length=4, memory="ram", init=[1.5, -1, 0, 0.25])}


def _vectors(n, k, seed, lo=-4.0, hi=4.0):
    rng = random.Random(seed)
    return [tuple(rng.uniform(lo, hi) for _ in range(k)) for _ in range(n)]


def test_the_function_is_left_as_written():
    assert lpf(1.0, 0.0) == 0.1
    assert mac(0.5, 0.5, 1) == 1.25


def test_formats_and_ports(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    p = Py2ver(mac, MAC)
    assert p.output_args_list == ["ret"] and p._fx_out == {"ret": 8}
    assert p.output_args_width_list == [16]
    p = Py2ver(fir, FIR)
    assert p._fx_out == {"y": 15, **{f"new_delay_{k}": 15 for k in range(4)}}
    assert Py2ver(lpf, LPF).state_init == {"y": 8192}


def test_bit_exact_reference_and_float_reference(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    p = Py2ver(mac, MAC)
    assert p.PY_batch()([(0.5, 0.5, 1)]) == [(1.25,)]
    assert p.PY_batch()([(1.9, 0.99, 127)]) == [(127.99609375,)]        # saturated
    assert p.PY_batch()([(0.3, 0.3, 0)]) == [(0.08984375,)]             # inputs floored, result rounded
    assert p.PY_float()([(0.3, 0.3, 0)]) == [(0.2998046875 * 0.2998046875,)]

    p = Py2ver(lpf, LPF)
    xs = [1, 1, 1, -1]
    exact, ideal = p.PY_stream()(xs), p.PY_float()(xs)
    assert ideal[0][0] == pytest.approx(0.55) and exact[0][0] == pytest.approx(0.55, abs=2 ** -13)
    err = p.quantization_error(xs)
    assert set(err) == {"yn"} and 0 < err["yn"][0] < 2 ** -12


def test_integer_designs_are_untouched(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    p = Py2ver(mac, {"x": {"signed": 1, "width": 8}, "c": {"signed": 1, "width": 8},
                     "a": {"signed": 1, "width": 8}})
    assert p._model is None and p._fx_types == {} and p._fx_out == {}


@pytest.mark.parametrize(
    "src, attr, match",
    [
        ("def f(a, b):\n    y = a / b\n    return y", {"a": fx(8, 4), "b": fx(8, 4)},
         "needs a constant divisor"),
        ("def f(a):\n    y = a // 2\n    return y", {"a": fx(8, 4)}, "'//' is not defined"),
        ("def f(a):\n    y = a\n    for i in range(3):\n        y = y * a\n    return y",
         {"a": fx(8, 4)}, "grows on every loop iteration"),
        ("def f(a):\n    z = complex(a, a)\n    y = abs(z)\n    return y", {"a": fx(8, 4)},
         "abs\\(\\) of a complex value"),
        ("def f(a):\n    y = round(a)\n    return y", {"a": fx(8, 4)}, "'round': True in attr"),
        ("def f(a, xs):\n    y = xs[a]\n    return y",
         {"a": fx(8, 4), "xs": {"signed": 0, "width": 4, "length": 2}}, "can't be an index"),
    ],
)
def test_unsupported_fixed_point_code_raises_clearly(monkeypatch, tmp_path, src, attr, match):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "design.py"
    path.write_text(src + "\n")
    ns = {"__name__": "design"}
    exec(compile(path.read_text(), str(path), "exec"), ns)
    with pytest.raises(NotImplementedError, match=match):
        Py2ver(ns["f"], attr)


def test_a_format_in_attr_fixes_a_growing_loop_variable(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    src = "def power(a):\n    y = a\n    for i in range(3):\n        y = y * a\n    return y\n"
    path = tmp_path / "power.py"
    path.write_text(src)
    ns = {"__name__": "power"}
    exec(compile(src, str(path), "exec"), ns)
    p = Py2ver(ns["power"], {"a": fx(8, 6), "y": fx(16, 12)})
    assert p.PY_batch()([(0.5,)]) == [(0.0625,)]


# ----------------------------- simulation vs bit-exact reference -----------------------------

@pytest.mark.sim
def test_fixed_point_designs_match_the_reference_in_simulation(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    for func, attr, vectors in [
        (mac, MAC, _vectors(40, 3, 1)),
        (mixed, MIXED, [(a, abs(b)) for a, b in _vectors(40, 2, 2)]),
        (lists_and_helpers, LISTS, [((x, y, z), abs(g)) for x, y, z, g in _vectors(30, 4, 3, -2, 2)]),
    ]:
        p = Py2ver(func, attr)
        got = p.TB_batch()(vectors)
        assert got == p.PY_batch()(vectors), func.__name__
        assert all(isinstance(v, float) for v in got[0][:1])

    p = Py2ver(mac, MAC, pipeline=2)
    vectors = _vectors(20, 3, 4)
    assert p.TB_batch()(vectors) == p.PY_batch()(vectors)

    rng = random.Random(5)
    xs = [rng.uniform(-1, 1) for _ in range(16)]
    for func, attr, samples in [
        (lpf, LPF, [0, 1, 1, 1, -2, 0.3, 1.9, -2] + xs),
        (fir, FIR, xs),
        (accumulate, ACC, [(rng.randrange(4), rng.uniform(-8, 8)) for _ in range(20)]),
    ]:
        p = Py2ver(func, attr)
        assert p.TB_stream()(samples) == p.PY_stream()(samples), func.__name__


@pytest.mark.sim
@pytest.mark.board
def test_fixed_point_on_the_virtual_board(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    p = Py2ver(mac, MAC)
    vectors = _vectors(4, 3, 6)
    assert p.HW_sim()(vectors) == p.PY_batch()(vectors)


@pytest.mark.sim
def test_fixed_point_through_the_handshake(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    p = Py2ver(mac, MAC)
    vectors = _vectors(4, 3, 6)
    assert p.TB_handshake()(vectors, valid=0.6, ready=0.5, seed=1) == p.PY_batch()(vectors)


def _fx_scale(x):
    return x * 0.3 + 0.1


def test_copy_tree_shares_tables_only():
    import ast as _ast
    from py2ver.fixed import copy_tree, table_node
    table = table_node(tuple(range(100)))
    tree = _ast.parse("y = T[i] + 1")
    tree.body[0].value.left.value = table
    copied = copy_tree(tree)
    assert copied is not tree and copied.body[0] is not tree.body[0]
    assert copied.body[0].value.left.value is table                    # (shared)
    assert _ast.unparse(copied) == _ast.unparse(tree)


def test_front_end_cache_keeps_formats_apart():
    from py2ver.core import front_end
    q = {"signed": 1, "width": 16}
    coarse = front_end(_fx_scale, {"x": dict(q, frac=8), "ret": dict(q, frac=4)})
    fine = front_end(_fx_scale, {"x": dict(q, frac=8), "ret": dict(q, frac=12)})
    # 1.0 * 0.3 + 0.1 = 0.4, floored to 4 / 12 fraction bits
    assert (coarse.model(256), fine.model(256)) == (6, 1638)
