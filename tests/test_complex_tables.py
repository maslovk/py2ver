import cmath
import math
import random
from math import sin

import pytest

from py2ver import Py2ver, auto_fixed
from py2ver.auto import model_outputs
from py2ver.core import front_end

# Float models with complex values and math functions, as written.


def Q(width, frac, **kw):
    return dict(signed=1, width=width, frac=frac, **kw)


def mixer(x, lo):
    y = x * lo.conjugate()
    return y


MIXER = {"x": Q(12, 10, complex=True), "lo": Q(12, 10, complex=True), "y": Q(14, 11)}

ROT = cmath.exp(2j * math.pi * 0.05)


def nco(z):
    zn = z * ROT
    return zn.real, zn


NCO = {"z": Q(18, 16, complex=True, next="zn", init=1, round=True)}

N = 8
W = tuple(cmath.exp(-2j * math.pi * k / N) for k in range(N // 2))


def butterfly(a, b, k):
    t = W[k] * b
    return a + t, (a - t) / 2


BUTTERFLY = {"a": Q(14, 12, complex=True), "b": Q(14, 12, complex=True),
             "k": {"signed": 0, "width": 2}}


def soft(x):
    y = math.tanh(2 * x) + 0.5 * math.exp(-x * x)
    return y


SOFT = {"x": Q(12, 9), "math.tanh": dict(Q(16, 14), range=(-4, 4)),
        "math.exp": dict(Q(16, 14), range=(-4, 0))}


def wave(phase):
    y = sin(math.pi * phase) * 0.5
    return y


WAVE = {"phase": Q(12, 10), "sin": dict(Q(16, 14), range=(-6, 6), bits=8)}


def cmul(a, b):
    return a * b


_rng = random.Random(3)


def cv():
    return complex(_rng.uniform(-1, 1), _rng.uniform(-1, 1))


def test_complex_ports_and_values(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    p = Py2ver(mixer, MIXER)
    assert p.input_args_list == ["x_re", "x_im", "lo_re", "lo_im"]
    assert p.output_args_list == ["y_re", "y_im"]
    (re, im), = p.PY_batch()([(0.5 + 0.25j, 1j)])
    assert (re, im) == (0.25, -0.5)
    assert p.PY_float()([(0.5 + 0.25j, 1j)]) == [(0.25, -0.5)]
    p = Py2ver(nco, NCO)
    assert p.output_args_list == ["ret0", "zn_re", "zn_im"]
    assert p.state_init == {"z_re": 65536, "z_im": 0}


def test_math_tables(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    p = Py2ver(soft, SOFT)
    vs = [(x,) for x in (-3.0, -0.5, 0.0, 0.7, 3.9)]
    err = p.quantization_error(vs)
    assert err["y"][0] < 0.01                         # 1024-entry tables over 8 and 4
    p = Py2ver(wave, WAVE)                            # from math import sin; math.pi
    assert p.PY_batch()([(0.5,)])[0][0] == pytest.approx(0.5, abs=0.02)
    # sin over more than a period: a table of one period, so an argument beyond the
    # range gets sin's own value (another function's table uses its end entries)
    far = p.PY_batch()([(1.99,)])[0][0]
    assert far == pytest.approx(0.5 * math.sin(math.pi * 1.99), abs=0.01)


@pytest.mark.sim
def test_a_periodic_table_in_hardware(monkeypatch, tmp_path):
    """sin's one-period table in Verilog (the phase: a wide product, a shift, a
    mask) the same as the model, arguments within the range and beyond it."""
    monkeypatch.chdir(tmp_path)
    p = Py2ver(wave, WAVE)
    vs = [(x,) for x in (-1.99, -1.2, -0.3, 0.0, 0.5, 1.0, 1.7, 1.99)]
    assert p.TB_batch()(vs) == p.PY_batch()(vs)
    for (x,), (y,) in zip(vs, p.PY_batch()(vs)):
        assert y == pytest.approx(0.5 * math.sin(math.pi * x), abs=0.02)


def test_integer_complex_parameters(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    p = Py2ver(cmul, {"a": {"signed": 1, "width": 8, "complex": True},
                      "b": {"signed": 1, "width": 8, "complex": True}})
    assert p.PY_batch()([(3 + 4j, 2 - 1j)]) == [(10, 5)]


@pytest.mark.parametrize(
    "src, attr, match",
    [
        ("import math\ndef f(a):\n    y = math.sin(a)\n    return y", {"a": Q(8, 4)},
         "Give math.sin a table in attr"),
        ("import math\ndef f(a):\n    y = math.log(a)\n    return y",
         {"a": Q(8, 4), "math.log": dict(Q(8, 4), range=(-1, 1))}, "Table math.log over"),
        ("def f(a, b):\n    z = complex(a, b)\n    y = z / z\n    return y", {"a": Q(8, 4), "b": Q(8, 4)},
         "/ by a real value"),
        ("def f(a, b):\n    z = complex(a, b)\n    y = abs(z)\n    return y",
         {"a": Q(8, 4), "b": Q(8, 4)}, "abs\\(\\) of a complex value"),
    ],
)
def test_unsupported_raise_clearly(monkeypatch, tmp_path, src, attr, match):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "design.py"
    path.write_text(src + "\n")
    ns = {"__name__": "design"}
    exec(compile(path.read_text(), str(path), "exec"), ns)
    with pytest.raises(NotImplementedError, match=match):
        Py2ver(ns["f"], attr)


@pytest.mark.parametrize(
    "func, attr, samples",
    [
        (mixer, MIXER, [(cv(), cv()) for _ in range(10)]),
        (nco, NCO, [()] * 10),
        (butterfly, BUTTERFLY, [(cv(), cv(), _rng.randrange(4)) for _ in range(10)]),
        (soft, SOFT, [(_rng.uniform(-2, 2),) for _ in range(10)]),
    ],
)
def test_fast_model_equals_the_reference(monkeypatch, tmp_path, func, attr, samples):
    monkeypatch.chdir(tmp_path)
    p = Py2ver(func, attr)
    stream = bool(p.state_map)
    ports, rows = model_outputs(front_end(func, attr), samples)
    assert rows == (p.PY_stream() if stream else p.PY_batch())(samples)
    assert ports == p.output_args_list


def test_auto_formats_for_complex_and_tables(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    vs = [(cv(), cmath.exp(1j * _rng.uniform(0, 6.28))) for _ in range(100)]
    attr = auto_fixed(mixer, vs, width=16)
    assert attr["x"]["complex"] and attr["lo"]["complex"] and "complex" not in attr["y"]
    vs = [(_rng.uniform(-2, 2),) for _ in range(200)]
    attr = auto_fixed(soft, vs, width=16, max_error=5e-3)
    assert attr["math.tanh"]["range"] == (-8.0, 8.0)          # (seen: -4..4; margin 1)
    assert Py2ver(soft, attr).quantization_error(vs)["y"][0] <= 5e-3


@pytest.mark.sim
@pytest.mark.parametrize(
    "func, attr, samples",
    [
        (mixer, MIXER, [(cv(), cv()) for _ in range(20)]),
        (nco, NCO, [()] * 20),
        (butterfly, BUTTERFLY, [(cv(), cv(), _rng.randrange(4)) for _ in range(20)]),
        (soft, SOFT, [(_rng.uniform(-2, 2),) for _ in range(20)]),
        (wave, WAVE, [(_rng.uniform(-2, 2),) for _ in range(20)]),
        (cmul, {"a": {"signed": 1, "width": 8, "complex": True},
                "b": {"signed": 1, "width": 8, "complex": True}},
         [(complex(_rng.randint(-99, 99), _rng.randint(-99, 99)),
           complex(_rng.randint(-99, 99), _rng.randint(-99, 99))) for _ in range(20)]),
    ],
)
def test_complex_and_tables_match_in_simulation(monkeypatch, tmp_path, func, attr, samples):
    monkeypatch.chdir(tmp_path)
    p = Py2ver(func, attr)
    if p.state_map:
        assert p.TB_stream()(samples) == p.PY_stream()(samples)
    else:
        assert p.TB_batch()(samples) == p.PY_batch()(samples)
