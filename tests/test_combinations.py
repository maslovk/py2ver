"""
Features combined: every design is built with each option that applies, and
the hardware must match the Python reference (simulation, the stream
wrapper with random stalls, and the fast model auto_fixed uses), with a
small error against the float model.
"""
import cmath
import math
import random

import pytest

from py2ver import Array, Py2ver, S, State, auto_fixed
from py2ver.auto import model_outputs
from py2ver.core import front_end

# ----------------------------- designs (float models) -----------------------------

N = 8
TAPS = tuple(math.sin(math.pi * (k + 1) / (N + 1)) / 5 for k in range(N))
W = tuple(cmath.exp(-2j * math.pi * k / N) for k in range(N))
PI = math.pi


def fir_state(x, delay):                   # list state, loop
    window = [x] + delay
    acc = 0.0
    for k in range(N):
        acc += TAPS[k] * window[k]
    new_delay = window[:N - 1]
    return acc, new_delay


def cfir(x, delay):                        # the same, complex
    window = [x] + delay
    acc = 0
    for k in range(N):
        acc += TAPS[k] * window[k]
    new_delay = window[:N - 1]
    return acc, new_delay


def nco(step, phase):                      # tables, state, branch
    s = math.sin(phase)
    c = math.cos(phase)
    nphase = phase + step
    if nphase > PI:
        nphase = nphase - 2 * PI
    return s, c, nphase


def biquad(x, s1, s2):                     # constants, scalar state
    y = 0.07 * x + s1
    n1 = 0.13 * x + 1.14 * y + s2
    n2 = 0.07 * x - 0.41 * y
    return y, n1, n2


def dft_bin(xs, k):                        # complex table, variable index
    acc = 0
    for n in range(N):
        acc += xs[n] * W[(n * k) % N]
    return acc.real * acc.real + acc.imag * acc.imag


def interp(x, d1, d2):                     # list writes, state from state
    ys = [0.0] * 4
    for i in range(4):
        ys[i] = (i * 0.25) * x + (1 - i * 0.25) * d1 + 0.1 * d2
    return ys, x, d1


def _mac(acc, a, b):
    return acc + a * b


def helpers(xs, g):                        # helper in a loop, max of a list
    acc = 0.0
    for i in range(4):
        acc = _mac(acc, xs[i], g)
    top = max(xs)
    return acc, top


def keywords(type, buf):                   # Verilog keywords as names
    wire = type * 0.5 + buf
    reg = wire - abs(buf)
    return wire, reg


def ram_acc(i, v, mem):                    # block RAM
    s = mem[i] + v * 0.5
    mem[i] = s
    return s


def int_complex(a, b):                     # integer complex parameters
    p = a * b.conjugate()
    q = p + complex(3, -2)
    return q, p.real


_r = random.Random(1)


def F(w, f, **kw):
    return dict(signed=1, width=w, frac=f, **kw)


def _cv():
    return complex(_r.uniform(-1, 1), _r.uniform(-1, 1))


# name: (function, attr, samples, stream?)
DESIGNS = {
    "fir_state": (fir_state, {"x": F(14, 12), "delay": F(14, 12, length=N - 1, next="new_delay"),
                              "acc": F(18, 14)},
                  [_r.uniform(-1, 1) for _ in range(12)], True),
    "cfir": (cfir, {"x": F(14, 12, complex=True),
                    "delay": F(14, 12, length=N - 1, next="new_delay", complex=True),
                    "acc": F(18, 14)},
             [_cv() for _ in range(12)], True),
    "nco": (nco, {"step": F(12, 9), "phase": F(16, 12, next="nphase"),
                  "math.sin": F(14, 12, range=(-4, 4), bits=8),
                  "math.cos": F(14, 12, range=(-4, 4), bits=8)},
            [_r.uniform(0, 0.5) for _ in range(12)], True),
    "biquad": (biquad, {"x": F(14, 12), "s1": F(18, 14, next="n1"), "s2": F(18, 14, next="n2"),
                        "y": F(18, 14)},
               [_r.uniform(-1, 1) for _ in range(12)], True),
    "dft_bin": (dft_bin, {"xs": F(12, 10, length=N), "k": {"signed": 0, "width": 3}},
                [([_r.uniform(-1, 1) for _ in range(N)], _r.randrange(N)) for _ in range(8)], False),
    "interp": (interp, {"x": F(14, 12), "d1": F(14, 12, next="x"), "d2": F(14, 12, next="d1"),
                        "ys": F(16, 12, length=4)},
               [_r.uniform(-1, 1) for _ in range(10)], True),
    "helpers": (helpers, {"xs": F(12, 10, length=4), "g": F(8, 6)},
                [([_r.uniform(-1, 1) for _ in range(4)], _r.uniform(-1, 1)) for _ in range(8)],
                False),
    "keywords": (keywords, {"type": F(12, 10), "buf": F(12, 10)},
                 [(_r.uniform(-1, 1), _r.uniform(-1, 1)) for _ in range(8)], False),
    "ram_acc": (ram_acc, {"i": {"signed": 0, "width": 2}, "v": F(10, 6),
                          "mem": F(14, 6, length=4, memory="ram")},
                [(_r.randrange(4), _r.uniform(-4, 4)) for _ in range(12)], True),
    "int_complex": (int_complex, {"a": {"signed": 1, "width": 8, "complex": True},
                                  "b": {"signed": 1, "width": 8, "complex": True}},
                    [(complex(_r.randint(-99, 99), _r.randint(-99, 99)),
                      complex(_r.randint(-99, 99), _r.randint(-99, 99))) for _ in range(8)], False),
}

OPTIONS = ["plain", "pipe2", "fold", "fold_u2", "auto_w", "auto_e"]
UNSUPPORTED = {("ram_acc", "pipe2"): "pipeline cannot be combined with block-RAM"}


def _base(attr):
    """What auto_fixed can't know: state, lists, memories, complex parameters."""
    keep = ("next", "init", "length", "memory", "complex")
    base = {k: {kk: v for kk, v in m.items() if kk in keep} for k, m in attr.items()
            if "range" not in m}
    return {k: m for k, m in base.items() if m}


def _build(name, option):
    func, attr, samples, stream = DESIGNS[name]
    kw = {"pipe2": {"pipeline": 2}, "fold": {"fold": True},
          "fold_u2": {"fold": True, "unroll": 2}}.get(option, {})
    if option == "auto_w":
        attr = auto_fixed(func, samples * 3, _base(attr), width=14)
    elif option == "auto_e":
        attr = auto_fixed(func, samples * 3, _base(attr), width=14, max_error=5e-3)
    return Py2ver(func, attr, **kw), attr


@pytest.mark.sim
@pytest.mark.parametrize("option", OPTIONS)
@pytest.mark.parametrize("name", list(DESIGNS))
def test_combination(monkeypatch, tmp_path, name, option):
    monkeypatch.chdir(tmp_path)
    if (name, option) in UNSUPPORTED:
        with pytest.raises(NotImplementedError, match=UNSUPPORTED[(name, option)]):
            _build(name, option)
        return
    func, _, samples, stream = DESIGNS[name]
    p, attr = _build(name, option)
    ref = (p.PY_stream() if stream else p.PY_batch())(samples)
    assert (p.TB_stream() if stream else p.TB_batch())(samples) == ref
    assert p.TB_handshake()(samples, valid=0.7, ready=0.6, seed=2) == ref
    if p._model is not None:
        assert model_outputs(front_end(func, attr), samples)[1] == ref
    if any("frac" in m for m in attr.values() if isinstance(m, dict)):
        # Within 1% of the outputs' size (2% for the 8-bit sin / cos tables)
        size = max(1.0, max(abs(float(v)) for row in p.PY_float()(samples) for v in row))
        worst = max(e for e, _ in p.quantization_error(samples).values())
        assert worst < (0.02 if name == "nco" else 0.01) * size


# ----------------------------- serial ports, stages, annotations -----------------------------

M = 4
DTAPS = (0.05, 0.15, 0.3, 0.3, 0.15, 0.05, 0.0, 0.0)


def cdecim(xs, delay):
    window = xs[::-1] + delay
    acc = 0
    for k in range(len(DTAPS)):
        acc += DTAPS[k] * window[k]
    new_delay = window[:len(DTAPS) - 1]
    return acc, new_delay


def cinterp(x, d1):
    ys = [0j] * 4
    for i in range(4):
        ys[i] = (i * 0.25) * x + (1 - i * 0.25) * d1
    return ys, x


def mag2(v):
    return v.real * v.real + v.imag * v.imag


def wave(p):
    return math.sin(p)


def mixer(x, z):
    y = x * z
    zn = z * cmath.exp(0.3j)                # a constant call, folded
    return y, zn


def tone_chain(xs, z, delay):
    ys = []
    for x in xs:
        y, z = mixer(x, z)
        ys.append(y)
    acc, delay = cdecim(ys, delay)
    m = mag2(acc)
    s = wave(m)                             # a table inside a helper
    return s, z, delay


def isum(xs):
    return sum(xs)


def iscale(v):
    return v * 3 - 1


def int_chain(xs):
    t = isum(xs)                            # no format: widths from the fused design
    u = iscale(t)
    return u


def ann_fir(x: S[12], taps: State(Array(S[12], 8), next="taps")):
    acc = 0
    for k in range(8):
        acc += taps[k] * (k - 3)
    for k in range(7, 0, -1):               # k again, in another loop
        taps[k] = taps[k - 1]
    taps[0] = x
    return acc, taps


CDECIM = {"xs": F(14, 12, length=M, serial=True, complex=True),
          "delay": F(14, 12, length=len(DTAPS) - 1, next="new_delay", complex=True),
          "acc": F(18, 14)}
BLOCKS = [([_cv() for _ in range(M)],) for _ in range(10)]


@pytest.mark.sim
@pytest.mark.parametrize("fold, unroll, auto", [(None, 1, False), ("k", 2, False), ("k", 2, True)])
def test_serial_complex_folded_decimator(monkeypatch, tmp_path, fold, unroll, auto):
    monkeypatch.chdir(tmp_path)
    attr = CDECIM
    if auto:
        attr = auto_fixed(cdecim, BLOCKS * 3, _base(CDECIM) | {"xs": dict(_base(CDECIM)["xs"],
                                                                         serial=True)},
                          width=14, max_error=5e-3)
    p = Py2ver(cdecim, attr, fold=fold, unroll=unroll)
    ref = p.PY_stream()(BLOCKS)
    assert p.TB_stream()(BLOCKS) == ref
    assert p.TB_handshake()(BLOCKS, valid=0.7, ready=0.6) == ref
    assert p.quantization_error(BLOCKS)["acc_re"][0] < 0.01


@pytest.mark.sim
@pytest.mark.parametrize("fold", [None, "i"])
def test_serial_complex_interpolator(monkeypatch, tmp_path, fold):
    monkeypatch.chdir(tmp_path)
    p = Py2ver(cinterp, {"x": F(14, 12, complex=True), "d1": F(14, 12, next="x", complex=True),
                         "ys": F(16, 12, length=4, serial=True)}, fold=fold)
    xs = [_cv() for _ in range(10)]
    ref = p.PY_stream()(xs)
    assert p.TB_stream()(xs) == ref
    assert p.TB_handshake()(xs, valid=0.8, ready=0.6) == ref


@pytest.mark.sim
@pytest.mark.parametrize("max_error", [None, 2e-2])
def test_stages_with_complex_tables_and_folding(monkeypatch, tmp_path, max_error):
    monkeypatch.chdir(tmp_path)
    base = {"xs": {"serial": True}, "z": {"next": "z", "init": 1, "complex": True},
            "delay": {"length": 7, "next": "delay", "complex": True}}
    xs = [math.cos(0.3 * n) * 0.8 for n in range(160)]
    blocks = [(xs[i:i + 4],) for i in range(0, len(xs), 4)]
    attr = auto_fixed(tone_chain, blocks, base, width=16, max_error=max_error)
    assert "math.sin" in attr
    p = Py2ver(tone_chain, attr, stages=True)
    ref = p.PY_stages()(blocks[:12])
    assert p.TB_stages()(blocks[:12]) == ref
    assert p.TB_stages()(blocks[:12], valid=0.6, ready=0.5, seed=4) == ref
    assert p.TB_stream()(blocks[:12]) == p.PY_stream()(blocks[:12])


@pytest.mark.sim
def test_integer_stages(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    p = Py2ver(int_chain, {"xs": {"signed": 1, "width": 8, "length": 4}}, stages=True)
    blocks = [([_r.randint(-100, 100) for _ in range(4)],) for _ in range(10)]
    assert p.TB_stages()(blocks) == p.PY_stages()(blocks)


@pytest.mark.sim
@pytest.mark.parametrize("fold, unroll, pipeline", [("k", 1, 0), (True, 4, 0), (None, 1, 2)])
def test_annotated_fir_folded_or_pipelined(monkeypatch, tmp_path, fold, unroll, pipeline):
    monkeypatch.chdir(tmp_path)
    p = Py2ver(ann_fir, fold=fold, unroll=unroll, pipeline=pipeline)
    xs = [_r.randint(-2000, 2000) for _ in range(12)]
    ref = p.PY_stream()(xs)
    assert p.TB_stream()(xs) == ref
    assert p.TB_handshake()(xs) == ref
