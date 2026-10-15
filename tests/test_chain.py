import cmath
import math
import random

import pytest

from py2ver import Py2ver, auto_fixed

# A signal chain is plain Python: a function calling the blocks' functions.
# py2ver inlines them into one design; state is passed in and returned as
# usual, and a rate change is a loop over the samples of a block.

ROT = cmath.exp(-2j * math.pi * 0.125)
TAPS = (0.1, 0.2, 0.4, 0.2, 0.1)


def downconvert(x, z):              # one sample: mix with the oscillator
    y = x * z
    zn = z * ROT
    return y, zn


def decimate(xs, delay):            # 4 samples in, 1 out
    window = xs[::-1] + delay
    acc = 0
    for k in range(len(TAPS)):
        acc += TAPS[k] * window[k]
    new_delay = window[:len(TAPS) - 1]
    return acc, new_delay


def power(v):                       # demodulate: |v|^2
    return v.real * v.real + v.imag * v.imag


def receiver(xs, z, delay):
    ys = []
    for x in xs:
        y, z = downconvert(x, z)
        ys.append(y)
    acc, delay = decimate(ys, delay)
    p = power(acc)
    return p, z, delay


STATE = {"delay": {"length": 4, "next": "delay", "complex": True},
         "z": {"next": "z", "init": 1, "complex": True}}

_rng = random.Random(0)
_xs = [0.8 * math.cos(2 * math.pi * 0.125 * n) + 0.05 * _rng.uniform(-1, 1) for n in range(256)]
BLOCKS = [_xs[i:i + 4] for i in range(0, len(_xs), 4)]


def test_the_chain_runs_as_python():
    p, z, delay = receiver(BLOCKS[0], 1, [0j] * 4)
    assert z == pytest.approx(ROT ** 4) and len(delay) == 4 and p > 0


def test_float_reference_feeds_back_a_state_reassigned_under_its_name(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    Q = {"signed": 1, "width": 24, "frac": 20}
    attr = {"xs": dict(Q, length=4), "z": dict(Q, next="z", init=1, complex=True),
            "delay": dict(Q, length=4, next="delay", complex=True),
            "acc": Q, "p": Q, "y": Q}
    p = Py2ver(receiver, attr)
    ideal = p.PY_float()(BLOCKS[:4])
    assert [round(r[1]) for r in ideal] == [-1, 1, -1, 1]      # z turns by pi per block
    assert p.quantization_error(BLOCKS[:16])["p"][0] < 1e-4


def test_auto_formats_for_the_chain_meet_the_target(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    attr = auto_fixed(receiver, BLOCKS, STATE, width=16, max_error=1e-3, round=True)
    assert attr["z"]["complex"] and attr["delay"]["length"] == 4
    p = Py2ver(receiver, attr)
    assert p.quantization_error(BLOCKS)["p"][0] <= 1e-3
    assert p.output_args_list[:3] == ["p", "z_re_out", "z_im_out"]


@pytest.mark.sim
@pytest.mark.parametrize("fold", [None, True])
def test_the_chain_matches_in_simulation(monkeypatch, tmp_path, fold):
    monkeypatch.chdir(tmp_path)
    attr = auto_fixed(receiver, BLOCKS, STATE, width=16, max_error=1e-3, round=True)
    p = Py2ver(receiver, attr, fold=fold)
    assert p.TB_stream()(BLOCKS[:16]) == p.PY_stream()(BLOCKS[:16])


# ----------------------------- lists of complex values -----------------------------

W = tuple(cmath.exp(-2j * math.pi * k / 4) for k in range(4))


def complex_lists(zs, g):
    out = []
    for z in zs:
        out.append(z * g)
    out.extend(zs[:2])
    total = sum(out)
    rot = list(zs)
    for k in range(len(rot)):
        rot[k] = rot[k] * W[k]
    both = rot[1:] + [total]
    return both, total.real, len(out)


def test_a_growing_complex_list_asks_for_a_format(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    Q = {"signed": 1, "width": 16, "frac": 12}
    with pytest.raises(NotImplementedError, match="give 'rot' a format"):
        Py2ver(complex_lists, {"zs": dict(Q, length=4, complex=True), "g": Q})


def test_complex_list_operations(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    Q = {"signed": 1, "width": 16, "frac": 12}
    p = Py2ver(complex_lists, {"zs": dict(Q, length=4, complex=True), "g": Q,
                               "rot": dict(Q, length=4)})
    zs = [0.5 + 0.25j, -0.5j, 1, 0.75 - 0.5j]
    exact = p.PY_batch()([(zs, 0.5)])[0]
    ideal = p.PY_float()([(zs, 0.5)])[0]
    assert len(exact) == len(ideal) == 4 + 4 + 2
    assert max(abs(e - f) for e, f in zip(exact, ideal)) < 1e-3
    assert p.output_args_list[:4] == [f"both_re_{k}" for k in range(4)]


def int_complex_list(zs):
    s = 0
    for z in zs:
        s += z * z.conjugate()
    return s.real


@pytest.mark.sim
def test_complex_lists_match_in_simulation(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    Q = {"signed": 1, "width": 16, "frac": 12}
    p = Py2ver(complex_lists, {"zs": dict(Q, length=4, complex=True), "g": Q,
                               "rot": dict(Q, length=4)})
    vectors = [([complex(_rng.uniform(-1, 1), _rng.uniform(-1, 1)) for _ in range(4)],
                _rng.uniform(-1, 1)) for _ in range(10)]
    assert p.TB_batch()(vectors) == p.PY_batch()(vectors)
    p = Py2ver(int_complex_list, {"zs": {"signed": 1, "width": 8, "length": 3, "complex": True}})
    vectors = [([complex(_rng.randint(-99, 99), _rng.randint(-99, 99)) for _ in range(3)],)
               for _ in range(10)]
    assert p.TB_batch()(vectors) == p.PY_batch()(vectors)
    assert p.PY_batch()([([3 + 4j, 1j, 2],)]) == [(30,)]
