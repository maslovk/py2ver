import random

import pytest

from py2ver import Py2ver

# Multi-rate streams: lists marked 'serial' move one element per handshake.

M = 4
TAPS = (0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.0)


def decimate(xs, delay, gain):
    window = xs[::-1] + delay          # newest first
    acc = 0.0
    for k in range(len(TAPS)):
        acc += TAPS[k] * window[k]
    y = acc * gain
    new_delay = window[:len(TAPS) - 1]
    return y, new_delay


def Q(**kw):
    return dict(signed=1, width=16, frac=14, **kw)


DECIMATE = {"xs": Q(length=M, serial=True), "delay": Q(length=len(TAPS) - 1, next="new_delay"),
            "gain": {"signed": 0, "width": 4, "frac": 2},
            "acc": {"signed": 1, "width": 20, "frac": 16}, "y": Q()}

P0 = (0.0, -0.1, -0.1, 0.0)
P1 = (1.0, 0.8, 0.5, 0.3)
P2 = (0.0, 0.3, 0.6, 0.8)


def interpolate(x, d1, d2):
    ys = [0.0] * 4
    for i in range(4):
        ys[i] = P0[i] * d2 + P1[i] * d1 + P2[i] * x
    nd1 = x
    nd2 = d1
    return ys, nd1, nd2


INTERPOLATE = {"x": Q(), "d1": Q(next="nd1"), "d2": Q(next="nd2"), "ys": Q(length=4, serial=True)}

_rng = random.Random(0)
BLOCKS = [([_rng.uniform(-1, 1) for _ in range(M)], _rng.choice([1, 1.5, 2])) for _ in range(10)]
XS = [_rng.uniform(-1, 1) for _ in range(10)]


def test_serial_ports_of_the_wrapper(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    p = Py2ver(decimate, DECIMATE)
    assert p._serial["m"] == M and p._serial["l"] == 1
    text = p.stream_wrapper().read_text()
    assert "module decimate_blocks" in text and "module decimate_stream" in text
    outer = text.split("module decimate_stream")[1]
    assert "input  signed [15:0] s_xs," in outer and "input  signed [15:0] s_xs_0" not in outer
    p = Py2ver(interpolate, INTERPOLATE)
    assert p._serial["l"] == 4 and list(p._serial["out"]) == ["ys"]


@pytest.mark.parametrize(
    "attr, match",
    [
        ({"x": Q(serial=True), "d1": Q(next="nd1"), "d2": Q(next="nd2")}, "must be a list input"),
        ({"x": Q(), "d1": Q(next="nd1", serial=True), "d2": Q(next="nd2")}, "must be a list input"),
        ({"x": Q(), "d1": Q(next="nd1"), "d2": Q(next="nd2"), "nd1": Q(serial=True)},
         "must be a returned list"),
    ],
)
def test_bad_serial_ports_raise_clearly(monkeypatch, tmp_path, attr, match):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match=match):
        Py2ver(interpolate, attr)


@pytest.mark.sim
@pytest.mark.parametrize("fold", [None, "k"])
def test_decimator_takes_one_sample_per_transfer(monkeypatch, tmp_path, fold):
    monkeypatch.chdir(tmp_path)
    p = Py2ver(decimate, DECIMATE, fold=fold, unroll=2)
    ref = p.PY_stream()(BLOCKS)
    assert p.TB_stream()(BLOCKS) == ref
    assert p.TB_handshake()(BLOCKS, valid=0.7, ready=0.6, seed=3) == ref
    # Folded onto 2 multipliers, 4 clocks per block: still about one sample per clock
    if fold:
        assert p.ram_phases == M
        assert p.TB_handshake()(BLOCKS, valid=1.0, ready=1.0) == ref
        assert p.handshake_cycles <= M * len(BLOCKS) + 3 * M


@pytest.mark.sim
@pytest.mark.parametrize("fold", [None, "i"])
def test_interpolator_gives_one_sample_per_transfer(monkeypatch, tmp_path, fold):
    monkeypatch.chdir(tmp_path)
    p = Py2ver(interpolate, INTERPOLATE, fold=fold)
    ref = p.PY_stream()(XS)
    assert p.TB_stream()(XS) == ref
    assert p.TB_handshake()(XS, valid=0.8, ready=0.7, seed=4) == ref
