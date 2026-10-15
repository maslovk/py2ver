import random

import pytest

from py2ver import Py2ver, auto_fixed

# Float models without any formats: auto_fixed chooses them.


def lpf(x, y):
    yn = y + 0.1 * (x - y)
    return yn


TAPS = (0.05, 0.25, 0.4, 0.25, 0.05)


def fir(x, delay):
    window = [x] + delay
    acc = 0.0
    for k in range(len(TAPS)):
        acc += TAPS[k] * window[k]
    y = acc
    new_delay = window[:-1]
    return y, new_delay


def gain(i, x, mem):
    g = mem[i] * 1.5
    mem[i] = x
    return g + x


def scaled(a, b):
    s = a * 300.0 + b
    return s, a * 0.001


_rng = random.Random(0)
XS = [_rng.uniform(-1, 1) for _ in range(200)]
FIR_STATE = {"delay": {"length": 4, "next": "new_delay"}}


def test_width_mode_sizes_every_value_from_its_range(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    vectors = [(_rng.uniform(-1, 1), _rng.uniform(-50, 50)) for _ in range(100)]
    attr = auto_fixed(scaled, vectors, width=16)
    # a: |a| < 1 -> 1 integer bit with the margin; s: |s| < 350 -> 10 bits
    assert attr["a"] == {"signed": 1, "width": 16, "frac": 14}
    assert attr["s"]["frac"] == 16 - 1 - 10
    assert attr["ret1"]["frac"] > 16            # tiny values: fraction bits beyond the width
    assert set(attr) == {"a", "b", "s", "ret1"}
    p = Py2ver(scaled, attr)
    err = p.quantization_error(vectors)
    assert err["s"][0] < 0.05 and err["ret1"][0] < 1e-6


def test_width_mode_keeps_state_details_and_given_formats(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    given = {"x": {"signed": 1, "width": 12, "frac": 11}, **FIR_STATE}
    attr = auto_fixed(fir, XS, given, width=16)
    assert attr["x"] == given["x"]
    assert attr["delay"]["next"] == "new_delay" and attr["delay"]["length"] == 4
    assert attr["window"]["length"] == 5
    assert "new_delay" not in attr               # takes the state's format
    assert all(m["width"] == 16 for k, m in attr.items() if k != "x")


def test_integer_inputs_get_integer_formats(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    samples = [(_rng.randrange(4), _rng.uniform(-2, 2)) for _ in range(50)]
    attr = auto_fixed(gain, samples, {"mem": {"length": 4, "memory": "ram"}}, width=12)
    assert attr["i"] == {"signed": 0, "width": 3}           # 0..3, plus the margin bit
    assert attr["mem"]["frac"] == 12 - 1 - 2 and attr["mem"]["memory"] == "ram"
    assert "frac" not in attr["i"]


@pytest.mark.parametrize("target", [1e-2, 1e-3, 1e-4])
def test_error_mode_meets_the_target_with_fewer_bits(monkeypatch, tmp_path, target):
    monkeypatch.chdir(tmp_path)
    attr = auto_fixed(lpf, XS, {"y": {"next": "yn"}}, width=14, max_error=target)
    assert attr["x"]["width"] == 14                          # inputs: the given width
    err = Py2ver(lpf, attr).quantization_error(XS)
    assert err["yn"][0] <= target
    # one fraction bit less on the state breaks the target
    tighter = dict(attr, y=dict(attr["y"], frac=attr["y"]["frac"] - 1,
                                width=attr["y"]["width"] - 1))
    assert Py2ver(lpf, tighter).quantization_error(XS)["yn"][0] > target
    assert list(tmp_path.iterdir()) == [tmp_path / "hdl"]   # trial designs built elsewhere


def test_error_mode_per_output_targets(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    attr = auto_fixed(fir, XS, FIR_STATE, width=16, max_error={"y": 2e-4})
    err = Py2ver(fir, attr).quantization_error(XS)
    assert err["y"][0] <= 2e-4
    loose = auto_fixed(fir, XS, FIR_STATE, width=16, max_error={"y": 2e-2})
    assert sum(m["width"] for m in loose.values()) < sum(m["width"] for m in attr.values())


def test_bad_requests_raise_clearly():
    with pytest.raises(ValueError, match="width=<bits> or max_error"):
        auto_fixed(lpf, XS, {"y": {"next": "yn"}})
    with pytest.raises(ValueError, match="needs example inputs"):
        auto_fixed(lpf, [], {"y": {"next": "yn"}}, width=16)
    with pytest.raises(ValueError, match="Give input 'x' a format"):
        auto_fixed(lpf, XS, {"y": {"next": "yn"}}, max_error=1e-3)


@pytest.mark.sim
def test_chosen_formats_match_in_simulation(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    attr = auto_fixed(fir, XS, FIR_STATE, width=16, max_error=1e-3)
    p = Py2ver(fir, attr)
    assert p.TB_stream()(XS[:40]) == p.PY_stream()(XS[:40])


# ----------------------------- Py2ver(func, samples=...) -----------------------------

def _printed_attr(text):
    ns = {}
    exec(text[text.index("attr = "):], ns)
    return ns["attr"]


def test_py2ver_without_attr_chooses_and_prints_formats(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    vectors = [(_rng.uniform(-1, 1), _rng.uniform(-1, 1)) for _ in range(50)]
    p = Py2ver(lpf, samples=vectors)
    out = capsys.readouterr().out
    assert "formats chosen for lpf" in out
    printed = _printed_attr(out)
    assert printed == auto_fixed(lpf, vectors, width=16)
    assert list(printed)[:2] == ["x", "y"]                  # parameters first
    assert p.PY_batch()(vectors) == Py2ver(lpf, printed).PY_batch()(vectors)


def test_py2ver_samples_fill_in_a_partial_attr(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    p = Py2ver(lpf, {"y": {"next": "yn"}}, samples=XS, max_error=1e-3)
    printed = _printed_attr(capsys.readouterr().out)
    assert printed["y"]["next"] == "yn" and printed["x"]["width"] == 16
    assert p.quantization_error(XS)["yn"][0] <= 1e-3


def test_py2ver_needs_attr_annotations_or_samples(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="samples=<example inputs>"):
        Py2ver(lpf)
