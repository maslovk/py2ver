import re
import shutil

import pytest

import py2ver.core as py2ver_mod
from py2ver import Py2ver


def lpf_step(x, y_prev):
    err = x - y_prev
    delta = err >> 3
    y = y_prev + delta
    return y


LPF_ATTR = {
    "x": {"signed": 1, "width": 12, "type": "wire"},
    "y_prev": {"signed": 1, "width": 16, "type": "wire", "next": "y"},
    "err": {"signed": 1, "width": 16, "type": "wire"},
    "delta": {"signed": 1, "width": 16, "type": "wire"},
    "y": {"signed": 1, "width": 16, "type": "wire"},
}


def fir7_step(x, d1, d2, d3, d4, d5, d6):
    acc = (x * 2) + (d1 * 8) + (d2 * 14) + (d3 * 16) + (d4 * 14) + (d5 * 8) + (d6 * 2)
    y = acc >> 6
    nd1 = x
    nd2 = d1
    nd3 = d2
    nd4 = d3
    nd5 = d4
    nd6 = d5
    return y, nd1, nd2, nd3, nd4, nd5, nd6


def _fir7_attr():
    attr = {
        "x": {"signed": 1, "width": 16, "type": "wire"},
        "acc": {"signed": 1, "width": 32, "type": "wire"},
        "y": {"signed": 1, "width": 16, "type": "wire"},
    }
    for i in range(1, 7):
        attr[f"d{i}"] = {"signed": 1, "width": 16, "type": "wire", "next": f"nd{i}"}
        attr[f"nd{i}"] = {"signed": 1, "width": 16, "type": "wire"}
    return attr


def acc_step(x, acc):
    nacc = acc + x
    return nacc


def ext_step(x, s):
    n = x - s
    m = s
    return n, m


def pipelined_state(x, s):
    n = x + s
    return n


@pytest.fixture
def sim_env(monkeypatch, tmp_path):
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("Icarus Verilog tools are not installed")
    pytest.importorskip("cocotb")
    pytest.importorskip("cocotb_tools")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SIM", "icarus")
    return tmp_path


# ----------------------------- simulation -----------------------------

@pytest.mark.sim
def test_lpf_stream_matches_hand_threaded_reference(sim_env):
    xs = [0, 0, 256, 256, 512, 1024, 1024, -2048, -2048, 0, 100, 2047, -2048]

    # Independent reference: thread y_prev by hand with 16-bit signed wrap
    expected = []
    y_prev = 0
    for x in xs:
        y = Py2ver._wrap(lpf_step(x, y_prev), 16, True)
        expected.append((y,))
        y_prev = y

    p = Py2ver(lpf_step, LPF_ATTR)
    assert p.PY_stream()(xs) == expected
    assert p.TB_stream()(xs) == expected


@pytest.mark.sim
def test_fir7_stream_impulse_response(sim_env):
    p = Py2ver(fir7_step, _fir7_attr())

    # Impulse of 64 through h = [2, 8, 14, 16, 14, 8, 2] / 64 gives h itself
    xs = [64] + [0] * 8
    sim = p.TB_stream()(xs)

    assert [row[0] for row in sim] == [2, 8, 14, 16, 14, 8, 2, 0, 0]
    assert sim == p.PY_stream()(xs)


@pytest.mark.sim
def test_fir7_stream_random_matches_python(sim_env):
    xs = [((i * 7919) % 4001) - 2000 for i in range(64)]
    p = Py2ver(fir7_step, _fir7_attr())
    assert p.TB_stream()(xs) == p.PY_stream()(xs)


@pytest.mark.sim
def test_accumulator_init_and_wraparound(sim_env):
    attr = {
        "x": {"signed": 0, "width": 8, "type": "wire"},
        "acc": {"signed": 0, "width": 8, "type": "wire", "next": "nacc", "init": 250},
        "nacc": {"signed": 0, "width": 8, "type": "wire"},
    }
    p = Py2ver(acc_step, attr)

    # 250+3=253, +3=256->0, +3=3, +3=6 ...
    xs = [3, 3, 3, 3]
    expected = [(253,), (0,), (3,), (6,)]
    assert p.PY_stream()(xs) == expected
    assert p.TB_stream()(xs) == expected


@pytest.mark.sim
@pytest.mark.parametrize(
    "s_meta, n_meta",
    [
        ({"signed": 1, "width": 16}, {"signed": 1, "width": 8}),  # sign-extend narrow next
        ({"signed": 0, "width": 12}, {"signed": 1, "width": 8}),  # signed next into unsigned state
        ({"signed": 1, "width": 8}, {"signed": 0, "width": 12}),  # wide unsigned next truncated
    ],
)
def test_state_width_and_sign_conversion(sim_env, s_meta, n_meta):
    attr = {
        "x": {"signed": 1, "width": 8, "type": "wire"},
        "s": {**s_meta, "type": "wire", "next": "n"},
        "n": {**n_meta, "type": "wire"},
        "m": {**s_meta, "type": "wire"},  # exposes all state bits
    }
    p = Py2ver(ext_step, attr)
    xs = [5, -100, 127, -128, 0, 77, -1, 33]
    assert p.TB_stream()(xs) == p.PY_stream()(xs)


# ----------------------------- generation / validation -----------------------------

def test_state_lowering_ports_and_verilog(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    p = Py2ver(lpf_step, LPF_ATTR)

    assert p.input_args_list == ["x"]
    assert p.output_args_list == ["y"]
    assert p.has_clk is True
    assert p.tb_latency_cycles == 0

    verilog = (tmp_path / "hdl" / "main.v").read_text(encoding="utf-8")
    assert re.search(r"input +rst,", verilog)
    assert re.search(r"reg +signed \[15:0\] y_prev;", verilog)
    assert "if (rst) y_prev <= 16'sd0;\n  else if (ce) y_prev <= y;" in verilog
    assert re.search(r"input +ce,", verilog)


def test_stream_empty_does_not_simulate(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr(py2ver_mod.tb_runner, "test_runner", lambda *a: calls.append(a))
    p = Py2ver(lpf_step, LPF_ATTR)

    assert p.TB_stream()([]) == []
    assert p.PY_stream()([]) == []
    assert calls == []


def test_stateful_design_rejects_tb_batch(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    p = Py2ver(lpf_step, LPF_ATTR)

    with pytest.raises(ValueError, match="TB_stream"):
        p.TB()(1)
    with pytest.raises(ValueError, match="TB_stream"):
        p.TB_batch()([(1,)])


def test_stateless_design_rejects_stream(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    attr = {
        "x": {"signed": 0, "width": 8},
        "s": {"signed": 0, "width": 8},
        "n": {"signed": 0, "width": 8},
    }
    p = Py2ver(ext_step, attr)

    with pytest.raises(ValueError, match="need state"):
        p.TB_stream()([1])


@pytest.mark.parametrize(
    "bad_attr, exc, match",
    [
        ({"x": {"width": 8}, "s": {"width": 8}, "n": {"width": 8}, "q": {"width": 8, "next": "n"}},
         ValueError, "must be a function argument"),
        ({"x": {"width": 8}, "s": {"width": 8, "next": "zz"}, "n": {"width": 8}},
         ValueError, "must be a returned value"),
        ({"x": {"width": 8}, "s": {"width": 8, "next": "n"}, "n": {"width": 8, "type": "reg"}},
         NotImplementedError, "cannot be combined"),
    ],
)
def test_invalid_state_declarations(monkeypatch, tmp_path, bad_attr, exc, match):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(exc, match=match):
        Py2ver(pipelined_state, bad_attr)
