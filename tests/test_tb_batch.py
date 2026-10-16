import pickle
import shutil
from pathlib import Path

import pytest

import py2ver.core as py2ver_mod
from py2ver import Py2ver


@pytest.fixture(autouse=True)
def _cocotb_testbench(monkeypatch):
    """These tests look at the cocotb testbench itself (not the native one)."""
    monkeypatch.setenv("PY2VER_SIM", "cocotb")


def _batch_add_sub(a, b):
    s = a + b
    d = a - b
    return s, d


def _batch_mixed_latency(a, b):
    fast = a ^ b
    s = a + b
    slow1 = s
    slow2 = slow1
    return fast, slow2


ADD_SUB_ATTR = {
    "a": {"signed": 0, "width": 8, "type": "wire"},
    "b": {"signed": 0, "width": 8, "type": "wire"},
    "s": {"signed": 0, "width": 9, "type": "wire"},
    "d": {"signed": 1, "width": 9, "type": "wire"},
}


def _fake_runner_writing(rows):
    calls = []

    def _runner(top, proj=None):
        calls.append(top)
        results_dir = Path(proj) / "results"
        results_dir.mkdir(exist_ok=True)
        with (results_dir / "results.pickle").open("wb") as handle:
            pickle.dump(rows, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return _runner, calls


def test_tb_batch_empty_does_not_simulate(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    runner, calls = _fake_runner_writing([])
    monkeypatch.setattr(py2ver_mod.tb_runner, "test_runner", runner)

    p = Py2ver(_batch_add_sub, ADD_SUB_ATTR)

    assert p.TB_batch()([]) == []
    assert calls == []


def test_tb_batch_maps_rows_per_vector(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    # d is 9-bit signed: 0x1FD -> -3. Second row missing -> zeros fallback.
    runner, calls = _fake_runner_writing([{"s": 10, "d": 0x1FD}])
    monkeypatch.setattr(py2ver_mod.tb_runner, "test_runner", runner)

    p = Py2ver(_batch_add_sub, ADD_SUB_ATTR)

    assert p.TB_batch()([(5, 5), (1, 1)]) == [(10, -3), (0, 0)]
    assert len(calls) == 1


def test_gen_tb_pads_missing_args_with_zero(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    p = Py2ver(_batch_add_sub, ADD_SUB_ATTR)

    p.gen_tb([(5,), (1, 2)])

    tb_src = (tmp_path / "tb.py").read_text(encoding="utf-8")
    assert "VECTORS = [(5, 0), (1, 2)]" in tb_src


def test_gen_tb_writes_inputs_as_port_bit_patterns(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    attr = dict(ADD_SUB_ATTR, a={"signed": 1, "width": 8}, b={"signed": 1, "width": 1})
    p = Py2ver(_batch_add_sub, attr)

    p.gen_tb([(-1, -1), (-128, 0)])

    tb_src = (tmp_path / "tb.py").read_text(encoding="utf-8")
    assert "VECTORS = [(255, 1), (128, 0)]" in tb_src


@pytest.mark.sim
def test_tb_batch_pipeline_single_simulation(monkeypatch, tmp_path):
    """A batch through a 2-stage pipeline matches Python and runs one simulation."""
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("Icarus Verilog tools are not installed")
    pytest.importorskip("cocotb")
    pytest.importorskip("cocotb_tools")

    # Run outside the repo to cover the output-directory handling too
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SIM", "icarus")

    real_runner = py2ver_mod.tb_runner.test_runner
    calls = []

    def _counting_runner(top, proj=None):
        calls.append(top)
        return real_runner(top, proj)

    monkeypatch.setattr(py2ver_mod.tb_runner, "test_runner", _counting_runner)

    attr = {
        "a": {"signed": 0, "width": 8, "type": "wire"},
        "b": {"signed": 0, "width": 8, "type": "wire"},
        "fast": {"signed": 0, "width": 8, "type": "wire"},
        "s": {"signed": 0, "width": 9, "type": "wire"},
        "slow1": {"signed": 0, "width": 9, "type": "reg"},
        "slow2": {"signed": 0, "width": 9, "type": "reg"},
    }
    p = Py2ver(_batch_mixed_latency, attr)
    assert p.tb_latency_cycles == 2

    vectors = [(a, (a * 37 + 11) % 256) for a in range(0, 256, 5)]
    outs = p.TB_batch()(vectors)

    assert outs == [_batch_mixed_latency(*v) for v in vectors]
    assert len(calls) == 1
