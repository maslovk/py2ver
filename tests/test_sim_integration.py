import importlib
import shutil
from pathlib import Path

import pytest


def _sim_add_sub(a, b):
    s = a + b
    d = a - b
    return s, d


def _sim_reg_pipeline(a, b):
    s = a + b
    p1 = s
    p2 = p1
    return p2


def _sim_mixed_latency(a, b):
    fast = a ^ b
    s = a + b
    slow1 = s
    slow2 = slow1
    return fast, slow2


@pytest.mark.sim
def test_icarus_cocotb_tb_end_to_end(monkeypatch):
    """Run the real cocotb+Icarus simulation path and verify decoded outputs."""
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("Icarus Verilog tools are not installed")

    pytest.importorskip("cocotb")
    pytest.importorskip("cocotb_tools")

    try:
        Py2ver = importlib.import_module("py2ver").Py2ver
    except ModuleNotFoundError as exc:
        pytest.skip(f"py2ver runtime dependency missing: {exc}")

    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repo_root)
    monkeypatch.setenv("SIM", "icarus")

    attr = {
        "a": {"signed": 0, "width": 8, "type": "wire"},
        "b": {"signed": 0, "width": 8, "type": "wire"},
        "s": {"signed": 0, "width": 9, "type": "wire"},
        "d": {"signed": 1, "width": 9, "type": "wire"},
    }

    p = Py2ver(_sim_add_sub, attr)
    tb = p.TB()

    assert tb(7, 3) == (10, 4)
    assert tb(2, 5) == (7, -3)


@pytest.mark.sim
def test_icarus_pipeline_latency_and_value(monkeypatch):
    """Verify reg-pipeline latency metadata and sampled value correctness."""
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("Icarus Verilog tools are not installed")

    pytest.importorskip("cocotb")
    pytest.importorskip("cocotb_tools")

    try:
        Py2ver = importlib.import_module("py2ver").Py2ver
    except ModuleNotFoundError as exc:
        pytest.skip(f"py2ver runtime dependency missing: {exc}")

    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repo_root)
    monkeypatch.setenv("SIM", "icarus")

    attr = {
        "a": {"signed": 0, "width": 8, "type": "wire"},
        "b": {"signed": 0, "width": 8, "type": "wire"},
        "s": {"signed": 0, "width": 9, "type": "wire"},
        "p1": {"signed": 0, "width": 9, "type": "reg"},
        "p2": {"signed": 0, "width": 9, "type": "reg"},
    }

    p = Py2ver(_sim_reg_pipeline, attr)
    assert p.has_clk is True
    assert p.out_latency.get("p2") == 2
    assert p.tb_latency_cycles == 2

    tb = p.TB()
    for vec in [(1, 2), (7, 3), (25, 50), (255, 1)]:
        assert tb(*vec) == (_sim_reg_pipeline(*vec),)


@pytest.mark.sim
def test_icarus_mixed_output_latencies(monkeypatch):
    """Verify per-output latency accounting for wire+reg mixed outputs."""
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("Icarus Verilog tools are not installed")

    pytest.importorskip("cocotb")
    pytest.importorskip("cocotb_tools")

    try:
        Py2ver = importlib.import_module("py2ver").Py2ver
    except ModuleNotFoundError as exc:
        pytest.skip(f"py2ver runtime dependency missing: {exc}")

    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repo_root)
    monkeypatch.setenv("SIM", "icarus")

    attr = {
        "a": {"signed": 0, "width": 8, "type": "wire"},
        "b": {"signed": 0, "width": 8, "type": "wire"},
        "fast": {"signed": 0, "width": 8, "type": "wire"},
        "s": {"signed": 0, "width": 9, "type": "wire"},
        "slow1": {"signed": 0, "width": 9, "type": "reg"},
        "slow2": {"signed": 0, "width": 9, "type": "reg"},
    }

    p = Py2ver(_sim_mixed_latency, attr)
    assert p.has_clk is True
    assert p.out_latency.get("fast") == 0
    assert p.out_latency.get("slow2") == 2
    assert p.tb_latency_cycles == 2

    tb = p.TB()
    for vec in [(1, 2), (7, 3), (0, 255), (42, 11)]:
        assert tb(*vec) == _sim_mixed_latency(*vec)
