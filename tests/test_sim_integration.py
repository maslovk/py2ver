import importlib
import shutil
from pathlib import Path

import pytest


def _sim_add_sub(a, b):
    s = a + b
    d = a - b
    return s, d


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
