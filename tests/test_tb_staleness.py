import importlib
import shutil
from pathlib import Path

import pytest


def _stale_sensitive(a, b):
    y = (a * 3) + (b * 5)
    return y


@pytest.mark.sim
def test_tb_calls_do_not_reuse_stale_results(monkeypatch):
    """Two consecutive TB calls with different inputs must produce different fresh outputs."""
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
        "y": {"signed": 0, "width": 16, "type": "wire"},
    }

    p = Py2ver(_stale_sensitive, attr)
    tb = p.TB()

    v1 = (2, 1)
    v2 = (7, 9)
    out1 = tb(*v1)
    out2 = tb(*v2)

    assert out1 == (_stale_sensitive(*v1),)
    assert out2 == (_stale_sensitive(*v2),)
    assert out1 != out2
