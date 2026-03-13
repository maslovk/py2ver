import pytest
import os
import shutil
from pathlib import Path

from py2ver import Py2ver

hypothesis = pytest.importorskip("hypothesis")

from hypothesis import given, settings, strategies as st


def prop_add_sub(a, b):
    s = a + b
    d = a - b
    return s, d


@given(
    vectors=st.lists(
        st.tuples(st.integers(min_value=0, max_value=255), st.integers(min_value=0, max_value=255)),
        min_size=4,
        max_size=8,
    )
)
@settings(max_examples=5, deadline=None)
def test_property_python_vs_sim_add_sub(vectors):
    """Property-based differential check: Python reference must match Icarus sim."""
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("Icarus Verilog tools are not installed")
    pytest.importorskip("cocotb")
    pytest.importorskip("cocotb_tools")

    attr = {
        "a": {"signed": 0, "width": 8, "type": "wire"},
        "b": {"signed": 0, "width": 8, "type": "wire"},
        "s": {"signed": 0, "width": 9, "type": "wire"},
        "d": {"signed": 1, "width": 9, "type": "wire"},
    }

    root = Path(__file__).resolve().parents[1]
    old_cwd = Path.cwd()
    old_sim = os.environ.get("SIM")
    try:
        os.chdir(root)
        os.environ["SIM"] = "icarus"
        p = Py2ver(prop_add_sub, attr)
        tb = p.TB()
        for a, b in vectors:
            py_out = prop_add_sub(a, b)
            sim_out = tb(a, b)
            assert sim_out == py_out
    finally:
        os.chdir(old_cwd)
        if old_sim is None:
            os.environ.pop("SIM", None)
        else:
            os.environ["SIM"] = old_sim
