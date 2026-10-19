"""The Icarus helpers (icarus.py) find what differs, and only that."""
import pytest

from py2ver import Py2ver

from icarus import assert_same_bits, assert_same_modules, mismatches
from test_fold import FIR, fir

INPUTS = {"a": (8, True), "b": (8, False), "w": (40, False)}


@pytest.mark.sim
def test_expressions_that_differ_are_found(tmp_path):
    pairs = [(16, True, "a + b", "b + a"),
             (16, True, "a - b", "b - a"),                       # differs
             (40, False, "w >> 3", "{3'b000, w[39:3]}"),         # (40 random bits)
             (1, False, "a < 0", "a[7]")]
    assert [k for k, _, _ in mismatches(tmp_path, INPUTS, pairs)] == [1]
    with pytest.raises(AssertionError, match="a - b"):
        assert_same_bits(tmp_path, INPUTS, pairs)


def _ports(p):
    return ({q.name: (q.width, bool(q.signed)) for q in p.ir.inputs},
            {q.name: (q.width, bool(q.signed)) for q in p.ir.outputs})


@pytest.mark.sim
def test_a_clocked_module_is_compared_from_reset(tmp_path):
    p = Py2ver(fir, FIR, write=False)                 # (its delay line: registers)
    inputs, outputs = _ports(p)
    assert_same_modules(tmp_path, p.verilog, p.verilog, "fir", inputs, outputs)
    reset = p.verilog.replace("delay_3 <= 16'sd0;", "delay_3 <= 16'sd1;")
    assert reset != p.verilog
    with pytest.raises(AssertionError, match="differ"):
        assert_same_modules(tmp_path, p.verilog, reset, "fir", inputs, outputs)
    tap = p.verilog.replace("* window_2", "* window_3", 1)
    assert tap != p.verilog
    with pytest.raises(AssertionError, match="differ"):
        assert_same_modules(tmp_path, p.verilog, tap, "fir", inputs, outputs)
