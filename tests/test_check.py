import contextlib
import io

import pytest

from py2ver import Py2ver
from py2ver.process import CheckFailed
from test_state_stream import LPF_ATTR, lpf_step

# check() for a function: Python as written, py2ver's model and the simulated
# Verilog agree, or the message says what to change.


def arithmetic(x, y):
    total = x + y
    difference = x - y
    product = x * y
    quotient = x // y
    return total, difference, product, quotient


def halves(x, y):
    q = x / y
    return q


GOOD = {"x": 8, "y": 8, "total": 9, "difference": {"signed": 1, "width": 9},
        "product": 16, "quotient": 8}
VECTORS = [(3, 1), (7, 2), (255, 255), (200, 3), (2, 5)]


def _quiet(f, attr):
    with contextlib.redirect_stdout(io.StringIO()):
        return Py2ver(f, attr)


def test_a_function_that_fits_its_widths():
    assert _quiet(arithmetic, GOOD).check(VECTORS, verilog=False) == \
        [(4, 2, 3, 3), (9, 5, 14, 3), (510, 0, 65025, 1), (203, 197, 600, 66), (7, -3, 10, 0)]


def test_an_output_too_narrow_says_how_wide_it_must_be():
    with pytest.raises(CheckFailed, match=r"arithmetic\(255, 255\): Python gives product = 65025, "
                                          r"but its 9 bits hold only 0 to 511, so the hardware "
                                          r"gives 1\. Give it more bits in attr: "
                                          r"\{'product': 16\}"):
        _quiet(arithmetic, dict(GOOD, product=9)).check(VECTORS, verilog=False)


def test_a_negative_value_in_an_unsigned_output():
    with pytest.raises(CheckFailed, match=r"Python gives difference = -3, .* It can be negative, "
                                          r"so make it signed in attr: \{'difference': "
                                          r"\{'signed': 1, 'width': 9\}\}"):
        _quiet(arithmetic, dict(GOOD, difference=9)).check(VECTORS, verilog=False)


def test_a_true_division():
    with pytest.raises(CheckFailed, match=r"halves\(7, 2\): Python gives q = 3\.5, but the "
                                          r"hardware works in whole numbers .* Use // to divide"):
        _quiet(halves, {"x": 8, "y": 8, "q": 8}).check(VECTORS, verilog=False)


def test_without_a_simulator_the_model_is_checked_and_a_warning_says_so(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.warns(UserWarning, match="no Verilog simulator is installed"):
        out = _quiet(arithmetic, GOOD).check(VECTORS)
    assert out[2] == (510, 0, 65025, 1)


def test_a_testbench_is_for_a_process():
    with pytest.raises(ValueError, match="takes a list of input vectors for a function"):
        _quiet(arithmetic, GOOD).check(lambda dut: (yield))


def test_a_design_with_state_its_samples():
    xs = [256, 256, 512, -2048, 100, 0]
    p = _quiet(lpf_step, LPF_ATTR)
    assert p.check(xs, verilog=False) == p.PY_stream()(xs)


def test_the_doctor_says_what_is_installed_and_what_it_is_for(monkeypatch):
    from py2ver import doctor
    text = doctor.report(look_for_board=False)
    assert text.startswith("py2ver ") and "Simulating the Verilog" in text
    assert "check() and TB_batch() run the generated Verilog" in text
    monkeypatch.setattr("shutil.which", lambda name: None)
    monkeypatch.setattr(doctor, "_quartus_tool", lambda name, d: None)
    bare = doctor.report(look_for_board=False)
    assert "[ ] Icarus Verilog         not found (apt install iverilog)" in bare
    assert "without it: no check() and TB_batch() run the generated Verilog" in bare
    assert "[ ] Quartus" in bare and "a board" not in bare
    assert doctor.main(["help"]) == 2


@pytest.mark.sim
def test_the_verilog_checked_too(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert _quiet(arithmetic, GOOD).check(VECTORS)[4] == (7, -3, 10, 0)
    xs = [256, 256, 512, -2048, 100, 0]
    p = _quiet(lpf_step, LPF_ATTR)
    assert p.check(xs) == p.PY_stream()(xs)
