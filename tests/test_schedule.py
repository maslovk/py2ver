"""A design over several clocks a sample: fold=, multipliers= and sequential=True
each give it a Schedule (schedule.py); options that can't go together are
refused before translating."""
import pytest

from py2ver import Py2ver

from test_fold import FIR, fir
from test_sequencer import X16, mlp


@pytest.mark.parametrize("func, attr, options, kind", [
    (fir, FIR, {"fold": True}, "folded"),
    (fir, FIR, {"multipliers": 2}, "shared"),
    (mlp, X16, {"sequential": True}, "sequenced"),
])
def test_each_structure_gives_its_schedule(func, attr, options, kind):
    p = Py2ver(func, attr, write=False, **options)
    s = p.schedule
    assert s.kind == kind and s.phases == p.ram_phases > 1
    assert s.phase in s.params and set(s.attr) >= {s.phase}
    assert str(s).startswith({"folded": "Folded loop 'k'", "shared": "Shared",
                              "sequenced": "Sequenced"}[kind])


def test_one_clock_a_sample_has_no_schedule():
    assert Py2ver(fir, FIR, write=False).schedule is None


@pytest.mark.parametrize("options, error, message", [
    ({"rate": 1e6, "multipliers": 2}, ValueError, "rate= chooses"),
    ({"sequential": True, "fold": True}, NotImplementedError, "sequential=True can't"),
    ({"sequential": True, "multipliers": 2, "pipeline": 1}, NotImplementedError,
     "sequential=True can't"),
    ({"multipliers": 2, "pipeline": 1}, NotImplementedError, "multipliers= can't"),
    ({"fold": "k", "pipeline": 1}, NotImplementedError, "fold can't"),
])
def test_options_that_cant_go_together(options, error, message):
    with pytest.raises(error, match=message):
        Py2ver(fir, FIR, write=False, **options)
