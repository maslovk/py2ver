import random
import shutil

import pytest

from py2ver import Py2ver, native
from test_fold import FIR, fir
from test_pipeline_state import ACC_ATTR, pre_acc_post

# The plain Verilog testbench (Icarus / Verilator, no cocotb) gives what the
# cocotb testbench gives, for every way a design is stepped.

I16 = {"signed": 1, "width": 16}
_rng = random.Random(21)


def comb(a, b):
    return a * b - (a >> 2), a ^ b


def _both(monkeypatch, run):
    """run() on the native testbench (Icarus), then on cocotb."""
    monkeypatch.setenv("PY2VER_SIM", "icarus")
    first = run()
    monkeypatch.setenv("PY2VER_SIM", "cocotb")
    return first, run()


CASES = [
    ("combinational", lambda: Py2ver(comb, {"a": I16, "b": I16}), "batch"),
    ("pipelined", lambda: Py2ver(comb, {"a": I16, "b": I16}, pipeline=2), "batch"),
    ("state", lambda: Py2ver(fir, FIR), "stream"),
    ("folded", lambda: Py2ver(fir, FIR, fold="k", unroll=4), "stream"),
    ("shared", lambda: Py2ver(fir, FIR, multipliers=2), "stream"),
    ("pipelined state", lambda: Py2ver(pre_acc_post, ACC_ATTR, pipeline=2), "stream"),
]


@pytest.mark.sim
@pytest.mark.parametrize("name, make, how", CASES, ids=[c[0] for c in CASES])
def test_native_testbench_matches_cocotb(monkeypatch, tmp_path, name, make, how):
    monkeypatch.chdir(tmp_path)
    p = make()
    if how == "batch":
        vectors = [(_rng.randint(-999, 999), _rng.randint(-999, 999)) for _ in range(12)]
        native_out, cocotb_out = _both(monkeypatch, lambda: p.TB_batch()(vectors))
        assert native_out == cocotb_out == p.PY_batch()(vectors)
    else:
        top = 100 if name == "pipelined state" else 3000         # (x is 8 bits there)
        xs = [_rng.randint(-top, top) for _ in range(12)]
        native_out, cocotb_out = _both(monkeypatch, lambda: p.TB_stream()(xs))
        assert native_out == cocotb_out == p.PY_stream()(xs)


@pytest.mark.sim
@pytest.mark.skipif(shutil.which("verilator") is None, reason="Verilator not found")
def test_verilator_builds_once(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PY2VER_SIM", "verilator")
    p = Py2ver(fir, FIR, multipliers=2)
    xs = [_rng.randint(-3000, 3000) for _ in range(40)]
    assert p.TB_stream()(xs) == p.PY_stream()(xs)
    builds = list((tmp_path / "native_build").glob("verilator_*"))
    assert len(builds) == 1
    assert p.TB_stream()(xs[:5]) == p.PY_stream()(xs[:5])
    assert list((tmp_path / "native_build").glob("verilator_*")) == builds    # (reused)


@pytest.mark.sim
def test_a_failed_verilator_build_falls_back_to_icarus(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PY2VER_SIM", "verilator")
    real_build = native.build

    def build(p, mode, simulator, workdir):
        if simulator == "verilator":
            raise RuntimeError("verilator --cc failed:\n%Error: too long")
        return real_build(p, mode, simulator, workdir)
    monkeypatch.setattr(native, "build", build)
    p = Py2ver(comb, {"a": I16, "b": I16})
    vectors = [(3, 4), (-5, 6)]
    assert p.TB_batch()(vectors) == p.PY_batch()(vectors)
    assert list((tmp_path / "native_build").glob("verilator_failed_*"))


def test_long_runs_choose_verilator(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PY2VER_SIM", raising=False)
    p = Py2ver(comb, {"a": I16, "b": I16})
    assert native.choose_simulator(p, 10) == "icarus"
    if shutil.which("verilator"):
        assert native.choose_simulator(p, 10 ** 9) == "verilator"
    monkeypatch.setenv("PY2VER_SIM", "cocotb")
    assert native.choose_simulator(p, 10 ** 9) == "cocotb"


@pytest.mark.sim
def test_tables_are_arrays_any_simulator_reads(monkeypatch, tmp_path):
    import contextlib
    import io
    from test_share import _samples, neuron
    monkeypatch.chdir(tmp_path)
    samples = _samples(neuron)
    with contextlib.redirect_stdout(io.StringIO()):
        p = Py2ver(neuron, samples=samples, width=18)         # a tanh table, unpipelined
    verilog = (tmp_path / "hdl" / "main.v").read_text()
    assert "// Table:" in verilog and max(map(len, verilog.splitlines())) < 2000
    expected = p.PY_batch()(samples[:15])
    monkeypatch.setenv("PY2VER_SIM", "icarus")
    assert p.TB_batch()(samples[:15]) == expected
    if shutil.which("verilator"):
        monkeypatch.setenv("PY2VER_SIM", "verilator")
        assert p.TB_batch()(samples[:15]) == expected
        assert not list((tmp_path / "native_build").glob("verilator_failed_*"))
