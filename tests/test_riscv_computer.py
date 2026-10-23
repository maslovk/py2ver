import contextlib
import functools
import io
import os
import sys
from pathlib import Path

import pytest

from py2ver import Py2ver

# examples/riscv_computer.py: an RV32IMA CPU written as a process, running
# programs built with the GNU RISC-V toolchain (skipped without one), and
# checked by riscv-tests (skipped without a checkout in ~/opt/riscv).

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
import riscv_computer as R  # noqa: E402

pytestmark = pytest.mark.skipif(R.toolchain() is None, reason="needs a RISC-V gcc (see riscv_computer.py)")


def _cpu(image):
    with contextlib.redirect_stdout(io.StringIO()):
        return Py2ver(functools.partial(R.rv32ima, image=image), R.CPU_ATTR)


@pytest.mark.parametrize("sim", [False, pytest.param(True, marks=pytest.mark.sim)])
@pytest.mark.parametrize("program", ["selftest.S", "traps.S"])
def test_the_cpu_passes_its_own_checks(program, sim, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    # (selftest.S: every RV32I instruction; traps.S: the CSRs, the traps, the
    # timer's interrupt and mret)
    assert _cpu(R.build(program)).check(R.console(b"\n"), verilog=sim).value == b"OK\n"


@pytest.mark.parametrize("sim", [False, pytest.param(True, marks=pytest.mark.sim)])
def test_a_c_program_prints_and_echoes(sim, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    run = _cpu(R.build("hello.c")).check(R.console(b"I echo:\nok", typed=b"ok"), verilog=sim)
    printed = run.value
    assert printed.startswith(b"Hello from RISC-V!\nFibonacci:\n0 1 1 2 3 5 8 13 21 34 55 89 144 ")
    assert b"610 987 \n" in printed and printed.endswith(b"I echo:\nok")


def _riscv_tests():
    names = []
    for suite in ("rv32ui", "rv32um", "rv32ua"):
        folder = Path(R.RISCV_TESTS) / "isa" / suite
        if folder.is_dir():
            names += [f"{suite}/{p.stem}" for p in sorted(folder.glob("*.S")) if p.stem != "ma_data"]
    return names


@pytest.mark.slow
@pytest.mark.skipif(not os.path.isdir(R.RISCV_TESTS), reason="needs riscv-tests in ~/opt/riscv")
@pytest.mark.parametrize("name", _riscv_tests())
def test_riscv_tests(name, monkeypatch, tmp_path):
    # (riscv-tests' rv32ui, rv32um, rv32ua: each prints OK, or FAIL and its
    # failing case; ma_data left out: misaligned accesses trap here)
    monkeypatch.chdir(tmp_path)
    printed = _cpu(R.riscv_test(name)).check(R.console(b"\n", limit=300_000), verilog=False).value
    assert printed == b"OK\n"
