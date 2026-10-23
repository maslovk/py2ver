import contextlib
import functools
import io
import sys
from pathlib import Path

import pytest

from py2ver import Py2ver

# examples/riscv_computer.py: an RV32I CPU written as a process, running
# programs built with the GNU RISC-V toolchain (skipped without one).

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
import riscv_computer as R  # noqa: E402

pytestmark = pytest.mark.skipif(R.toolchain() is None, reason="needs a RISC-V gcc (see riscv_computer.py)")


def _cpu(*sources):
    with contextlib.redirect_stdout(io.StringIO()):
        return Py2ver(functools.partial(R.rv32i, program=R.build(*sources)), R.CPU_ATTR)


@pytest.mark.parametrize("sim", [False, pytest.param(True, marks=pytest.mark.sim)])
def test_the_cpu_passes_its_own_check(sim, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    # (every RV32I instruction against what it should give: selftest.S)
    assert _cpu("selftest.S").check(R.console(b"\n"), verilog=sim).value == b"OK\n"


@pytest.mark.parametrize("sim", [False, pytest.param(True, marks=pytest.mark.sim)])
def test_a_c_program_prints_and_echoes(sim, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    printed = _cpu("hello.c").check(R.console(b"I echo:\nok", typed=b"ok"), verilog=sim).value
    assert printed.startswith(b"Hello from RISC-V!\nFibonacci:\n0 1 1 2 3 5 8 13 21 34 55 89 144 ")
    assert b"610 987 \n" in printed and printed.endswith(b"I echo:\nok")
