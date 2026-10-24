import contextlib
import functools
import io
import os
import sys
from pathlib import Path

import pytest

from py2ver import Py2ver
from py2ver.devices.sdram import SdramModel, SdramTiming
from py2ver.protocol import alongside, together

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


def _memtest(words, stride, passes, sim, chip):
    image = R.build("memtest.c", defines=(f"WORDS={words}", f"STRIDE={stride}", f"PASSES={passes}"))
    with contextlib.redirect_stdout(io.StringIO()):
        p = Py2ver(R.cpu_with_sdram(image, timing=SdramTiming(power_up=20, refresh=60)))
    return p.check(R.console(b"done\n", chip=chip), verilog=sim).value


@pytest.mark.parametrize("sim", [False, pytest.param(True, marks=pytest.mark.sim)])
def test_the_sdram_holds_words_bytes_halves_and_code(sim, monkeypatch, tmp_path):
    # (memtest.c on the CPU and the SDRAM controller, the chip's model behind
    # them: 24 words spread over the banks, each a line of its own, written,
    # changed a byte or a half at a time, read back by loads and atomic adds;
    # and a function run from the SDRAM, then changed and run again)
    monkeypatch.chdir(tmp_path)
    chip = SdramModel()
    printed = _memtest(24, 211_001, 1, sim, chip)
    assert printed == b"SDRAM test\nrunning from the SDRAM: ok\npass 1: 0 words wrong\ndone\n"
    assert chip.word(0) == 0x42021 ^ 0xFF        # (the first random number, its low byte inverted)


@pytest.mark.parametrize("sim", [False, pytest.param(True, marks=pytest.mark.sim)])
@pytest.mark.parametrize("stride", [1, 2049])
def test_the_cache_keeps_up_with_stores(stride, sim, monkeypatch, tmp_path):
    # (stride 1: 64 words in 8 lines, the second pass storing into lines the
    # cache holds; 2049, 8 KB and a word: lines taking each other's place)
    monkeypatch.chdir(tmp_path)
    printed = _memtest(64, stride, 2, sim, SdramModel())
    assert printed.endswith(b"pass 1: 0 words wrong\npass 2: 0 words wrong\ndone\n")
    assert b"running from the SDRAM: ok\n" in printed


def test_the_boot_loader_loads_a_program_and_starts_it(monkeypatch, tmp_path):
    # (boot.c on the CPU and the SDRAM: a program typed in as a load record,
    # started by a go record with a1 'Z'; it prints 'K', then a1's low byte;
    # the console's copy for the PC, its room there, the same as the display's)
    monkeypatch.chdir(tmp_path)
    import riscv_linux as L
    program = b"".join(n.to_bytes(4, "little") for n in (
        0x100002B7,                      # lui  t0, 0x10000 (the UART)
        0x04B00313,                      # li   t1, 'K'
        0x00628023,                      # sb   t1, 0(t0)
        0x00B28023,                      # sb   a1, 0(t0)
        0x0000006F))                     # j    .
    typed = L.load_record(0x80000000, program) + L.go_record(0x80000000, ord("Z"))
    with contextlib.redirect_stdout(io.StringIO()):
        p = Py2ver(R.cpu_with_sdram(R.build("boot.c"), timing=SdramTiming(power_up=20, refresh=60)))
    chip = SdramModel()

    def copy_for_the_pc(dut):
        copied = bytearray()
        while True:
            dut.jtag_tx_room = 1
            if dut.jtag_tx_send:
                copied.append(dut.jtag_tx)
            yield
            if copied.endswith(b"KZ"):
                return bytes(copied)

    def tb(dut):
        printed, copied = yield from together(R.console(b"KZ", typed, chip=chip)(dut),
                                              copy_for_the_pc(dut))
        return printed, copied

    printed, copied = p.check(tb, verilog=False).value
    assert printed == (b"py2ver RISC-V boot loader: waiting for the PC\n" + L.loaded_line(0x80000000, program)
                       + b"starting at 80000000\nKZ")
    assert copied == printed
    assert [chip.word(k) for k in range(5)] == [int.from_bytes(program[k:k + 4], "little")
                                                for k in range(0, 20, 4)]


@pytest.mark.parametrize("every", [60, None])
def test_the_pc_s_copy_of_the_console_waits_for_room_while_the_pc_reads(every, monkeypatch, tmp_path):
    # (a FIFO to the PC of a byte (the board top's has 16), taken from every
    # 60 clocks: each byte of the copy waiting for room; or never any room: no
    # one reading, the CPU going on without it)
    monkeypatch.chdir(tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        p = Py2ver(functools.partial(R.rv32ima, image=R.build("traps.S"), pc_patience=200),
                   R.CPU_ATTR)

    copied = bytearray()

    def pc(dut):
        clock, waiting = 0, 0
        while True:
            if dut.jtag_tx_send:
                copied.append(dut.jtag_tx)
                waiting += 1
            if every and clock % every == 0 and waiting:
                waiting -= 1
            dut.jtag_tx_room = 1 if every and waiting < 1 else 0
            clock += 1
            yield

    def tb(dut):
        printed = yield from alongside(R.console(b"\n")(dut), pc(dut))
        return printed

    assert p.check(tb, verilog=False).value == b"OK\n"
    assert copied == (b"OK\n" if every else b"")


def test_the_device_tree_gives_linux_the_sdram():
    import riscv_linux as L
    tree = L.device_tree()
    assert tree[:4] == bytes.fromhex("d00dfeed") and len(tree) % 4 == 0
    assert L.TREE_AT + len(tree) <= 0x82000000                  # (in the SDRAM's last 16 KB)
    assert bytes.fromhex("00000000 80000000 00000000 01ffc000") in tree     # (memory: up to it)


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
