import contextlib
import functools
import io
import os
import sys
from pathlib import Path

import pytest

from py2ver import Py2ver
from py2ver.process import drive
from py2ver.devices.ili9341 import ILI9341, ILI9341Model
from py2ver.devices.sdram import SdramModel, SdramTiming
from py2ver.protocol import SPI, alongside, together

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


@pytest.mark.parametrize("sim", [False, pytest.param(True, marks=pytest.mark.sim)])
def test_the_display_shows_the_framebuffer(sim, monkeypatch, tmp_path):
    # (screen.c's test pattern on a 16 x 4 screen: drawn by the CPU into the
    # SDRAM, read by the framebuffer's scanner as the controller's second
    # client, between the CPU's accesses, and sent to the display)
    monkeypatch.chdir(tmp_path)
    width, height = 16, 4
    image = R.build("screen.c", defines=(f"WIDTH={width}", f"HEIGHT={height}", "PATTERN_ONLY"))
    lcd = ILI9341(spi=SPI(1, mode=0, miso=""), wake_clocks=50)
    with contextlib.redirect_stdout(io.StringIO()):
        p = Py2ver(R.computer(image, lcd=lcd, height=height, width=width, screen="framebuffer",
                              timing=SdramTiming(power_up=20, refresh=60)))
    chip = SdramModel(cs_n="sdram_cs_n")
    screen = ILI9341Model(lcd._replace(cs_n="display_cs_n"), width, height)

    def tb(dut):
        copied, frames_after = bytearray(), None
        dut.keys = 3
        while frames_after is None or screen.written < frames_after:
            dut.jtag_tx_room = 1
            if dut.jtag_tx_send:
                copied.append(dut.jtag_tx)
                if copied.endswith(b"done\n"):           # (then two whole frames more)
                    frames_after = screen.written + 2 * width * height
            yield
        return bytes(copied)

    def run(dut):
        printed = yield from alongside(tb(dut), chip.run(dut), screen.run(dut))
        return printed

    assert p.check(run, verilog=sim).value == b"framebuffer: a test pattern\ndone\n"

    def rgb(red, green, blue):
        return (red >> 3) << 11 | (green >> 2) << 5 | blue >> 3
    assert screen.pixels == [[rgb(x * 256 // width, y * 256 // height, 128) for x in range(width)]
                             for y in range(height)]


def test_key1_sends_the_pc_where_the_cpu_is(monkeypatch, tmp_path):
    # (traps.S's last loop: KEY1 held, the report sent once, its mcause the
    # last trap's: the ebreak back to machine mode from supervisor mode)
    monkeypatch.chdir(tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        p = Py2ver(functools.partial(R.rv32ima, image=R.build("traps.S")), R.CPU_ATTR)

    def tb(dut):
        printed = yield from R.console(b"\n")(dut)
        report = bytearray()
        dut.keys = 1                         # (KEY1 pressed, KEY0 not)
        for _ in range(3000):
            dut.jtag_tx_room = 1
            dut.tx_ready = 1
            if dut.jtag_tx_send:
                report.append(dut.jtag_tx)
            yield
        return printed, bytes(report)

    printed, report = p.check(tb, verilog=False).value
    assert printed == b"OK\n"
    assert report.startswith(b"\n[pc ") and report.endswith(b"]\n") and report.count(b"[pc") == 1
    fields = report.decode().strip("\n[]").split()
    assert fields[0::2] == ["pc", "mcause", "mepc", "mtval"]
    assert fields[3] == "00000003"


@pytest.mark.parametrize("model", [False, True])
def test_a_page_mapped_elsewhere_reads_the_new_page_after_sfence_vma(model, monkeypatch, tmp_path):
    # (paging.S in supervisor mode, its pages in the RAM and the SDRAM: the
    # TLB's old mapping gone after sfence.vma, an unmapped page a fault; on the
    # generator, and through the hardware's model)
    monkeypatch.chdir(tmp_path)
    system = R.cpu_with_sdram(R.build("paging.S"), timing=SdramTiming(power_up=20, refresh=60))
    tb = R.console(b"\n", chip=SdramModel())
    if model:
        with contextlib.redirect_stdout(io.StringIO()):
            printed = Py2ver(system).check(tb, verilog=False).value
    else:
        printed = drive(system, tb, {}).value
    assert printed == b"OK\n"


def test_the_mmu_device_tree_and_build_suit_the_cpu():
    import riscv_linux as L
    tree = L.device_tree("linux_mmu.dts", room=4096)
    assert tree[:4] == bytes.fromhex("d00dfeed") and L.TREE_AT + len(tree) <= 0x82000000
    assert b"riscv,sv32" in tree and b"rv32ima_zicsr_zifencei" in tree and b"ns16550a" in tree
    config = L.MMU_CONFIG.format(overlay="o", kernel_options="k", busybox_options="b", patches="p")
    assert "# BR2_RISCV_ISA_RVC is not set" in config and "PLATFORM_RISCV_ISA=rv32ima_zicsr_zifencei" in config
    # (OpenSBI must take the device tree's address from us: see OPENSBI_PATCH)
    assert "FW_JUMP_FDT_ADDR=0x81FFC000" in config and 'BR2_GLOBAL_PATCH_DIR="p"' in config
    assert "+FW_JUMP_FDT_ADDR?=" in L.OPENSBI_PATCH
    options = L.fragment(L.MMU_KERNEL_OPTIONS)
    assert "# CONFIG_RISCV_ISA_C is not set\n" in options and "# CONFIG_EFI is not set\n" in options
    assert "CONFIG_FB_SIMPLE=y\n" in options


def test_our_kernel_s_options_take_the_place_of_mini_rv32ima_s():
    import riscv_linux as L
    assert L.git_hash(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"   # (git's own)
    config = "CONFIG_A=y\n# CONFIG_FB is not set\nCONFIG_DEVMEM=n\nCONFIG_FBX=y\n"
    out = L.with_options(config, {"CONFIG_FB": "y", "CONFIG_DEVMEM": "y"})
    assert out == "CONFIG_A=y\nCONFIG_FBX=y\nCONFIG_FB=y\nCONFIG_DEVMEM=y\n"


def test_the_device_tree_gives_linux_the_sdram():
    import riscv_linux as L
    tree = L.device_tree()
    assert tree[:4] == bytes.fromhex("d00dfeed") and len(tree) % 4 == 0
    assert L.TREE_AT + len(tree) <= 0x82000000                  # (in the SDRAM's last 16 KB)
    assert bytes.fromhex("00000000 80000000 00000000 01fd0000") in tree     # (memory: up to the
    assert bytes.fromhex("00000000 81fd0000 00000000 00025800") in tree     # framebuffer, then it)
    assert R.FRAMEBUFFER == 0x81FD0000 and 0x81FD0000 + 240 * 320 * 2 <= L.TREE_AT


NOT_YET = set()


def _riscv_tests():
    names = []
    for suite in ("rv32ui", "rv32um", "rv32ua", "rv32si"):
        folder = Path(R.RISCV_TESTS) / "isa" / suite
        if folder.is_dir():
            names += [f"{suite}/{p.stem}" for p in sorted(folder.glob("*.S"))
                      if f"{suite}/{p.stem}" not in NOT_YET]
    return names


@pytest.mark.slow
@pytest.mark.skipif(not os.path.isdir(R.RISCV_TESTS), reason="needs riscv-tests in ~/opt/riscv")
@pytest.mark.parametrize("name", _riscv_tests())
def test_riscv_tests(name):
    # (riscv-tests' rv32ui, rv32um, rv32ua, rv32si: each prints OK, or FAIL and
    # its failing case; ma_data among them: loads and stores at any address;
    # rv32si's in supervisor mode. On the CPU as written, the generator: that
    # its hardware does the same, test_riscv_tests_in_the_hardware and the
    # CPU's own programs check)
    run = drive(functools.partial(R.rv32ima, image=R.riscv_test(name)), R.console(b"\n", limit=300_000),
                R.CPU_ATTR, limit=400_000)
    assert run.value == b"OK\n"


@pytest.mark.slow
@pytest.mark.skipif(not os.path.isdir(R.RISCV_TESTS), reason="needs riscv-tests in ~/opt/riscv")
@pytest.mark.parametrize("name", [n for n in _riscv_tests() if not n.startswith("rv32si/")])
def test_riscv_tests_with_paging(name):
    # (each user-mode test again, in user mode with Sv32 paging on: every
    # fetch, load and store through a two-level walk; the page tables at
    # 0x2000 and 0x3000, the test below them)
    image = R.riscv_test(name, paging=True)
    assert not any(image[0x2000 // 4:])
    run = drive(functools.partial(R.rv32ima, image=image), R.console(b"\n", limit=1_000_000),
                R.CPU_ATTR, limit=1_200_000)
    assert run.value == b"OK\n"


@pytest.mark.slow
@pytest.mark.skipif(not os.path.isdir(R.RISCV_TESTS), reason="needs riscv-tests in ~/opt/riscv")
@pytest.mark.parametrize("name, paging", [("rv32ui/add", False), ("rv32ui/ma_data", False),
                                          ("rv32ua/lrsc", False), ("rv32si/csr", False),
                                          ("rv32si/dirty", False), ("rv32ui/ld_st", True),
                                          ("rv32ua/amoadd_w", True)])
def test_riscv_tests_in_the_hardware(name, paging, monkeypatch, tmp_path):
    # (a few, through py2ver's model of the hardware as well, some with paging
    # on: the TLBs, the walk, page faults and sfence.vma in the hardware too)
    monkeypatch.chdir(tmp_path)
    printed = _cpu(R.riscv_test(name, paging=paging)).check(R.console(b"\n", limit=1_000_000),
                                                            verilog=False).value
    assert printed == b"OK\n"
