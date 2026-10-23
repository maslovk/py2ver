import contextlib
import io
import sys
from pathlib import Path

import pytest

from py2ver import Py2ver
from py2ver.devices.sdram import SdramModel

# py2ver/devices/sdram.py: the SDRAM controller written as a process, checked
# by examples/sdram_test.py's memory test on the chip's model. (The controller
# in Verilog, for the sequencer's networks, is test_sdram.py's.)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
import sdram_test as S  # noqa: E402


def _quiet(f, attr=None, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return Py2ver(f, attr, **kw)


def xorshift(x):
    """The memory test's random numbers."""
    x ^= (x << 13) & 0xFFFFFFFF
    x ^= x >> 17
    return x ^ ((x << 5) & 0xFFFFFFFF)


def _small():
    """40 words, spread over every bank and many rows; a short start."""
    return _quiet(S.sdram_test(words=40, stride=211_001, timing=S.QUICK))


@pytest.mark.parametrize("sim", [False, pytest.param(True, marks=pytest.mark.sim)])
def test_the_memory_test_passes_on_the_chip(sim, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    chip = SdramModel()
    run = _small().check(S.console(passes=2, chip=chip), verilog=sim)
    assert run.value == (b"sdram test\npass 0001: 000000 words wrong\n"
                         b"pass 0002: 000000 words wrong\n")
    names = [c[1] for c in chip.commands]
    assert names[:5] == ["precharge", "refresh", "refresh", "mode", "active"]
    assert chip.commands[3][3] == 0b000_1_00_010_0_000               # (CAS latency 2, bursts of 1)


def test_each_word_holds_what_was_written_and_its_changed_byte(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    chip = SdramModel()
    _small().check(S.console(chip=chip), verilog=False)
    x, address = 1, 0
    for k in range(40):
        x = xorshift(x)
        assert chip.word(address) == x ^ (0xFF << (8 * (k & 3))), k
        address = (address + 211_001) % S.WORDS


def test_refreshes_come_often_enough(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    chip = SdramModel()
    _small().check(S.console(passes=2, chip=chip), verilog=False)
    refreshes = [c[0] for c in chip.commands if c[1] == "refresh"][2:]   # (after the start's two)
    gaps = [b - a for a, b in zip(refreshes, refreshes[1:])]
    assert len(gaps) > 10
    assert max(gaps) <= S.QUICK.refresh + 16                         # (an access finishing first)


@pytest.mark.parametrize("delay", [3, 5])
def test_a_read_a_clock_early_or_late_is_caught(delay, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    run = _small().check(S.console(chip=SdramModel(delay=delay)), verilog=False)
    assert run.value.endswith(b"pass 0001: 000028 words wrong\n")      # (all 40)


def test_the_board_drives_the_data_bus_only_to_write():
    top = _quiet(S.sdram_test(), pins=S.PINS).board_top()
    assert "assign DRAM_CLK = ~clk;" in top
    assert "assign DRAM_DQ[15] = dq_oe_w ? dq_out_w[15] : 1'bz;" in top
    assert top.index("wire dq_oe_w;") < top.index("assign DRAM_DQ[0]")
    assert ".dq_in(dq_in_s2)" in top                                   # (read through two registers)
