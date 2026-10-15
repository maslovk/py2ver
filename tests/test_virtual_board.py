import stat

import pytest

from py2ver import Py2ver
from test_state_stream import LPF_ATTR, lpf_step
from test_syn_report import FIXTURES, FOO_ATTR, foo

# The virtual board simulates the whole DE0-Nano project (UART RX/TX, the
# latency handshake and the core) and talks to it over its UART pins with
# the bytes hw_fun() would send, so these tests cover the board path end to
# end except the physical cable.


def board_add_sub(a, b):
    s = a + b
    d = a - b
    return s, d


def board_reg_latency(a, b):
    fast = a ^ b
    s = a + b
    slow1 = s
    slow2 = slow1
    return fast, slow2


def board_muladd(a, b, c):
    p = a * b
    y = p + c
    z = (p >> 3) - c
    return y, z


def board_odd_widths(flag, mid, wide):
    lo = wide & 4095
    y = (wide >> 3) + mid
    z = flag ^ 1
    return y, z, lo


@pytest.fixture
def board_env(monkeypatch, tmp_path):
    import shutil

    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("Icarus Verilog tools are not installed")
    pytest.importorskip("cocotb")
    pytest.importorskip("cocotb_tools")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SIM", "icarus")
    return tmp_path


@pytest.mark.sim
def test_board_combinational(board_env):
    attr = {
        "a": {"signed": 0, "width": 8},
        "b": {"signed": 0, "width": 8},
        "s": {"signed": 0, "width": 9},
        "d": {"signed": 1, "width": 9},
    }
    p = Py2ver(board_add_sub, attr)
    vectors = [(7, 3), (2, 5), (255, 255), (0, 200)]

    assert p.HW_sim()(vectors) == p.TB_batch()(vectors) == [(10, 4), (7, -3), (510, 0), (200, -200)]


@pytest.mark.sim
def test_board_waits_for_reg_latency(board_env):
    attr = {
        "a": {"signed": 0, "width": 8},
        "b": {"signed": 0, "width": 8},
        "fast": {"signed": 0, "width": 8},
        "s": {"signed": 0, "width": 9},
        "slow1": {"signed": 0, "width": 9, "type": "reg"},
        "slow2": {"signed": 0, "width": 9, "type": "reg"},
    }
    p = Py2ver(board_reg_latency, attr)
    vectors = [(7, 3), (100, 50), (200, 13)]

    assert p.tb_latency_cycles == 2
    assert p.HW_sim()(vectors) == [board_reg_latency(*v) for v in vectors]


@pytest.mark.sim
def test_board_pipelined_signed(board_env):
    attr = {
        "a": {"signed": 1, "width": 8},
        "b": {"signed": 1, "width": 8},
        "c": {"signed": 1, "width": 8},
        "p": {"signed": 1, "width": 16},
        "y": {"signed": 1, "width": 17},
        "z": {"signed": 1, "width": 17},
    }
    p = Py2ver(board_muladd, attr, pipeline=2)
    vectors = [(7, 3, 1), (100, 50, -4), (-56, 13, 9), (-128, -128, 127)]

    assert p.HW_sim()(vectors) == [board_muladd(*v) for v in vectors]


@pytest.mark.sim
def test_board_state_advances_once_per_transaction(board_env):
    p = Py2ver(lpf_step, LPF_ATTR)
    xs = [256, 256, 512, -2048, 100, 0]

    assert p.HW_sim()([(x,) for x in xs]) == p.PY_stream()(xs)


@pytest.mark.sim
def test_board_packs_odd_widths(board_env):
    attr = {
        "flag": {"signed": 0, "width": 1},
        "mid": {"signed": 1, "width": 12},
        "wide": {"signed": 0, "width": 20},
        "lo": {"signed": 0, "width": 12},
        "y": {"signed": 1, "width": 20},
        "z": {"signed": 0, "width": 1},
    }
    p = Py2ver(board_odd_widths, attr)
    vectors = [(1, -2048, (1 << 20) - 1), (0, 2047, 0x5A5A5), (1, -1, 8)]

    assert p.HW_sim()(vectors) == p.TB_batch()(vectors)


def board_sum4(a, b, c, d):
    y = a + b + c + d
    return y


@pytest.mark.sim
def test_board_receives_many_back_to_back_bytes(board_env):
    # 8 input bytes sent back to back, like a USB-serial adapter does. The
    # receiver used to lag a little more on every byte and misread the 5th.
    attr = {n: {"signed": 0, "width": 16} for n in "abcd"}
    attr["y"] = {"signed": 0, "width": 18}
    p = Py2ver(board_sum4, attr)
    vectors = [(1000, 2000, 3, 65535), (65535, 65535, 65535, 65535)]

    assert p.rx_width // 8 == 8
    assert p.HW_sim()(vectors) == [(sum(v),) for v in vectors]


# ----------------------------- HW() build checks (fake Quartus) -----------------------------

def _fake_quartus_with_reports(tmp_path, reports_dir):
    tool = tmp_path / "quartus_bin"
    tool.mkdir()
    exe = tool / "quartus_sh"
    # Stand-in for a board compile: drop real reports as <revision>.<kind>
    exe.write_text(
        "#!/bin/sh\n"
        'rev="$3"\n'
        f'for f in "{reports_dir}"/*; do b=$(basename "$f"); cp "$f" "./$rev.${{b#*.}}"; done\n',
        encoding="utf-8",
    )
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return str(tool)


def test_hw_refuses_board_design_that_misses_timing(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    tool = _fake_quartus_with_reports(tmp_path, FIXTURES / "foo" / "output_files")  # 27.6 MHz
    p = Py2ver(foo, FOO_ATTR)

    with pytest.raises(RuntimeError, match="misses 50 MHz timing.*pipeline=N"):
        p.HW({"QUARTUS_DIR": tool})
    assert p.hw_report.meets_timing is False


def test_hw_timing_override_and_no_silent_fallback(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    tool = _fake_quartus_with_reports(tmp_path, FIXTURES / "foo" / "output_files")
    p = Py2ver(foo, FOO_ATTR)

    hw = p.HW({"QUARTUS_DIR": tool, "ALLOW_TIMING_FAILURE": True,
               "UART_PORT": str(tmp_path / "no_such_tty")})

    # No board on that port: raise rather than quietly return simulation results
    with pytest.raises(RuntimeError, match="UART"):
        hw(3, 1)


def test_hw_accepts_stateful_design_that_meets_timing(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    tool = _fake_quartus_with_reports(tmp_path, FIXTURES / "lpf_step" / "output_files")
    p = Py2ver(lpf_step, LPF_ATTR)

    hw = p.HW({"QUARTUS_DIR": tool})

    assert callable(hw)
    assert p.hw_report.meets_timing is True
    top = (tmp_path / "hw" / "lpf_step" / "DE0_Nano.v").read_text(encoding="utf-8")
    assert ".clk(CLOCK_50)" in top and ".ce(core_step)" in top and ".rst(core_rst)" in top
