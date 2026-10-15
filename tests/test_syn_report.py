import os
import shutil
import stat
from pathlib import Path

import pytest

import py2ver as py2ver_mod
from py2ver import Py2ver
from syn_report import parse_setup_slack, parse_summary, read_report, summary_int
from test_state_stream import LPF_ATTR, lpf_step

# Trimmed reports from real Quartus 22.1std compiles of the SYN() wrapper
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "quartus"


def foo(a, b):
    s = a + b
    m = a * b
    q = a / b
    return s, m, q


FOO_ATTR = {
    "a": {"width": 16},
    "b": {"width": 16},
    "s": {"width": 17},
    "m": {"width": 32},
    "q": {"width": 16},
}


def _quartus_dir():
    env = os.environ.get("QUARTUS_DIR")
    if env and os.path.exists(os.path.join(env, "quartus_sh")):
        return env
    found = shutil.which("quartus_sh")
    return os.path.dirname(found) if found else None


# ----------------------------- parsers on real reports -----------------------------

def test_report_divider_design_fails_timing():
    r = read_report(FIXTURES / "foo", "foo_syn_top", "foo", 20.0)

    assert r.device == "EP4CE22F17C6"
    assert r.combinational == 317
    assert r.registers == 0
    assert r.multipliers_9bit == 2
    assert r.memory_bits == 0
    assert r.logic_elements == 328
    assert r.fmax_mhz == 27.62          # worst of the 85C/0C slow corners
    assert r.setup_slack_ns == -16.201  # worst over all corners
    assert r.meets_timing is False
    assert "FAILED" in str(r)


def test_report_stateful_design_counts_core_registers():
    r = read_report(FIXTURES / "lpf_step", "lpf_step_syn_top", "lpf_step", 20.0)

    assert r.combinational == 48
    assert r.registers == 16             # y_prev only, not the wrapper's I/O registers
    assert r.fmax_mhz == 194.21
    assert r.meets_timing is True
    assert r.fit_summary["Total virtual pins"] == "29"


def test_report_missing_files_gives_none(tmp_path):
    r = read_report(tmp_path, "nothing", "nothing", 20.0)

    assert r.combinational is None
    assert r.fmax_mhz is None
    assert r.meets_timing is None
    assert "n/a" in str(r)


def test_summary_helpers():
    s = parse_summary("Total logic elements : 1,328 / 22,320 ( 6 % )\n    Total registers : 65\n")
    assert summary_int(s, "Total logic elements") == 1328
    assert summary_int(s, "Total registers") == 65
    assert summary_int(s, "Missing") is None


def test_setup_slack_ignores_hold_and_pulse_width():
    text = (
        "Type  : Slow 1200mV 85C Model Setup 'clk'\nSlack : 2.5\n\n"
        "Type  : Slow 1200mV 85C Model Hold 'clk'\nSlack : -1.0\n\n"
        "Type  : Fast 1200mV 0C Model Setup 'clk'\nSlack : 1.25\n"
    )
    assert parse_setup_slack(text) == 1.25


# ----------------------------- SYN() flow without Quartus -----------------------------

def _fake_quartus(tmp_path, body):
    tool_dir = tmp_path / "quartus_bin"
    tool_dir.mkdir()
    exe = tool_dir / "quartus_sh"
    exe.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return str(tool_dir)


def test_syn_parses_reports_written_by_quartus(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    src = FIXTURES / "lpf_step" / "output_files"
    # Stand-in for quartus_sh: emit the real reports into output_files/
    tool = _fake_quartus(tmp_path, f'mkdir -p output_files && cp "{src}"/* output_files/\n')

    r = Py2ver(lpf_step, LPF_ATTR).SYN({"QUARTUS_DIR": tool})

    assert r.registers == 16
    assert r.meets_timing is True
    assert r.project_dir == tmp_path / "syn" / "lpf_step"


def test_syn_compile_failure_reports_errors(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    tool = _fake_quartus(
        tmp_path,
        'echo "Info: starting"\necho "Error (10161): object x is not declared"\nexit 3\n',
    )

    with pytest.raises(RuntimeError, match=r"rc=3[\s\S]*object x is not declared"):
        Py2ver(foo, FOO_ATTR).SYN({"QUARTUS_DIR": tool})


def test_syn_without_quartus_raises(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path))  # no quartus_sh on PATH

    with pytest.raises(RuntimeError, match="QUARTUS_DIR"):
        Py2ver(foo, FOO_ATTR).SYN()


def test_syn_project_wraps_stateful_core(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    p = Py2ver(lpf_step, LPF_ATTR)

    proj = p.gen_syn_project({"CLK_PERIOD_NS": 8, "DEVICE": "EP4CE6E22C8"})

    wrapper = (proj / "lpf_step_syn_top.v").read_text(encoding="utf-8")
    assert "input [11:0] x_in" in wrapper
    assert "input rst_in" in wrapper
    assert "output reg [15:0] y_out" in wrapper
    assert ".clk(clk)" in wrapper and ".rst(rst_q)" in wrapper
    assert "y_prev" not in wrapper  # state stays inside the core

    qsf = (proj / "lpf_step_syn_top.qsf").read_text(encoding="utf-8")
    assert "DEVICE EP4CE6E22C8" in qsf
    assert "TOP_LEVEL_ENTITY lpf_step_syn_top" in qsf
    assert "create_clock -name clk -period 8.0" in (proj / "lpf_step_syn_top.sdc").read_text()
    assert (proj / "lpf_step.v").exists()


def test_syn_project_combinational_core_has_no_clk_connection(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    proj = Py2ver(foo, FOO_ATTR).gen_syn_project({})

    wrapper = (proj / "foo_syn_top.v").read_text(encoding="utf-8")
    assert ".clk(" not in wrapper
    assert "create_clock -name clk -period 20.0" in (proj / "foo_syn_top.sdc").read_text()


# ----------------------------- real Quartus -----------------------------

@pytest.mark.quartus
def test_syn_real_quartus_lpf(monkeypatch, tmp_path):
    tool = _quartus_dir()
    if tool is None:
        pytest.skip("Quartus not found (set QUARTUS_DIR or put quartus_sh on PATH)")
    monkeypatch.chdir(tmp_path)

    r = Py2ver(lpf_step, LPF_ATTR).SYN({"QUARTUS_DIR": tool})

    assert r.registers == 16
    assert r.combinational and r.combinational > 0
    assert r.fmax_mhz and r.fmax_mhz > 50
    assert r.meets_timing is True
    assert r.fit_summary["Total virtual pins"] == "29"  # x(12) + rst(1) + y(16)
