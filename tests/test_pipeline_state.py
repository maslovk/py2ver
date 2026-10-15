import os
import random
import shutil

import pytest

from py2ver import Py2ver
from test_lists import fir_step
from test_state_stream import LPF_ATTR, lpf_step
from test_stream_and_ram import HIST_RAM_ATTR, hist_ram

FIR_ATTR = {
    "x": {"signed": 1, "width": 16},
    "delay": {"signed": 1, "width": 16, "length": 6, "next": "new_delay"},
    "y": {"signed": 1, "width": 16},
}


def pre_acc_post(x, acc):
    u = (x * 3 + 1) * (x - 5)        # input-only: pipelined before the loop
    v = (u >> 2) + x * x
    acc2 = acc + v                   # the state loop: stays in one stage
    y = (acc2 * acc2) >> 8           # state -> output: pipelined after it
    z = y - acc2 * 3
    return z, acc2


ACC_ATTR = {
    "x": {"signed": 1, "width": 8},
    "acc": {"signed": 1, "width": 20, "next": "acc2"},
    "acc2": {"signed": 1, "width": 20},
    "z": {"signed": 1, "width": 32},
}


@pytest.fixture
def sim_dir(monkeypatch, tmp_path):
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("Icarus Verilog tools are not installed")
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.sim
@pytest.mark.parametrize("stages", [1, 2, 3])
def test_pipelined_fir_with_delay_line_state(sim_dir, stages):
    p = Py2ver(fir_step, FIR_ATTR, pipeline=stages)
    rng = random.Random(stages)
    xs = [64] + [0] * 8 + [rng.randint(-2000, 2000) for _ in range(20)]
    sim = p.TB_stream()(xs)

    assert [row[0] for row in sim[:9]] == [2, 8, 14, 16, 14, 8, 2, 0, 0]
    assert sim == p.PY_stream()(xs)


@pytest.mark.sim
@pytest.mark.parametrize("stages", [2, 3])
def test_loop_goes_after_input_logic(sim_dir, stages):
    p = Py2ver(pre_acc_post, ACC_ATTR, pipeline=stages)
    rng = random.Random(10 + stages)
    xs = [rng.randint(-128, 127) for _ in range(40)]

    assert p._pipe.loop_stage == 1          # input logic in stage 0, loop in stage 1
    assert p.TB_stream()(xs) == p.PY_stream()(xs)


@pytest.mark.sim
@pytest.mark.parametrize("func, attr", [(lpf_step, LPF_ATTR), (pre_acc_post, ACC_ATTR)])
@pytest.mark.parametrize("valid, ready", [(1.0, 1.0), (0.5, 0.5), (0.8, 0.3)])
def test_bubbles_and_stalls_leave_state_alone(sim_dir, func, attr, valid, ready):
    # Gaps in the input (bubbles) and a stalling consumer must not update the
    # state: only real samples reaching the loop stage do.
    p = Py2ver(func, attr, pipeline=2)
    rng = random.Random(int(valid * 10 + ready))
    xs = [rng.randint(-128, 127) for _ in range(30)]
    assert p.TB_handshake()(xs, valid=valid, ready=ready, seed=5) == p.PY_stream()(xs)


@pytest.mark.sim
def test_pipelined_state_on_the_virtual_board(sim_dir):
    p = Py2ver(fir_step, FIR_ATTR, pipeline=2)
    xs = [64, 0, 0, 1000, -1000]
    assert p.HW_sim()([(x,) for x in xs]) == p.PY_stream()(xs)


def test_core_has_valid_ports_and_gated_state(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    p = Py2ver(pre_acc_post, ACC_ATTR, pipeline=2)
    verilog = (tmp_path / "hdl" / "main.v").read_text(encoding="utf-8")

    assert p.input_args_list == ["x"] and p.output_args_list == ["z", "acc2"]
    assert "input vin;" in verilog and "output vout;" in verilog
    assert "acc <= rst ? 20'sd0 : ((ce && __v1) ?" in verilog   # loop in stage 1


def test_pipeline_with_ram_is_rejected(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(NotImplementedError, match="block-RAM"):
        Py2ver(hist_ram, HIST_RAM_ATTR, pipeline=2)


@pytest.mark.quartus
def test_pipelining_a_stateful_fir_raises_fmax(monkeypatch, tmp_path):
    tool = os.environ.get("QUARTUS_DIR")
    if not (tool and os.path.exists(os.path.join(tool, "quartus_sh"))):
        found = shutil.which("quartus_sh")
        tool = os.path.dirname(found) if found else None
    if tool is None:
        pytest.skip("Quartus not found (set QUARTUS_DIR or put quartus_sh on PATH)")
    monkeypatch.chdir(tmp_path)

    flat = Py2ver(fir_step, FIR_ATTR).SYN({"QUARTUS_DIR": tool})
    piped = Py2ver(fir_step, FIR_ATTR, pipeline=2).SYN({"QUARTUS_DIR": tool})
    assert piped.fmax_mhz > 1.3 * flat.fmax_mhz         # ~84 -> ~126 MHz
