import random
import shutil

import pytest

from py2ver import Py2ver
from test_lists import histogram as histogram_regs
from test_pipeline import SHA_ATTR, sha4_round_core
from test_state_stream import LPF_ATTR, lpf_step

S8 = {"signed": 1, "width": 8}


# ----------------------------- designs -----------------------------

def mac(a, b, c):
    return a * b + c


def reg_design(a, b):
    s = a + b
    return s


def hist_ram(bin_, counts):
    c = counts[bin_] + 1
    counts[bin_] = c
    return c


def delay_ram(x, i, buf):
    # 'buf' is a Verilog keyword: it is renamed in the HDL, not in Python
    old = buf[i]
    buf[i] = x
    return old


def cond_write(x, addr, mem):
    v = mem[addr]
    if x > 0:
        mem[addr] = v + x
    return v * 2


def read_after_write(x, mem):
    mem[0] = x
    return mem[0]


def whole_memory(x, mem):
    return sum(mem)


def returns_memory(x, mem):
    mem[0] = x
    return x, mem


def keywords(time, buf):
    table = time + buf
    type = table * 2
    return type


HIST_RAM_ATTR = {
    "bin_": {"signed": 0, "width": 4},
    "counts": {"signed": 0, "width": 8, "length": 16, "memory": "ram"},
}
DELAY_RAM_ATTR = {
    "x": S8,
    "i": {"signed": 1, "width": 4},
    "buf": {"signed": 1, "width": 8, "length": 8, "memory": "ram",
            "init": [1, 2, 3, 4, 5, 6, 7, 8]},
}
COND_ATTR = {
    "x": {"signed": 1, "width": 6},
    "addr": {"signed": 0, "width": 3},
    "mem": {"signed": 1, "width": 10, "length": 8, "memory": "ram"},
}


@pytest.fixture
def sim_dir(monkeypatch, tmp_path):
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("Icarus Verilog tools are not installed")
    monkeypatch.chdir(tmp_path)
    return tmp_path


# ----------------------------- valid/ready wrapper -----------------------------

MIXES = [(1.0, 1.0), (0.5, 0.5), (0.9, 0.2), (0.2, 0.9)]


@pytest.mark.sim
@pytest.mark.parametrize("valid, ready", MIXES)
def test_wrapper_combinational(sim_dir, valid, ready):
    p = Py2ver(mac, {n: S8 for n in "abc"})
    rng = random.Random(1)
    vectors = [tuple(rng.randint(-128, 127) for _ in range(3)) for _ in range(30)]
    assert p.TB_handshake()(vectors, valid=valid, ready=ready, seed=3) == p.PY_batch()(vectors)


@pytest.mark.sim
@pytest.mark.parametrize("valid, ready", MIXES)
def test_wrapper_pipelined_stalls_correctly(sim_dir, valid, ready):
    p = Py2ver(sha4_round_core, SHA_ATTR, pipeline=3)
    rng = random.Random(2)
    vectors = [tuple(rng.getrandbits(32) for _ in range(16)) for _ in range(25)]
    assert p.TB_handshake()(vectors, valid=valid, ready=ready, seed=4) == p.PY_batch()(vectors)


@pytest.mark.sim
def test_wrapper_pipelined_takes_one_input_per_clock(sim_dir):
    p = Py2ver(sha4_round_core, SHA_ATTR, pipeline=3)
    rng = random.Random(3)
    vectors = [tuple(rng.getrandbits(32) for _ in range(16)) for _ in range(40)]

    assert p.TB_handshake()(vectors, valid=1.0, ready=1.0) == p.PY_batch()(vectors)
    assert p.handshake_cycles <= len(vectors) + 3 + 2      # n inputs + 3 stages (+ slack)


@pytest.mark.sim
@pytest.mark.parametrize(
    "func, attr, samples",
    [
        (lpf_step, LPF_ATTR, [random.Random(4).randint(-2048, 2047) for _ in range(25)]),
        (histogram_regs, {"bin_": {"signed": 0, "width": 3},
                          "counts": {"signed": 0, "width": 8, "length": 8, "next": "counts"}},
         [random.Random(5).randint(0, 7) for _ in range(25)]),
        (cond_write, COND_ATTR,
         [(random.Random(6).randint(-32, 31), k % 8) for k in range(25)]),
    ],
)
def test_wrapper_with_state_advances_once_per_input(sim_dir, func, attr, samples):
    p = Py2ver(func, attr)
    got = p.TB_handshake()(samples, valid=0.6, ready=0.5, seed=7)
    assert got == p.PY_stream()(samples)


def test_wrapper_rejects_unbalanced_reg_design(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    p = Py2ver(reg_design, {"a": S8, "b": S8, "s": {"signed": 1, "width": 9, "type": "reg"}})
    with pytest.raises(NotImplementedError, match="pipeline=N"):
        p.stream_wrapper()


# ----------------------------- block-RAM memories -----------------------------

@pytest.mark.sim
@pytest.mark.parametrize(
    "func, attr, make",
    [
        (hist_ram, HIST_RAM_ATTR, lambda r: r.randint(0, 15)),
        (delay_ram, DELAY_RAM_ATTR, lambda r: (r.randint(-128, 127), r.randint(-8, 7))),
        (cond_write, COND_ATTR, lambda r: (r.randint(-32, 31), r.randint(0, 7))),
    ],
)
def test_ram_matches_python(sim_dir, func, attr, make):
    rng = random.Random(func.__name__)
    samples = [make(rng) for _ in range(40)]
    if func is hist_ram:
        samples += [3] * 6           # same address back to back
    p = Py2ver(func, attr)
    assert p.TB_stream()(samples) == p.PY_stream()(samples)


@pytest.mark.sim
@pytest.mark.board
def test_ram_on_the_virtual_board(sim_dir):
    p = Py2ver(cond_write, COND_ATTR)
    samples = [(5, 3), (7, 3), (-2, 3), (31, 0), (1, 0)]
    assert p.HW_sim()(samples) == p.PY_stream()(samples)


def test_ram_verilog_uses_an_array(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    Py2ver(delay_ram, DELAY_RAM_ATTR)
    verilog = (tmp_path / "hdl" / "main.v").read_text(encoding="utf-8")

    assert "reg signed [7:0] buf___m0 [0:7];" in verilog   # keyword renamed
    assert "buf___m0[0] = 8'd1;" in verilog                # initial contents
    assert "if (ce && (__phase == 1'd0)) buf___rdata0 <= buf___m0[" in verilog


@pytest.mark.parametrize(
    "func, match",
    [
        (read_after_write, "read after being written"),
        (whole_memory, "only be used as"),
        (returns_memory, "don't return it"),
    ],
)
def test_ram_restrictions_raise_clearly(monkeypatch, tmp_path, func, match):
    monkeypatch.chdir(tmp_path)
    attr = {"x": S8, "mem": {"signed": 1, "width": 8, "length": 4, "memory": "ram"}}
    with pytest.raises(NotImplementedError, match=match):
        Py2ver(func, attr)


@pytest.mark.sim
def test_verilog_keywords_are_renamed(sim_dir):
    p = Py2ver(keywords, {"time": S8, "buf": S8})
    vectors = [(1, 2), (-128, 127), (100, 100)]

    assert p.input_args_list == ["time_", "buf_"]
    assert p.TB_batch()(vectors) == p.PY_batch()(vectors)


def test_ram_uses_block_ram_in_quartus(quartus):
    attr = {"bin_": {"signed": 0, "width": 8},
            "counts": {"signed": 0, "width": 16, "length": 256, "memory": "ram"}}
    r = Py2ver(hist_ram, attr).SYN(quartus)

    assert r.memory_bits == 256 * 16
    assert r.combinational < 200
    assert r.meets_timing is True
