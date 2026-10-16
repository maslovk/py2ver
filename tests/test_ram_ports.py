import os
import random
import shutil

import pytest

from py2ver import Py2ver

MEM = {"signed": 1, "width": 10, "length": 8, "memory": "ram", "init": [3, 1, 4, 1, 5, 9, 2, 6]}
U3 = {"signed": 0, "width": 3}


def swap(i, j, mem):
    a = mem[i]
    b = mem[j]
    mem[i] = b
    mem[j] = a
    return a - b


def chase(i, mem):
    first = mem[i]
    second = mem[first & 7]          # address depends on another read
    mem[i] = second + 1
    return first, second


def double_write(i, x, mem):
    old = mem[i]
    mem[i] = x
    mem[i] = x + 100                 # last write wins, as in Python
    return old


def fill(i, x, mem):
    for k in range(3):
        mem[(i + k) & 7] = x + k
    return x


def branch_writes(i, x, mem):
    v = mem[i]
    if x > 0:
        mem[i] = v + x
    else:
        mem[(i + 1) & 7] = v - x
    return v


def write_once(i, x, mem):
    mem[i] = x
    return x


def two_memories(i, ptr, data):
    p = ptr[i]
    v = data[p & 7]
    data[p & 7] = v + i
    return p, v


CASES = [
    (swap, {"i": U3, "j": U3, "mem": MEM}, lambda r: (r.randint(0, 7), r.randint(0, 7)), 3),
    (chase, {"i": U3, "mem": MEM}, lambda r: r.randint(0, 7), 3),
    (double_write, {"i": {"signed": 1, "width": 4}, "x": {"signed": 1, "width": 8}, "mem": MEM},
     lambda r: (r.randint(-8, 7), r.randint(-128, 127)), 3),
    (fill, {"i": U3, "x": {"signed": 1, "width": 8}, "mem": MEM},
     lambda r: (r.randint(0, 7), r.randint(-128, 127)), 3),
    (branch_writes, {"i": U3, "x": {"signed": 1, "width": 6}, "mem": MEM},
     lambda r: (r.randint(0, 7), r.randint(-32, 31)), 3),
    (write_once, {"i": U3, "x": {"signed": 1, "width": 8}, "mem": MEM},
     lambda r: (r.randint(0, 7), r.randint(-128, 127)), 1),
    (two_memories, {"i": U3, "ptr": dict(MEM, init=[5, 0, 7, 2, 2, 6, 1, 3]), "data": MEM},
     lambda r: r.randint(0, 7), 3),
]


@pytest.fixture
def sim_dir(monkeypatch, tmp_path):
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("Icarus Verilog tools are not installed")
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.sim
@pytest.mark.parametrize("func, attr, make, clocks", CASES, ids=[c[0].__name__ for c in CASES])
def test_ram_ports_match_python(sim_dir, func, attr, make, clocks):
    p = Py2ver(func, attr)
    rng = random.Random(func.__name__)
    samples = [make(rng) for _ in range(30)]

    assert p.ram_phases == clocks
    assert p.TB_stream()(samples) == p.PY_stream()(samples)
    assert p.TB_handshake()(samples, valid=0.6, ready=0.5, seed=1) == p.PY_stream()(samples)


@pytest.mark.sim
@pytest.mark.board
def test_ram_ports_on_the_virtual_board(sim_dir):
    p = Py2ver(swap, {"i": U3, "j": U3, "mem": MEM})
    samples = [(1, 5), (5, 1), (2, 2), (0, 7)]
    assert p.HW_sim()(samples) == p.PY_stream()(samples)


def test_one_copy_per_read_port(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    Py2ver(swap, {"i": U3, "j": U3, "mem": MEM})
    verilog = (tmp_path / "hdl" / "main.v").read_text(encoding="utf-8")

    assert "reg signed [9:0] mem__m0 [0:7];" in verilog
    assert "reg signed [9:0] mem__m1 [0:7];" in verilog
    assert "mem__m2" not in verilog
    # both copies take every write through the one shared write port
    assert verilog.count("<= mem__wd;") == 2


@pytest.mark.quartus
def test_ram_ports_stay_block_ram_in_quartus(monkeypatch, tmp_path):
    tool = os.environ.get("QUARTUS_DIR")
    if not (tool and os.path.exists(os.path.join(tool, "quartus_sh"))):
        found = shutil.which("quartus_sh")
        tool = os.path.dirname(found) if found else None
    if tool is None:
        pytest.skip("Quartus not found (set QUARTUS_DIR or put quartus_sh on PATH)")
    monkeypatch.chdir(tmp_path)

    attr = {"i": {"signed": 0, "width": 8}, "j": {"signed": 0, "width": 8},
            "mem": {"signed": 1, "width": 16, "length": 256, "memory": "ram"}}
    r = Py2ver(swap, attr).SYN({"QUARTUS_DIR": tool})

    assert r.memory_bits == 2 * 256 * 16      # two read ports -> two copies
    assert r.combinational < 300
    assert r.meets_timing is True
