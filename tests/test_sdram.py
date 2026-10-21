import os
import shutil
import subprocess
from pathlib import Path

import pytest

# The SDRAM controller (templates/hw/sdram) against a model of the DE0-Nano's
# chip: random reads and writes, held until ready, read data in order.

HERE = Path(__file__).resolve().parent.parent / "py2ver" / "templates" / "hw" / "sdram"

TB = r"""
`timescale 1ns/1ps
module tb;
    reg clk = 0;
    always #10 clk = ~clk;
    reg rst = 1;
    reg req = 0, we = 0;
    reg [23:0] addr = 0;
    reg [15:0] wdata = 0;
    wire ready, rvalid, init_done;
    wire [15:0] rdata;
    wire [12:0] sa; wire [1:0] ba, dqm; wire cs_n, ras_n, cas_n, we_n, cke;
    wire [15:0] dq;
    py2ver_sdram #(.T_INIT(20), .T_REF(380)) c(
        .clk(clk), .rst(rst), .req(req), .we(we), .addr(addr), .wdata(wdata), .ready(ready),
        .rvalid(rvalid), .rdata(rdata), .init_done(init_done), .sa(sa), .ba(ba), .cs_n(cs_n),
        .ras_n(ras_n), .cas_n(cas_n), .we_n(we_n), .cke(cke), .dqm(dqm), .dq(dq));
    CHIP

    reg [15:0] refm [0:16383];        // {bank, row 0..7, col}
    reg [15:0] want_q [0:65535];
    integer head = 0, tail = 0, n = 0, reads = 0, bad = 0, seed = SEED, clocks = 0;
    integer k;
    function [23:0] place(input [13:0] a);
        place = {a[13:12], 10'b0, a[11:9], a[8:0]};
    endfunction
    reg [13:0] a;
    always @(posedge clk) begin
        clocks = clocks + 1;
        if (rvalid) begin
            if (rdata !== want_q[tail]) begin
                bad = bad + 1;
                if (bad < 5) $display("read %0d: %h, wanted %h", tail, rdata, want_q[tail]);
            end
            tail = tail + 1;
        end
        if (req && ready) begin                     // issued: the next one (or a gap)
            if (!we) begin want_q[head] = refm[a]; head = head + 1; reads = reads + 1; end
            else refm[a] = wdata;
            n = n + 1;
            req <= 0;
        end
        if (!rst && init_done && (!req || ready) && n < NOPS && ($random(seed) & 3) != 0) begin
            if (n < 16384) begin                    // (everything written first)
                a = n; we <= 1;
            end else begin
                a = (SEQ ? n : $random(seed)) & 16383; we <= ($random(seed) & 3) == 0;
            end
            addr <= place(a);
            wdata <= $random(seed);
            req <= 1;
        end
        if (n >= NOPS && tail == head && !req) begin
            $display("RESULT bad=%0d reads=%0d model_errors=%0d clocks=%0d", bad, reads,
                     ERRORS, clocks);
            $finish;
        end
    end
    initial begin
        #100 rst = 0;
        #(20 * 400000) $display("RESULT timeout"); $finish;
    end
endmodule
"""


PY2VER_CHIP = ("sdram_model m(.clk(~clk), .cke(cke), .cs_n(cs_n), .ras_n(ras_n), .cas_n(cas_n), "
               ".we_n(we_n), .ba(ba), .sa(sa), .dqm(dqm), .dq(dq));")
MICRON_CHIP = ("mt48lc16m16a2 m(.Dq(dq), .Addr(sa), .Ba(ba), .Clk(~clk), .Cke(cke), .Cs_n(cs_n), "
               ".Ras_n(ras_n), .Cas_n(cas_n), .We_n(we_n), .Dqm(dqm));")


def _run(tmp_path, seq, nops=60000, seed=7, micron=None):
    """The controller against py2ver's model of the chip, or (micron: the path of
    Micron's MT48LC16M16A2 model, the same organization) against Micron's,
    whose timing errors are its 'ERROR' lines."""
    tb = tmp_path / "tb.v"
    tb.write_text(TB.replace("SEED", str(seed)).replace("NOPS", str(nops))
                  .replace("SEQ", "1" if seq else "0")
                  .replace("CHIP", MICRON_CHIP if micron else PY2VER_CHIP)
                  .replace("ERRORS", "0" if micron else "m.errors"))
    if micron:                       # (its includes: the time unit, no defines)
        (tmp_path / "timescale.v").write_text("`timescale 1ns/1ps\n")
        (tmp_path / "test-defines.v").write_text("\n")
    chip = str(micron) if micron else str(HERE / "sdram_model.v")
    subprocess.run(["iverilog", "-g2012", "-I", str(tmp_path), "-o", str(tmp_path / "sim.vvp"),
                    str(tb), str(HERE / "py2ver_sdram.v"), chip], check=True)
    out = subprocess.run(["vvp", "-n", str(tmp_path / "sim.vvp")], capture_output=True,
                         text=True, check=True).stdout
    line = next(x for x in out.splitlines() if x.startswith("RESULT"))
    assert "timeout" not in line, out[-2000:]
    r = dict(kv.split("=") for kv in line.split()[1:])
    if micron:
        errors = [x for x in out.splitlines() if "ERROR" in x]
        r["model_errors"] = str(len(errors))
        assert not errors, "\n".join(errors[:5])
    return r


@pytest.mark.sim
@pytest.mark.skipif(not shutil.which("iverilog"), reason="needs Icarus")
def test_random_accesses_read_back_what_was_written(tmp_path):
    r = _run(tmp_path, seq=False)
    assert int(r["bad"]) == 0 and int(r["model_errors"]) == 0 and int(r["reads"]) > 30000


@pytest.mark.sim
@pytest.mark.skipif(not shutil.which("iverilog"), reason="needs Icarus")
def test_accesses_within_a_row_run_one_a_clock(tmp_path):
    # In order, with the driver's gaps (a request 3 clocks in 4): a clock an access,
    # but for the rows opened, the refreshes and the writes after reads
    r = _run(tmp_path, seq=True)
    assert int(r["bad"]) == 0 and int(r["model_errors"]) == 0
    assert int(r["clocks"]) < 60000 * 4 / 3 * 1.35


@pytest.mark.sim
@pytest.mark.skipif(not shutil.which("iverilog"), reason="needs Icarus")
@pytest.mark.skipif(not os.environ.get("PY2VER_MICRON_MODEL"),
                    reason="PY2VER_MICRON_MODEL: the path of Micron's mt48lc16m16a2.v")
@pytest.mark.parametrize("seq", [False, True])
def test_the_controller_against_microns_model(tmp_path, seq):
    """Micron's model of a chip organized as the DE0-Nano's (MT48LC16M16A2, not in
    the repository: its own license): its timing checks, and the read data on
    the bus when the chip has it."""
    r = _run(tmp_path, seq=seq, micron=Path(os.environ["PY2VER_MICRON_MODEL"]))
    assert int(r["bad"]) == 0 and int(r["reads"]) > (0 if seq else 30000)


# ----------------------------- a sequenced design with its lists in the SDRAM -----------------------------

_W = tuple((k * 37) % 101 - 50 for k in range(128))


def _with_a_list(x):
    h = [0] * 64
    for i in range(64):
        h[i] = x[i % 16] * (i + 1) + _W[i * 2]
    s = 0
    for i in range(64):
        s = s + h[63 - i] * x[i % 16]
    return s


@pytest.mark.sim
@pytest.mark.skipif(not shutil.which("iverilog"), reason="needs Icarus")
@pytest.mark.parametrize("sim", ["icarus", "verilator"])
def test_a_design_with_its_list_and_table_in_the_sdram(monkeypatch, tmp_path, sim):
    import contextlib
    import io
    import random
    from py2ver import Py2ver
    if sim == "verilator" and not shutil.which("verilator"):
        pytest.skip("Verilator not found")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PY2VER_SIM", sim)
    with contextlib.redirect_stdout(io.StringIO()):
        p = Py2ver(_with_a_list, {"x": {"signed": 1, "width": 8, "length": 16}}, sequential=True,
                   sdram=0)                                  # (everything in the SDRAM)
    sd = p.schedule.sdram
    assert list(sd["lists"]) == ["h"] and len(sd["tables"]) == 1
    assert "dram_dq" in (tmp_path / "hdl" / "main.v").read_text()
    r = random.Random(3)
    vectors = [([r.randint(-128, 127) for _ in range(16)],) for _ in range(4)]
    # (written and read back through the controller, the design stopped while the
    # SDRAM opens rows and refreshes: the same results as the model)
    assert p.TB_batch()(vectors) == p.PY_batch()(vectors)
    # (by the simulator asked for: Verilator builds the chip's model too)
    assert not list((tmp_path / "native_build").glob("verilator_failed_*"))


def _input_in_the_sdram(x):
    s = 0
    for i in range(64):
        s = s + x[i] * _W[i * 2]
    t = 0
    for i in range(64):
        t = t + x[63 - i] * x[i]
    return s, t


@pytest.mark.sim
@pytest.mark.skipif(not shutil.which("iverilog"), reason="needs Icarus")
def test_a_long_input_list_is_written_into_the_sdram(monkeypatch, tmp_path):
    import contextlib
    import io
    import random
    from py2ver import Py2ver
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PY2VER_SIM", "icarus")
    with contextlib.redirect_stdout(io.StringIO()):
        p = Py2ver(_input_in_the_sdram, {"x": {"signed": 1, "width": 8, "length": 64}},
                   sequential=True, sdram=0)
    # (no ports for x: each sample's values written into the SDRAM before it runs)
    assert len(p._ext_inputs) == 64 and [q.name for q in p.ir.inputs] == ["clk", "rst", "ce_in"]
    r = random.Random(1)
    vectors = [([r.randint(-128, 127) for _ in range(64)],) for _ in range(3)]
    assert p.TB_batch()(vectors) == p.PY_batch()(vectors)


def _wide_list(x):
    h = [0] * 64
    for i in range(64):
        h[i] = x[i % 16] * x[(i * 7) % 16] * (i + 1)
    s = 0
    for i in range(64):
        s = s + h[63 - i] - h[i] // 3
    t = 0
    for i in range(64):
        h[i] = h[i] + i
        t = t + h[i] * (i % 5)
    return s, t


@pytest.mark.sim
@pytest.mark.skipif(not shutil.which("iverilog"), reason="needs Icarus")
def test_a_list_of_values_wider_than_a_word_takes_two(monkeypatch, tmp_path):
    import contextlib
    import io
    import random
    from py2ver import Py2ver
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PY2VER_SIM", "icarus")
    with contextlib.redirect_stdout(io.StringIO()):
        p = Py2ver(_wide_list, {"x": {"signed": 1, "width": 8, "length": 16}},
                   sequential=True, sdram=0)
    # (h's values are 22 bits: two words each, read and written in turn)
    h = p.schedule.sdram["lists"]["h"]
    assert h["meta"]["width"] > 16 and h["words"] == 2
    r = random.Random(1)
    vectors = [([r.randint(-128, 127) for _ in range(16)],) for _ in range(3)]
    assert p.TB_batch()(vectors) == p.PY_batch()(vectors)


_WIDE = tuple((k * 40503) % 200003 - 100001 for k in range(128))


def _wide_table(x):
    s = 0
    for i in range(64):
        s = s + x[i % 16] * _WIDE[i * 2 + 1]
    return s


@pytest.mark.sim
@pytest.mark.skipif(not shutil.which("iverilog"), reason="needs Icarus")
def test_a_table_of_values_wider_than_a_word_takes_two(monkeypatch, tmp_path):
    import contextlib
    import io
    import random
    from py2ver import Py2ver
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PY2VER_SIM", "icarus")
    with contextlib.redirect_stdout(io.StringIO()):
        p = Py2ver(_wide_table, {"x": {"signed": 1, "width": 8, "length": 16}},
                   sequential=True, sdram=0)
    # (the table's values are 18 bits: two words each, loaded low word first)
    (t,) = p.schedule.sdram["tables"].values()
    assert t["meta"]["width"] > 16 and t["words"] == 2
    r = random.Random(1)
    vectors = [([r.randint(-128, 127) for _ in range(16)],) for _ in range(3)]
    assert p.TB_batch()(vectors) == p.PY_batch()(vectors)


def _lists_in_turn(x):
    a = [0] * 64
    for i in range(64):
        a[i] = x[i % 16] * (i + 1)
    s = 0
    for i in range(64):
        s = s + a[63 - i]
    b = [0] * 64
    for i in range(64):
        b[i] = x[(i * 5) % 16] - i
    for i in range(64):
        s = s + b[i] * b[63 - i]
    c = [0] * 64
    for i in range(64):
        c[i] = x[(i * 3) % 16] + s
    for i in range(64):
        s = s - c[63 - i]
    d = [0] * 64
    for i in range(64):
        d[i] = x[(i * 7) % 16] * i
    for i in range(64):
        s = s + d[i] // 4
    e = [0] * 64
    for i in range(64):
        e[i] = x[(i * 9) % 16] - s
    t = 0
    for i in range(64):
        t = t + e[63 - i] * (i % 3)
    return s, t


@pytest.mark.sim
@pytest.mark.skipif(not shutil.which("iverilog"), reason="needs Icarus")
def test_lists_not_needed_at_once_share_the_sdram(monkeypatch, tmp_path):
    import contextlib
    import io
    import random
    from py2ver import Py2ver
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PY2VER_SIM", "icarus")
    with contextlib.redirect_stdout(io.StringIO()):
        p = Py2ver(_lists_in_turn, {"x": {"signed": 1, "width": 8, "length": 16}},
                   sequential=True, sdram=0)
    # (five lists, each done with when the next is filled, in four banks: two of
    # them at the same addresses, written over)
    sd = p.schedule.sdram["lists"]
    assert sorted(sd) == ["a", "b", "c", "d", "e"]
    assert len({info["base"] for info in sd.values()}) == 4
    r = random.Random(1)
    vectors = [([r.randint(-128, 127) for _ in range(16)],) for _ in range(3)]
    assert p.TB_batch()(vectors) == p.PY_batch()(vectors)


def _returns_a_list(x):
    h = [0] * 64
    for i in range(64):
        h[i] = x[i % 16] * (i + 1)
    s = 0
    for i in range(16):
        s = s + x[i]
    return s, h


@pytest.mark.sim
@pytest.mark.skipif(not shutil.which("iverilog"), reason="needs Icarus")
def test_a_long_returned_list_is_read_back_from_the_sdram(monkeypatch, tmp_path):
    import contextlib
    import io
    import random
    from py2ver import Py2ver
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PY2VER_SIM", "icarus")
    with contextlib.redirect_stdout(io.StringIO()):
        p = Py2ver(_returns_a_list, {"x": {"signed": 1, "width": 8, "length": 16}},
                   sequential=True, sdram=0)
    # (h has no ports: each sample's values read from the SDRAM after it, in their
    # place among the outputs)
    assert len(p._ext_outputs) == 64 and [q.name for q in p.ir.outputs if q.name == "s"] == ["s"]
    assert p.output_args_list[:3] == ["s", "h_0", "h_1"]
    r = random.Random(1)
    vectors = [([r.randint(-128, 127) for _ in range(16)],) for _ in range(3)]
    assert p.TB_batch()(vectors) == p.PY_batch()(vectors)


def _returns_only_a_list(x):
    h = [0] * 64
    for i in range(64):
        h[i] = x[i % 16] - x[(i + 1) % 16]
    return h


def _wide_ports(x):
    h = [0] * 64
    for i in range(64):
        h[i] = x[i] * ((i % 7) - 3) + x[63 - i]
    s = 0
    for i in range(64):
        s = s + x[i] // 8
    return s, h


@pytest.mark.sim
@pytest.mark.skipif(not shutil.which("iverilog"), reason="needs Icarus")
def test_input_and_returned_lists_wider_than_a_word_take_two(monkeypatch, tmp_path):
    import contextlib
    import io
    import random
    from py2ver import Py2ver
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PY2VER_SIM", "icarus")
    with contextlib.redirect_stdout(io.StringIO()):
        p = Py2ver(_wide_ports, {"x": {"signed": 1, "width": 20, "length": 64}},
                   sequential=True, sdram=0)
    # (x's 20-bit values and h's 23-bit ones: two words each, x's written before
    # a sample, h's read back after)
    sd = p.schedule.sdram["lists"]
    assert sd["x"]["words"] == 2 and sd["h"]["words"] == 2
    assert len(p._ext_inputs) == 64 and len(p._ext_outputs) == 64
    r = random.Random(1)
    vectors = [([r.randint(-2 ** 19, 2 ** 19 - 1) for _ in range(64)],) for _ in range(3)]
    assert p.TB_batch()(vectors) == p.PY_batch()(vectors)


@pytest.mark.board
@pytest.mark.skipif(not shutil.which("iverilog"), reason="needs Icarus")
@pytest.mark.parametrize("f, n, w", [(_with_a_list, 16, 8), (_input_in_the_sdram, 64, 8),
                                     (_returns_a_list, 16, 8), (_returns_only_a_list, 16, 8),
                                     (_wide_table, 16, 8), (_wide_ports, 64, 20)])
def test_the_board_loads_the_sdram_over_its_uart(monkeypatch, tmp_path, f, n, w):
    # The DE0-Nano's project (UART, the commands, the design, the controller) with
    # a model of the chip on its pins: the tables written with 'W' commands, a
    # sample's input lists too, then 'R' and its outputs, as HW() does on the board
    import contextlib
    import io
    import random
    from py2ver import Py2ver
    monkeypatch.chdir(tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        p = Py2ver(f, {"x": {"signed": 1, "width": w, "length": n}}, sequential=True, sdram=0)
    r = random.Random(2)
    vectors = [([r.randint(-2 ** (w - 1), 2 ** (w - 1) - 1) for _ in range(n)],)
               for _ in range(2)]
    assert p.HW_sim()(vectors) == p.PY_batch()(vectors)
