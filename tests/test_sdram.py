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
    sdram_model m(.clk(~clk), .cke(cke), .cs_n(cs_n), .ras_n(ras_n), .cas_n(cas_n),
                  .we_n(we_n), .ba(ba), .sa(sa), .dqm(dqm), .dq(dq));

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
                     m.errors, clocks);
            $finish;
        end
    end
    initial begin
        #100 rst = 0;
        #(20 * 400000) $display("RESULT timeout"); $finish;
    end
endmodule
"""


def _run(tmp_path, seq, nops=60000, seed=7):
    tb = tmp_path / "tb.v"
    tb.write_text(TB.replace("SEED", str(seed)).replace("NOPS", str(nops))
                  .replace("SEQ", "1" if seq else "0"))
    subprocess.run(["iverilog", "-g2012", "-o", str(tmp_path / "sim.vvp"), str(tb),
                    str(HERE / "py2ver_sdram.v"), str(HERE / "sdram_model.v")], check=True)
    out = subprocess.run(["vvp", "-n", str(tmp_path / "sim.vvp")], capture_output=True,
                         text=True, check=True).stdout
    line = next(x for x in out.splitlines() if x.startswith("RESULT"))
    assert "timeout" not in line, out[-2000:]
    return dict(kv.split("=") for kv in line.split()[1:])


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
def test_a_design_with_its_list_and_table_in_the_sdram(monkeypatch, tmp_path):
    import contextlib
    import io
    import random
    from py2ver import Py2ver
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PY2VER_SIM", "icarus")
    with contextlib.redirect_stdout(io.StringIO()):
        p = Py2ver(_with_a_list, {"x": {"signed": 1, "width": 8, "length": 16}}, sequential=True,
                   sdram=0)                                  # (everything in the SDRAM)
    sd = p._fold["sdram"]
    assert list(sd["lists"]) == ["h"] and len(sd["tables"]) == 1
    assert "dram_dq" in (tmp_path / "hdl" / "main.v").read_text()
    r = random.Random(3)
    vectors = [([r.randint(-128, 127) for _ in range(16)],) for _ in range(4)]
    # (written and read back through the controller, the design stopped while the
    # SDRAM opens rows and refreshes: the same results as the model)
    assert p.TB_batch()(vectors) == p.PY_batch()(vectors)
