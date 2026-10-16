# native.py
"""
Simulation without cocotb: a plain Verilog testbench, run by Icarus or Verilator.

cocotb drives a design from Python, a round trip per clock; fine for short
runs, slow for many vectors or clocks. Here the testbench is Verilog: it reads
one input vector per sample from a file ($fscanf), steps the design as the
cocotb testbenches do, and writes each sample's outputs to a file. It uses no
delays (its clock comes from outside), so the same file runs on

  - Icarus (iverilog / vvp), with a two-line clock module, and
  - Verilator, even 4.x, compiled to C++ with a small main: much faster per
    clock on a large design, after a compile that is cached

The vectors aren't part of the testbench, so a compiled simulation is reused
(keyed by the design's Verilog and the testbench) until the design changes.

Which: PY2VER_SIM=icarus | verilator | cocotb (default: icarus, or verilator
when it is installed and the run is long, see LONG_RUN).
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Icarus takes about a microsecond per clock and line of the design's Verilog;
# above this many clock-lines (some seconds) Verilator is worth its compile
LONG_RUN = 5_000_000

log = logging.getLogger(__name__)

_MAIN_CPP = """#include "Vpy2ver_tb.h"
#include "verilated.h"
int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Vpy2ver_tb* top = new Vpy2ver_tb;
    top->clk = 0;
    top->eval();
    while (!Verilated::gotFinish()) {
        top->clk = !top->clk;
        top->eval();
    }
    top->final();
    delete top;
    return 0;
}
"""

_CLOCK_V = """module py2ver_clock;
  reg clk = 0;
  always #5 clk = ~clk;
  py2ver_tb tb(.clk(clk));
endmodule
"""


def mode_of(p) -> Optional[Dict[str, Any]]:
    """How the testbench steps this design, or None if cocotb must (latency 'reg'
    signals)."""
    stateful = bool(p.state_map or p.rams or p._fold)
    if stateful:
        return {"kind": "stream", "phases": int(getattr(p, "ram_phases", 1)),
                "latency": p.pipeline_stages if p._has_vin else 0, "vin": p._has_vin}
    if p.pipeline_stages:
        return {"kind": "pipeline", "latency": int(p.pipeline_stages)}
    if p.has_clk:
        return None
    return {"kind": "comb"}


def testbench(p, mode: Dict[str, Any]) -> str:
    """The Verilog testbench (top py2ver_tb, clocked by its clk input)."""
    ins = list(zip(p.input_args_list, p.input_args_width_list))
    outs = list(zip(p.output_args_list, p.output_args_width_list))
    iw = max(1, sum(w for _, w in ins))
    ports = {q.name for q in p.ir.inputs}
    lines = ["module py2ver_tb(input clk);",
             f"  reg [{iw - 1}:0] cur, nxt;",
             "  integer fin, fout, n, k, c, got, total;"]
    for name, w in outs:
        lines.append(f"  wire [{w - 1}:0] o_{name};")
    for ctl in ("rst", "ce", "vin"):
        if ctl in ports:
            lines.append(f"  reg {ctl};")
    if "vout" in {q.name for q in p.ir.outputs}:
        lines.append("  wire vout;")
    conns = []
    lo = iw
    for name, w in ins:                         # the vector holds inputs in port order, first highest
        lo -= w
        conns.append(f".{name}(cur[{lo + w - 1}:{lo}])")
    conns += [f".{name}(o_{name})" for name, _ in outs]
    conns += [f".{c}({c})" for c in ("clk", "rst", "ce", "vin") if c in ports]
    if "vout" in {q.name for q in p.ir.outputs}:
        conns.append(".vout(vout)")
    lines.append(f"  {p.t_name} dut({', '.join(conns)});")
    fmt = " ".join(["%h"] * len(outs))
    write = f'$fwrite(fout, "{fmt}\\n", {", ".join(f"o_{n}" for n, _ in outs)});'
    # The next vector, into nxt (applied with <= after the edge that reads the last)
    lines += [
        "  task next_vector;",
        "    begin",
        "      if (n > 0) begin got = $fscanf(fin, \"%h\\n\", nxt); n = n - 1; end",
        "      else nxt = 0;",
        "    end",
        "  endtask",
        "  initial begin",
        '    fin = $fopen("vectors.hex", "r");',
        '    fout = $fopen("outputs.hex", "w");',
        '    if (!$value$plusargs("n=%d", total)) total = 0;',
        "    n = total; k = 0; c = 0; cur = 0; nxt = 0;",
    ]
    kind = mode["kind"]
    if "rst" in ports:
        lines.append("    rst = 1;")
    if "ce" in ports:
        lines.append("    ce = 1;")
    if "vin" in ports:
        lines.append("    vin = 0;")
    if kind in ("comb", "pipeline"):
        lines.append("    next_vector; cur = nxt;")
    lines += ["  end", "  always @(posedge clk) begin"]
    if kind == "comb":
        # A vector per clock; outputs settle in the clock it is applied in
        lines += [f"    {write}",
                  "    k = k + 1;",
                  "    if (k == total) begin $fclose(fout); $finish; end",
                  "    next_vector; cur <= nxt;"]
    elif kind == "pipeline":
        lat = mode["latency"]
        lines += [f"    if (k >= {lat}) {write}",
                  "    k = k + 1;",
                  f"    if (k == total + {lat}) begin $fclose(fout); $finish; end",
                  "    next_vector; cur <= nxt;"]
    else:
        P, lat = mode["phases"], mode["latency"]
        lines += ["    if (c == 0) begin                     // the reset clock",
                  "      rst <= 0;" if "rst" in ports else "",
                  "      c = 1; k = 0;",
                  "      next_vector; cur <= nxt;"]
        if mode["vin"]:
            lines.append("      vin <= (total > 0);")
        lines.append("    end else begin")
        if lat:
            # Pipelined with state: a sample per clock, its result `lat` clocks later
            lines += [f"      if (c > {lat}) {write}",
                      "      c = c + 1;",
                      f"      if (c > total + {lat}) begin $fclose(fout); $finish; end",
                      "      next_vector; cur <= nxt;",
                      "      vin <= (c <= total);"]
        else:
            # Each sample holds its inputs for P clocks; outputs are read in the last
            lines += [f"      if (k == {P - 1}) begin",
                      f"        {write}",
                      "        k = 0; c = c + 1;",
                      "        if (c > total) begin $fclose(fout); $finish; end",
                      "        next_vector; cur <= nxt;",
                      "      end else k = k + 1;"]
        lines.append("    end")
    lines += ["  end", "endmodule", ""]
    return "\n".join(x for x in lines if x != "")


def _key(*texts: str) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode())
    return h.hexdigest()[:16]


def _run(cmd: List[str], cwd: Path) -> None:
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:2])} failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")


def build(p, mode: Dict[str, Any], simulator: str, workdir: Path) -> Path:
    """The compiled simulation (cached by design and testbench); returns its directory."""
    core = (p.hdl_dir / "main.v").read_text(encoding="utf-8")
    tb = testbench(p, mode)
    d = workdir / f"{simulator}_{_key(core, tb, simulator)}"
    exe = d / ("sim.vvp" if simulator == "icarus" else "obj/Vpy2ver_tb")
    if exe.exists():
        return d
    tmp = d.with_name(d.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    (tmp / "main.v").write_text(core, encoding="utf-8")
    (tmp / "tb.v").write_text(tb, encoding="utf-8")
    if simulator == "icarus":
        (tmp / "clock.v").write_text(_CLOCK_V, encoding="utf-8")
        _run(["iverilog", "-g2012", "-o", "sim.vvp", "-s", "py2ver_clock",
              "clock.v", "tb.v", "main.v"], tmp)
    else:
        (tmp / "main.cpp").write_text(_MAIN_CPP, encoding="utf-8")
        _run(["verilator", "--cc", "--exe", "--build", "-O2", "-Wno-fatal", "-Wno-lint",
              "-Wno-style", "-Wno-TIMESCALEMOD", "--top-module", "py2ver_tb", "-Mdir", "obj",
              "tb.v", "main.v", "main.cpp"], tmp)
    if d.exists():
        shutil.rmtree(d)
    tmp.rename(d)
    return d


def run(p, vectors: List[Tuple[int, ...]], simulator: str) -> List[Dict[str, int]]:
    """Simulate the vectors (raw port values, one per sample); rows of raw outputs.
    A design Verilator can't build (e.g. a line too long for its preprocessor)
    runs on Icarus."""
    mode = mode_of(p)
    if mode is None:
        raise ValueError("This design needs the cocotb testbench")
    workdir = p.outdir / "native_build"
    if simulator == "verilator":
        core = (p.hdl_dir / "main.v").read_text(encoding="utf-8")
        failed = workdir / f"verilator_failed_{_key(core, testbench(p, mode))}"
        if failed.exists():
            simulator = "icarus"
        else:
            try:
                build(p, mode, simulator, workdir)
            except RuntimeError as e:
                log.warning("Verilator can't build this design, using Icarus: %s",
                            str(e).strip().splitlines()[1] if "\n" in str(e) else e)
                workdir.mkdir(parents=True, exist_ok=True)
                failed.write_text(str(e))
                simulator = "icarus"
    d = build(p, mode, simulator, workdir)
    widths = list(p.input_args_width_list)
    with (d / "vectors.hex").open("w") as f:
        for vec in vectors:
            word = 0
            for v, w in zip(vec, widths):
                word = (word << w) | (int(v) & ((1 << w) - 1))
            f.write(f"{word:x}\n")
    out = d / "outputs.hex"
    if out.exists():
        out.unlink()
    arg = f"+n={len(vectors)}"
    if simulator == "icarus":
        _run(["vvp", "-n", "sim.vvp", arg], d)
    else:
        _run(["obj/Vpy2ver_tb", arg], d)
    rows = []
    names = list(p.output_args_list)
    for line in out.read_text().split("\n"):
        if line.strip():
            vals = line.split()
            rows.append({n: int(v, 16) if "x" not in v.lower() and "z" not in v.lower() else 0
                         for n, v in zip(names, vals)})
    return rows


def choose_simulator(p, clocks: int) -> str:
    """PY2VER_SIM, else Icarus, or Verilator for a long run when it is installed."""
    sim = os.environ.get("PY2VER_SIM", "").strip().lower()
    if sim:
        return sim
    lines = (p.hdl_dir / "main.v").read_text(encoding="utf-8").count("\n")
    if clocks * lines >= LONG_RUN and shutil.which("verilator"):
        return "verilator"
    return "icarus"
