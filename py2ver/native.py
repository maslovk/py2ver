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
CHUNK = 4096              # an input vector wider than this is read in parts of this many bits
OUT_GROUP = 256           # outputs written by one $fwrite

log = logging.getLogger(__name__)

_MAIN_CPP = """#include "Vpy2ver_tb.h"
#include "verilated.h"
double sc_time_stamp() { return 0; }   // ($time in the SDRAM's model: Verilator 4 asks for it)
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


def _sdram(p) -> Optional[Dict[str, Any]]:
    """The design's SDRAM (its lists and tables), or None."""
    return p.schedule.sdram if p.schedule else None


def mode_of(p) -> Optional[Dict[str, Any]]:
    """How the testbench steps this design, or None if cocotb must (latency 'reg'
    signals)."""
    stateful = bool(p.state_map or p.rams or p.schedule)
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
    ext_in = getattr(p, "_ext_inputs", {})
    ext_out = getattr(p, "_ext_outputs", {})
    ins = [(n, w) for n, w in zip(p.input_args_list, p.input_args_width_list) if n not in ext_in]
    outs = [(n, w) for n, w in zip(p.output_args_list, p.output_args_width_list)
            if n not in ext_out]
    iw = max(1, sum(w for _, w in ins))
    ports = {q.name for q in p.ir.inputs}
    lines = ["module py2ver_tb(input clk);",
             f"  reg [{iw - 1}:0] cur, nxt;",
             "  integer fin, fout, n, k, c, got, total;"]
    for name, w in outs:
        lines.append(f"  wire [{w - 1}:0] o_{name};")
    for ctl in ("rst", "ce", "vin"):
        if ctl in ports or (ctl == "ce" and "ce_in" in ports):
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
    sdram = _sdram(p)
    if "ce_in" in ports:
        conns.append(".ce_in(ce)")
    if sdram:
        # The SDRAM: a model of the chip on the design's pins, its tables loaded
        for d, w, n in p.SDRAM_PORTS:
            tie = " = 0" if d == "input" else ""          # (the host's port: unused here)
            lines.append(f"  wire {'[' + str(w - 1) + ':0] ' if w > 1 else ''}{n}{tie};")
            conns.append(f".{n}({n})")
        lines.append("  sdram_model __mem(.clk(dram_clk), .cke(dram_cke), .cs_n(dram_cs_n), "
                     ".ras_n(dram_ras_n), .cas_n(dram_cas_n), .we_n(dram_we_n), .ba(dram_ba), "
                     ".sa(dram_addr), .dqm(dram_dqm), .dq(dram_dq));")
        lines.append("  initial begin")
        for k, (base, values) in enumerate(sdram_images(p)):
            lines.append(f'    $readmemh("sdram_{k}.hex", __mem.mem, {base}, {base + len(values) - 1});')
        lines.append("  end")
        if ext_in:
            lines.append("  integer fext, e;")
            lines.append("  reg [15:0] word;")
        if ext_out:
            # (returned lists in the SDRAM: read from the chip's model at the clock
            # after a sample, its last write in and the next sample's not yet, a
            # line a sample)
            addrs = [a for _, a, _ in ext_words(p.output_args_list, p.output_args_width_list,
                                                ext_out)]
            lines += ["  integer fx, ex, dump, ending;",
                      "  function integer out_addr(input integer e);", "    begin"]
            for start, base, count in _runs(addrs):
                lines.append(f"      if (e >= {start} && e < {start + count}) "
                             f"out_addr = {base} + e - {start};")
            lines += ["    end", "  endfunction"]
            dump = ["    if (dump) begin",
                    "      dump = 0;",
                    f"      for (ex = 0; ex < {len(addrs)}; ex = ex + 1)",
                    '        $fwrite(fx, "%h ", __mem.mem[out_addr(ex)]);',
                    '      $fwrite(fx, "\\n");',
                    "      if (ending) begin $fclose(fout); $fclose(fx); $finish; end",
                    "    end"]
    if "vout" in {q.name for q in p.ir.outputs}:
        conns.append(".vout(vout)")
    # (a port a line: Verilator reads at most 20000 tokens a line)
    lines.append(f"  {p.t_name} dut(\n    " + ",\n    ".join(conns) + ");")
    # The outputs a line a vector, written a group at a time (Verilator reads at most
    # 20000 tokens a line: a model's thousands of scores)
    groups = [outs[i:i + OUT_GROUP] for i in range(0, len(outs), OUT_GROUP)] or [[]]
    lines.append("  task write_outputs;")
    lines.append("    begin")
    for gi, grp in enumerate(groups):
        end = "\\n" if gi == len(groups) - 1 else " "
        fmt = " ".join(["%h"] * len(grp)) + end
        args = "".join(f", o_{n}" for n, _ in grp)
        lines.append(f'      $fwrite(fout, "{fmt}"{args});')
    lines.append("    end")
    lines.append("  endtask")
    write = "write_outputs;"
    # The next vector, into nxt (applied with <= after the edge that reads the last);
    # a wide one in parts (Verilator reads at most 8192 bits an argument)
    if iw <= CHUNK:
        read = ["      if (n > 0) begin got = $fscanf(fin, \"%h\\n\", nxt); n = n - 1; end"]
    else:
        lines.append(f"  reg [{CHUNK - 1}:0] part;")
        read = ["      if (n > 0) begin"]
        for hi, lo in _chunks(iw):
            read.append(f"        got = $fscanf(fin, \"%h\", part); nxt[{hi}:{lo}] = part[{hi - lo}:0];")
        read.append("        n = n - 1;")
        read.append("      end")
    if ext_in:
        # (the sample's inputs in the SDRAM: written into the chip's model, in order)
        addrs = [a for _, a, _ in ext_words(p.input_args_list, p.input_args_width_list, ext_in)]
        read = read[:-1] + [f"        for (e = 0; e < {len(addrs)}; e = e + 1) begin",
                            '          got = $fscanf(fext, "%h", word);',
                            "          __mem.mem[ext_addr(e)] = word;",
                            "        end"] + read[-1:] if len(read) > 1 else read
        if len(read) == 1:                       # (the one-line form: open it up)
            read = ["      if (n > 0) begin", '        got = $fscanf(fin, "%h\\n", nxt);',
                    f"        for (e = 0; e < {len(addrs)}; e = e + 1) begin",
                    '          got = $fscanf(fext, "%h", word);',
                    "          __mem.mem[ext_addr(e)] = word;",
                    "        end",
                    "        n = n - 1;",
                    "      end"]
        runs = _runs(addrs)
        lines.append("  function integer ext_addr(input integer e);")
        lines.append("    begin")
        for start, base, count in runs:
            lines.append(f"      if (e >= {start} && e < {start + count}) ext_addr = {base} + e - {start};")
        lines.append("    end")
        lines.append("  endfunction")
    lines += [
        "  task next_vector;",
        "    begin",
        *read,
        "      else nxt = 0;",
        "    end",
        "  endtask",
        "  initial begin",
        '    fin = $fopen("vectors.hex", "r");',
        *(['    fext = $fopen("inputs.hex", "r");'] if ext_in else []),
        *(['    fx = $fopen("outputs_ext.hex", "w"); dump = 0; ending = 0;'] if ext_out else []),
        '    fout = $fopen("outputs.hex", "w");',
        '    if (!$value$plusargs("n=%d", total)) total = 0;',
        "    n = total; k = 0; c = 0; cur = 0; nxt = 0;",
    ]
    kind = mode["kind"]
    if "rst" in ports:
        lines.append("    rst = 1;")
    if "ce" in ports or "ce_in" in ports:
        lines.append("    ce = 1;")
    if "vin" in ports:
        lines.append("    vin = 0;")
    if kind in ("comb", "pipeline"):
        lines.append("    next_vector; cur = nxt;")
    lines += ["  end", "  always @(posedge clk) begin"]
    if ext_out:
        lines += dump
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
            # (with the SDRAM: P clocks the design advanced in)
            done = ("        dump = 1; if (c > total) ending = 1;" if ext_out
                    else "        if (c > total) begin $fclose(fout); $finish; end")
            lines += [f"      if ({'adv && ' if sdram else ''}k == {P - 1}) begin",
                      f"        {write}",
                      "        k = 0; c = c + 1;",
                      done,
                      "        next_vector; cur <= nxt;",
                      f"      end else{' if (adv)' if sdram else ''} k = k + 1;"]
        lines.append("    end")
    lines += ["  end", "endmodule", ""]
    return "\n".join(x for x in lines if x != "")


def _runs(addrs: List[int]) -> List[Tuple[int, int, int]]:
    """Consecutive addresses as (first position, first address, count)."""
    out: List[Tuple[int, int, int]] = []
    for i, a in enumerate(addrs):
        if out and out[-1][1] + out[-1][2] == a:
            out[-1] = (out[-1][0], out[-1][1], out[-1][2] + 1)
        else:
            out.append((i, a, 1))
    return out


def sdram_images(p) -> List[Tuple[int, List[int]]]:
    """(base address, 16-bit words) of each SDRAM list or table with values to start from."""
    sdram = _sdram(p) or {}
    out = []
    for kind in ("tables", "lists"):
        for key, info in sdram.get(kind, {}).items():
            init = info.get("init")
            if init is not None:
                if info.get("words", 1) == 2:            # (a value in two words: low, high)
                    words = [int(v) >> s & 0xFFFF for v in init for s in (0, 16)]
                else:
                    words = [int(v) & 0xFFFF for v in init]
                out.append((int(info["base"]), words))
    return out


def ext_words(names: List[str], widths: List[int], ext: Dict[str, int]) -> List[Tuple[str, int, int]]:
    """(port, address, shift) of each SDRAM word of the ports in ext (input or
    output lists in the SDRAM), in the ports' order: a value over 16 bits in two
    words, its low one first."""
    return [(n, ext[n] + k, 16 * k) for n, w in zip(names, widths) if n in ext
            for k in range(-(-int(w) // 16))]


def ext_value(words: List[int], shifts: List[int], width: int) -> int:
    """A port's value from its SDRAM words (unsigned, width bits)."""
    return sum(w << s for w, s in zip(words, shifts)) & ((1 << width) - 1)


def _chunks(width: int) -> List[Tuple[int, int]]:
    """(high, low) bits of a wide input vector's parts, the highest first."""
    out, hi = [], width - 1
    while hi >= 0:
        lo = hi // CHUNK * CHUNK
        out.append((hi, lo))
        hi = lo - 1
    return out


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
    core = p.verilog
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
    extra = []
    if _sdram(p):
        model = Path(__file__).parent / "templates" / "hw" / "sdram" / "sdram_model.v"
        (tmp / "sdram_model.v").write_text(model.read_text(encoding="utf-8"), encoding="utf-8")
        extra = ["sdram_model.v"]
    if simulator == "icarus":
        (tmp / "clock.v").write_text(_CLOCK_V, encoding="utf-8")
        _run(["iverilog", "-g2012", "-o", "sim.vvp", "-s", "py2ver_clock",
              "clock.v", "tb.v", "main.v", *extra], tmp)
    else:
        (tmp / "main.cpp").write_text(_MAIN_CPP, encoding="utf-8")
        _run(["verilator", "--cc", "--exe", "--build", "-O2", "-Wno-fatal", "-Wno-lint",
              "-Wno-style", "-Wno-TIMESCALEMOD", "--top-module", "py2ver_tb", "-Mdir", "obj",
              "tb.v", "main.v", *extra, "main.cpp"], tmp)
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
    p.write_hdl()                               # (its tables' files, read by the simulation)
    workdir = p.outdir / "native_build"
    if simulator == "verilator":
        failed = workdir / f"verilator_failed_{_key(p.verilog, testbench(p, mode))}"
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
    ext_in = getattr(p, "_ext_inputs", {})
    keep = [i for i, n in enumerate(p.input_args_list) if n not in ext_in]
    if ext_in:
        pos = {n: i for i, n in enumerate(p.input_args_list)}
        words = ext_words(p.input_args_list, p.input_args_width_list, ext_in)
        with (d / "inputs.hex").open("w") as f:
            for vec in vectors:
                f.write("".join(f"{int(vec[pos[n]]) >> s & 0xFFFF:04x}\n" for n, _, s in words))
    widths = [p.input_args_width_list[i] for i in keep]
    vectors = [tuple(vec[i] for i in keep) for vec in vectors]
    with (d / "vectors.hex").open("w") as f:
        for vec in vectors:
            word = 0
            for v, w in zip(vec, widths):
                word = (word << w) | (int(v) & ((1 << w) - 1))
            iw = max(1, sum(widths))
            if iw <= CHUNK:
                f.write(f"{word:x}\n")
            else:
                f.write(" ".join(f"{(word >> lo) & ((1 << (hi - lo + 1)) - 1):x}"
                                 for hi, lo in _chunks(iw)) + "\n")
    for k, (base, values) in enumerate(sdram_images(p)):      # (the SDRAM's starting contents)
        (d / f"sdram_{k}.hex").write_text("".join(f"{v:04x}\n" for v in values))
    out = d / "outputs.hex"
    if out.exists():
        out.unlink()
    arg = f"+n={len(vectors)}"
    if simulator == "icarus":
        _run(["vvp", "-n", "sim.vvp", arg], d)
    else:
        _run(["obj/Vpy2ver_tb", arg], d)
    rows = []
    ext_out = getattr(p, "_ext_outputs", {})
    names = [n for n in p.output_args_list if n not in ext_out]
    for line in out.read_text().split("\n")[:len(vectors)]:
        if line.strip() or ext_out:              # (no ports left: an empty line a sample)
            vals = line.split()
            rows.append({n: int(v, 16) if "x" not in v.lower() and "z" not in v.lower() else 0
                         for n, v in zip(names, vals)})
    if ext_out:                                # (the returned lists in the SDRAM)
        widths = dict(zip(p.output_args_list, p.output_args_width_list))
        words = ext_words(p.output_args_list, p.output_args_width_list, ext_out)
        lines = [x for x in (d / "outputs_ext.hex").read_text().split("\n") if x.strip()]
        for row, line in zip(rows, lines):
            got: Dict[str, Tuple[List[int], List[int]]] = {}
            for (n, _, s), v in zip(words, line.split()):
                ok = "x" not in v.lower() and "z" not in v.lower()
                got.setdefault(n, ([], []))[0].append(int(v, 16) if ok else 0)
                got[n][1].append(s)
            for n, (ws, ss) in got.items():
                row[n] = ext_value(ws, ss, widths[n])
    return rows


def choose_simulator(p, clocks: int) -> str:
    """PY2VER_SIM, else Icarus, or Verilator for a long run when it is installed."""
    sim = os.environ.get("PY2VER_SIM", "").strip().lower()
    if sim:
        return sim
    lines = p.verilog.count("\n")
    if clocks * lines >= LONG_RUN and shutil.which("verilator"):
        return "verilator"
    return "icarus"
