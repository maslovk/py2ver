"""
Checking that Verilog computes the same bits as other Verilog, in Icarus, on
random inputs (for tests of how the Verilog is written):

  assert_same_bits(tmp_path, inputs, pairs)       expressions, two by two
  assert_same_modules(tmp_path, a, b, top, ...)   two modules with the same ports

inputs: {name: (width, signed)}, set from $random every sample (a clocked
module's clk, rst and ce are driven by the testbench: rst for the first two
clocks, ce always).
"""
import re
import shutil
import subprocess
from typing import Dict, List, Sequence, Tuple

import pytest

Ports = Dict[str, Tuple[int, bool]]
_CONTROL = ("clk", "rst", "ce")


def _decl(name: str, width: int, signed: bool) -> str:
    return f"{'signed ' if signed else ''}[{width - 1}:0] {name}"


def _random(width: int) -> str:
    """$random enough times for width bits (the assignment keeps the low ones)."""
    n = -(-width // 32)
    return "$random" if n == 1 else "{" + ", ".join(["$random"] * n) + "}"


def mismatches(tmp_path, inputs: Ports, pairs: Sequence[Tuple[int, bool, str, str]],
               sources: str = "", body: Sequence[str] = (), samples: int = 2000,
               clocked: bool = False) -> List[Tuple[int, str, str]]:
    """(pair, bits of its first expression, of its second) for each pair
    (width, signed, a, b) whose expressions differ on a sample, the first such
    sample of each. sources: modules the testbench uses; body: lines in it
    (instances and their wires)."""
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("Icarus Verilog tools are not installed")
    tb = ["module same_bits_tb;"]
    tb += [f"  reg {_decl(n, w, s)};" for n, (w, s) in inputs.items()]
    if clocked:
        tb.append("  reg clk = 0, rst = 1, ce = 1;")
    tb += [f"  {line}" for line in body]
    for k, (w, s, a, b) in enumerate(pairs):
        tb += [f"  wire {_decl(f'a{k}', w, s)} = {a};", f"  wire {_decl(f'b{k}', w, s)} = {b};"]
    tb += [f"  reg [{max(1, len(pairs)) - 1}:0] seen = 0;", "  integer t;", "  initial begin",
           f"    for (t = 0; t < {samples}; t = t + 1) begin"]
    tb += [f"      {n} = {_random(w)};" for n, (w, _) in inputs.items()]
    tb.append("      #1;")
    tb += [f'      if (a{k} !== b{k} && !seen[{k}]) begin seen[{k}] = 1; '
           f'$display("DIFF {k} %b %b", a{k}, b{k}); end' for k in range(len(pairs))]
    if clocked:
        tb += ["      clk = 1; #1; clk = 0;", "      rst = t < 1;"]
    tb += ["    end", '    $display("DONE");', "  end", "endmodule"]
    (tmp_path / "same_bits.v").write_text(sources + "\n" + "\n".join(tb) + "\n")
    subprocess.run(["iverilog", "-g2012", "-o", "same_bits.vvp", "-s", "same_bits_tb",
                    "same_bits.v"], cwd=tmp_path, check=True)
    out = subprocess.run(["vvp", "-n", "same_bits.vvp"], cwd=tmp_path, capture_output=True,
                         text=True).stdout
    assert "DONE" in out, out[-2000:]
    return [(int(k), a, b) for k, a, b in re.findall(r"^DIFF (\d+) (\S+) (\S+)$", out, re.M)]


def assert_same_bits(tmp_path, inputs: Ports, pairs: Sequence[Tuple[int, bool, str, str]],
                     **kwargs) -> None:
    """Each pair's two expressions give the same bits on every sample."""
    bad = mismatches(tmp_path, inputs, pairs, **kwargs)
    assert not bad, f"{len(bad)} pair(s) differ, e.g.:\n" + "\n".join(
        f"  {pairs[k][2]}\n  {pairs[k][3]}\n    {a} != {b}" for k, a, b in bad[:5])


def _without_dumps(verilog: str) -> str:
    """The rendered module without its $dumpfile / $dumpvars block (an instance
    of the module under another name couldn't name itself)."""
    return re.sub(r"initial begin\n\s*\$dumpfile\([^)]*\);\n\s*\$dumpvars\([^)]*\);\nend\n", "",
                  verilog)


def assert_same_modules(tmp_path, a: str, b: str, top: str, inputs: Ports, outputs: Ports,
                        samples: int = 2000) -> None:
    """Modules a and b (Verilog text; module top in each, the same ports)
    give the same outputs, sample by sample: clocked from reset if they
    have a clk port."""
    control = [c for c in _CONTROL if re.search(rf"\binput\b[^\n]*\b{c}\b", a)]
    data = {n: m for n, m in inputs.items() if n not in _CONTROL}
    renamed = re.sub(rf"\bmodule {re.escape(top)}\b", f"module {top}__b", _without_dumps(b), 1)
    body = []
    for side, module in (("a", top), ("b", f"{top}__b")):
        body += [f"wire {_decl(f'{side}_{o}', w, s)};" for o, (w, s) in outputs.items()]
        conns = [f".{n}({n})" for n in [*control, *data]] + [f".{o}({side}_{o})" for o in outputs]
        body.append(f"{module} dut_{side}({', '.join(conns)});")
    pairs = [(w, s, f"a_{o}", f"b_{o}") for o, (w, s) in outputs.items()]
    assert_same_bits(tmp_path, data, pairs, sources=_without_dumps(a) + "\n" + renamed,
                     body=body, samples=samples, clocked="clk" in control)
