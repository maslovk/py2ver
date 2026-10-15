# names.py
"""
Keep Python names that are Verilog/SystemVerilog keywords out of the
generated HDL: `buf`, `time`, `table`, `type`, `int`, ... become `buf_`,
`time_`, ... (the simulator compiles with SystemVerilog enabled, so its
keywords count too).
"""

from __future__ import annotations

import ast
import copy
from typing import Dict, Tuple

# IEEE 1364-2005 and IEEE 1800-2017 reserved words
VERILOG_KEYWORDS = frozenset("""
accept_on alias always always_comb always_ff always_latch and assert assign assume
automatic before begin bind bins binsof bit break buf bufif0 bufif1 byte case casex
casez cell chandle checker class clocking cmos config const constraint context
continue cover covergroup coverpoint cross deassign default defparam design disable
dist do edge else end endcase endchecker endclass endclocking endconfig endfunction
endgenerate endgroup endinterface endmodule endpackage endprimitive endprogram
endproperty endsequence endspecify endtable endtask enum event eventually expect
export extends extern final first_match for force foreach forever fork forkjoin
function generate genvar global highz0 highz1 if iff ifnone ignore_bins illegal_bins
implements implies import incdir include initial inout input inside instance int
integer interconnect interface intersect join join_any join_none large let liblist
library local localparam logic longint macromodule matches medium modport module
nand negedge nettype new nexttime nmos nor noshowcancelled not notif0 notif1 null or
output package packed parameter pmos posedge primitive priority program property
protected pull0 pull1 pulldown pullup pulsestyle_ondetect pulsestyle_onevent pure
rand randc randcase randsequence rcmos real realtime ref reg reject_on release
repeat restrict return rnmos rpmos rtran rtranif0 rtranif1 s_always s_eventually
s_nexttime s_until s_until_with scalared sequence shortint shortreal showcancelled
signed small soft solve specify specparam static string strong strong0 strong1
struct super supply0 supply1 sync_accept_on sync_reject_on table tagged task this
throughout time timeprecision timeunit tran tranif0 tranif1 tri tri0 tri1 triand
trior trireg type typedef union unique unique0 unsigned until until_with untyped use
uwire var vectored virtual void wait wait_order wand weak weak0 weak1 while wildcard
wire with within wor xnor xor
""".split())


def rename_keywords(tree: ast.Module) -> Tuple[ast.Module, Dict[str, str]]:
    """Rename keyword identifiers in the module's first function; returns (tree, py -> hw)."""
    tree = copy.deepcopy(tree)
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    used |= {a.arg for a in ast.walk(tree) if isinstance(a, ast.arg)}
    mapping: Dict[str, str] = {}

    def hw(name: str) -> str:
        if name not in VERILOG_KEYWORDS:
            return name
        if name not in mapping:
            new = name + "_"
            while new in used or new in VERILOG_KEYWORDS:
                new += "_"
            used.add(new)
            mapping[name] = new
        return mapping[name]

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            node.id = hw(node.id)
        elif isinstance(node, ast.arg):
            node.arg = hw(node.arg)
        elif isinstance(node, ast.FunctionDef):
            node.name = hw(node.name)
    return tree, mapping
