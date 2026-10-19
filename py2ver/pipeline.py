# pipeline.py
"""
Automatic pipelining of a combinational ModuleIR.

The design is cut into `stages + 1` combinational segments of roughly equal
estimated delay, with a register boundary between segments:

  - every signal gets a stage from its estimated arrival time
  - a value used `k` stages after it is produced is read through `k` delay
    registers (<name>__p1 .. <name>__pk), so the pipeline is balanced
  - every output appears exactly `stages` clocks after its inputs, so a new
    input vector can be applied every clock
  - constants (signals that do not depend on any input) are never registered
  - all pipeline registers share a clock enable `ce` (tie it to 1, or use it
    to stall the pipeline, as the valid/ready stream wrapper does)
  - a table read in block RAM (`roms`, from the translator) is a register
    stage of its own: the ROM latches the index at the end of one stage and
    gives the entry in the next

Delay is estimated per assignment by summing operator costs in units of
roughly one 32-bit adder. It is a heuristic; SYN() measures the real result.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple

from .ast_ir import Assignment, ModuleIR, Port, resolve_meta as _meta

_IDENT = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")

# (pattern, cost) per operator occurrence, in ~32-bit-adder units
_OP_COSTS: List[Tuple[re.Pattern, float]] = [
    (re.compile(r" / "), 10.0),                         # combinational divider
    (re.compile(r" \* "), 2.5),                         # multiplier
    (re.compile(r" [+-] "), 1.0),                       # adder / subtractor
    (re.compile(r"(?:^|[(\s])-\("), 1.0),               # unary minus
    (re.compile(r" (?:<|<=|>|>=) "), 1.0),              # magnitude compare
    (re.compile(r" (?:==|!=) "), 0.5),                  # equality
    (re.compile(r" (?:>>>|<<) \((?!\d+\))"), 1.0),      # variable shift (constant: wiring)
    (re.compile(r" \? "), 0.5),                         # mux
    (re.compile(r" (?:&|\||\^|&&|\|\|) "), 0.25),       # one logic level
]


def expr_cost(expr: str) -> float:
    return sum(len(p.findall(expr)) * c for p, c in _OP_COSTS)


@dataclass
class PipelineResult:
    ir: ModuleIR
    attr: Dict[str, Dict[str, Any]]
    stage: Dict[str, int] = field(default_factory=dict)
    register_bits: int = 0
    stage_delay: List[float] = field(default_factory=list)
    # With state: the stage holding the state loop, the valid bit of the
    # sample in that stage, and each next-state output's value at that stage
    loop_stage: int = 0
    valid_at_loop: str = ""
    next_at_loop: Dict[str, str] = field(default_factory=dict)
    roms: int = 0


def _rename(expr: str, mapping: Dict[str, str]) -> str:
    return _IDENT.sub(lambda m: mapping.get(m.group(0), m.group(0)), expr)


ROM_COST = 1.5          # a block RAM's clock-to-output, in adder units


def _rom_items(name: str, rom: Dict[str, Any], index: str, index_width: int) -> List[str]:
    entries, w, aw = list(rom["entries"]), int(rom["width"]), int(rom["address_bits"])
    entries += [entries[-1]] * ((1 << aw) - len(entries))      # (padded: no index reads x)
    sig = "signed " if rom.get("signed") else ""
    mask = (1 << w) - 1
    arr = f"{name}__rom"
    if index_width > aw:                    # (a table index is never out of range)
        index = f"{index}[{aw - 1}:0]"
    return ([f"// Table in block RAM: {len(entries)} x {w} bits",
             f"reg {sig}[{w - 1}:0] {arr} [0:{len(entries) - 1}];",
             f"reg {sig}[{w - 1}:0] {name};",
             "initial begin"]
            + [f"  {arr}[{k}] = {w}'d{int(v) & mask};" for k, v in enumerate(entries)]
            + ["end", f"always @(posedge clk) if (ce) {name} <= {arr}[{index}];"])


def pipeline_ir(ir: ModuleIR, attr: Dict[str, Dict[str, Any]], stages: int,
                state: Dict[str, str] | None = None,
                roms: Dict[str, Dict[str, Any]] | None = None) -> PipelineResult:
    """
    state maps state inputs to their next-state outputs (before the state
    registers are added). The state loop (state -> ... -> next state) stays in
    one stage so a sample per clock still works; logic that only depends on
    inputs is pipelined before it, logic from the state to the outputs after
    it. A valid bit (vin -> vout) travels with each sample so the state only
    updates for real samples.
    """
    if stages < 0:
        raise ValueError("pipeline stages must be >= 0")
    if stages == 0:
        return PipelineResult(ir, {k: dict(v) for k, v in attr.items()})
    if ir.has_clk:
        raise NotImplementedError(
            "Automatic pipelining needs a combinational design "
            "(no 'reg' signals or state registers)."
        )

    attr = {k: dict(v) for k, v in attr.items()}
    state = dict(state or {})
    roms = dict(roms or {})
    inputs = [p.name for p in ir.inputs if p.kind == "input"]
    outputs = [p.name for p in ir.outputs]
    ports = {p.name: p for p in ir.inputs + ir.outputs}

    for y in outputs:
        if y in inputs:
            raise NotImplementedError(f"Output '{y}' is also an input; cannot pipeline it.")

    # Outputs are computed under an internal name and re-exposed after the pipeline
    out_c = {y: f"{y}__c" for y in outputs}
    for y, c in out_c.items():
        attr[c] = _meta(y, attr, ports)
    assigns = [
        Assignment(out_c.get(a.left, a.left), _rename(a.right, out_c), a.is_reg, a.line)
        for a in ir.assigns
    ]

    defined = {a.left for a in assigns}
    known = defined | set(inputs)
    deps: Dict[str, List[str]] = {
        a.left: sorted({i for i in _IDENT.findall(a.right) if i in known and i != a.left})
        for a in assigns
    }

    # Topological order (program order normally already is)
    order: List[str] = []
    mark: Dict[str, int] = {}

    def _visit(n: str) -> None:
        if n not in deps or mark.get(n) == 2:
            return
        if mark.get(n) == 1:
            raise ValueError(f"Combinational loop through '{n}'")
        mark[n] = 1
        for d in deps[n]:
            _visit(d)
        mark[n] = 2
        order.append(n)

    for a in assigns:
        _visit(a.left)
    by_left = {a.left: a for a in assigns}

    # Constants: no transitive dependency on any input
    const: Set[str] = set()
    for n in order:
        if all(d in const for d in deps[n]):
            const.add(n)

    nonconst = [n for n in order if n not in const]
    cost = {n: ROM_COST if n in roms else expr_cost(by_left[n].right) for n in nonconst}

    # State loop: signals on a path from a state input to a next-state output
    state_inputs = set(state)
    next_c = {out_c[n] for n in state.values()}
    fwd: Set[str] = set(state_inputs)            # depends on the state
    for n in order:
        if any(d in fwd for d in deps[n]):
            fwd.add(n)
    bwd: Set[str] = set(next_c)                  # feeds a next-state output
    for n in reversed(order):
        if n in bwd:
            bwd.update(deps[n])
    loop = fwd & bwd
    if loop & set(roms):
        raise NotImplementedError("A table in block RAM can't be in the state loop of a pipeline")

    def _schedule(limit: float) -> Tuple[Dict[str, int], Dict[str, float]]:
        """
        Greedy ASAP stages: a signal joins the latest stage of its inputs unless
        that would push the stage's delay past `limit`, then it starts the next
        stage (its inputs from earlier stages arrive through registers). With
        state, input-only logic is placed first, the loop goes in the stage
        after its latest input-only operand, and is never split.
        """
        stg: Dict[str, int] = {i: 0 for i in inputs}
        rel: Dict[str, float] = {i: 0.0 for i in inputs}

        def place(n: str, split: bool) -> None:
            ds = [d for d in deps[n] if d not in const]
            s = max((stg[d] for d in ds), default=0)
            begin = max((rel[d] for d in ds if stg[d] == s), default=0.0)
            if n in roms:                       # the ROM's own register: the next stage
                s, begin = s + 1, 0.0
            elif split and begin > 0 and begin + cost[n] > limit + 1e-9:
                s, begin = s + 1, 0.0
            stg[n] = s
            rel[n] = begin + cost[n]

        for n in nonconst:
            if n not in fwd:
                place(n, split=True)
        if state:
            k = max((stg[d] for n in loop if n in deps for d in deps[n]
                     if d not in fwd and d not in const), default=0)
            for si in state_inputs:
                stg[si], rel[si] = k, 0.0
            for n in nonconst:
                if n in fwd:
                    place(n, split=n not in loop)
        return stg, rel

    # Smallest per-stage delay limit that fits in stages + 1 segments
    lo = max(cost.values(), default=0.0)
    hi = max(lo, sum(cost.values()))
    for _ in range(60):
        mid = (lo + hi) / 2
        stg, _rel = _schedule(mid)
        if max(stg.values(), default=0) <= stages:
            hi = mid
        else:
            lo = mid
    stage, rel = _schedule(hi)
    if max(stage.values(), default=0) > stages:
        if roms:
            raise NotImplementedError(
                f"The design needs at least {max(stage.values())} pipeline stages "
                "(each table read in block RAM takes one)")
        raise NotImplementedError("Could not fit the design into the requested stages")
    loop_stage = stage[next(iter(state_inputs))] if state else 0

    stage_delay = [0.0] * (stages + 1)
    for n in nonconst:
        stage_delay[stage[n]] = max(stage_delay[stage[n]], rel[n])

    # Rewrite each assignment to read its inputs at its own stage
    need: Dict[str, int] = {}

    def _at(name: str, s: int) -> str:
        k = s - stage[name]
        if k == 0:
            return name
        need[name] = max(need.get(name, 0), k)
        return f"{name}__p{k}"

    new_assigns: List[Assignment] = []
    rom_items: List[str] = []
    for n in order:
        a = by_left[n]
        if n in const:
            new_assigns.append(a)
            continue
        if n in roms:                           # reads the index of the stage before
            idx = roms[n]["index"]
            rom_items += _rom_items(n, roms[n], _at(idx, stage[n] - 1),
                                    int(_meta(idx, attr, ports).get("width", 1)))
            continue
        mapping = {d: _at(d, stage[n]) for d in deps[n] if d not in const}
        new_assigns.append(Assignment(n, _rename(a.right, mapping), False, a.line))

    for y, c in out_c.items():
        src = c if c in const else _at(c, stages)
        new_assigns.append(Assignment(y, src, False))

    # Next-state values as the state registers see them (in the loop stage)
    next_at_loop = {n: (out_c[n] if out_c[n] in const else _at(out_c[n], loop_stage))
                    for n in state.values()}

    register_bits = 0
    for name, k in sorted(need.items()):
        meta = _meta(name, attr, ports)
        for j in range(1, k + 1):
            attr[f"{name}__p{j}"] = dict(meta, type="reg")
            prev = name if j == 1 else f"{name}__p{j - 1}"
            # Clock enable: the whole pipeline holds while ce is low (stall)
            new_assigns.append(Assignment(f"{name}__p{j}", f"ce ? {prev} : {name}__p{j}", True))
        register_bits += k * int(meta.get("width", 1))

    new_inputs = list(ir.inputs) + [
        Port(name="clk", width=1, signed=False, kind="clk"),
        Port(name="ce", width=1, signed=False, kind="ce"),
    ]
    new_outputs = list(ir.outputs)
    valid_at_loop = ""
    if state:
        # Valid bit travelling with the samples: vin -> __v1 .. __vN -> vout
        new_inputs.append(Port(name="vin", width=1, signed=False, kind="vin"))
        new_outputs.append(Port(name="vout", width=1, signed=False, kind="vout"))
        for j in range(1, stages + 1):
            prev = "vin" if j == 1 else f"__v{j - 1}"
            attr[f"__v{j}"] = {"width": 1, "signed": 0, "type": "reg"}
            # Cleared by rst (added with the state registers), so no stale valids
            new_assigns.append(Assignment(f"__v{j}", f"rst ? 1'b0 : (ce ? {prev} : __v{j})", True))
        new_assigns.append(Assignment("vout", f"__v{stages}", False))
        valid_at_loop = "vin" if loop_stage == 0 else f"__v{loop_stage}"
    new_ir = ModuleIR(
        name=ir.name,
        inputs=new_inputs,
        outputs=new_outputs,
        assigns=new_assigns,
        has_clk=True,
        items=list(ir.items) + rom_items,
    )
    return PipelineResult(new_ir, attr, stage, register_bits, stage_delay,
                          loop_stage, valid_at_loop, next_at_loop, len(roms))
