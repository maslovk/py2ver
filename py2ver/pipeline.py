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


def _rename(expr: str, mapping: Dict[str, str]) -> str:
    return _IDENT.sub(lambda m: mapping.get(m.group(0), m.group(0)), expr)


def pipeline_ir(ir: ModuleIR, attr: Dict[str, Dict[str, Any]], stages: int) -> PipelineResult:
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
        Assignment(out_c.get(a.left, a.left), _rename(a.right, out_c), a.is_reg)
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
    state: Dict[str, int] = {}

    def _visit(n: str) -> None:
        if n not in deps or state.get(n) == 2:
            return
        if state.get(n) == 1:
            raise ValueError(f"Combinational loop through '{n}'")
        state[n] = 1
        for d in deps[n]:
            _visit(d)
        state[n] = 2
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
    cost = {n: expr_cost(by_left[n].right) for n in nonconst}

    def _schedule(limit: float) -> Tuple[Dict[str, int], Dict[str, float]]:
        """
        Greedy ASAP stages: a signal joins the latest stage of its inputs unless
        that would push the stage's delay past `limit`, then it starts the next
        stage (its inputs from earlier stages arrive through registers).
        """
        stg: Dict[str, int] = {i: 0 for i in inputs}
        rel: Dict[str, float] = {i: 0.0 for i in inputs}
        for n in nonconst:
            ds = [d for d in deps[n] if d not in const]
            s = max((stg[d] for d in ds), default=0)
            begin = max((rel[d] for d in ds if stg[d] == s), default=0.0)
            if begin > 0 and begin + cost[n] > limit + 1e-9:
                s, begin = s + 1, 0.0
            stg[n] = s
            rel[n] = begin + cost[n]
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
    for n in order:
        a = by_left[n]
        if n in const:
            new_assigns.append(a)
            continue
        mapping = {d: _at(d, stage[n]) for d in deps[n] if d not in const}
        new_assigns.append(Assignment(n, _rename(a.right, mapping), False))

    for y, c in out_c.items():
        src = c if c in const else _at(c, stages)
        new_assigns.append(Assignment(y, src, False))

    register_bits = 0
    for name, k in sorted(need.items()):
        meta = _meta(name, attr, ports)
        for j in range(1, k + 1):
            attr[f"{name}__p{j}"] = dict(meta, type="reg")
            prev = name if j == 1 else f"{name}__p{j - 1}"
            new_assigns.append(Assignment(f"{name}__p{j}", prev, True))
        register_bits += k * int(meta.get("width", 1))

    new_inputs = list(ir.inputs) + [Port(name="clk", width=1, signed=False, kind="clk")]
    new_ir = ModuleIR(
        name=ir.name,
        inputs=new_inputs,
        outputs=list(ir.outputs),
        assigns=new_assigns,
        has_clk=True,
    )
    return PipelineResult(new_ir, attr, stage, register_bits, stage_delay)
