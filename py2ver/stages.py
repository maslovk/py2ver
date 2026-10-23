# stages.py
"""
Build a chain as separate hardware stages instead of one fused block.

A chain is a function calling the blocks' functions (see "Signal chains" in
docs/reference.md). Built as one design, everything runs once per block: a stage
called for each of a block's M samples is built M times. With stages=True,
each call becomes its own core behind a valid/ready stream wrapper, and a
top module, <chain>_stages, connects them:

    def receiver(xs, z, delay):
        ys = []
        for x in xs:                         # a sample stage: one core,
            y, z = downconvert(x, z)         #   called once per sample
            ys.append(y)                     # ys: a serial input of the next
        acc, delay = decimate(ys, delay)     # a block stage, once per block
        p = power(acc)                       # another block stage
        return p, z, delay

The chain function itself must be such a list of stages: calls of your
functions at its top level (block stages) and `for v in xs:` loops holding
one call and appends of its results (sample stages), `ys = []` for the lists
the loops fill, and a return. Data flows only through the calls' arguments
and results, each stage takes its inputs from the stage before it (or the
chain's input), and state ('next' in attr) belongs to one stage. The chain's
returned values that aren't state come out of the last stage.

Each stage is a py2ver design of a generated function

    def receiver_s1_decimate(ys, delay):
        acc, delay = decimate(ys, delay)
        return acc, delay

whose parameters and results are the chain's variables, so the chain's attr
(formats, state) applies to them: the stages compute exactly what the fused
design does, and the chain's PY_stream() is the reference. A block stage that
gets several clocks per block (its input arrives one sample per clock) has
its longest loop folded onto as few copies as keep up.
"""

from __future__ import annotations

import ast
import inspect
import linecache
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .fold import find_loop


class NotSplittable(NotImplementedError):
    """The chain function isn't a list of stages (see the module docstring)."""


@dataclass
class Stage:
    index: int
    kind: str                         # "block" | "sample"
    callee: str
    args: List[str]
    targets: List[str]
    loop_var: Optional[str] = None    # sample stage: for <loop_var> in <loop_list>
    loop_list: Optional[str] = None
    appends: Dict[str, str] = field(default_factory=dict)    # list -> appended variable
    source: Any = None                # producer: "input" or a stage index
    params: List[str] = field(default_factory=list)          # the generated function's
    name: str = ""
    func: Any = None
    design: Any = None                # its Py2ver


def _err(node: ast.AST, msg: str):
    line = getattr(node, "lineno", None)
    raise NotSplittable(f"{msg} (line {line})" if line else msg)


def _call(stmt: ast.stmt, ns: Dict[str, Any]) -> Optional[Tuple[str, List[str], List[str]]]:
    """`targets = f(args)` with names only: (f, args, targets), or None."""
    if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
            and isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Name)):
        return None
    call, tgt = stmt.value, stmt.targets[0]
    if not inspect.isfunction(ns.get(call.func.id)):
        return None
    if call.keywords or not all(isinstance(a, ast.Name) for a in call.args):
        _err(stmt, f"A stage call passes names only: {ast.unparse(call)}")
    elts = tgt.elts if isinstance(tgt, ast.Tuple) else [tgt]
    if not all(isinstance(t, ast.Name) for t in elts):
        _err(stmt, "A stage's results are assigned to names")
    return call.func.id, [a.id for a in call.args], [t.id for t in elts]


def plan(func, attr: Dict[str, Dict[str, Any]]) -> Tuple[List[Stage], Dict[str, Any]]:
    """The chain's stages, and {'input', 'length', 'state', 'returns'}."""
    fd = ast.parse(inspect.getsource(func)).body[0]
    ns = func.__globals__
    params = [a.arg for a in fd.args.args]
    state = [p for p in params if {"next", "memory"} & set(attr.get(p, {}))]
    inputs = [p for p in params if p not in state]
    if len(inputs) != 1:
        raise NotSplittable(f"A chain built as stages takes one input (a block of samples); "
                            f"{fd.name} takes {inputs}")
    chain_in = inputs[0]
    length = int(attr.get(chain_in, {}).get("length", 1))
    stages: List[Stage] = []
    lists: set = set()
    returns: List[str] = []
    for stmt in fd.body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            continue                                            # docstring
        if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name) and isinstance(stmt.value, ast.List)
                and not stmt.value.elts):
            lists.add(stmt.targets[0].id)                       # ys = []
            continue
        c = _call(stmt, ns)
        if c is not None:
            stages.append(Stage(len(stages), "block", *c))
            continue
        if isinstance(stmt, ast.For) and isinstance(stmt.target, ast.Name) \
                and isinstance(stmt.iter, ast.Name) and not stmt.orelse:
            body = [s for s in stmt.body]
            c = _call(body[0], ns) if body else None
            if c is None:
                _err(stmt, "A sample loop starts with one call of a stage function")
            appends = {}
            for s in body[1:]:
                ok = (isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)
                      and isinstance(s.value.func, ast.Attribute) and s.value.func.attr == "append"
                      and isinstance(s.value.func.value, ast.Name) and len(s.value.args) == 1
                      and isinstance(s.value.args[0], ast.Name) and s.value.args[0].id in c[2])
                if not ok:
                    _err(s, "A sample loop holds one call and appends of its results")
                if s.value.func.value.id not in lists:
                    _err(s, f"'{s.value.func.value.id}' must be set to [] before the loop")
                appends[s.value.func.value.id] = s.value.args[0].id
            stages.append(Stage(len(stages), "sample", *c, loop_var=stmt.target.id,
                                loop_list=stmt.iter.id, appends=appends))
            continue
        if isinstance(stmt, ast.Return) and stmt.value is not None:
            vals = stmt.value.elts if isinstance(stmt.value, ast.Tuple) else [stmt.value]
            if not all(isinstance(v, ast.Name) for v in vals):
                _err(stmt, "A chain built as stages returns names")
            returns = [v.id for v in vals]
            continue
        _err(stmt, "A chain built as stages holds only stage calls, sample loops, "
                   "`xs = []` and a return")
    if not stages:
        raise NotSplittable("The chain calls no stage functions")

    # Where each variable comes from, and from which stage each stage takes its inputs
    producer: Dict[str, Any] = {chain_in: "input"}
    per_block = {chain_in: length}           # list lengths per block
    owner: Dict[str, int] = {}
    for st in stages:
        sources = set()
        for a in st.args:
            if a in state:
                if a in owner:
                    raise NotSplittable(f"State '{a}' is used by two stages")
                owner[a] = st.index
                nxt = attr[a].get("next")
                if nxt is not None and nxt != a and nxt not in st.targets:
                    raise NotSplittable(f"The next value of state '{a}' must come from the "
                                        f"stage that uses it ({st.callee})")
                continue
            if st.kind == "sample" and a == st.loop_var:
                sources.add(producer.get(st.loop_list))
                continue
            if a not in producer:
                raise NotSplittable(f"'{a}' (an argument of {st.callee}) isn't the chain's input "
                                    "or a stage's result")
            sources.add(producer[a])
        if len(sources) != 1 or None in sources:
            raise NotSplittable(f"{st.callee} must take its inputs from one stage (or the "
                                "chain's input)")
        st.source = sources.pop()
        expect = "input" if st.index == 0 else st.index - 1
        if st.source != expect:
            raise NotSplittable(f"{st.callee} must take its inputs from the stage before it")
        if st.kind == "sample":
            if st.loop_list not in per_block:
                raise NotSplittable(f"Can't tell how many samples '{st.loop_list}' has per block")
            for lst in st.appends:
                producer[lst] = st.index
                per_block[lst] = per_block[st.loop_list]
        for t in st.targets:
            producer[t] = st.index
            if t in attr and "length" in attr[t]:
                per_block[t] = int(attr[t]["length"])
    last = stages[-1].index
    outs = [r for r in returns if r not in state and r not in
            {attr[p].get("next") for p in state}]
    for r in outs:
        if producer.get(r) != last:
            raise NotSplittable(f"The chain's result '{r}' must come from its last stage")
    return stages, {"input": chain_in, "length": length, "state": state, "returns": outs,
                    "per_block": per_block, "producer": producer}


def make_functions(chain, stages: List[Stage]) -> None:
    """The generated function of each stage (source registered for inspect)."""
    for st in stages:
        st.name = f"{chain.__name__}_s{st.index}_{st.callee}"
        params = list(dict.fromkeys(st.args))
        st.params = params
        tgt = ", ".join(st.targets)
        src = (f"def {st.name}({', '.join(params)}):\n"
               f"    {tgt} = {st.callee}({', '.join(st.args)})\n"
               f"    return {tgt}\n")
        filename = f"<py2ver stage {st.name}>"
        linecache.cache[filename] = (len(src), None, src.splitlines(True), filename)
        ns = dict(chain.__globals__)
        exec(compile(src, filename, "exec"), ns)
        st.func = ns[st.name]


def _element(meta: Dict[str, Any]) -> Dict[str, Any]:
    """A list's format as one element's."""
    return {k: v for k, v in meta.items() if k not in ("length", "next", "init", "serial", "memory")}


def stage_attr(st: Stage, stages: List[Stage], info: Dict[str, Any], attr: Dict[str, Any],
               complex_names: set, inferred) -> Dict[str, Dict[str, Any]]:
    """The chain's attr for a stage's generated function. A port without a
    format in attr takes the one the fused design inferred: inferred(var)."""
    out: Dict[str, Dict[str, Any]] = {k: dict(v) for k, v in attr.items() if "range" in v}   # tables
    appended = {lst: var for s in stages for lst, var in s.appends.items()}

    def fmt(var: str) -> Dict[str, Any]:
        m = dict(attr.get(var, {}))
        if "width" not in m:
            m = dict(inferred(var), **m)
        if var in complex_names:
            m["complex"] = True
        return m

    for p in st.params:
        if p in info["state"]:
            out[p] = fmt(p)
        elif st.kind == "sample" and p == st.loop_var:
            out[p] = _element(fmt(st.loop_list))           # one sample of the list
            if st.loop_list in complex_names:
                out[p]["complex"] = True
        elif p in appended:                                # filled by a sample stage
            out[p] = dict(_element(fmt(appended[p])), length=info["per_block"][p], serial=True)
        elif p == info["input"] and info["length"] > 1:    # the chain's block of samples
            out[p] = dict(fmt(p), serial=True)
        else:
            out[p] = fmt(p)
        out[p] = {k: v for k, v in out[p].items() if v is not None}
        if not out[p]:
            del out[p]
    nxt_stage = stages[st.index + 1] if st.index + 1 < len(stages) else None
    for t in st.targets:
        if t in st.params:
            continue
        m = {k: v for k, v in attr.get(t, {}).items() if k != "complex"}
        if nxt_stage is not None and nxt_stage.kind == "sample" and nxt_stage.loop_list == t:
            m["serial"] = True                             # a list the next stage loops over
        if m:
            out[t] = m
    return out


def auto_fold(st: Stage, sattr: Dict[str, Any], clocks: int, options: Dict[str, Any],
              const_bits: int) -> Tuple[Any, int]:
    """(fold, unroll) for a stage: the user's options, else the longest loop
    folded onto the fewest copies that finish in `clocks` clocks."""
    if "fold" in options:
        return options.get("fold"), int(options.get("unroll", 1))
    if st.kind == "sample" or clocks < 2:
        return None, 1
    from .core import front_end
    try:
        _, _, values = find_loop(front_end(st.func, sattr, const_bits).tree.body[0], True)
    except ValueError:
        return None, 1
    n = len(values)
    for unroll in range(1, n + 1):
        if n % unroll == 0 and n // unroll <= clocks:
            return (True, unroll) if n // unroll > 1 else (None, 1)
    return None, 1


def stream_ports(p) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """(input ports, output ports) of a design's stream wrapper: (name, range)."""
    def meta(name: str) -> Tuple[int, bool]:
        m = p.attr.get(name, {})
        return int(m.get("width", 1)), int(m.get("signed", 0)) == 1

    def rng(names: List[str]) -> str:
        signed = any(meta(n)[1] for n in names)
        w = max(meta(n)[0] + (1 if signed and not meta(n)[1] else 0) for n in names)
        return f"{'signed ' if signed else ''}[{w - 1}:0]"

    s_in = {q for ports in p._serial["in"].values() for q in ports}
    s_out = {q for ports in p._serial["out"].values() for q in ports}
    ins = [(n, rng(ports)) for n, ports in p._serial["in"].items()]
    ins += [(n, rng([n])) for n in p.input_args_list if n not in s_in]
    outs = [(n, rng(ports)) for n, ports in p._serial["out"].items()]
    outs += [(n, rng([n])) for n in p.output_args_list if n not in s_out]
    return ins, outs


def _var_of(port: str, names: List[str]) -> Tuple[str, str]:
    """(variable, suffix) of a port: the longest variable name it starts with."""
    for v in sorted(names, key=len, reverse=True):
        if port == v or port.startswith(v + "_"):
            return v, port[len(v):]
    raise NotSplittable(f"Can't tell which variable port '{port}' carries")


def top_module(chain: str, stages: List[Stage], info: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Verilog of <chain>_stages, and its ports."""
    ports = [stream_ports(st.design) for st in stages]
    appended = {lst: var for s in stages for lst, var in s.appends.items()}
    lines: List[str] = []
    top_in = [(n, r) for n, r in ports[0][0]]
    last = stages[-1]
    # The chain's results: ports of the last stage carrying them
    carried = {appended.get(r, r) for r in info["returns"]}
    out_names = last.targets + list(last.params)
    top_out = [(n, r) for n, r in ports[-1][1] if _var_of(n, out_names)[0] in carried]

    decl = ["    input  clk", "    input  rst", "    input  s_valid", "    output s_ready"]
    decl += [f"    input  {r} s_{n}" for n, r in top_in]
    decl += ["    output m_valid", "    input  m_ready"]
    decl += [f"    output {r} m_{n}" for n, r in top_out]
    lines.append(f"// {chain} as {len(stages)} stages connected by valid/ready streams")
    lines.append(f"module {chain}_stages (\n" + ",\n".join(decl) + "\n);")
    for st, (ins, outs) in zip(stages, ports):
        i = st.index
        lines.append(f"    wire sv{i}, sr{i}, mv{i}, mr{i};")
        lines += [f"    wire {r} o{i}_{n};" for n, r in outs]
    for st, (ins, outs) in zip(stages, ports):
        i = st.index
        conns = [".clk(clk)", ".rst(rst)", f".s_valid(sv{i})", f".s_ready(sr{i})",
                 f".m_valid(mv{i})", f".m_ready(mr{i})"]
        if i == 0:
            conns += [f".s_{n}(s_{n})" for n, _ in ins]
        else:
            prev = stages[i - 1]
            prev_outs = {n for n, _ in ports[i - 1][1]}
            for n, _ in ins:
                var, suffix = _var_of(n, st.params)
                src = appended.get(var, var)
                if st.kind == "sample" and var == st.loop_var:
                    src = st.loop_list
                for cand in (src + suffix, src + suffix + "_out"):
                    if cand in prev_outs:
                        conns.append(f".s_{n}(o{i - 1}_{cand})")
                        break
                else:
                    raise NotSplittable(f"{st.callee}'s input '{n}' has no matching output of "
                                        f"{prev.callee}")
        conns += [f".m_{n}(o{i}_{n})" for n, _ in outs]
        lines.append(f"    {st.name}_stream u{i} (\n        " + ",\n        ".join(conns) + "\n    );")
    lines.append("    assign sv0 = s_valid;")
    lines.append("    assign s_ready = sr0;")
    for st in stages[1:]:
        lines.append(f"    assign sv{st.index} = mv{st.index - 1};")
        lines.append(f"    assign mr{st.index - 1} = sr{st.index};")
    lines.append(f"    assign m_valid = mv{last.index};")
    lines.append(f"    assign mr{last.index} = m_ready;")
    lines += [f"    assign m_{n} = o{last.index}_{n};" for n, _ in top_out]
    lines.append("endmodule")
    return "\n".join(lines) + "\n", {"in": top_in, "out": top_out}
