# fold.py
"""
Fold a loop: run its body over several clocks instead of unrolling it.

A `for k in range(...)` loop with N iterations normally becomes N copies of
its body. Folded with `unroll=U`, it becomes U copies run once per clock for
N / U clocks (phases) per sample, so a 64-tap FIR needs one multiplier
instead of 64. The design then takes N / U clocks per sample, like a design
with block RAM (same phase counter; state commits in the last phase).

The function is rewritten for the translator (its Python meaning, and so the
references, are unchanged):

    before the loop                      before the loop
    for k in range(a, b, s):      ->     c = c if fphase == 0 else fc_c    (each carried c)
        body                             k = a + s * (fphase * U + j)       (j = 0 .. U-1)
    after the loop                       body                               (each j)
                                         fcn_c = c
                                         after the loop, also returning fcn_c

fphase is the phase counter and fc_c a register loaded with fcn_c every
clock; core turns these extra ports into the phase counter and registers.
A variable is carried if the body assigns it and it has a value before the
loop. Its register is as wide as the widest value the unrolled design gives
it, so folding never changes a result.
"""

from __future__ import annotations

import ast
import copy
from typing import Any, Dict, List, Optional, Set, Tuple


def _assigned(stmts: List[ast.stmt]) -> Set[str]:
    out: Set[str] = set()
    for s in stmts:
        for n in ast.walk(s):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                out.add(n.id)
    return out


def _range_values(it: ast.expr) -> Optional[List[int]]:
    if not (isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "range"
            and not it.keywords):
        return None
    try:
        return list(range(*[ast.literal_eval(a) for a in it.args]))
    except (ValueError, TypeError):
        return None


def _fresh(name: str, used: Set[str]) -> str:
    while name in used:
        name += "_"
    used.add(name)
    return name


def carried_widths(attr: Dict[str, Dict[str, Any]], names: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Register format for each carried variable: wide enough for every version
    of it in the unrolled design (attr from the translator: c, c__v1, ...).
    """
    out: Dict[str, Dict[str, Any]] = {}
    for c in names:
        metas = [m for k, m in attr.items()
                 if k == c or (k.startswith(f"{c}__v") and k[len(c) + 3:].isdigit())]
        if not metas:
            raise NotImplementedError(f"Can't size the folded register for '{c}'")
        signed = any(int(m.get("signed", 0)) for m in metas)
        width = max(int(m["width"]) + (1 if signed and not int(m.get("signed", 0)) else 0)
                    for m in metas)
        out[c] = {"signed": int(signed), "width": width, "type": "wire"}
    return out


def find_loop(fd: ast.FunctionDef, loop: Any) -> Tuple[int, ast.For, List[int]]:
    """The top-level range loop to fold: (its index in the body, the loop, k values)."""
    loops = [(i, s, _range_values(s.iter)) for i, s in enumerate(fd.body)
             if isinstance(s, ast.For) and isinstance(s.target, ast.Name)]
    loops = [(i, s, v) for i, s, v in loops if v is not None]
    if isinstance(loop, str):
        loops = [x for x in loops if x[1].target.id == loop]
        if not loops:
            raise ValueError(f"No top-level 'for {loop} in range(...)' loop to fold")
    if not loops:
        raise ValueError("No top-level 'for ... in range(...)' loop to fold")
    return max(loops, key=lambda x: len(x[2]))


def fold_loop(tree: ast.Module, loop: Any, unroll: int,
              widths: Dict[str, Dict[str, Any]]) -> Tuple[ast.Module, Dict[str, Any]]:
    """
    Rewrite the module's first function with the loop folded. `widths` is
    the translator's attr for the unfolded function (to size registers).
    Returns (tree, info): info has 'phases', 'phase' (the phase port name),
    'regs' ({register: next value}) and 'attr' (formats of the new signals).
    """
    tree = copy.deepcopy(tree)
    fd = tree.body[0]
    index, node, values = find_loop(fd, loop)
    unroll = int(unroll)
    if unroll < 1 or len(values) % unroll:
        raise ValueError(f"unroll={unroll} must divide the {len(values)} iterations of the loop")
    phases = len(values) // unroll
    k = node.target.id
    if node.orelse:
        raise NotImplementedError("A folded loop can't have an else block")
    before, after = fd.body[:index], fd.body[index + 1:]
    if k in {n.id for s in after for n in ast.walk(s) if isinstance(n, ast.Name)}:
        raise NotImplementedError(f"The folded loop's variable '{k}' is used after the loop")
    for s in node.body:
        for n in ast.walk(s):
            if (isinstance(n, (ast.Assign, ast.AugAssign))
                    and any(isinstance(t, ast.Subscript)
                            for t in (n.targets if isinstance(n, ast.Assign) else [n.target]))):
                raise NotImplementedError("A folded loop can't assign list elements yet")
    params = [a.arg for a in fd.args.args]
    carried = sorted((_assigned(node.body) - {k}) & (_assigned(before) | set(params)))
    sizes = carried_widths(widths, carried)

    used = {n.id for n in ast.walk(fd) if isinstance(n, ast.Name)} | set(params)
    phase = _fresh("fphase", used)
    regs = {c: (_fresh(f"fc_{c}", used), _fresh(f"fcn_{c}", used)) for c in carried}

    def name(n: str, store: bool = False) -> ast.Name:
        return ast.Name(id=n, ctx=ast.Store() if store else ast.Load())

    first = ast.Compare(left=name(phase), ops=[ast.Eq()], comparators=[ast.Constant(0)])
    body: List[ast.stmt] = list(before)
    for c, (reg, _) in regs.items():
        body.append(ast.Assign(targets=[name(c, True)],
                               value=ast.IfExp(test=copy.deepcopy(first), body=name(c), orelse=name(reg))))
    start, step = values[0], (values[1] - values[0]) if len(values) > 1 else 1
    base = phase
    if unroll > 1:                       # first iteration of this clock: phase * U, once
        base = _fresh("fbase", used)
        body.append(ast.Assign(targets=[name(base, True)], value=ast.BinOp(
            left=name(phase), op=ast.Mult(), right=ast.Constant(unroll))))
    for j in range(unroll):
        i = name(base) if j == 0 else ast.BinOp(left=name(base), op=ast.Add(), right=ast.Constant(j))
        kv = i if step == 1 else ast.BinOp(left=ast.Constant(step), op=ast.Mult(), right=i)
        if start:
            kv = ast.BinOp(left=ast.Constant(start), op=ast.Add(), right=kv)
        body.append(ast.Assign(targets=[name(k, True)], value=kv))
        body += copy.deepcopy(node.body)
    for c, (_, nxt) in regs.items():
        body.append(ast.Assign(targets=[name(nxt, True)], value=name(c)))
    ret = after[-1] if after and isinstance(after[-1], ast.Return) else None
    if ret is None or ret.value is None:
        raise NotImplementedError("A function with a folded loop must end with 'return ...'")
    vals = ret.value.elts if isinstance(ret.value, ast.Tuple) else [ret.value]
    named = []
    for i, v in enumerate(vals):          # a returned expression keeps its output name
        if not isinstance(v, ast.Name):
            out = f"ret{i}" if len(vals) > 1 else "ret"
            if out in used:
                raise NotImplementedError(f"Output name '{out}' is already used")
            after.insert(len(after) - 1, ast.Assign(targets=[name(out, True)], value=v))
            v = name(out)
        named.append(v)
    vals = named
    ret.value = ast.Tuple(elts=list(vals) + [name(nxt) for _, nxt in regs.values()], ctx=ast.Load())
    fd.body = body + after
    fd.args.args = list(fd.args.args) + [ast.arg(arg=phase)] + [ast.arg(arg=r) for r, _ in regs.values()]
    ast.fix_missing_locations(tree)

    attr = {phase: {"signed": 0, "width": max(1, (phases - 1).bit_length()), "type": "wire"}}
    for c, (reg, nxt) in regs.items():
        attr[reg] = dict(sizes[c])
        attr[nxt] = dict(sizes[c])
    return tree, {"phases": phases, "phase": phase, "loop": k,
                  "regs": {reg: nxt for reg, nxt in regs.values()}, "attr": attr}
