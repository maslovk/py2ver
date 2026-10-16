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


def _const(node: ast.expr) -> int:
    """A constant int expression: literals, len() of a literal list, + - * //."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "len" \
            and len(node.args) == 1 and isinstance(node.args[0], (ast.List, ast.Tuple)):
        return len(node.args[0].elts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv)):
        a, b = _const(node.left), _const(node.right)
        return {ast.Add: a + b, ast.Sub: a - b, ast.Mult: a * b}.get(type(node.op), None) \
            if not isinstance(node.op, ast.FloorDiv) else a // b
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_const(node.operand)
    value = ast.literal_eval(node)
    if not isinstance(value, int):
        raise ValueError("not an int")
    return value


def _range_values(it: ast.expr) -> Optional[List[int]]:
    if not (isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "range"
            and not it.keywords):
        return None
    try:
        return list(range(*[_const(a) for a in it.args]))
    except (ValueError, TypeError, ZeroDivisionError):
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


def _list_writes(stmts: List[ast.stmt]) -> Set[str]:
    """Lists whose elements the statements assign (xs[i] = ..., xs[i] += ...)."""
    out: Set[str] = set()
    for s in stmts:
        for n in ast.walk(s):
            targets = n.targets if isinstance(n, ast.Assign) else [n.target] \
                if isinstance(n, ast.AugAssign) else []
            for t in targets:
                if isinstance(t, ast.Subscript):
                    if not isinstance(t.value, ast.Name) or isinstance(t.slice, ast.Slice):
                        raise NotImplementedError(
                            "A folded loop assigns list elements as xs[i] = ... only")
                    out.add(t.value.id)
    return out


def fold_loop(tree: ast.Module, loop: Any, unroll: int,
              widths: Dict[str, Dict[str, Any]],
              lists: Dict[str, int]) -> Tuple[ast.Module, Dict[str, Any]]:
    """
    Rewrite the module's first function with the loop folded. `widths` is
    the translator's attr for the unfolded function (to size registers) and
    `lists` its list lengths.
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
    for s in after:                     # k after the loop, before something sets it again
        if isinstance(s, ast.For) and isinstance(s.target, ast.Name) and s.target.id == k:
            break
        if isinstance(s, ast.Assign) and any(isinstance(t, ast.Name) and t.id == k
                                             for t in s.targets) \
                and k not in {n.id for n in ast.walk(s.value) if isinstance(n, ast.Name)}:
            break
        if k in {n.id for n in ast.walk(s) if isinstance(n, ast.Name)}:
            raise NotImplementedError(f"The folded loop's variable '{k}' is used after the loop")
    params = [a.arg for a in fd.args.args]
    defined = _assigned(before) | set(params)
    carried = sorted((_assigned(node.body) - {k}) & defined)
    # A list whose elements the loop assigns is carried element by element
    carried_lists = sorted(_list_writes(node.body) & defined)
    for c in carried_lists:
        if c in carried:
            raise NotImplementedError(f"A folded loop can't both assign list '{c}' and its elements")
        if c not in lists:
            raise NotImplementedError(f"Can't tell the length of list '{c}'")
    sizes = carried_widths(widths, carried)
    for c in carried_lists:
        elems = carried_widths(widths, [f"{c}_{i}" for i in range(lists[c])])
        signed = any(m["signed"] for m in elems.values())
        sizes[c] = {"signed": int(signed), "type": "wire", "length": lists[c],
                    "width": max(m["width"] + (1 if signed and not m["signed"] else 0)
                                 for m in elems.values())}

    used = {n.id for n in ast.walk(fd) if isinstance(n, ast.Name)} | set(params)
    phase = _fresh("fphase", used)
    regs = {c: (_fresh(f"fc_{c}", used), _fresh(f"fcn_{c}", used)) for c in carried + carried_lists}

    def name(n: str, store: bool = False) -> ast.Name:
        return ast.Name(id=n, ctx=ast.Store() if store else ast.Load())

    first = ast.Compare(left=name(phase), ops=[ast.Eq()], comparators=[ast.Constant(0)])
    body: List[ast.stmt] = list(before)
    for c, (reg, _) in regs.items():
        if c in carried_lists:            # element by element: [c[i] if phase 0 else reg[i], ...]
            value = ast.List(elts=[ast.IfExp(
                test=copy.deepcopy(first),
                body=ast.Subscript(value=name(c), slice=ast.Constant(i), ctx=ast.Load()),
                orelse=ast.Subscript(value=name(reg), slice=ast.Constant(i), ctx=ast.Load()))
                for i in range(lists[c])], ctx=ast.Load())
        else:
            value = ast.IfExp(test=copy.deepcopy(first), body=name(c), orelse=name(reg))
        body.append(ast.Assign(targets=[name(c, True)], value=value))
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
        value = (ast.Call(func=name("list"), args=[name(c)], keywords=[]) if c in carried_lists
                 else name(c))
        body.append(ast.Assign(targets=[name(nxt, True)], value=value))
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
    ports: Dict[str, str] = {}               # register port -> its next value's port
    for c, (reg, nxt) in regs.items():
        attr[reg] = dict(sizes[c])
        attr[nxt] = {key: v for key, v in sizes[c].items() if key != "length"}
        if c in carried_lists:
            ports.update({f"{reg}_{i}": f"{nxt}_{i}" for i in range(lists[c])})
        else:
            ports[reg] = nxt
    return tree, {"phases": phases, "phase": phase, "loop": k, "regs": ports,
                  "params": {phase} | {reg for reg, _ in regs.values()}, "attr": attr}
