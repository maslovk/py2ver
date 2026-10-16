# control.py
"""
while loops, break / continue and early return, as hardware takes them.

Hardware has no jumps: every statement is built, and what runs is chosen by
conditions. So before translation:

  - a return that isn't the last statement sets the results and a flag; the
    statements after it run only while the flag is clear, and the function
    returns the results at the end:

        if x > 255:                     ret = 0          (the results, a flag)
            return 255                  __done1 = 0
        y = x * 3                 ->    if x > 255:
        return y >> 2                       ret = 255
                                            __done1 = 1
                                        if __done1 == 0:
                                            y = x * 3
                                            ret = y >> 2
                                        return ret

  - break and continue in a loop work the same way, with a flag per loop
    (break also skips the loop's remaining iterations)
  - a while loop becomes a for loop over its largest number of iterations,
    each guarded by the loop's condition (and stopping for good once it is
    false). That number is worked out for a counting loop
    (while i < 10: ... i += 1, with i set to a constant before), or given as
    attr['while'] = N (for every while loop of the function), or measured on
    the examples by auto_fixed (samples=). If Python would need more
    iterations than that, the hardware stops early.

A returned name that every return returns at the same position keeps its
name (it becomes that output's name as usual); other positions are called
ret / ret0, ret1, ..., like returned expressions.
"""

from __future__ import annotations

import ast
import copy
import itertools
from typing import Any, Dict, List, Optional, Set

MAX_STATIC = 1 << 16          # largest iteration count worked out for a counting loop


class _Ctx:
    def __init__(self) -> None:
        self.ret_flag: Optional[str] = None
        self.ret_names: List[str] = []
        self.loops: List[Dict[str, Optional[str]]] = []      # innermost last: brk / cont


def _has(node: ast.AST, kinds) -> bool:
    """node contains one of kinds (not counting nested functions)."""
    return any(isinstance(n, kinds) for n in ast.walk(node))


def _name(i: int, n: int) -> str:
    return "ret" if n == 1 else f"ret{i}"


def _assign(name: str, value: ast.expr) -> ast.Assign:
    return ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=value)


def _not_any(flags: List[str]) -> ast.expr:
    """None of the flags (0 / 1) is set."""
    total: ast.expr = ast.Name(id=flags[0], ctx=ast.Load())
    for f in flags[1:]:
        total = ast.BinOp(left=total, op=ast.BitOr(), right=ast.Name(id=f, ctx=ast.Load()))
    return ast.Compare(left=total, ops=[ast.Eq()], comparators=[ast.Constant(0)])


class _Structure:
    def __init__(self, bound: Optional[int], used: Set[str]) -> None:
        self.bound = bound
        self.used = used
        self.count = itertools.count(1)
        self.ctx = _Ctx()

    def fresh(self, base: str) -> str:
        while True:
            name = f"__{base}{next(self.count)}"
            if name not in self.used:
                self.used.add(name)
                return name

    # ------------------------------------------------------------ statements
    def flags_set(self, s: ast.stmt) -> List[str]:
        """The flags statement s may set (for guarding what follows it)."""
        out = []
        loop = self.ctx.loops[-1] if self.ctx.loops else None
        if self.ctx.ret_flag and _has(s, ast.Return):
            out.append(self.ctx.ret_flag)
        if loop is not None:
            # (a break / continue in a loop nested in s belongs to that loop)
            def mine(node: ast.AST, kind) -> bool:
                if isinstance(node, kind):
                    return True
                if isinstance(node, (ast.For, ast.While)) and node is not s:
                    return False
                return any(mine(c, kind) for c in ast.iter_child_nodes(node))
            if loop["brk"] and mine(s, ast.Break):
                out.append(loop["brk"])
            if loop["cont"] and mine(s, ast.Continue):
                out.append(loop["cont"])
        return out

    def block(self, stmts: List[ast.stmt]) -> List[ast.stmt]:
        out: List[ast.stmt] = []
        for i, s in enumerate(stmts):
            if isinstance(s, ast.Return):
                out += self.ret(s)
                return out                               # (the rest never runs)
            if isinstance(s, ast.Break):
                out.append(_assign(self.ctx.loops[-1]["brk"], ast.Constant(1)))
                return out
            if isinstance(s, ast.Continue):
                out.append(_assign(self.ctx.loops[-1]["cont"], ast.Constant(1)))
                return out
            flags = self.flags_set(s)
            out += self.stmt(s)
            rest = stmts[i + 1:]
            if flags and rest:
                body = self.block(rest)
                if body:
                    out.append(ast.If(test=_not_any(flags), body=body, orelse=[]))
                return out
        return out

    def stmt(self, s: ast.stmt) -> List[ast.stmt]:
        if isinstance(s, ast.If):
            s.body = self.block(s.body) or [ast.Pass()]
            s.orelse = self.block(s.orelse)
            return [s]
        if isinstance(s, ast.While):
            return self.loop(s, self.while_items(s))
        if isinstance(s, ast.For):
            if not _has(s, (ast.Break, ast.Continue)) and not (
                    self.ctx.ret_flag and _has(s, ast.Return)):
                s.body = self.block(s.body)
                return [s]
            return self.loop(s, None)
        if isinstance(s, (ast.With, ast.Try)):
            raise NotImplementedError(f"'{type(s).__name__.lower()}' (line {s.lineno}) "
                                      "is not supported")
        return [s]

    def ret(self, s: ast.Return) -> List[ast.stmt]:
        if self.ctx.ret_flag is None:                   # (the final return)
            return [s]
        vals = (list(s.value.elts) if isinstance(s.value, ast.Tuple)
                else [] if s.value is None else [s.value])
        if len(vals) != len(self.ctx.ret_names):
            raise NotImplementedError(
                f"Every return must give {len(self.ctx.ret_names)} value(s); the one on line "
                f"{s.lineno} gives {len(vals)}")
        out = [_assign(n, v) for n, v in zip(self.ctx.ret_names, vals)
               if not (isinstance(v, ast.Name) and v.id == n)]
        out.append(_assign(self.ctx.ret_flag, ast.Constant(1)))
        return out

    # ------------------------------------------------------------ loops
    def loop(self, s, items: Optional[ast.expr]) -> List[ast.stmt]:
        """A for loop (or a while loop over `items`) with flags for break /
        continue (and an early return inside it)."""
        is_while = isinstance(s, ast.While)
        if s.orelse:
            raise NotImplementedError(f"A loop with an 'else' (line {s.lineno}) is not supported")
        brk = self.fresh("brk") if (is_while or _has(s, ast.Break)) else None
        cont = self.fresh("cont") if _has(s, ast.Continue) else None
        self.ctx.loops.append({"brk": brk, "cont": cont})
        body = self.block(s.body)
        self.ctx.loops.pop()
        if is_while:                                    # stops for good once its test fails
            body = [ast.If(test=copy.deepcopy(s.test), body=body,
                           orelse=[_assign(brk, ast.Constant(1))])]
        if cont:
            body = [_assign(cont, ast.Constant(0))] + body
        stop = [f for f in (brk, self.ctx.ret_flag) if f]
        if stop:
            body = [ast.If(test=_not_any(stop), body=body, orelse=[])]
        if is_while:
            target = ast.Name(id=self.fresh("it"), ctx=ast.Store())
            new = ast.For(target=target, iter=items, body=body, orelse=[], type_comment=None)
        else:
            new = ast.For(target=s.target, iter=s.iter, body=body, orelse=[],
                          type_comment=None)
        pre = [_assign(brk, ast.Constant(0))] if brk else []
        return pre + [ast.copy_location(new, s)]

    def while_items(self, s: ast.While) -> ast.expr:
        n = self.static_count(s)
        if n is None:
            n = self.bound
        if n is None:
            raise NotImplementedError(
                f"The while loop on line {s.lineno} needs a largest number of iterations: give "
                "attr['while'] = N, or samples= to measure it")
        return ast.Call(func=ast.Name(id="range", ctx=ast.Load()), args=[ast.Constant(int(n))],
                        keywords=[])

    def static_count(self, s: ast.While) -> Optional[int]:
        """Iterations of `while i <op> C:` with i a constant before the loop and
        changed only by one unconditional i += / -= constant in its body."""
        t = s.test
        if not (isinstance(t, ast.Compare) and len(t.ops) == 1):
            return None
        left, right, op = t.left, t.comparators[0], t.ops[0]
        if isinstance(right, ast.Name) and not isinstance(left, ast.Name):
            left, right = right, left
            op = {ast.Lt: ast.Gt(), ast.Gt: ast.Lt(), ast.LtE: ast.GtE(), ast.GtE: ast.LtE()}.get(
                type(op), op)
        if not isinstance(left, ast.Name) or not isinstance(right, ast.Constant) \
                or type(right.value) is not int:
            return None
        var, limit = left.id, right.value
        start = self.before.get(id(s), {}).get(var)
        if start is None:
            return None
        steps = []
        for b in s.body:
            if isinstance(b, ast.AugAssign) and isinstance(b.target, ast.Name) \
                    and b.target.id == var and isinstance(b.value, ast.Constant) \
                    and type(b.value.value) is int and isinstance(b.op, (ast.Add, ast.Sub)):
                steps.append(b.value.value if isinstance(b.op, ast.Add) else -b.value.value)
        stores = [n for n in ast.walk(ast.Module(body=s.body, type_ignores=[]))
                  if isinstance(n, ast.Name) and n.id == var and isinstance(n.ctx, ast.Store)]
        if len(steps) != 1 or len(stores) != 1 or steps[0] == 0:
            return None
        test = {ast.Lt: int.__lt__, ast.LtE: int.__le__, ast.Gt: int.__gt__, ast.GtE: int.__ge__,
                ast.NotEq: int.__ne__}.get(type(op))
        if test is None:
            return None
        i, n = start, 0
        while test(i, limit):
            i += steps[0]
            n += 1
            if n > MAX_STATIC:
                return None
        return n


def _constants_before(fd: ast.FunctionDef) -> Dict[int, Dict[str, int]]:
    """For each while loop: the variables set to an int constant by the statements
    just before it in its block (and not changed in between)."""
    out: Dict[int, Dict[str, int]] = {}

    def walk(stmts: List[ast.stmt]) -> None:
        known: Dict[str, int] = {}
        for s in stmts:
            if isinstance(s, ast.While):
                out[id(s)] = dict(known)
            if isinstance(s, ast.Assign) and len(s.targets) == 1 \
                    and isinstance(s.targets[0], ast.Name) and isinstance(s.value, ast.Constant) \
                    and type(s.value.value) is int:
                known[s.targets[0].id] = s.value.value
            else:
                for n in ast.walk(s):
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                        known.pop(n.id, None)
            for field in ("body", "orelse"):
                inner = getattr(s, field, None)
                if isinstance(inner, list) and inner and isinstance(inner[0], ast.stmt):
                    walk(inner)
    walk(fd.body)
    return out


def needs_structuring(fd: ast.FunctionDef) -> bool:
    if _has(fd, (ast.While, ast.Break, ast.Continue)):
        return True
    returns = [n for n in ast.walk(fd) if isinstance(n, ast.Return)]
    return len(returns) > 1 or (len(returns) == 1 and fd.body and returns[0] is not fd.body[-1])


def structure_function(fd: ast.FunctionDef, bound: Optional[int] = None) -> ast.FunctionDef:
    """The function with while / break / continue / early returns rewritten."""
    if not needs_structuring(fd):
        return fd
    fd = copy.deepcopy(fd)
    used = {n.id for n in ast.walk(fd) if isinstance(n, ast.Name)} | {a.arg for a in fd.args.args}
    st = _Structure(bound, used)
    st.before = _constants_before(fd)
    returns = [n for n in ast.walk(fd) if isinstance(n, ast.Return)]
    early = len(returns) > 1 or (returns and returns[0] is not fd.body[-1])
    pre: List[ast.stmt] = []
    if early:
        shapes = {len(r.value.elts) if isinstance(r.value, ast.Tuple) else (r.value is not None)
                  for r in returns}
        if len(shapes) != 1:
            raise NotImplementedError("Every return must give the same number of values")
        n = shapes.pop()
        vals = [list(r.value.elts) if isinstance(r.value, ast.Tuple) else [r.value]
                for r in returns] if n else []
        names = []
        for i in range(int(n)):
            col = [v[i] for v in vals]
            same = all(isinstance(v, ast.Name) and v.id == col[0].id for v in col) \
                if isinstance(col[0], ast.Name) else False
            names.append(col[0].id if same else _name(i, int(n)))
        st.ctx.ret_flag = st.fresh("done")
        st.ctx.ret_names = names
        params = {a.arg for a in fd.args.args}
        for nm, col in zip(names, zip(*vals) if vals else []):
            if nm in params or (isinstance(col[0], ast.Name) and col[0].id == nm):
                continue
            pre.append(_assign(nm, ast.Constant(0)))      # (every path sets it)
        pre.append(_assign(st.ctx.ret_flag, ast.Constant(0)))
    body = st.block(fd.body)
    if early:
        ret = (ast.Tuple(elts=[ast.Name(id=n, ctx=ast.Load()) for n in st.ctx.ret_names],
                         ctx=ast.Load()) if len(st.ctx.ret_names) != 1
               else ast.Name(id=st.ctx.ret_names[0], ctx=ast.Load()))
        body.append(ast.Return(value=ret if st.ctx.ret_names else None))
    fd.body = pre + body
    return ast.fix_missing_locations(fd)


def structure(tree: ast.Module, attr: Optional[Dict[str, Any]]) -> ast.Module:
    fd = tree.body[0]
    if not isinstance(fd, ast.FunctionDef) or not needs_structuring(fd):
        return tree
    tree = copy.copy(tree)
    tree.body = [structure_function(fd, while_bound(attr))] + tree.body[1:]
    return tree


def while_bound(attr: Optional[Dict[str, Any]]) -> Optional[int]:
    """attr['while']: N, or {'iterations': N}."""
    v = (attr or {}).get("while")
    if isinstance(v, dict):
        v = v.get("iterations")
    return None if v is None else int(v)
