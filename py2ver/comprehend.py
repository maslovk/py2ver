# comprehend.py
"""
List comprehensions and generator expressions, written out as lists.

Lists in hardware have a fixed length, so a comprehension over something
whose length is known when translating becomes a list literal with one
element per item (before fixed point is lowered and the translator runs):

    ys = [x * g for x in xs]              ys = [xs[0] * g, xs[1] * g, xs[2] * g]
    s = sum(y * y for y in ys)      ->    s = sum([ys[0] * ys[0], ys[1] * ys[1], ...])
    zs = [ys[i] + i for i in range(3)]    zs = [ys[0] + 0, ys[1] + 1, ys[2] + 2]

It iterates over a list argument or variable of known length (attr
'length', a list literal, a slice of one, xs + ys, [v] * n), a module-level
list or tuple, range(...) of constants (and len() of such lists), or
enumerate / zip of these; several 'for's nest. An 'if' must be decidable
when translating (it may test the index, not the values).
"""

from __future__ import annotations

import ast
import copy
from typing import Any, Dict, List, Optional


class _Expand(ast.NodeTransformer):
    def __init__(self, lengths: Dict[str, int], ns: Dict[str, Any]) -> None:
        self.lengths = dict(lengths)
        self.ns = ns
        self.local: set = set()

    # ------------------------------------------------------------ lengths, constants
    def const(self, n: ast.expr) -> Optional[int]:
        """An int known when translating, or None."""
        if isinstance(n, ast.Constant) and type(n.value) is int:
            return n.value
        if isinstance(n, ast.Name) and n.id not in self.local and type(self.ns.get(n.id)) is int:
            return self.ns[n.id]
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "len" \
                and len(n.args) == 1:
            return self.length(n.args[0])
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            v = self.const(n.operand)
            return None if v is None else -v
        if isinstance(n, ast.BinOp):
            a, b = self.const(n.left), self.const(n.right)
            if a is None or b is None:
                return None
            ops = {ast.Add: lambda: a + b, ast.Sub: lambda: a - b, ast.Mult: lambda: a * b,
                   ast.FloorDiv: lambda: a // b if b else None, ast.Mod: lambda: a % b if b else None}
            f = ops.get(type(n.op))
            return f() if f else None
        return None

    def length(self, n: ast.expr) -> Optional[int]:
        if isinstance(n, (ast.List, ast.Tuple)):
            return len(n.elts)
        if isinstance(n, ast.Name):
            if n.id in self.lengths:
                return self.lengths[n.id]
            v = self.ns.get(n.id) if n.id not in self.local else None
            return len(v) if isinstance(v, (list, tuple)) else None
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add):
            a, b = self.length(n.left), self.length(n.right)
            return None if a is None or b is None else a + b
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult):
            for lst, k in ((n.left, n.right), (n.right, n.left)):
                if isinstance(lst, (ast.List, ast.Tuple)) and self.const(k) is not None:
                    return len(lst.elts) * self.const(k)
        if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Slice):
            m = self.length(n.value)
            s = n.slice
            parts = [None if x is None else self.const(x) for x in (s.lower, s.upper, s.step)]
            if m is None or any(x is None and y is not None for x, y in
                                zip(parts, (s.lower, s.upper, s.step))):
                return None
            return len(range(m)[slice(*parts)])
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and len(n.args) == 1 \
                and n.func.id in ("list", "tuple", "reversed"):
            return self.length(n.args[0])
        if isinstance(n, ast.ListComp):
            return None
        return None

    # ------------------------------------------------------------ iteration
    def items(self, it: ast.expr) -> Optional[List[ast.expr]]:
        """The items an iterable gives, as expressions, or None if unknown."""
        if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and not it.keywords:
            fn, args = it.func.id, it.args
            if fn == "range" and 1 <= len(args) <= 3:
                vals = [self.const(a) for a in args]
                if None in vals or (len(vals) == 3 and vals[2] == 0):
                    return None
                return [ast.Constant(v) for v in range(*vals)]
            if fn == "enumerate" and len(args) == 1:
                xs = self.items(args[0])
                return None if xs is None else [
                    ast.Tuple(elts=[ast.Constant(k), x], ctx=ast.Load()) for k, x in enumerate(xs)]
            if fn == "zip" and args:
                cols = [self.items(a) for a in args]
                if any(c is None for c in cols):
                    return None
                return [ast.Tuple(elts=list(row), ctx=ast.Load()) for row in zip(*cols)]
            if fn == "reversed" and len(args) == 1:
                xs = self.items(args[0])
                return None if xs is None else xs[::-1]
            if fn in ("list", "tuple") and len(args) == 1:
                return self.items(args[0])
        if isinstance(it, (ast.List, ast.Tuple)):
            return [copy.deepcopy(e) for e in it.elts]
        n = self.length(it)
        if n is None:
            return None
        return [ast.Subscript(value=copy.deepcopy(it), slice=ast.Constant(k), ctx=ast.Load())
                for k in range(n)]

    @staticmethod
    def bind(target: ast.expr, value: ast.expr, env: Dict[str, ast.expr]) -> bool:
        if isinstance(target, ast.Name):
            env[target.id] = value
            return True
        if isinstance(target, ast.Tuple) and isinstance(value, ast.Tuple) \
                and len(target.elts) == len(value.elts):
            return all(_Expand.bind(t, v, env) for t, v in zip(target.elts, value.elts))
        return False

    def expand(self, node) -> ast.expr:
        """[elt for ... in ...] as a list literal."""
        envs: List[Dict[str, ast.expr]] = [{}]
        for gen in node.generators:
            if gen.is_async:
                raise NotImplementedError("async comprehensions are not supported")
            nxt = []
            for env in envs:
                its = self.items(_subst(gen.iter, env))
                if its is None:
                    raise NotImplementedError(
                        f"Can't tell how many items '{ast.unparse(gen.iter)}' has in "
                        f"'{ast.unparse(node)}'; a list's length must be known when translating "
                        "(give it a 'length' in attr, or iterate over range(n))")
                for item in its:
                    e = dict(env)
                    if not self.bind(gen.target, item, e):
                        raise NotImplementedError(f"Can't unpack {ast.unparse(item)} into "
                                                  f"{ast.unparse(gen.target)}")
                    keep = True
                    for cond in gen.ifs:
                        v = _decide(_subst(cond, e), self)
                        if v is None:
                            raise NotImplementedError(
                                f"The condition '{ast.unparse(cond)}' in '{ast.unparse(node)}' "
                                "depends on values, so the list's length would too")
                        keep = keep and v
                    if keep:
                        nxt.append(e)
            envs = nxt
        elts = [self.visit(_subst(node.elt, env)) for env in envs]
        return ast.copy_location(ast.List(elts=elts, ctx=ast.Load()), node)

    # ------------------------------------------------------------ visiting
    def visit_ListComp(self, node: ast.ListComp):
        return self.expand(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp):
        return self.expand(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.local |= {a.arg for a in node.args.args}
        self.local |= {n.id for n in ast.walk(node) if isinstance(n, ast.Name)
                       and isinstance(n.ctx, ast.Store)}
        node.body = self.block(node.body)
        return node

    def block(self, stmts: List[ast.stmt]) -> List[ast.stmt]:
        out = []
        for s in stmts:
            s = self.visit(s)
            out.append(s)
            # Lengths of lists assigned on the way (statements run in order)
            if isinstance(s, ast.Assign) and len(s.targets) == 1 \
                    and isinstance(s.targets[0], ast.Name):
                n = self.length(s.value)
                if n is not None:
                    self.lengths[s.targets[0].id] = n
        return out

    def generic_visit(self, node):
        for field in ("body", "orelse", "finalbody"):
            stmts = getattr(node, field, None)
            if isinstance(stmts, list) and stmts and isinstance(stmts[0], ast.stmt):
                setattr(node, field, self.block(stmts))
        for field, value in ast.iter_fields(node):
            if field in ("body", "orelse", "finalbody") and isinstance(value, list) \
                    and value and isinstance(value[0], ast.stmt):
                continue
            if isinstance(value, list):
                setattr(node, field, [self.visit(v) if isinstance(v, ast.AST) else v
                                      for v in value])
            elif isinstance(value, ast.AST):
                setattr(node, field, self.visit(value))
        return node


def _subst(node: ast.expr, env: Dict[str, ast.expr]) -> ast.expr:
    """node with the comprehension's variables replaced by their items."""
    node = copy.deepcopy(node)
    if not env:
        return node

    class _S(ast.NodeTransformer):
        def __init__(self, env):
            self.env = env

        def visit_Name(self, n: ast.Name):
            if isinstance(n.ctx, ast.Load) and n.id in self.env:
                return copy.deepcopy(self.env[n.id])
            return n

        def _scoped(self, n):
            # An inner comprehension's own variables hide the outer ones
            inner = {t.id for g in n.generators for t in ast.walk(g.target)
                     if isinstance(t, ast.Name)}
            for g in n.generators:
                g.iter = self.visit(g.iter)
            sub = _S({k: v for k, v in self.env.items() if k not in inner})
            for g in n.generators:
                g.ifs = [sub.visit(c) for c in g.ifs]
            n.elt = sub.visit(n.elt)
            return n

        visit_ListComp = visit_GeneratorExp = _scoped

    return _S(env).visit(node)


def _decide(cond: ast.expr, ex: _Expand) -> Optional[bool]:
    """A condition known when translating (of constants), or None."""
    if isinstance(cond, ast.Compare):
        vals = [ex.const(x) for x in [cond.left] + cond.comparators]
        if None in vals:
            return None
        ops = {ast.Eq: int.__eq__, ast.NotEq: int.__ne__, ast.Lt: int.__lt__,
               ast.LtE: int.__le__, ast.Gt: int.__gt__, ast.GtE: int.__ge__}
        for op, a, b in zip(cond.ops, vals, vals[1:]):
            f = ops.get(type(op))
            if f is None:
                return None
            if not f(a, b):
                return False
        return True
    if isinstance(cond, ast.BoolOp):
        vals = [_decide(v, ex) for v in cond.values]
        if None in vals:
            return None
        return all(vals) if isinstance(cond.op, ast.And) else any(vals)
    if isinstance(cond, ast.UnaryOp) and isinstance(cond.op, ast.Not):
        v = _decide(cond.operand, ex)
        return None if v is None else not v
    v = ex.const(cond)
    return None if v is None else bool(v)


def expand_comprehensions(tree: ast.Module, attr: Dict[str, Any],
                          ns: Dict[str, Any]) -> ast.Module:
    """The module with every list comprehension / generator expression written out."""
    if not any(isinstance(n, (ast.ListComp, ast.GeneratorExp)) for n in ast.walk(tree)):
        return tree
    lengths = {k: int(m["length"]) for k, m in (attr or {}).items()
               if isinstance(m, dict) and "length" in m}
    tree = _Expand(lengths, ns).visit(copy.deepcopy(tree))
    return ast.fix_missing_locations(tree)
