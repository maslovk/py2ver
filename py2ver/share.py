# share.py
"""
Share multipliers and block-RAM tables: a design over several clocks.

A design normally becomes one big combinational circuit: every product its
own multiplier, every table lookup its own copy of the table in logic (a
numpy design written out element by element, say). With `multipliers=N` it
is scheduled over as many clocks as it needs instead:

  - loops over constants or lists are unrolled, and if / else becomes a
    selection (both branches computed, each variable chosen by the test)
  - every product runs on one of N shared multipliers (a product by a
    constant too, unless the constant is a power of two)
  - every function table (a fixed-point np.tanh, sigmoid, math.sin, ...) is
    a ROM in block RAM with one read per clock; a read gives its value one
    clock later
  - the rest (additions, shifts, comparisons, clamps) is built as it is and
    chained within a clock as far as the clock period allows
  - a value used in a later clock than the one computing it is kept in a
    register; nothing else is

The function is first written as one operation per line, the operations are
list-scheduled (the longest remaining path first) and the multiplier inputs
become multiplexers on the clock's number (the phase). The result is the
same contract fold.py gives core: a phase input, registers loaded every
clock, plus the ROMs. Every operation keeps the exact width the unshared
design gives it, so sharing never changes a result.
"""

from __future__ import annotations

import ast
import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .fixed import FIT_LIST, wrap

ROM_MIN = 64              # a constant table with at least this many entries goes into block RAM
DEFAULT_PERIOD_NS = 12.0  # the clock period the chaining aims for

# Rough delays (ns) on a Cyclone IV E, speed grade 6; SYN() measures the real ones
T_REG = 1.0               # clock to output of a register, plus routing
ROM_TCO = 3.5             # clock to output of a block RAM
ROM_SETUP = 0.5           # its address setup
MUX_LEVEL = 0.9           # one level of a 2:1 multiplexer
SEL_DECODE = 2.0          # which input a multiplexer takes in this clock


class NotShareable(NotImplementedError):
    """Code the scheduler can't take (while, break, whole-list uses, ...)."""


@dataclass
class Op:
    name: str
    kind: str                         # 'mul' | 'rom' | 'free'
    node: ast.expr                    # operands are Name(<op>) or leaves
    deps: List[str] = field(default_factory=list)
    table: Optional[Tuple[int, ...]] = None
    declared: Optional[Dict[str, int]] = None     # format of a declared variable's copy


def _names(node: ast.AST) -> List[str]:
    return [n.id for n in ast.walk(node) if isinstance(n, ast.Name)]


def _is_const(node: ast.expr) -> bool:
    return not any(isinstance(n, (ast.Name, ast.Subscript, ast.Call)) for n in ast.walk(node))


def _const_value(node: ast.expr) -> Any:
    return eval(compile(ast.Expression(ast.fix_missing_locations(copy.deepcopy(node))),
                        "<const>", "eval"), {})


def _clamped_index(node: ast.expr, n: int, ops: Dict[str, Op]) -> bool:
    """idx is max(0, min(n - 1, ...)) (as fixed.py writes a table index)."""
    if isinstance(node, ast.Name) and node.id in ops:
        node = ops[node.id].node
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "max" and len(node.args) == 2):
        return False
    lo, inner = node.args
    if isinstance(inner, ast.Name) and inner.id in ops:
        inner = ops[inner.id].node
    return (isinstance(lo, ast.Constant) and lo.value == 0
            and isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name)
            and inner.func.id == "min" and len(inner.args) == 2
            and isinstance(inner.args[0], ast.Constant) and inner.args[0].value == n - 1)


class _ToOps:
    """The function's body as operations, one per line (three-address code)."""

    def __init__(self, fd: ast.FunctionDef, attr: Dict[str, Dict[str, Any]]) -> None:
        self.attr = attr
        self.params = [a.arg for a in fd.args.args]
        self.lists = {p for p in self.params if "length" in attr.get(p, {})}
        self.env: Dict[str, Any] = {}             # variable -> ref, or [refs] for a list
        self.ops: Dict[str, Op] = {}
        self.cse: Dict[str, str] = {}
        used = {n.id for n in ast.walk(fd) if isinstance(n, ast.Name)} | set(self.params)
        self.used = used
        self.count = 0
        self.depth = 0                            # inside a loop or an if
        self.outputs: List[Tuple[str, Any]] = []

    def fresh(self, base: str) -> str:
        while True:
            self.count += 1
            name = f"{base}{self.count}"
            if name not in self.used:
                self.used.add(name)
                return name

    def op(self, kind: str, node: ast.expr, table=None, declared=None) -> ast.Name:
        key = (kind, ast.dump(node), repr(declared))
        if key in self.cse:
            return ast.Name(id=self.cse[key], ctx=ast.Load())
        name = self.fresh("s")
        deps = sorted({n for n in _names(node) if n in self.ops})
        self.ops[name] = Op(name, kind, node, deps, table, declared)
        self.cse[key] = name
        return ast.Name(id=name, ctx=ast.Load())

    def _fmt(self, var: str) -> Optional[Dict[str, int]]:
        m = self.attr.get(var)
        if isinstance(m, dict) and "width" in m:
            return {"width": int(m["width"]), "signed": int(m.get("signed", 0))}
        return None

    def declare(self, ref: ast.expr, var: str) -> ast.expr:
        """ref assigned to declared variable var: wrapped to its format."""
        fmt = self._fmt(var)
        if fmt is None:
            return ref
        if _is_const(ref):
            return ast.Constant(wrap(int(_const_value(ref)), fmt["width"], bool(fmt["signed"])))
        return self.op("free", ref, declared=fmt)

    # ---------------------------------------------------------------- expressions
    def elements(self, node: ast.expr) -> Optional[List[ast.expr]]:
        """The elements of a list-valued expression, or None if it isn't one."""
        if isinstance(node, (ast.List, ast.Tuple)):
            return [self.expr(e) for e in node.elts]
        if isinstance(node, ast.Name):
            v = self.env.get(node.id)
            if isinstance(v, list):
                return [copy.deepcopy(e) for e in v]
            if v is None and node.id in self.lists:
                return [ast.Subscript(value=ast.Name(id=node.id, ctx=ast.Load()),
                                      slice=ast.Constant(k), ctx=ast.Load())
                        for k in range(int(self.attr[node.id]["length"]))]
            return None
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            a, b = self.elements(node.left), self.elements(node.right)
            return None if a is None or b is None else a + b
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            for lst, k in ((node.left, node.right), (node.right, node.left)):
                xs = self.elements(lst) if isinstance(lst, (ast.List, ast.Tuple)) else None
                if xs is not None:
                    n = self.expr(k)
                    if not _is_const(n):
                        raise NotShareable(f"[...] * n needs a constant n: {ast.unparse(node)}")
                    return [copy.deepcopy(x) for _ in range(int(_const_value(n))) for x in xs]
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice):
            xs = self.elements(node.value)
            if xs is None:
                return None
            bounds = []
            for b in (node.slice.lower, node.slice.upper, node.slice.step):
                v = None if b is None else self.expr(b)
                if v is not None and not _is_const(v):
                    raise NotShareable(f"A slice needs constant bounds: {ast.unparse(node)}")
                bounds.append(None if v is None else int(_const_value(v)))
            return xs[slice(*bounds)]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in ("list", "tuple") and len(node.args) == 1:
                return self.elements(node.args[0])
            if node.func.id == FIT_LIST:
                xs = self.elements(node.args[0])
                shift, rnd, lo, hi = (_const_value(a) for a in node.args[1:])
                out = []
                for x in xs:
                    if shift > 0:
                        if rnd:
                            x = self._bin(x, ast.Add(), ast.Constant(1 << (shift - 1)))
                        v = self._bin(x, ast.RShift(), ast.Constant(shift))
                    else:
                        v = self._bin(x, ast.LShift(), ast.Constant(-shift))
                    if lo is not None:
                        v = self._call("max", [ast.Constant(lo),
                                               self._call("min", [ast.Constant(hi), v])])
                    out.append(v)
                return out
        return None

    def _bin(self, l: ast.expr, op: ast.operator, r: ast.expr) -> ast.expr:
        node = ast.BinOp(left=l, op=op, right=r)
        if _is_const(l) and _is_const(r):
            return node
        mul = isinstance(op, ast.Mult) and not (_is_const(l) and _is_const(r))
        if mul and (_is_const(l) or _is_const(r)):
            c = abs(int(_const_value(l if _is_const(l) else r)))
            mul = c & (c - 1) != 0                     # (0, 1 and powers of two: shifts)
        return self.op("mul" if mul else "free", node)

    def _call(self, fn: str, args: List[ast.expr]) -> ast.expr:
        node = ast.Call(func=ast.Name(id=fn, ctx=ast.Load()), args=args, keywords=[])
        if all(_is_const(a) for a in args):
            return ast.Constant(_const_value(node)) if fn in ("min", "max", "abs") else node
        return self.op("free", node)

    def expr(self, n: ast.expr) -> ast.expr:
        if isinstance(n, ast.Constant):
            return n
        if isinstance(n, ast.Name):
            v = self.env.get(n.id)
            if isinstance(v, list) or (v is None and n.id in self.lists):
                raise NotShareable(f"'{n.id}' is used as a whole list")
            if v is not None:
                return copy.deepcopy(v)
            return ast.Name(id=n.id, ctx=ast.Load())       # a parameter (or a global)
        if isinstance(n, ast.Subscript):
            if isinstance(n.slice, ast.Slice):
                raise NotShareable("Slices are not supported")
            base = n.value
            if isinstance(base, (ast.Tuple, ast.List)) and all(
                    isinstance(e, ast.Constant) for e in base.elts):
                idx = self.expr(n.slice)
                values = tuple(e.value for e in base.elts)
                if _is_const(idx):
                    return ast.Constant(values[int(_const_value(idx))])
                table = ast.Subscript(value=ast.Tuple(elts=[ast.Constant(v) for v in values],
                                                      ctx=ast.Load()), slice=idx, ctx=ast.Load())
                if len(values) >= ROM_MIN and _clamped_index(idx, len(values), self.ops):
                    return self.op("rom", table, table=values)
                return self.op("free", table)
            elems = self.elements(base)
            if elems is None:
                raise NotShareable(f"Can't index {ast.unparse(base)}")
            idx = self.expr(n.slice)
            if _is_const(idx):
                k = int(_const_value(idx))
                if not -len(elems) <= k < len(elems):
                    raise NotShareable(f"Index {k} is out of range in {ast.unparse(n)}")
                return elems[k]
            return self.op("free", ast.Subscript(
                value=ast.Tuple(elts=elems, ctx=ast.Load()), slice=idx, ctx=ast.Load()))
        if isinstance(n, ast.BinOp):
            return self._bin(self.expr(n.left), n.op, self.expr(n.right))
        if isinstance(n, ast.UnaryOp):
            v = self.expr(n.operand)
            node = ast.UnaryOp(op=n.op, operand=v)
            return node if _is_const(v) else self.op("free", node)
        if isinstance(n, ast.Compare):
            node = ast.Compare(left=self.expr(n.left), ops=list(n.ops),
                               comparators=[self.expr(c) for c in n.comparators])
            return node if _is_const(node) else self.op("free", node)
        if isinstance(n, ast.BoolOp):
            node = ast.BoolOp(op=n.op, values=[self.expr(v) for v in n.values])
            return node if _is_const(node) else self.op("free", node)
        if isinstance(n, ast.IfExp):
            node = ast.IfExp(test=self.expr(n.test), body=self.expr(n.body),
                             orelse=self.expr(n.orelse))
            return node if _is_const(node) else self.op("free", node)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and not n.keywords:
            fn = n.func.id
            if fn in ("min", "max") and len(n.args) == 1:
                elems = self.elements(n.args[0])
                if elems is None:
                    raise NotShareable(f"Unsupported call: {ast.unparse(n)}")
                return self._call(fn, elems)
            if fn == "sum" and len(n.args) == 1:
                elems = self.elements(n.args[0])
                if elems is None:
                    raise NotShareable(f"Unsupported call: {ast.unparse(n)}")
                acc: ast.expr = ast.Constant(0)
                for e in elems:
                    acc = self._bin(acc, ast.Add(), e)
                return acc
            if fn == "len" and len(n.args) == 1:
                elems = self.elements(n.args[0])
                if elems is not None:
                    return ast.Constant(len(elems))
            if fn in ("min", "max", "abs", "int", "bool"):
                return self._call(fn, [self.expr(a) for a in n.args])
        raise NotShareable(f"Unsupported expression: {ast.unparse(n)}")

    # ---------------------------------------------------------------- statements
    def assign(self, var: str, value: ast.expr) -> None:
        elems = self.elements(value)
        if elems is not None:
            self.env[var] = [self.declare(e, var) for e in elems]
        else:
            self.env[var] = self.declare(self.expr(value), var)

    def store(self, target: ast.Subscript, value: ast.expr) -> None:
        if not isinstance(target.value, ast.Name):
            raise NotShareable(f"Unsupported assignment to {ast.unparse(target)}")
        var = target.value.id
        elems = self.elements(target.value)
        idx = self.expr(target.slice)
        if elems is None or not _is_const(idx):
            raise NotShareable(f"Only xs[constant] = ... is supported: {ast.unparse(target)}")
        k = int(_const_value(idx)) % len(elems)
        elems[k] = self.declare(self.expr(value), var)
        self.env[var] = elems

    def body(self, stmts: List[ast.stmt]) -> None:
        for s in stmts:
            if isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant):
                continue                                     # docstring
            if isinstance(s, ast.Assign) and len(s.targets) == 1:
                t = s.targets[0]
                if isinstance(t, ast.Name):
                    self.assign(t.id, s.value)
                    continue
                if isinstance(t, ast.Subscript):
                    self.store(t, s.value)
                    continue
                if isinstance(t, ast.Tuple) and isinstance(s.value, ast.Tuple) \
                        and len(t.elts) == len(s.value.elts) \
                        and all(isinstance(e, ast.Name) for e in t.elts):
                    vals = [(e.id, self.elements(v) if isinstance(v, (ast.List, ast.Tuple))
                             else self.expr(v)) for e, v in zip(t.elts, s.value.elts)]
                    for name, v in vals:
                        self.env[name] = ([self.declare(x, name) for x in v]
                                          if isinstance(v, list) else self.declare(v, name))
                    continue
            if isinstance(s, ast.For):
                self.loop(s)
                continue
            if isinstance(s, ast.If):
                self.branch(s)
                continue
            if isinstance(s, ast.Pass):
                continue
            if isinstance(s, ast.Return):
                if self.depth:
                    raise NotShareable(f"A return inside a loop or an if (line {s.lineno}) "
                                       "isn't supported")
                vals = (s.value.elts if isinstance(s.value, ast.Tuple)
                        else [] if s.value is None else [s.value])
                for i, v in enumerate(vals):
                    name = v.id if isinstance(v, ast.Name) else (f"ret{i}" if len(vals) > 1
                                                                 else "ret")
                    elems = self.elements(v)
                    self.outputs.append((name, elems if elems is not None else self.expr(v)))
                continue
            kind = type(s).__name__
            raise NotShareable(
                f"multipliers= can't schedule a '{kind}' statement "
                f"(line {getattr(s, 'lineno', '?')})")

    def loop(self, s: ast.For) -> None:
        """A for loop over items known now, unrolled."""
        if s.orelse:
            raise NotShareable(f"A for loop with an else (line {s.lineno}) isn't supported")
        for n in ast.walk(s):
            if isinstance(n, (ast.Break, ast.Continue, ast.While)):
                raise NotShareable(f"'{type(n).__name__.lower()}' in a loop (line {n.lineno}) "
                                   "isn't supported")
        items = self.items(s.iter)
        if items is None:
            raise NotShareable(f"Can't unroll 'for ... in {ast.unparse(s.iter)}' (line {s.lineno}):"
                               " it needs constant bounds or a list of known length")
        self.depth += 1
        for item in items:
            self.bind(s.target, item)
            self.body(s.body)
        self.depth -= 1

    def items(self, it: ast.expr) -> Optional[List[ast.expr]]:
        if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and not it.keywords:
            fn, args = it.func.id, it.args
            if fn == "range":
                vals = [self.expr(a) for a in args]
                if not all(_is_const(v) for v in vals):
                    return None
                return [ast.Constant(v) for v in range(*(int(_const_value(v)) for v in vals))]
            if fn in ("enumerate", "zip", "reversed"):
                cols = [self.items(a) for a in args]
                if any(c is None for c in cols):
                    return None
                if fn == "reversed":
                    return cols[0][::-1]
                if fn == "enumerate":
                    return [ast.Tuple(elts=[ast.Constant(k), x], ctx=ast.Load())
                            for k, x in enumerate(cols[0])]
                return [ast.Tuple(elts=list(row), ctx=ast.Load()) for row in zip(*cols)]
        return self.elements(it)

    def bind(self, target: ast.expr, item: ast.expr) -> None:
        if isinstance(target, ast.Name):
            self.env[target.id] = item
        elif isinstance(target, ast.Tuple) and isinstance(item, ast.Tuple) \
                and len(target.elts) == len(item.elts):
            for t, v in zip(target.elts, item.elts):
                self.bind(t, v)
        else:
            raise NotShareable(f"Can't unpack {ast.unparse(item)} into {ast.unparse(target)}")

    def branch(self, s: ast.If) -> None:
        """if / else as selections: both branches computed, each variable they
        assign then chosen by the condition (element by element for a list)."""
        test = self.expr(s.test)
        if _is_const(test):                          # known now: only that branch
            self.body(s.body if _const_value(test) else s.orelse)
            return
        before = dict(self.env)
        self.depth += 1
        self.body(s.body)
        taken = self.env
        self.env = dict(before)
        self.body(s.orelse)
        other = self.env
        self.depth -= 1
        self.env = dict(before)
        for var in dict.fromkeys([*taken, *other]):
            a, b = taken.get(var), other.get(var)
            if a is None or b is None:               # (assigned in one branch only)
                self.env[var] = a if b is None else b
                continue
            if isinstance(a, list) != isinstance(b, list) or (
                    isinstance(a, list) and len(a) != len(b)):
                raise NotShareable(f"'{var}' is a different kind of value in the two branches "
                                   f"of the if on line {s.lineno}")
            if isinstance(a, list):
                self.env[var] = [self.pick(test, x, y) for x, y in zip(a, b)]
            else:
                self.env[var] = self.pick(test, a, b)

    def pick(self, test: ast.expr, a: ast.expr, b: ast.expr) -> ast.expr:
        if ast.dump(a) == ast.dump(b):
            return a
        node = ast.IfExp(test=copy.deepcopy(test), body=a, orelse=b)
        return node if _is_const(node) else self.op("free", node)


def _name(node: ast.expr) -> Optional[str]:
    return node.id if isinstance(node, ast.Name) else None


# ==================================================================== widths

def _output_stmts(outputs, resolve) -> List[ast.stmt]:
    stmts: List[ast.stmt] = []
    names = []
    for name, v in outputs:
        value = (ast.List(elts=[resolve(e) for e in v], ctx=ast.Load()) if isinstance(v, list)
                 else resolve(v))
        stmts.append(ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=value))
        names.append(name)
    ret = (ast.Tuple(elts=[ast.Name(id=n, ctx=ast.Load()) for n in names], ctx=ast.Load())
           if len(names) != 1 else ast.Name(id=names[0], ctx=ast.Load()))
    stmts.append(ast.Return(value=ret))
    return stmts


def _fmt_range(meta: Dict[str, Any]) -> Tuple[int, int]:
    w, sgn = int(meta.get("width", 1)), int(meta.get("signed", 0))
    return (-(1 << (w - 1)), (1 << (w - 1)) - 1) if sgn else (0, (1 << w) - 1)


def _fmt_of(r: Tuple[int, int]) -> Dict[str, int]:
    """The narrowest format holding every value in r."""
    lo, hi = r
    if lo >= 0:
        return {"width": max(1, hi.bit_length()), "signed": 0}
    return {"width": max((-lo - 1).bit_length(), hi.bit_length()) + 1, "signed": 1}


def _hull(*rs: Tuple[int, int]) -> Tuple[int, int]:
    return min(r[0] for r in rs), max(r[1] for r in rs)


def _bits_range(*rs: Tuple[int, int]) -> Tuple[int, int]:
    w = max(_fmt_of(r)["width"] + (0 if r[0] < 0 else 1) for r in rs)
    return -(1 << (w - 1)), (1 << (w - 1)) - 1


def _interval(n: ast.expr, rng: Dict[str, Tuple[int, int]],
              attr: Dict[str, Dict[str, Any]]) -> Tuple[int, int]:
    """Every value n can take (an interval; exact arithmetic, as the model)."""
    def iv(x: ast.expr) -> Tuple[int, int]:
        return _interval(x, rng, attr)

    if isinstance(n, ast.Constant) and isinstance(n.value, (int, bool)):
        return int(n.value), int(n.value)
    if isinstance(n, ast.Name):
        if n.id in rng:
            return rng[n.id]
        if "width" in attr.get(n.id, {}):
            return _fmt_range(attr[n.id])
        raise NotShareable(f"Unknown value '{n.id}'")
    if isinstance(n, ast.Subscript):
        if isinstance(n.value, ast.Name) and "width" in attr.get(n.value.id, {}):
            return _fmt_range(attr[n.value.id])
        if isinstance(n.value, ast.Tuple):
            return _hull(*[iv(e) for e in n.value.elts])
    if isinstance(n, ast.BinOp):
        (a, b), (c, d) = iv(n.left), iv(n.right)
        op = n.op
        if isinstance(op, ast.Add):
            return a + c, b + d
        if isinstance(op, ast.Sub):
            return a - d, b - c
        if isinstance(op, ast.Mult):
            ps = (a * c, a * d, b * c, b * d)
            return min(ps), max(ps)
        if isinstance(op, (ast.LShift, ast.RShift)) and c >= 0:
            f = (lambda x, k: x << k) if isinstance(op, ast.LShift) else (lambda x, k: x >> k)
            ps = (f(a, c), f(a, d), f(b, c), f(b, d))
            return min(ps), max(ps)
        if isinstance(op, ast.FloorDiv) and c == d and c != 0:
            ps = (a // c, b // c)
            return min(ps), max(ps)
        if isinstance(op, ast.Mod) and c == d and c != 0:
            return (0, c - 1) if c > 0 else (c + 1, 0)
        if isinstance(op, (ast.Div, ast.FloorDiv)) and (c > 0 or d < 0):
            m = max(abs(a), abs(b))
            return -m, m
        if isinstance(op, ast.BitAnd):
            if a >= 0 and c >= 0:
                return 0, min(b, d)
            if a >= 0 or c >= 0:
                return 0, b if a >= 0 else d
            return _bits_range((a, b), (c, d))
        if isinstance(op, (ast.BitOr, ast.BitXor)):
            if a >= 0 and c >= 0:
                return 0, (1 << max(b.bit_length(), d.bit_length())) - 1
            return _bits_range((a, b), (c, d))
    if isinstance(n, ast.UnaryOp):
        if isinstance(n.op, ast.Not):
            return 0, 1
        a, b = iv(n.operand)
        if isinstance(n.op, ast.USub):
            return -b, -a
        if isinstance(n.op, ast.UAdd):
            return a, b
        if isinstance(n.op, ast.Invert):
            return ~b, ~a
    if isinstance(n, (ast.Compare, ast.BoolOp)):
        return 0, 1                                    # (and / or give 0 / 1 here)
    if isinstance(n, ast.IfExp):
        return _hull(iv(n.body), iv(n.orelse))
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
        rs = [iv(x) for x in n.args]
        fn = n.func.id
        if fn == "min":
            return min(r[0] for r in rs), min(r[1] for r in rs)
        if fn == "max":
            return max(r[0] for r in rs), max(r[1] for r in rs)
        if fn == "abs":
            a, b = rs[0]
            return (a, b) if a >= 0 else (-b, -a) if b <= 0 else (0, max(-a, b))
        if fn == "int":
            return rs[0]
        if fn == "bool":
            return 0, 1
    raise NotShareable(f"Can't bound the values of {ast.unparse(n)}")


def _ranges(ops: List[Op], attr: Dict[str, Dict[str, Any]]) -> Dict[str, Tuple[int, int]]:
    rng: Dict[str, Tuple[int, int]] = {}
    for o in ops:
        r = _interval(o.node, rng, attr)
        if o.declared is not None:
            f = _fmt_range(o.declared)
            if not (f[0] <= r[0] and r[1] <= f[1]):     # it wraps
                r = f
        rng[o.name] = r
    return rng


# ==================================================================== delays

def _adder(w: int) -> float:
    return 1.2 + 0.06 * w


def _mult_delay(wa: int, wb: int) -> float:
    """An embedded multiplier, with the adders joining its 18 x 18 parts."""
    big, small = max(wa, wb), min(wa, wb)
    if big <= 18:
        return 4.0
    if small <= 18:
        return 6.0 if big <= 36 else 8.0
    return 7.0 if big <= 27 else 8.5 if big <= 36 else 11.5


def _free_delay(o: Op, w: int) -> float:
    n = o.node
    if o.declared is not None or isinstance(n, (ast.Name, ast.Constant)):
        return 0.0
    if isinstance(n, ast.BinOp):
        op = n.op
        if isinstance(op, (ast.Add, ast.Sub)):
            return _adder(w)
        if isinstance(op, (ast.LShift, ast.RShift)):
            return 0.0 if _is_const(n.right) else 2.0
        if isinstance(op, ast.Mult):                 # by a constant: shifts and adds
            return 2 * _adder(w)
        if isinstance(op, (ast.BitAnd, ast.BitOr, ast.BitXor)):
            return 0.4
        if isinstance(op, (ast.FloorDiv, ast.Mod)) and _is_const(n.right):
            c = abs(int(_const_value(n.right)))
            return 0.0 if c & (c - 1) == 0 else 8.0
        return 10.0
    if isinstance(n, ast.UnaryOp):
        return _adder(w) if isinstance(n.op, ast.USub) else 0.4
    if isinstance(n, ast.Compare):
        return _adder(w) * len(n.ops)
    if isinstance(n, ast.BoolOp):
        return 0.4
    if isinstance(n, ast.IfExp):
        return 0.6
    if isinstance(n, ast.Call):
        return _adder(w) + 0.6 if n.func.id in ("min", "max") else _adder(w)
    if isinstance(n, ast.Subscript):
        elts = n.value.elts if isinstance(n.value, ast.Tuple) else []
        return MUX_LEVEL * max(1, math.ceil(math.log2(max(2, len(elts)))))
    return 1.0


# ==================================================================== scheduling

@dataclass
class Schedule:
    phase: Dict[str, int]                 # op -> clock it runs in (a ROM: its read)
    ready: Dict[str, int]                 # op -> clock its value is there
    unit: Dict[str, Tuple[str, int]]      # mul / rom op -> (unit class, index)
    phases: int
    mul_widths: List[Tuple[int, int]]
    invariant: Set[str] = field(default_factory=set)   # of the inputs only: any clock
    swap: Dict[str, bool] = field(default_factory=dict)  # mul op: its factors swapped (b, a)


def _mac_parts(node: ast.expr) -> Tuple[Optional[ast.expr], ast.expr, ast.expr, bool]:
    """A multiplier's operation: c + a * b, c - a * b or a * b -> (c, a, b, minus)."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return None, node.left, node.right, False
    prod = node.right
    return node.left, prod.left, prod.right, isinstance(node.op, ast.Sub)


def _schedule(ops: List[Op], widths, rng, attr, multipliers: int, period: float) -> Schedule:
    by = {o.name: o for o in ops}
    users: Dict[str, List[str]] = {o.name: [] for o in ops}
    for o in ops:
        for d in o.deps:
            users[d].append(o.name)

    def width(ref: ast.expr) -> int:
        return _fmt_of(_interval(ref, rng, attr))["width"]

    def sides(o: Op) -> Tuple[Optional[ast.expr], ast.expr, ast.expr]:
        """(c, a, b) with the wider factor on side a (as the multiplier takes them)."""
        c, a, b, _ = _mac_parts(o.node)
        return (c, b, a) if width(b) > width(a) else (c, a, b)

    n_mul = sum(o.kind == "mul" for o in ops)
    mux = SEL_DECODE + MUX_LEVEL * max(1, math.ceil(math.log2(max(2, n_mul / max(1, multipliers)))))
    delay: Dict[str, float] = {}
    for o in ops:
        if o.kind == "mul":             # operands into registers, then multiply (and add)
            c, a, b = sides(o)
            delay[o.name] = period + _mult_delay(width(a), width(b)) + (
                _adder(widths[o.name]["width"]) if c is not None else 0.0)
        elif o.kind == "rom":
            delay[o.name] = period
        else:
            delay[o.name] = _free_delay(o, widths[o.name]["width"])

    prio: Dict[str, float] = {}
    for o in reversed(ops):
        prio[o.name] = delay[o.name] + max((prio[u] for u in users[o.name]), default=0.0)
    order = {o.name: i for i, o in enumerate(ops)}

    tables: Dict[Tuple[int, ...], int] = {}
    for o in ops:
        if o.kind == "rom" and o.table not in tables:
            tables[o.table] = len(tables)

    # Multiplier sizes: unit i can take the smallest (N - i N / K) products, so
    # a few wide products share one wide multiplier instead of widening them all
    def shape(o: Op) -> Tuple[int, int]:
        _, a, b = sides(o)
        return width(a), width(b)
    shapes = sorted((shape(o) for o in ops if o.kind == "mul"), key=lambda t: -t[0] * t[1])
    k = min(multipliers, len(shapes))
    dims = [(max(x for x, _ in shapes[i * len(shapes) // k:]),
             max(y for _, y in shapes[i * len(shapes) // k:])) for i in range(k)]
    # The operands each multiplier has taken so far: a new one widens its multiplexer
    seen: List[Dict[str, Set[str]]] = [{"a": set(), "b": set(), "c": set()} for _ in range(k)]
    swap: Dict[str, bool] = {}

    phase: Dict[str, int] = {}
    ready: Dict[str, int] = {}
    finish: Dict[str, float] = {}
    unit: Dict[str, Tuple[str, int]] = {}
    busy: Set[Tuple[str, int, int]] = set()
    # Logic of the inputs (and state) only: they hold still for the whole sample,
    # so it is there in every clock, with no register
    inv: Set[str] = set()
    for o in ops:
        if o.kind == "free" and all(d in inv for d in o.deps):
            t = max([T_REG] + [finish[d] for d in o.deps]) + delay[o.name]
            if t <= period / 2:
                inv.add(o.name)
                phase[o.name] = ready[o.name] = 0
                finish[o.name] = t
    left = [o.name for o in ops if o.name not in inv]
    p = 0

    def at_p(d: str) -> float:
        return finish[d] if (d in inv or ready[d] == p) else T_REG

    def arrival(ref: Optional[ast.expr]) -> float:
        if ref is None:
            return T_REG
        return max([T_REG] + [at_p(d) for d in _names(ref) if d in by])

    while left:
        progress = True
        while progress:
            progress = False
            cands = [n for n in left if all(d in ready and ready[d] <= p for d in by[n].deps)]
            cands.sort(key=lambda n: (-prio[n], order[n]))
            for n in cands:
                o = by[n]
                start = max([T_REG] + [at_p(d) for d in o.deps])
                end = start + delay[n]
                earliest = max([T_REG] + [finish[d] for d in o.deps if d in inv])
                if o.kind == "mul":
                    # The operands are chosen and registered in this clock; the
                    # product (plus c, read then) comes in the next one
                    c, a, b = sides(o)
                    pick_t = max(arrival(a), arrival(b)) + mux
                    if pick_t > period and max(arrival(a), arrival(b)) > earliest:
                        continue
                    prod = T_REG + _mult_delay(*shape(o))
                    c_t = T_REG if c is None else max([T_REG] + [
                        finish[d] for d in _names(c) if d in inv]) + mux
                    end = prod if c is None else max(prod, c_t) + _adder(widths[n]["width"])
                # Too late in the clock: wait for the next one (with the operands in
                # registers), unless it adds no delay or can't start any earlier
                elif end > period and start > earliest and delay[n] > 0 and o.kind != "rom":
                    continue
                if o.kind == "free":
                    phase[n], ready[n], finish[n] = p, p, end
                elif o.kind == "mul":
                    big, small = shape(o)
                    fit = [u for u in range(k) if ("mul", u, p) not in busy
                           and big <= dims[u][0] and small <= dims[u][1]]
                    if not fit:
                        continue
                    keys = {"a": ast.dump(a), "b": ast.dump(b),
                            "c": ast.dump(c) if c is not None else "0"}
                    # The multiplier that has had these operands before, if free
                    u = min(fit, key=lambda u: (sum(keys[sd] not in seen[u][sd] for sd in keys),
                                                dims[u][0] * dims[u][1], u))
                    for sd in keys:
                        seen[u][sd].add(keys[sd])
                    swap[n] = a is not _mac_parts(o.node)[1]
                    busy.add(("mul", u, p))
                    unit[n] = ("mul", u)
                    phase[n], ready[n], finish[n] = p, p + 1, end
                else:                                        # a ROM read
                    t = tables[o.table]
                    if ("rom", t, p) in busy:
                        continue
                    if start + SEL_DECODE + MUX_LEVEL * 2 + ROM_SETUP > period and start > earliest:
                        continue
                    busy.add(("rom", t, p))
                    unit[n] = ("rom", t)
                    phase[n], ready[n], finish[n] = p, p + 1, ROM_TCO
                left.remove(n)
                progress = True
        p += 1
        if p > 4 * len(ops) + 8:
            raise RuntimeError("share: scheduling made no progress")
    last = max([ready[n] for n in ready] + [0])
    return Schedule(phase, ready, unit, last + 1, dims, inv, swap)


def _simplify(ops: List[Op], outputs) -> None:
    """
    Peepholes that make operations narrower, in place:
      x * 1, x + 0, x << 0 ... -> x
      (A + C) >> k, the sum read only there, with C a constant -> the constant
      added after the shift, as far as A's known low zero bits (or C's) allow
      (floor shifts: (A + C) >> k == ((A >> m) + (C >> m)) >> (k - m) if A is
      a multiple of 2**m, m <= k; and == (A >> k) + (C >> k) if C is one of 2**k)
    """
    by = {o.name: o for o in ops}
    uses = _uses(ops, outputs)

    def zeros(ref: ast.expr) -> int:
        """Known trailing zero bits of ref."""
        if isinstance(ref, ast.Constant) and isinstance(ref.value, int):
            v = ref.value
            return 1 << 20 if v == 0 else (v & -v).bit_length() - 1
        if not (isinstance(ref, ast.Name) and ref.id in by):
            return 0
        o = by[ref.id]
        n = o.node
        if o.declared is not None or isinstance(n, (ast.Name, ast.Constant)):
            return zeros(n) if not isinstance(n, ast.Name) or n.id in by else 0
        if isinstance(n, ast.BinOp):
            if isinstance(n.op, ast.LShift) and isinstance(n.right, ast.Constant):
                return zeros(n.left) + n.right.value
            if isinstance(n.op, ast.Mult):
                return zeros(n.left) + zeros(n.right)
            if isinstance(n.op, (ast.Add, ast.Sub)):
                return min(zeros(n.left), zeros(n.right))
        return 0

    def const(ref: ast.expr) -> Optional[int]:
        return int(_const_value(ref)) if _is_const(ref) else None

    for o in ops:
        n = o.node
        if o.kind != "free" or o.declared is not None or not isinstance(n, ast.BinOp):
            continue
        l, r = const(n.left), const(n.right)
        if (isinstance(n.op, ast.Mult) and (r == 1 or l == 1)) or \
                (isinstance(n.op, (ast.Add, ast.Sub, ast.LShift, ast.RShift, ast.BitOr,
                                   ast.BitXor)) and r == 0) or (isinstance(n.op, ast.Add) and l == 0):
            o.node = n.left if (r is not None and not isinstance(n.op, ast.Mult)) or r == 1 \
                else n.right
            continue
        if not (isinstance(n.op, ast.RShift) and r is not None and r > 0
                and isinstance(n.left, ast.Name) and n.left.id in by and uses[n.left.id] == 1):
            continue
        inner = by[n.left.id]
        m = inner.node
        if not (inner.kind == "free" and inner.declared is None and isinstance(m, ast.BinOp)
                and isinstance(m.op, (ast.Add, ast.Sub)) and const(m.right) is not None):
            continue
        c = const(m.right) * (1 if isinstance(m.op, ast.Add) else -1)
        k = r
        if c % (1 << k) == 0:
            a_shift, c_shift, rest = k, k, 0
        else:
            mz = min(zeros(m.left), k)
            if mz == 0:
                continue
            a_shift, c_shift, rest = mz, mz, k - mz
        # inner becomes A >> m, and this op (A >> m) + (C >> m), shifted by the rest
        inner.node = ast.BinOp(left=m.left, op=ast.RShift(), right=ast.Constant(a_shift))
        add = ast.BinOp(left=ast.Name(id=inner.name, ctx=ast.Load()), op=ast.Add(),
                        right=ast.Constant(c >> c_shift))
        o.node = add if rest == 0 else ast.BinOp(left=add, op=ast.RShift(), right=ast.Constant(rest))
        if c >> c_shift == 0:
            o.node = (ast.Name(id=inner.name, ctx=ast.Load()) if rest == 0 else
                      ast.BinOp(left=ast.Name(id=inner.name, ctx=ast.Load()), op=ast.RShift(),
                                right=ast.Constant(rest)))


def _uses(ops: List[Op], outputs) -> Dict[str, int]:
    """How many times each operation's value is read (by operations and outputs)."""
    count: Dict[str, int] = {o.name: 0 for o in ops}
    refs = [o.node for o in ops] + [r for _, v in outputs for r in (v if isinstance(v, list) else [v])]
    for node in refs:
        for n in _names(node):
            if n in count:
                count[n] += 1
    return count


def _fuse_macs(ops: List[Op], outputs, fresh, chains: int = 1) -> List[Op]:
    """
    Sums with products in them as multiply-accumulates: a sum of terms (an
    Add / Sub / negation tree whose inner sums are read only there) adds its
    other terms first and then each product read only there on a multiplier,
    c + a * b, so the additions happen in the multipliers' adders. (Integer
    addition is exact, so the order doesn't change the value.)
    """
    by = {o.name: o for o in ops}
    uses = _uses(ops, outputs)

    def is_sum(o: Op) -> bool:
        return o.kind == "free" and o.declared is None and (
            (isinstance(o.node, ast.BinOp) and isinstance(o.node.op, (ast.Add, ast.Sub)))
            or (isinstance(o.node, ast.UnaryOp) and isinstance(o.node.op, ast.USub)))

    def is_product(ref: ast.expr) -> bool:
        if not (isinstance(ref, ast.Name) and ref.id in by and uses[ref.id] == 1):
            return False
        o = by[ref.id]
        return o.kind == "mul" and isinstance(o.node, ast.BinOp) and isinstance(o.node.op, ast.Mult)

    absorbed: Set[str] = set()
    plans: Dict[str, List[Tuple[int, ast.expr]]] = {}
    for o in reversed(ops):
        if o.name in absorbed or not is_sum(o):
            continue
        terms: List[Tuple[int, ast.expr]] = []
        inner: List[str] = []

        def flatten(ref: ast.expr, sign: int, root: bool = False) -> None:
            if isinstance(ref, ast.Name) and ref.id in by and ref.id not in absorbed and (
                    root or (uses[ref.id] == 1 and is_sum(by[ref.id]))):
                node = by[ref.id].node
                if not root:
                    inner.append(ref.id)
                if isinstance(node, ast.UnaryOp):
                    flatten(node.operand, -sign)
                else:
                    flatten(node.left, sign)
                    flatten(node.right, sign if isinstance(node.op, ast.Add) else -sign)
                return
            terms.append((sign, ref))
        flatten(ast.Name(id=o.name, ctx=ast.Load()), 1, root=True)
        if not any(is_product(r) for _, r in terms):
            continue
        plans[o.name] = terms
        absorbed.update(inner)
        absorbed.update(r.id for _, r in terms if is_product(r))

    out: List[Op] = []

    def make(kind: str, node: ast.expr, name: Optional[str] = None) -> ast.Name:
        name = name or fresh()
        known = {x.name for x in out}
        out.append(Op(name, kind, node, sorted({n for n in _names(node) if n in known})))
        return ast.Name(id=name, ctx=ast.Load())

    for o in ops:
        if o.name in absorbed:
            continue
        if o.name not in plans:
            o.deps = sorted({n for n in _names(o.node) if n in {x.name for x in out}})
            out.append(o)
            continue
        terms = plans[o.name]
        prods = [(sg, by[r.id].node) for sg, r in terms if is_product(r)]
        const = sum(sg * int(_const_value(r)) for sg, r in terms if _is_const(r))
        acc: Optional[ast.expr] = None
        for sg, r in terms:
            if is_product(r) or _is_const(r):
                continue
            if acc is None:
                acc = r if sg > 0 else make("free", ast.UnaryOp(op=ast.USub(), operand=r))
            else:
                acc = make("free", ast.BinOp(left=acc, op=ast.Add() if sg > 0 else ast.Sub(),
                                             right=r))
        if const:
            acc = ast.Constant(const) if acc is None else make(
                "free", ast.BinOp(left=acc, op=ast.Add(), right=ast.Constant(const)))
        # Up to `chains` accumulations side by side (one per multiplier), then summed
        g = max(1, min(chains, len(prods) // 4))      # (at least 4 products each)
        parts: List[Optional[ast.expr]] = []
        for j in range(g):
            part = acc if j == 0 else None
            mine = prods[j::g]
            for i, (sg, prod) in enumerate(mine):
                prod = copy.deepcopy(prod)
                if part is None and sg < 0:
                    part = ast.Constant(0)
                node = prod if part is None else ast.BinOp(
                    left=part, op=ast.Add() if sg > 0 else ast.Sub(), right=prod)
                last = g == 1 and i == len(mine) - 1
                part = make("mul", node, o.name if last else None)
            parts.append(part)
        total = parts[0]
        for j, part in enumerate(parts[1:]):
            total = make("free", ast.BinOp(left=total, op=ast.Add(), right=part),
                         o.name if j == len(parts) - 2 else None)
    return out


# ==================================================================== the shared design

def _fresh(name: str, used: Set[str]) -> str:
    while name in used:
        name += "_"
    used.add(name)
    return name


def _select(sources: List[ast.expr], at_phase: List[int], phase: str,
            stmts: List[ast.stmt], target: str, used: Set[str]) -> None:
    """target = sources[at_phase[fphase]]: a balanced multiplexer on the
    phase (through a small table of which source each phase takes)."""
    if len(sources) == 1:
        stmts.append(ast.Assign(targets=[ast.Name(id=target, ctx=ast.Store())],
                                value=sources[0]))
        return
    sel = _fresh(f"{target}_sel", used)
    stmts.append(ast.Assign(targets=[ast.Name(id=sel, ctx=ast.Store())], value=ast.Subscript(
        value=ast.Tuple(elts=[ast.Constant(k) for k in at_phase], ctx=ast.Load()),
        slice=ast.Name(id=phase, ctx=ast.Load()), ctx=ast.Load())))
    stmts.append(ast.Assign(targets=[ast.Name(id=target, ctx=ast.Store())], value=ast.Subscript(
        value=ast.Tuple(elts=sources, ctx=ast.Load()), slice=ast.Name(id=sel, ctx=ast.Load()),
        ctx=ast.Load())))


def _toposort(stmts: List[ast.stmt]) -> List[ast.stmt]:
    """Assignments in an order where each name is assigned before it is read."""
    target = {s.targets[0].id: i for i, s in enumerate(stmts) if isinstance(s, ast.Assign)}
    deps = [{target[n] for n in _names(s) if n in target and target[n] != i}
            for i, s in enumerate(stmts)]
    out: List[ast.stmt] = []
    state = [0] * len(stmts)

    def visit(i: int) -> None:
        if state[i] == 2:
            return
        if state[i] == 1:
            raise RuntimeError("share: combinational loop in the shared design")
        state[i] = 1
        for d in sorted(deps[i]):
            visit(d)
        state[i] = 2
        out.append(stmts[i])

    for i in range(len(stmts)):
        visit(i)
    return out


def _live(ops: List[Op], outputs) -> List[Op]:
    """The operations an output depends on, in order."""
    by = {o.name: o for o in ops}
    need: Set[str] = set()
    stack = []
    for _, v in outputs:
        for ref in (v if isinstance(v, list) else [v]):
            stack += [n for n in _names(ref) if n in by]
    while stack:
        n = stack.pop()
        if n not in need:
            need.add(n)
            stack += by[n].deps
    return [o for o in ops if o.name in need]


def share(tree: ast.Module, attr: Dict[str, Dict[str, Any]], multipliers: int,
          period: float = DEFAULT_PERIOD_NS) -> Tuple[ast.Module, Dict[str, Any]]:
    """
    Schedule the module's first function (lowered, straight-line) on
    `multipliers` shared multipliers and one block-RAM port per table.
    Returns (tree, info) in fold.py's form ('phases', 'phase', 'regs',
    'params', 'attr'), plus 'roms': {data register: {'index', 'entries',
    'width', 'signed'}}; info['phases'] == 1 means nothing to share.
    """
    multipliers = int(multipliers)
    if multipliers < 1:
        raise ValueError("multipliers must be at least 1")
    fd = tree.body[0]
    if not isinstance(fd, ast.FunctionDef):
        raise NotShareable("Expected a function")
    from .visitor import _Desugar
    fd = _Desugar().visit(copy.deepcopy(fd))
    conv = _ToOps(fd, attr)
    conv.body(fd.body)
    if not conv.outputs:
        raise NotShareable("The function must return something")
    ops = _live(list(conv.ops.values()), conv.outputs)
    _simplify(ops, conv.outputs)
    ops = _live(_fuse_macs(ops, conv.outputs, lambda: conv.fresh("s"), multipliers), conv.outputs)
    rng = _ranges(ops, attr)
    widths = {n: _fmt_of(r) for n, r in rng.items()}
    by = {o.name: o for o in ops}
    sched = _schedule(ops, widths, rng, attr, multipliers, float(period))
    P = sched.phases
    info: Dict[str, Any] = {"phases": P, "loop": "(shared)", "mul_widths": sched.mul_widths}
    if P <= 1:
        return tree, info

    used = set(conv.used)
    phase = _fresh("fphase", used)
    last = P - 1

    # Values read after the clock that makes them (and outputs made before the
    # last clock) are kept in registers. Values from the same multiplier or ROM
    # whose lives don't overlap share one (its input is the same wire, so the
    # sharing costs no multiplexer).
    users: Dict[str, List[str]] = {o.name: [] for o in ops}
    for o in ops:
        for d in o.deps:
            users[d].append(o.name)
    out_refs = {n for _, v in conv.outputs for ref in (v if isinstance(v, list) else [v])
                for n in _names(ref) if n in by}
    life: Dict[str, Tuple[int, int]] = {}
    def read_at(u: str, v: str) -> int:
        """The clock in which operation u reads v (a multiplier reads c a clock after a, b)."""
        o = by[u]
        if o.kind == "mul":
            c = _mac_parts(o.node)[0]
            if c is not None and v in _names(c):
                return sched.phase[u] + 1
        return sched.phase[u]

    for o in ops:
        if o.name in sched.invariant:
            continue
        r = sched.ready[o.name]
        reads = [read_at(u, o.name) for u in users[o.name] if read_at(u, o.name) > r]
        if o.name in out_refs and r < last:
            reads.append(last)
        if reads:
            life[o.name] = (r, max(reads))
    slots: List[Dict[str, Any]] = []
    slot_of: Dict[str, int] = {}
    for n in sorted(life, key=lambda n: (life[n][0], life[n][1], n)):
        w, e = life[n]
        sig = sched.unit.get(n, ("op", n))
        lo, hi = rng[n]

        def cost(k: int) -> Tuple[int, int, int]:
            sl = slots[k]
            grow = (_fmt_of((min(lo, sl["lo"]), max(hi, sl["hi"])))["width"]
                    - _fmt_of((sl["lo"], sl["hi"]))["width"])
            return (0 if sig in sl["sigs"] else 1), grow, k
        free = [k for k, sl in enumerate(slots) if sl["end"] <= w and sig in sl["sigs"]]
        if free:
            k = min(free, key=cost)
        else:
            k = len(slots)
            slots.append({"end": e, "lo": lo, "hi": hi, "sigs": set(), "members": []})
        sl = slots[k]
        sl["end"], sl["lo"], sl["hi"] = e, min(lo, sl["lo"]), max(hi, sl["hi"])
        sl["sigs"].add(sig)
        sl["members"].append(n)
        slot_of[n] = k
    slot_reg = [(_fresh(f"r{k}", used), _fresh(f"rn{k}", used)) for k in range(len(slots))]

    def at(ref: ast.expr, p: int) -> ast.expr:
        """ref as an operation in clock p reads it."""
        ref = copy.deepcopy(ref)

        class _R(ast.NodeTransformer):
            def visit_Name(self, node: ast.Name):
                n = node.id
                if n not in by:
                    return node
                if sched.ready[n] == p or n in sched.invariant:
                    return node
                if sched.ready[n] > p or n not in slot_of:
                    raise RuntimeError(f"share: {n} is not ready in clock {p}")
                return ast.Name(id=slot_reg[slot_of[n]][0], ctx=ast.Load())
        return _R().visit(ref)

    stmts: List[ast.stmt] = []
    new_attr: Dict[str, Dict[str, Any]] = {phase: {"signed": 0, "width": max(1, (P - 1).bit_length()),
                                                   "type": "wire"}}

    def assign(name: str, value: ast.expr) -> None:
        stmts.append(ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=value))

    # Shared multipliers: operands chosen by the phase
    units: Dict[Tuple[str, int], List[str]] = {}
    for n, u in sched.unit.items():
        units.setdefault(u, []).append(n)
    roms: Dict[str, Dict[str, Any]] = {}
    source: Dict[str, str] = {}               # mul / rom op -> the unit output it reads
    unit_regs: Dict[str, str] = {}            # a multiplier's operand register -> its input
    mux_inputs = [0]                          # multiplier input bits (for the log)
    params_extra: List[str] = []
    returns_extra: List[str] = []
    for u, members in sorted(units.items()):
        members.sort(key=lambda n: sched.phase[n])
        if u[0] == "mul":
            # Pipelined: a and b are chosen and registered in the clock an operation
            # issues, c is chosen and added in the next, when the product is there
            out = _fresh(f"mul{u[1]}", used)
            parts = {}
            for n in members:
                c, x, y, minus = _mac_parts(by[n].node)
                if sched.swap.get(n):
                    x, y = y, x
                parts[n] = {"a": x, "b": y, "c": c if c is not None else ast.Constant(0),
                            "minus": minus}
            has_c = any(_mac_parts(by[n].node)[0] is not None for n in members)
            for side in ("a", "b", "c") if has_c else ("a", "b"):
                srcs: List[ast.expr] = []
                keys: List[str] = []
                pick = [0] * P
                when = [sched.phase[n] + (side == "c") for n in members]
                for n, p in zip(members, when):
                    ref = at(parts[n][side], p)
                    key = ast.dump(ref)
                    if key not in keys:
                        keys.append(key)
                        srcs.append(ref)
                    pick[p] = keys.index(key)
                _fill(pick, when)
                _select(srcs, pick, phase, stmts, f"{out}_{side}", used)
                fmt = dict(_fmt_of(_hull(*[_interval(parts[n][side], rng, attr)
                                           for n in members])), type="wire")
                mux_inputs[0] += (len(srcs) - 1) * fmt["width"]
                new_attr[f"{out}_{side}"] = fmt
                if side != "c":
                    reg = _fresh(f"{out}_r{side}", used)
                    unit_regs[reg] = f"{out}_{side}"
                    new_attr[reg] = dict(fmt)
            ra, rb = (ast.Name(id=r, ctx=ast.Load()) for r in list(unit_regs)[-2:])
            prod = ast.BinOp(left=ra, op=ast.Mult(), right=rb)
            if has_c:
                cn = ast.Name(id=f"{out}_c", ctx=ast.Load())
                minus = [parts[n]["minus"] for n in members]
                if all(minus) or not any(minus):
                    assign(out, ast.BinOp(left=cn, op=ast.Sub() if minus[0] else ast.Add(),
                                          right=prod))
                else:                                    # add in some clocks, subtract in others
                    pick = [0] * P
                    when = [sched.phase[n] + 1 for n in members]
                    for n, p in zip(members, when):
                        pick[p] = int(parts[n]["minus"])
                    _fill(pick, when)
                    assign(f"{out}_p", prod)
                    pn = ast.Name(id=f"{out}_p", ctx=ast.Load())
                    _select([ast.BinOp(left=cn, op=ast.Add(), right=pn),
                             ast.BinOp(left=copy.deepcopy(cn), op=ast.Sub(), right=copy.deepcopy(pn))],
                            pick, phase, stmts, out, used)
            else:
                assign(out, prod)
            for n in members:
                source[n] = out
        else:
            o0 = by[members[0]]
            q = _fresh(f"rom{u[1]}_q", used)
            idx = _fresh(f"rom{u[1]}_i", used)
            srcs, keys, pick = [], [], [0] * P
            for n in members:
                ref = at(by[n].node.slice, sched.phase[n])
                key = ast.dump(ref)
                if key not in keys:
                    keys.append(key)
                    srcs.append(ref)
                pick[sched.phase[n]] = keys.index(key)
            _fill(pick, [sched.phase[n] for n in members])
            _select(srcs, pick, phase, stmts, idx, used)
            entries = list(o0.table)
            width, signed = widths[o0.name]["width"], widths[o0.name]["signed"]
            roms[q] = {"index": idx, "entries": entries, "width": width, "signed": int(signed),
                       "address_bits": max(1, (len(entries) - 1).bit_length())}
            new_attr[q] = {"signed": int(signed), "width": width, "type": "wire"}
            params_extra.append(q)
            returns_extra.append(idx)
            for n in members:
                source[n] = q

    # Every operation in its clock
    for o in ops:
        if o.kind == "free":
            value = at(o.node, sched.phase[o.name])
        else:
            value = ast.Name(id=source[o.name], ctx=ast.Load())
        assign(o.name, value)
        new_attr[o.name] = dict(widths[o.name], type="wire")

    # The registers: each loads its values in the clocks that make them
    for k, sl in enumerate(slots):
        r, rn = slot_reg[k]
        srcs: List[ast.expr] = []
        keys: List[str] = []
        pick = [-1] * P
        for n in sl["members"]:
            src = source.get(n, n)          # a product / table entry: the unit's output
            if src not in keys:
                keys.append(src)
                srcs.append(ast.Name(id=src, ctx=ast.Load()))
            pick[sched.ready[n]] = keys.index(src)
        hold = len(srcs)
        srcs.append(ast.Name(id=r, ctx=ast.Load()))
        _select(srcs, [hold if x < 0 else x for x in pick], phase, stmts, rn, used)
        fmt = dict(_fmt_of((sl["lo"], sl["hi"])), type="wire")
        new_attr[r], new_attr[rn] = fmt, dict(fmt)

    stmts = _toposort(stmts)
    out_stmts = _output_stmts(conv.outputs, lambda ref: at(ref, last))
    ret = out_stmts[-1]
    names = ret.value.elts if isinstance(ret.value, ast.Tuple) else [ret.value]
    regs = dict({r: rn for r, rn in slot_reg}, **unit_regs)
    ret.value = ast.Tuple(elts=list(names) + [ast.Name(id=rn, ctx=ast.Load()) for rn in regs.values()]
                          + [ast.Name(id=i, ctx=ast.Load()) for i in returns_extra], ctx=ast.Load())

    new_fd = copy.deepcopy(fd)
    new_fd.body = stmts + out_stmts
    new_fd.args.args = (list(new_fd.args.args) + [ast.arg(arg=phase)]
                        + [ast.arg(arg=r) for r in regs] + [ast.arg(arg=q) for q in params_extra])
    new_tree = ast.fix_missing_locations(ast.Module(body=[new_fd], type_ignores=[]))
    info.update({
        "phase": phase,
        "regs": regs,
        "roms": roms,
        "params": {phase} | set(regs) | set(params_extra),
        "attr": new_attr,
        "register_bits": sum(new_attr[r]["width"] for r in regs),
        "registers": {slot_reg[k][0]: list(sl["members"]) for k, sl in enumerate(slots)},
        "multipliers_used": len([u for u in units if u[0] == "mul"]),
        "mux_input_bits": mux_inputs[0],
    })
    return new_tree, info


def _fill(pick: List[int], used_phases: List[int]) -> None:
    """Phases where a unit is idle repeat the choice before them (fewer switches)."""
    busy = set(used_phases)
    first = min(used_phases)
    prev = pick[first]
    for p in range(len(pick)):
        if p in busy:
            prev = pick[p]
        else:
            pick[p] = prev


def simulate(info: Dict[str, Any], model, args: Tuple[Any, ...], state: Tuple[Any, ...] = ()):
    """
    Run a shared design's bit-exact model clock by clock (for tests): args are
    the function's own arguments; returns its outputs after the last clock.
    """
    regs = {r: 0 for r in info["regs"]}
    qs = {q: 0 for q in info["roms"]}
    n_regs, n_roms = len(regs), len(info["roms"])
    out = None
    for p in range(info["phases"]):
        res = model(*args, p, *regs.values(), *qs.values())
        res = list(res) if isinstance(res, tuple) else [res]
        k = len(res) - n_regs - n_roms
        out, nxt, idx = res[:k], res[k:k + n_regs], res[k + n_regs:]
        regs = dict(zip(regs, nxt))
        qs = {q: info["roms"][q]["entries"][int(i)] for q, i in zip(qs, idx)}
    return tuple(out) if len(out) != 1 else out[0]
