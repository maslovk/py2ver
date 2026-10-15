# fixed.py
"""
Fixed point from an ordinary float model.

The function stays plain Python (floats and ints). attr gives a variable a
fixed-point format: 'frac' fraction bits within its 'width' (and 'signed'),
plus optionally 'round' (round half up instead of floor) and 'saturate'
(clamp instead of wrap) for the values assigned to it. In hardware a
fixed-point value is the integer raw = value * 2**frac.

What the hardware computes (and the bit-exact Python model with it):
  - an assignment to a variable with a format quantizes the value: floor
    (or round) to its fraction bits, wrap (or saturate) to its width.
    Parameters keep their format; a state's next value takes the state's
  - everything else is exact: + and - align the binary points, * adds the
    fraction bits, comparisons, min, max, abs and conditional expressions
    align, and a variable without a format holds the widest value assigned
  - / by a constant: a power of two moves the binary point, any other
    constant multiplies by its reciprocal (a constant, see below)
  - a float constant is exact if it fits in const_bits significant bits,
    else it is rounded to const_bits significant bits
  - int(x) truncates toward zero, math.floor / math.ceil, float(x) is x

lower_fixed() rewrites the function into integer code on raw values, which
the rest of py2ver translates as usual; build_model() runs that same code in
Python, wrapping every declared signal like the hardware does.
"""

from __future__ import annotations

import ast
import copy
import math
from fractions import Fraction
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

DEFAULT_CONST_BITS = 18
# Beyond this many fraction bits a variable's format is taken to grow forever
MAX_FRAC = 256


def int_range(width: int, signed: bool) -> Tuple[int, int]:
    return (-(1 << (width - 1)), (1 << (width - 1)) - 1) if signed else (0, (1 << width) - 1)


def wrap(n: int, width: int, signed: bool) -> int:
    n &= (1 << width) - 1
    return n - (1 << width) if signed and n >> (width - 1) else n


def quantize(v: Any, frac: int, width: int, signed: bool,
             round: bool = False, saturate: bool = False) -> int:
    """Raw value of the number v with `frac` fraction bits: floor (or round half
    up), then wrap (or saturate) to width."""
    scaled = Fraction(v) * (1 << frac)
    raw = math.floor(scaled + Fraction(1, 2)) if round else math.floor(scaled)
    if saturate:
        lo, hi = int_range(width, signed)
        return max(lo, min(hi, raw))
    return wrap(raw, width, signed)


def const_raw(v: Any, bits: int) -> Tuple[int, int]:
    """A constant as (raw, frac): exact if it fits in `bits` significant bits,
    else rounded to that many."""
    x = Fraction(v)
    num, den = x.numerator, x.denominator
    if num == 0:
        return 0, 0
    if den == 1 or (den & (den - 1) == 0 and abs(num).bit_length() <= bits):
        return num, den.bit_length() - 1
    e = abs(num).bit_length() - den.bit_length()        # 2**e <= |x| < 2**(e+2)
    if abs(x) >= Fraction(2) ** (e + 1):
        e += 1
    frac = max(0, bits - 1 - e)
    raw = math.floor(x * (1 << frac) + Fraction(1, 2))
    while frac > 0 and raw % 2 == 0:
        raw //= 2
        frac -= 1
    return raw, frac


# ====================================================================
# Lowering: fixed-point code -> integer code on raw values
# ====================================================================
#
# Types: None (not known yet), INT, CPLX, ("f", frac), ("l", elem)

INT = ("i",)
CPLX = ("c",)


def FX(frac: int):
    return ("f", int(frac))


def LST(elem):
    return ("l", elem)


def _is_fx(t) -> bool:
    return t is not None and (t[0] == "f" or (t[0] == "l" and _is_fx(t[1])))


def _frac(t) -> int:
    return t[1] if t is not None and t[0] == "f" else 0


def _fmt(t) -> str:
    if t is None or t == INT:
        return "an integer"
    if t == CPLX:
        return "a complex value"
    if t[0] == "l":
        return "a list of " + _fmt(t[1]).split(" ", 1)[1] + "s"
    return f"a fixed-point value with {t[1]} fraction bits"


def _join(a, b, what: str):
    if a is None:
        return b
    if b is None:
        return a
    if a[0] == "l" or b[0] == "l":
        if a[0] != b[0]:
            raise NotImplementedError(f"'{what}' is both a list and a single value")
        return LST(_join(a[1], b[1], what))
    if CPLX in (a, b):
        if _is_fx(a) or _is_fx(b):
            raise NotImplementedError(f"'{what}' mixes complex and fixed-point values")
        return CPLX
    if a == INT:
        return b
    if b == INT:
        return a
    return FX(max(a[1], b[1]))


def attr_type(meta: Dict[str, Any]):
    t = FX(meta["frac"]) if meta.get("frac") is not None else INT
    return LST(t) if "length" in meta else t


def _number(node: ast.expr, ns: Dict[str, Any]) -> Optional[Any]:
    """The value of a constant expression (literal, -literal, global number), or None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _number(node.operand, ns)
        return None if v is None else (-v if isinstance(node.op, ast.USub) else v)
    if isinstance(node, ast.Name) and isinstance(ns.get(node.id), (int, float)) \
            and not isinstance(ns.get(node.id), bool):
        return ns[node.id]
    return None


class _Lower:
    def __init__(self, fd: ast.FunctionDef, attr: Dict[str, Dict[str, Any]],
                 ns: Dict[str, Any], const_bits: int, py_names: Dict[str, str]) -> None:
        self.ns = ns
        self.py_names = py_names      # keyword-renamed names (int_ -> int)
        self.const_bits = const_bits
        self.params = [a.arg for a in fd.args.args]
        self.locals: Set[str] = set(self.params) | {
            n.id for n in ast.walk(fd) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
        }
        self.used = {n.id for n in ast.walk(fd) if isinstance(n, ast.Name)} | set(self.params)
        # Names with a format: assignments to them quantize
        self.meta: Dict[str, Dict[str, Any]] = {
            k: m for k, m in attr.items() if isinstance(m, dict) and "width" in m
        }
        for p in self.params:
            self.meta.setdefault(p, dict(attr.get(p, {})))
        # A state's next value takes the state's format
        for name, meta in attr.items():
            nxt = meta.get("next") if isinstance(meta, dict) else None
            if nxt is None or nxt == name:
                continue
            if nxt in self.meta and attr_type(self.meta[nxt]) != attr_type(meta):
                raise ValueError(f"Next state '{nxt}' of '{name}' must have {_fmt(attr_type(meta))}")
            self.meta.setdefault(nxt, {k: v for k, v in meta.items() if k not in ("next", "init")})
        self.pinned = {k: attr_type(m) for k, m in self.meta.items()}
        self.env: Dict[str, Any] = dict(self.pinned)
        self.final = False
        self.changed = False
        self.pre: List[ast.stmt] = []
        self.new_attr: Dict[str, Dict[str, Any]] = {}
        self.n_temps = 0

    # ---------------- helpers ----------------
    def _err(self, node: ast.AST, msg: str):
        line = getattr(node, "lineno", None)
        raise NotImplementedError(f"{msg} (line {line})" if line else msg)

    def _temp(self) -> str:
        while True:
            name = f"fxq{self.n_temps}"
            self.n_temps += 1
            if name not in self.used:
                return name

    def _named(self, node: ast.expr) -> ast.expr:
        """node as something cheap to repeat: a name or constant (else a temporary)."""
        if isinstance(node, (ast.Name, ast.Constant)):
            return node
        tmp = self._temp()
        self.pre.append(ast.Assign(targets=[ast.Name(id=tmp, ctx=ast.Store())], value=node))
        return ast.Name(id=tmp, ctx=ast.Load())

    def _bind(self, name: str, t, node: ast.AST) -> None:
        if self.final or name in self.pinned or t is None:
            return
        new = _join(self.env.get(name), t, name)
        if new != self.env.get(name):
            if _frac(new if new[0] != "l" else new[1]) > MAX_FRAC:
                self._err(node, f"The fixed-point format of '{name}' grows on every loop "
                                f"iteration; give '{name}' a format in attr, e.g. "
                                f"{{'{name}': {{'signed': 1, 'width': 32, 'frac': 16}}}}")
            self.env[name] = new
            self.changed = True

    @staticmethod
    def _shl(node: ast.expr, k: int) -> ast.expr:
        if k <= 0:
            return node
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return ast.copy_location(ast.Constant(node.value << k), node)
        return ast.BinOp(left=node, op=ast.LShift(), right=ast.Constant(k))

    def _align(self, node: ast.expr, t, target, what: str, where: ast.AST) -> ast.expr:
        """node (of type t) as an exact value of a type with at least t's fraction."""
        if t is None or target is None or t == target:
            return node
        if t[0] == "l" or target[0] == "l":
            return self._fit(node, t, target, {}, what, where)
        if CPLX in (t, target):
            if _is_fx(t) or _is_fx(target):
                self._err(where, f"'{what}' mixes complex and fixed-point values")
            return node
        return self._shl(node, _frac(target) - _frac(t))

    def _fit(self, node: ast.expr, t, target, meta: Dict[str, Any], what: str,
             where: ast.AST) -> ast.expr:
        """node (of type t) assigned to a value of type target with format meta:
        floor / round to its fraction, saturate to its range if asked."""
        if t is None or target is None:
            return node
        if t[0] == "l" or target[0] == "l":
            if t[0] != target[0]:
                if self.final:
                    self._err(where, f"'{what}' holds {_fmt(target)}, not {_fmt(t)}")
                return node
            if t[1] == target[1] or not (_is_fx(t[1]) or _is_fx(target[1])):
                return node
            if isinstance(node, (ast.List, ast.Tuple)):
                node.elts = [self._fit(e, t[1], target[1], meta, what, where) for e in node.elts]
                return node
            if (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult)
                    and isinstance(node.left, (ast.List, ast.Tuple))):
                node.left = self._fit(node.left, t, target, meta, what, where)
                return node
            if self.final:
                self._err(where, f"'{what}' holds {_fmt(target)}; this list has {_fmt(t)}. "
                                 "Assign it element by element")
            return node
        if CPLX in (t, target):
            if _is_fx(t) or _is_fx(target):
                self._err(where, f"'{what}' mixes complex and fixed-point values")
            return node
        shift = _frac(t) - _frac(target)
        if shift > 0:
            if meta.get("round"):
                node = ast.BinOp(left=node, op=ast.Add(), right=ast.Constant(1 << (shift - 1)))
            node = ast.BinOp(left=node, op=ast.RShift(), right=ast.Constant(shift))
        else:
            node = self._shl(node, -shift)
        if meta.get("saturate") and "width" in meta:
            lo, hi = int_range(int(meta["width"]), bool(meta.get("signed")))
            v = self._named(node)
            lo_node = (ast.UnaryOp(op=ast.USub(), operand=ast.Constant(-lo)) if lo < 0
                       else ast.Constant(lo))
            node = ast.Call(func=ast.Name(id="max", ctx=ast.Load()), keywords=[], args=[
                lo_node,
                ast.Call(func=ast.Name(id="min", ctx=ast.Load()), keywords=[],
                         args=[ast.Constant(hi), v]),
            ])
        return node

    def _constant(self, v: Any):
        """A global float or table with floats as (node, type), or None."""
        def num(x):
            return isinstance(x, (int, float)) and not isinstance(x, bool)

        if isinstance(v, float):
            raw, frac = const_raw(v, self.const_bits)
            return ast.Constant(raw), FX(frac)
        if (isinstance(v, (tuple, list)) and v and all(num(x) for x in v)
                and any(isinstance(x, float) for x in v)):
            vals = [const_raw(x, self.const_bits) for x in v]
            frac = max(f for _, f in vals)
            table = ast.Tuple(elts=[ast.Constant(r << (frac - f)) for r, f in vals], ctx=ast.Load())
            return table, LST(FX(frac))
        return None

    # ---------------- expressions ----------------
    def ex(self, n: ast.expr):
        """Lower an expression: (new node, type)."""
        if isinstance(n, ast.Constant):
            v = n.value
            if isinstance(v, float):
                raw, frac = const_raw(v, self.const_bits)
                return ast.copy_location(ast.Constant(raw), n), FX(frac)
            return n, (CPLX if isinstance(v, complex) else INT)

        if isinstance(n, ast.Name):
            if n.id in self.locals:
                return n, self.env.get(n.id)
            const = self._constant(self.ns.get(n.id)) if n.id in self.ns else None
            if const is not None:
                return ast.copy_location(const[0], n), const[1]
            return n, INT

        if isinstance(n, ast.BinOp):
            return self._binop(n)

        if isinstance(n, ast.UnaryOp):
            v, t = self.ex(n.operand)
            n.operand = v
            if isinstance(n.op, ast.Not):
                return n, INT
            if isinstance(n.op, ast.Invert) and _is_fx(t):
                self._err(n, "~ is not defined for fixed-point values")
            return n, t

        if isinstance(n, ast.Compare):
            parts = [self.ex(x) for x in [n.left] + n.comparators]
            types = [t for _, t in parts]
            if any(t is None for t in types):
                return n, INT
            if any(_is_fx(t) for t in types):
                if any(t[0] == "l" or t == CPLX for t in types):
                    self._err(n, "Fixed-point values compare only with numbers")
                f = max(_frac(t) for t in types)
                parts = [(self._align(v, t, FX(f), "comparison", n), t) for v, t in parts]
            n.left = parts[0][0]
            n.comparators = [v for v, _ in parts[1:]]
            return n, INT

        if isinstance(n, ast.BoolOp):
            n.values = [self.ex(v)[0] for v in n.values]
            return n, INT

        if isinstance(n, ast.IfExp):
            n.test = self.ex(n.test)[0]
            (b, tb), (o, to) = self.ex(n.body), self.ex(n.orelse)
            t = _join(tb, to, "conditional expression")
            n.body = self._align(b, tb, t, "conditional expression", n)
            n.orelse = self._align(o, to, t, "conditional expression", n)
            return n, t

        if isinstance(n, ast.Call):
            return self._call(n)

        if isinstance(n, ast.Subscript):
            v, tv = self.ex(n.value)
            n.value = v
            if isinstance(n.slice, ast.Slice):
                for part in ("lower", "upper", "step"):
                    if getattr(n.slice, part) is not None:
                        setattr(n.slice, part, self.ex(getattr(n.slice, part))[0])
                return n, tv
            i, ti = self.ex(n.slice)
            if _is_fx(ti):
                self._err(n, "A fixed-point value can't be an index; use int(...)")
            n.slice = i
            if tv is None:
                return n, None
            return n, (tv[1] if tv[0] == "l" else INT)

        if isinstance(n, (ast.List, ast.Tuple)):
            parts = [self.ex(e) for e in n.elts]
            t = None
            for _, te in parts:
                t = _join(t, te, "list")
            n.elts = [self._align(v, te, t, "list", n) for v, te in parts]
            return n, LST(t)

        if isinstance(n, ast.Attribute):
            v, t = self.ex(n.value)
            n.value = v
            if _is_fx(t):
                self._err(n, f"Fixed-point values have no '.{n.attr}'")
            return n, INT

        return n, INT

    def _binop(self, n: ast.BinOp):
        divisor = _number(n.right, self.ns) if isinstance(n.op, ast.Div) else None
        (l, tl), (r, tr) = self.ex(n.left), self.ex(n.right)
        n.left, n.right = l, r
        if tl is None or tr is None:
            return n, None
        if tl[0] == "l" or tr[0] == "l":
            if isinstance(n.op, ast.Mult):                      # [v] * n
                return n, (tl if tl[0] == "l" else tr)
            if isinstance(n.op, ast.Add) and tl[0] == tr[0] == "l":
                t = _join(tl, tr, "list")
                n.left = self._align(l, tl, t, "list", n)
                n.right = self._align(r, tr, t, "list", n)
                return n, t
            return n, INT
        if CPLX in (tl, tr):
            if _is_fx(tl) or _is_fx(tr):
                self._err(n, "Complex values can't be mixed with fixed-point values")
            return n, CPLX
        if not (_is_fx(tl) or _is_fx(tr)):
            return n, INT
        fl, fr = _frac(tl), _frac(tr)
        op = n.op
        if isinstance(op, (ast.Add, ast.Sub)):
            f = max(fl, fr)
            n.left = self._align(l, tl, FX(f), "sum", n)
            n.right = self._align(r, tr, FX(f), "sum", n)
            return n, FX(f)
        if isinstance(op, ast.Mult):
            return n, FX(fl + fr)
        if isinstance(op, ast.Div):
            return self._divide(n, l, fl, divisor)
        if isinstance(op, (ast.LShift, ast.RShift)):
            if _is_fx(tr) or not (isinstance(r, ast.Constant) and isinstance(r.value, int)
                                  and r.value >= 0):
                self._err(n, "A fixed-point value shifts only by a constant int")
            k = r.value
            if isinstance(op, ast.RShift):                      # exact: k more fraction bits
                return l, FX(fl + k)
            if k <= fl:
                return l, FX(fl - k)
            n.right = ast.Constant(k - fl)
            return n, FX(0)
        self._err(n, f"'{_OPS.get(type(op), type(op).__name__)}' is not defined for fixed-point "
                     "values")

    def _divide(self, n: ast.BinOp, l: ast.expr, fl: int, d: Any):
        """x / constant: move the binary point for a power of two, else multiply
        by the reciprocal as a constant."""
        if d is None:
            self._err(n, "Fixed-point division needs a constant divisor (x / 8, x / 3.5)")
        if d == 0:
            self._err(n, "Division by zero")
        q = Fraction(d)
        num, den = abs(q.numerator), q.denominator
        if num & (num - 1) == 0 and den & (den - 1) == 0:
            k = (num.bit_length() - 1) - (den.bit_length() - 1)   # |d| = 2**k
            node, f = (l, fl + k) if fl + k >= 0 else (self._shl(l, -(fl + k)), 0)
            if q < 0:
                node = ast.UnaryOp(op=ast.USub(), operand=node)
            return node, FX(f)
        raw, frac = const_raw(1 / q, self.const_bits)
        return ast.BinOp(left=l, op=ast.Mult(), right=ast.Constant(raw)), FX(fl + frac)

    def _call(self, n: ast.Call):
        f = n.func
        name = self.py_names.get(f.id, f.id) if isinstance(f, ast.Name) else None
        is_math = (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
                   and self.ns.get(f.value.id) is math)
        parts = [self.ex(a) for a in n.args]
        n.args = [v for v, _ in parts]
        types = [t for _, t in parts]
        one = len(types) == 1 and not n.keywords
        if one and (name in ("int", "float") or (is_math and f.attr in ("floor", "ceil"))):
            v, t = parts[0]
            if t is None:
                return n, None
            if name == "float" or not _is_fx(t):
                return v, (INT if name != "float" and t == INT else t)
            k = _frac(t)
            if k == 0:
                return v, INT
            v = self._named(v)
            floor = ast.BinOp(left=v, op=ast.RShift(), right=ast.Constant(k))
            neg = ast.UnaryOp(op=ast.USub(), operand=ast.BinOp(
                left=ast.UnaryOp(op=ast.USub(), operand=v), op=ast.RShift(), right=ast.Constant(k)))
            if is_math:
                return (floor if f.attr == "floor" else neg), INT
            return ast.IfExp(test=ast.Compare(left=v, ops=[ast.Lt()], comparators=[ast.Constant(0)]),
                             body=neg, orelse=floor), INT          # int(): toward zero
        if name == "abs" and one:
            return n, (INT if types[0] == CPLX else types[0])
        if name in ("min", "max", "sum", "list") and one \
                and types[0] is not None and types[0][0] == "l":
            return n, (types[0] if name == "list" else types[0][1])
        if name in ("min", "max") and not n.keywords:
            if any(t is None for t in types):
                return n, None
            t = None
            for te in types:
                t = _join(t, te, name)
            n.args = [self._align(v, te, t, name, n) for v, te in parts]
            return n, t
        if name == "len":
            return n, INT
        if name == "round" and any(_is_fx(t) for t in types):
            self._err(n, "round() is not supported; give the variable 'round': True in attr")
        if isinstance(f, ast.Attribute) and not is_math:
            base, tb = self.ex(f.value)
            f.value = base
            if _is_fx(tb) or any(_is_fx(t) for t in types):
                self._err(n, f"'.{f.attr}()' is not supported on fixed-point values")
            return n, (CPLX if tb == CPLX else INT)
        if any(_is_fx(t) for t in types):
            self._err(n, f"'{name or ast.unparse(f)}' can't take fixed-point values")
        if name == "complex":
            return n, CPLX
        return n, INT

    # ---------------- statements ----------------
    def block(self, stmts: List[ast.stmt]) -> List[ast.stmt]:
        saved, out = self.pre, []
        for s in stmts:
            self.pre = []
            new = self.stmt(s)
            out.extend(self.pre)
            out.extend(new)
        self.pre = saved
        return out

    def _assign_name(self, name: str, value: ast.expr, s: ast.stmt) -> ast.expr:
        v, t = self.ex(value)
        self._bind(name, t, s)
        if not self.final:
            return v
        if name in self.meta:
            return self._fit(v, t, self.env.get(name), self.meta[name], name, s)
        return self._align(v, t, self.env.get(name), name, s)

    def stmt(self, s: ast.stmt) -> List[ast.stmt]:
        if isinstance(s, ast.AugAssign):
            load = copy.deepcopy(s.target)
            for node in ast.walk(load):
                if hasattr(node, "ctx"):
                    node.ctx = ast.Load()
            s = ast.copy_location(ast.Assign(
                targets=[s.target], value=ast.BinOp(left=load, op=s.op, right=s.value)), s)

        if isinstance(s, ast.Assign):
            if len(s.targets) != 1:
                return [s]
            tgt = s.targets[0]
            if isinstance(tgt, ast.Name):
                s.value = self._assign_name(tgt.id, s.value, s)
            elif (isinstance(tgt, ast.Tuple) and isinstance(s.value, ast.Tuple)
                  and len(tgt.elts) == len(s.value.elts)
                  and all(isinstance(e, ast.Name) for e in tgt.elts)):
                s.value.elts = [self._assign_name(e.id, v, s)
                                for e, v in zip(tgt.elts, s.value.elts)]
            elif isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Name) \
                    and not isinstance(tgt.slice, ast.Slice):
                name = tgt.value.id
                tgt.slice = self.ex(tgt.slice)[0]
                v, t = self.ex(s.value)
                self._bind(name, LST(t), s)
                lt = self.env.get(name)
                elem = lt[1] if lt and lt[0] == "l" else lt
                if self.final:
                    v = (self._fit(v, t, elem, self.meta[name], name, s) if name in self.meta
                         else self._align(v, t, elem, name, s))
                s.value = v
            else:
                v, t = self.ex(s.value)
                if _is_fx(t):
                    self._err(s, "Assign fixed-point values to plain names or list elements")
                s.value = v
            return [s]

        if isinstance(s, ast.Expr):
            c = s.value
            if (isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                    and isinstance(c.func.value, ast.Name) and c.func.attr in ("append", "extend")
                    and len(c.args) == 1):
                name = c.func.value.id
                v, t = self.ex(c.args[0])
                t = LST(t) if c.func.attr == "append" else t
                self._bind(name, t, s)
                if self.final and t is not None:
                    lt = self.env.get(name)
                    meta = self.meta.get(name, {})
                    if c.func.attr == "append":
                        v = self._fit(v, t[1], lt[1] if lt and lt[0] == "l" else lt, meta, name, s)
                    else:
                        v = self._fit(v, t, lt, meta, name, s)
                c.args = [v]
                return [s]
            if isinstance(c, ast.Constant):
                return [s]
            s.value = self.ex(c)[0]
            return [s]

        if isinstance(s, ast.If):
            s.test = self.ex(s.test)[0]
            s.body = self.block(s.body)
            s.orelse = self.block(s.orelse)
            return [s]

        if isinstance(s, ast.For):
            it, ti = self.ex(s.iter)
            s.iter = it
            elem = ti[1] if ti is not None and ti[0] == "l" else (None if ti is None else INT)
            if isinstance(s.target, ast.Name):
                name = s.target.id
                self._bind(name, elem, s)
                if self.final and elem is not None and self.env.get(name) not in (None, elem) \
                        and (_is_fx(elem) or _is_fx(self.env.get(name))):
                    self._err(s, f"Loop variable '{name}' is also assigned {_fmt(self.env[name])}; "
                                 "use another name")
            s.body = self.block(s.body)
            s.orelse = self.block(s.orelse)
            return [s]

        if isinstance(s, ast.Return) and s.value is not None:
            vals = s.value.elts if isinstance(s.value, ast.Tuple) else [s.value]
            out = []
            for i, v in enumerate(vals):
                if isinstance(v, ast.Name):
                    out.append(self.ex(v)[0])
                    continue
                # A returned expression is the output ret / ret<i>: name it here,
                # since lowering may turn it into a temporary's name
                name = f"ret{i}" if len(vals) > 1 else "ret"
                if name in self.locals:
                    self._err(s, f"Output name '{name}' is already used")
                value = self._assign_name(name, v, s)
                self.pre.append(ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())],
                                           value=value))
                out.append(ast.Name(id=name, ctx=ast.Load()))
            if isinstance(s.value, ast.Tuple):
                s.value.elts = out
            else:
                s.value = out[0]
            return [s]

        return [s]


_OPS = {ast.Div: "/", ast.FloorDiv: "//", ast.Mod: "%", ast.Pow: "**", ast.BitAnd: "&",
        ast.BitOr: "|", ast.BitXor: "^", ast.MatMult: "@"}


def uses_fixed(attr: Dict[str, Dict[str, Any]]) -> bool:
    return any(isinstance(m, dict) and m.get("frac") is not None for m in attr.values())


def lower_fixed(tree: ast.Module, attr: Dict[str, Dict[str, Any]], ns: Dict[str, Any],
                const_bits: int = DEFAULT_CONST_BITS, py_names: Optional[Dict[str, str]] = None):
    """
    Rewrite the module's first function from fixed-point to integer code, if
    attr gives anything a 'frac'. Returns (tree, attr of the temporaries it
    adds, name -> type).
    """
    fd = tree.body[0]
    if not isinstance(fd, ast.FunctionDef) or not uses_fixed(attr):
        return tree, {}, {}
    low = _Lower(fd, attr, ns, const_bits, py_names or {})
    # Types first: a variable without a format holds the widest value assigned
    for _ in range(MAX_FRAC + 2):
        low.changed = False
        low.n_temps = 0
        low.block(copy.deepcopy(fd.body))
        if not low.changed:
            break
    low.final = True
    low.n_temps = 0
    fd.body = low.block(fd.body)
    ast.fix_missing_locations(tree)
    return tree, low.new_attr, dict(low.env)


# ====================================================================
# Bit-exact Python model of the lowered code
# ====================================================================

def _tdiv(a: int, b: int) -> int:
    """Integer a / b truncated toward zero, like int(a / b) without float rounding."""
    q = abs(a) // abs(b)
    return -q if (a < 0) != (b < 0) else q


def build_model(tree: ast.Module, attr: Dict[str, Dict[str, Any]],
                ns: Dict[str, Any]) -> Callable[..., Any]:
    """
    The lowered function as Python on raw values, with every declared signal
    wrapped to its width on assignment (as its hardware wire is), integer /
    truncating, and and/or giving 0/1.
    """
    fd = copy.deepcopy(tree.body[0])
    fd.decorator_list = []
    fd.name = "__py2ver_model"
    declared = {k: (int(m["width"]), bool(m.get("signed")), "length" in m)
                for k, m in attr.items() if isinstance(m, dict) and "width" in m}

    def wrapped(value: ast.expr, name: str, element: bool) -> ast.expr:
        w, sgn, is_list = declared[name]
        fn = "__py2ver_wrapl" if is_list and not element else "__py2ver_wrap"
        return ast.Call(func=ast.Name(id=fn, ctx=ast.Load()), keywords=[],
                        args=[value, ast.Constant(w), ast.Constant(sgn)])

    class _Model(ast.NodeTransformer):
        def visit_Assign(self, node: ast.Assign):
            self.generic_visit(node)
            tgt = node.targets[0]
            if isinstance(tgt, ast.Name) and tgt.id in declared:
                node.value = wrapped(node.value, tgt.id, False)
            elif (isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Name)
                  and tgt.value.id in declared and not isinstance(tgt.slice, ast.Slice)):
                node.value = wrapped(node.value, tgt.value.id, True)
            elif (isinstance(tgt, ast.Tuple) and isinstance(node.value, ast.Tuple)
                  and len(tgt.elts) == len(node.value.elts)):
                node.value.elts = [wrapped(v, t.id, False)
                                   if isinstance(t, ast.Name) and t.id in declared else v
                                   for t, v in zip(tgt.elts, node.value.elts)]
            return node

        def visit_AugAssign(self, node: ast.AugAssign):
            load = copy.deepcopy(node.target)
            for n in ast.walk(load):
                if hasattr(n, "ctx"):
                    n.ctx = ast.Load()
            return self.visit_Assign(ast.copy_location(ast.Assign(
                targets=[node.target], value=ast.BinOp(left=load, op=node.op, right=node.value)),
                node))

        def visit_BinOp(self, node: ast.BinOp):
            self.generic_visit(node)
            if isinstance(node.op, ast.Div):
                return ast.Call(func=ast.Name(id="__py2ver_tdiv", ctx=ast.Load()), keywords=[],
                                args=[node.left, node.right])
            return node

        def visit_BoolOp(self, node: ast.BoolOp):
            self.generic_visit(node)
            return ast.IfExp(test=node, body=ast.Constant(1), orelse=ast.Constant(0))

    fd = _Model().visit(fd)
    module = ast.fix_missing_locations(ast.Module(body=[fd], type_ignores=[]))
    env = dict(ns)
    env.update(__py2ver_wrap=wrap, __py2ver_tdiv=_tdiv,
               __py2ver_wrapl=lambda xs, w, s: [wrap(int(x), w, s) for x in xs])
    exec(compile(module, "<py2ver bit-exact model>", "exec"), env)
    return env[fd.name]
