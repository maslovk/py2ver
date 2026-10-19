# vectorize.py
"""
Loop programs run with numpy: the same results, many times faster.

numpy_loops.py writes a design as loops over flat lists, and the format run
(auto.py), the translation's check and the bit-exact model (fixed.py) all
used to run those loops in Python, an element at a time: minutes for a
network. Here the same program (its AST) runs with numpy, giving exactly what
Python gives:

  - a loop whose iterations don't depend on each other runs all of them at
    once: its variable is an array of its values (one axis of a grid, the
    loops around it the others), so `a[i * 4 + j] = x[j] * w[i]` is one
    fancy-indexed store. Iterations depend on each other when a name is read
    before it is assigned in the body and assigned in it (a sum carried
    from one to the next: acc = acc + ...), or a list is written and read at
    another index; such a loop runs one iteration at a time, its body still
    over the grid of the loops around it (a matrix product: the sums of all
    its elements carried together, a product a step)
  - a store whose indices repeat over the grid (col2im adding overlapping
    windows back, g[p] = g[p] + c) runs in rounds: each round the iterations
    whose index is that index's k-th occurrence, in order, so each element
    sees its updates in the order Python makes them. Otherwise (an index
    computed in the body) the loop falls back to one iteration at a time
  - an if inside a loop is a mask
  - integers are int64, or Python's integers (object arrays) wherever an
    operation could pass 62 bits; floats are float64; comparisons give
    booleans, used as Python's 0 / 1 in arithmetic
  - the fixed-point model's wrap(), truncating division, integer square root
    and list fitting are element-wise numpy

A function it can't run (a construct it doesn't know) runs as Python.
PY2VER_VECTOR=0 turns it off.
"""

from __future__ import annotations

import ast
import math
import os
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

VECTOR_KEY = "__py2ver_vector__"   # (in a namespace) a loop program: models run with numpy
MAX_ROUNDS = 64              # more repeats of a store's index: one iteration at a time
_BIG = 2.0 ** 62             # int64 results stay below this, else Python's integers


def enabled() -> bool:
    return os.environ.get("PY2VER_VECTOR", "1").strip() not in ("0", "false", "no")


_NP = None


def _np():
    global _NP
    if _NP is None:
        import numpy
        _NP = numpy
    return _NP


class ListValue:
    """A whole list given to record() (its values, as an array): a list's range and
    length, as the traced Python records a list."""

    def __init__(self, arr: Any) -> None:
        self.arr = arr


def _makes_list(n: ast.expr) -> bool:
    """[v] * n, a list literal, or the model's wrapped list."""
    if isinstance(n, ast.List):
        return True
    if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult):
        return isinstance(n.left, ast.List) or isinstance(n.right, ast.List)
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
        from .fixed import FIT_LIST
        return n.func.id in ("__py2ver_wrapl", FIT_LIST)
    return False


class Unsupported(Exception):
    """A construct the runner doesn't know: the function runs as Python."""


class _Hazard(Exception):
    """Iterations that depend on each other, found while running: one at a time."""


# ==================================================================== analysis

def _names(node: ast.AST, ctx=None) -> Set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)
            and (ctx is None or isinstance(n.ctx, ctx))}


def _assigned(stmts: List[ast.stmt]) -> Set[str]:
    out: Set[str] = set()
    for s in stmts:
        for n in ast.walk(s):
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    out |= _target_names(t)
            elif isinstance(n, ast.AugAssign):
                out |= _target_names(n.target)
            elif isinstance(n, ast.For):
                out |= _target_names(n.target)
    return out


def _target_names(t: ast.expr) -> Set[str]:
    if isinstance(t, ast.Name):
        return {t.id}
    if isinstance(t, (ast.Tuple, ast.List)):
        return {x for e in t.elts for x in _target_names(e)}
    return set()


def _read_first(stmts: List[ast.stmt], defined: Set[str]) -> Set[str]:
    """Names read (in program order) before this code assigns them."""
    out: Set[str] = set()
    defined = set(defined)
    for s in stmts:
        if isinstance(s, (ast.Assign, ast.AugAssign)):
            targets = s.targets if isinstance(s, ast.Assign) else [s.target]
            reads = _names(s.value, ast.Load)
            for t in targets:
                if not isinstance(t, ast.Name):
                    reads |= _names(t, ast.Load)
            if isinstance(s, ast.AugAssign) and isinstance(s.target, ast.Name):
                reads.add(s.target.id)
            out |= reads - defined
            for t in targets:
                defined |= _target_names(t)
        elif isinstance(s, ast.For):
            out |= _names(s.iter, ast.Load) - defined
            out |= _read_first(s.body, defined | _target_names(s.target))
            defined |= _assigned(s.body) | _target_names(s.target)
        elif isinstance(s, ast.If):
            out |= _names(s.test, ast.Load) - defined
            out |= _read_first(s.body, defined) | _read_first(s.orelse, defined)
            defined |= _assigned(s.body) & _assigned(s.orelse)
        else:
            out |= _names(s, ast.Load) - defined
    return out


def _accesses(stmts: List[ast.stmt]):
    """{list: [index dumps read]}, {list: [index dumps written]}."""
    reads: Dict[str, List[str]] = {}
    writes: Dict[str, List[str]] = {}
    for s in stmts:
        for n in ast.walk(s):
            if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name):
                side = writes if isinstance(n.ctx, ast.Store) else reads
                side.setdefault(n.value.id, []).append(ast.dump(n.slice))
    return reads, writes


class _Loop:
    """What a for loop's body allows (computed once)."""

    def __init__(self, node: ast.For) -> None:
        self.var = node.target.id if isinstance(node.target, ast.Name) else None
        body = node.body
        assigned = _assigned(body)
        carried = _read_first(body, {self.var} if self.var else set()) & assigned
        reads, writes = _accesses(body)
        conflict = any(set(reads.get(a, [])) - set(ix) or len(set(ix)) > 1
                       for a, ix in writes.items() if a in reads)
        # (a list made in it, a new one each turn: a loop run in turn, as repeats.py's
        # loop over a model's layers)
        makes = any(isinstance(s, ast.Assign) and _makes_list(s.value)
                    for x in body for s in ast.walk(x))
        self.parallel = self.var is not None and not carried and not conflict and not makes
        self.written = set(writes)
        self.assigned = assigned
        # A perfect nest of parallel loops down to straight code: its stores' indices
        # can be computed first (for rounds when they repeat)
        self.straight = all(isinstance(s, (ast.Assign, ast.AugAssign, ast.Expr)) for s in body)


# ==================================================================== values

def _lift(x: Any, nd: int) -> Any:
    """x with trailing axes of 1 up to the grid's nd (the grid's axes are the loops,
    outermost first)."""
    np = _np()
    if isinstance(x, np.ndarray) and 0 < x.ndim < nd:
        return x.reshape(x.shape + (1,) * (nd - x.ndim))
    return x


def _arith(x: Any) -> Any:
    """A boolean as Python's 0 / 1 in arithmetic."""
    np = _np()
    if isinstance(x, np.ndarray) and x.dtype == bool:
        return x.astype(np.int64)
    if isinstance(x, (bool, np.bool_)):
        return int(x)
    return x


def _bound(x: Any) -> Optional[float]:
    """The largest magnitude of an integer value (None: a float, inf: unbounded)."""
    np = _np()
    if isinstance(x, np.ndarray):
        kind = x.dtype.kind
        if kind in "iu":
            return float(abs(x).max()) if x.size else 0.0
        if kind == "O":
            return math.inf
        if x.dtype == bool:
            return 1.0
        return None
    if isinstance(x, (bool, np.bool_)):
        return 1.0
    if isinstance(x, (int, np.integer)):
        return abs(float(x))
    return None


def _big(x: Any) -> Any:
    np = _np()
    if isinstance(x, np.ndarray):
        return x.astype(object) if x.dtype != object else x
    return int(x)


def _small(x: Any) -> Any:
    """An object array of integers back to int64 where they fit."""
    np = _np()
    if isinstance(x, np.ndarray) and x.dtype == object and x.size:
        try:
            if max(abs(int(v)) for v in x.flat) < _BIG:
                return x.astype(np.int64)
        except (TypeError, ValueError):
            pass
    return x


def _ints(*xs: Any) -> bool:
    return all(_bound(x) is not None for x in xs)


def _binop(op: ast.operator, a: Any, b: Any) -> Any:
    np = _np()
    a, b = _arith(a), _arith(b)
    arrays = isinstance(a, np.ndarray) or isinstance(b, np.ndarray)
    if not arrays:
        return _PY_OPS[type(op)](a, b)                 # (Python's own, exact)
    if _ints(a, b):
        ba, bb = _bound(a), _bound(b)
        grow = {ast.Add: ba + bb, ast.Sub: ba + bb, ast.Mult: ba * bb,
                ast.LShift: ba * 2.0 ** min(float(np.max(b)) if isinstance(b, np.ndarray)
                                            else float(b), 1023.0)}.get(type(op))
        if grow is not None and grow >= _BIG:
            return _small(_PY_OPS[type(op)](_big(a), _big(b)))
        if isinstance(op, ast.Div):
            return np.true_divide(a, b)
    with np.errstate(all="ignore"):
        return _PY_OPS[type(op)](a, b)


_PY_OPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
           ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b,
           ast.FloorDiv: lambda a, b: a // b, ast.Mod: lambda a, b: a % b,
           ast.Pow: lambda a, b: a ** b, ast.LShift: lambda a, b: a << b,
           ast.RShift: lambda a, b: a >> b, ast.BitAnd: lambda a, b: a & b,
           ast.BitOr: lambda a, b: a | b, ast.BitXor: lambda a, b: a ^ b}
_CMP = {ast.Gt: lambda a, b: a > b, ast.Lt: lambda a, b: a < b, ast.GtE: lambda a, b: a >= b,
        ast.LtE: lambda a, b: a <= b, ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b}


def wrap(n: Any, width: int, signed: bool) -> Any:
    """fixed.wrap, element-wise."""
    np = _np()
    if not isinstance(n, np.ndarray):
        from .fixed import wrap as one
        return one(int(n), width, signed)
    if n.dtype == object or width > 62:
        from .fixed import wrap as one
        return _small(np.array([one(int(v), width, signed) for v in n.flat],
                               dtype=object).reshape(n.shape))
    n = n.astype(np.int64, copy=False) & ((1 << width) - 1)
    if signed:
        n = np.where(n >> (width - 1), n - (1 << width), n)
    return n


def tdiv(a: Any, b: Any) -> Any:
    """fixed._tdiv, element-wise: a / b truncated toward zero, 0 for b == 0."""
    np = _np()
    if not (isinstance(a, np.ndarray) or isinstance(b, np.ndarray)):
        from .fixed import _tdiv
        return _tdiv(int(a), int(b))
    a, b = np.broadcast_arrays(_arith(a), _arith(b))
    if a.dtype == object or b.dtype == object:
        from .fixed import _tdiv
        return _small(np.array([_tdiv(int(x), int(y)) for x, y in zip(a.flat, b.flat)],
                               dtype=object).reshape(a.shape))
    safe = np.where(b == 0, 1, b)
    q = np.abs(a) // np.abs(safe)
    q = np.where((a < 0) != (b < 0), -q, q)
    return np.where(b == 0, 0, q)


def isqrt(n: Any) -> Any:
    """fixed.isqrt, element-wise."""
    np = _np()
    if not isinstance(n, np.ndarray):
        from .fixed import isqrt as one
        return one(int(n))
    if n.dtype == object or (n.size and float(np.max(n)) >= 2.0 ** 52):
        from .fixed import isqrt as one
        return _small(np.array([one(int(v)) for v in n.flat], dtype=object).reshape(n.shape))
    n = n.astype(np.int64)
    r = np.sqrt(np.maximum(n, 0).astype(float)).astype(np.int64)
    for _ in range(3):                                      # (exact: correct the float root)
        r = np.where(r * r > n, r - 1, r)
        r = np.where((r + 1) * (r + 1) <= n, r + 1, r)
    return np.where(n > 0, r, 0)


def fit_list(xs: Any, shift: int, rnd: bool, lo: Any, hi: Any) -> Any:
    np = _np()
    xs = _arith(np.asarray(xs))
    if shift > 0:
        v = _binop(ast.RShift(), _binop(ast.Add(), xs, (1 << (shift - 1)) if rnd else 0), shift)
    else:
        v = _binop(ast.LShift(), xs, -shift)
    return v if lo is None else np.maximum(lo, np.minimum(hi, v))


# ==================================================================== the runner

class _Run:
    def __init__(self, prog: "Program", args: Dict[str, Any]) -> None:
        self.p = prog
        self.env: Dict[str, Any] = {}
        self.grid: List[Tuple[ast.For, Any]] = []        # (loop, its values) per axis
        self.lists: Set[str] = set()

    # ------------------------------------------------------------ expressions
    def ev(self, n: ast.expr) -> Any:
        np = _np()
        nd = len(self.grid)
        if isinstance(n, ast.Constant):
            return n.value
        if isinstance(n, ast.Name):
            if n.id in self.env:
                v = self.env[n.id]
                return v if n.id in self.lists else _lift(v, nd)
            v = self.p.globals_value(n.id)
            return v
        if isinstance(n, ast.BinOp):
            if isinstance(n.op, ast.Mult) and isinstance(n.left, ast.List) \
                    and len(n.left.elts) == 1:               # ([v] * n: a new list)
                v, k = self.ev(n.left.elts[0]), self.ev(n.right)
                if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) \
                        and isinstance(k, (int, np.integer)):
                    if isinstance(v, (float, np.floating)):
                        return np.full(int(k), float(v))
                    if abs(int(v)) < _BIG:
                        return np.full(int(k), int(v), dtype=np.int64)
            return _binop(n.op, self.ev(n.left), self.ev(n.right))
        if isinstance(n, ast.UnaryOp):
            v = self.ev(n.operand)
            if isinstance(n.op, ast.USub):
                v = _arith(v)
                if _bound(v) is not None and _bound(v) >= _BIG:
                    return _small(-_big(v))
                return -v
            if isinstance(n.op, ast.UAdd):
                return _arith(v)
            if isinstance(n.op, ast.Not):
                return np.logical_not(v) if isinstance(v, np.ndarray) else (not v)
            if isinstance(n.op, ast.Invert):
                return ~_arith(v)
        if isinstance(n, ast.Compare):
            left = self.ev(n.left)
            out = None
            for op, c in zip(n.ops, n.comparators):
                if type(op) not in _CMP:
                    raise Unsupported(ast.unparse(n))
                right = self.ev(c)
                r = _CMP[type(op)](_arith(left), _arith(right))
                out = r if out is None else (np.logical_and(out, r) if isinstance(r, np.ndarray)
                                             or isinstance(out, np.ndarray) else (out and r))
                left = right
            return out
        if isinstance(n, ast.BoolOp):
            vals = [self.ev(v) for v in n.values]
            if not any(isinstance(v, np.ndarray) for v in vals):
                out = vals[0]
                for v in vals[1:]:
                    out = (out and v) if isinstance(n.op, ast.And) else (out or v)
                return out
            fn = np.logical_and if isinstance(n.op, ast.And) else np.logical_or
            out = vals[0]
            for v in vals[1:]:
                out = fn(out, v)
            return out
        if isinstance(n, ast.IfExp):
            test = self.ev(n.test)
            if not isinstance(test, np.ndarray):
                return self.ev(n.body) if test else self.ev(n.orelse)
            return np.where(test, _arith(self.ev(n.body)), _arith(self.ev(n.orelse)))
        if isinstance(n, ast.Subscript):
            base = self.ev(n.value)
            idx = self.ev(n.slice)
            if isinstance(base, np.ndarray):
                return base[_index(idx)]
            if isinstance(base, (list, tuple)):
                arr = self.p.table(n.value, base)
                return arr[_index(idx)]
            raise Unsupported(ast.unparse(n))
        if isinstance(n, ast.Call):
            return self.call(n)
        if isinstance(n, ast.List):
            return [self.ev(e) for e in n.elts]
        if isinstance(n, ast.Tuple):
            return tuple(self.ev(e) for e in n.elts)
        raise Unsupported(ast.unparse(n))

    def call(self, n: ast.Call) -> Any:
        np = _np()
        if n.keywords:
            raise Unsupported(ast.unparse(n))
        fname = ast.unparse(n.func)
        args = [self.ev(a) for a in n.args]
        record = self.p.record
        if fname in self.p.table_keys and record is not None:
            out = self._ufunc(n, args)
            self.rec(fname + ":in", args[0])
            self.rec(fname, out)
            return out
        if fname in ("max", "min") and len(args) >= 2:
            if not any(isinstance(a, np.ndarray) for a in args):
                return (max if fname == "max" else min)(*args)
            fn = np.maximum if fname == "max" else np.minimum
            out = _arith(args[0])
            for a in args[1:]:
                out = fn(out, _arith(a))
            return out
        if fname == "abs":
            v = _arith(args[0])
            return np.abs(v) if isinstance(v, np.ndarray) else abs(v)
        if fname == "int":
            v = args[0]
            if isinstance(v, np.ndarray):
                return np.trunc(v).astype(np.int64) if v.dtype.kind == "f" else _arith(v)
            return int(v)
        if fname == "float":
            v = args[0]
            return v.astype(float) if isinstance(v, np.ndarray) else float(v)
        if fname == "__py2ver_wrap":
            return wrap(args[0], int(args[1]), bool(args[2]))
        if fname == "__py2ver_wrapl":
            return wrap(np.asarray(args[0]) if not isinstance(args[0], np.ndarray) else args[0],
                        int(args[1]), bool(args[2]))
        if fname == "__py2ver_tdiv":
            return tdiv(args[0], args[1])
        from .fixed import FIT_LIST, ISQRT
        if fname == ISQRT:
            return isqrt(args[0])
        if fname == FIT_LIST:
            return fit_list(*args)
        return self._ufunc(n, args)

    def _ufunc(self, n: ast.Call, args: List[Any]) -> Any:
        np = _np()
        fn = self.ev(n.func) if not isinstance(n.func, ast.Attribute) else self.p.attr_value(n.func)
        if isinstance(fn, np.ufunc) or getattr(fn, "__module__", None) == "numpy":
            with np.errstate(all="ignore"):
                return fn(*[_arith(a) for a in args])
        if callable(fn) and (getattr(fn, "__py2ver_vectorized__", False)
                             or not any(isinstance(a, np.ndarray) for a in args)):
            return fn(*args)
        raise Unsupported(ast.unparse(n))

    # ------------------------------------------------------------ recording
    def rec(self, name: str, v: Any, mask: Any = None) -> None:
        np = _np()
        if self.p.record is None:
            return
        if isinstance(v, ListValue):
            self.p.record(name, v)
            return
        if isinstance(v, np.ndarray):
            if v.dtype == bool:
                return
            if mask is not None:
                v = np.broadcast_to(_lift(v, len(self.grid)), np.shape(mask))[mask]
            self.p.record(name, v)
        elif isinstance(v, (list, tuple)):
            self.p.record(name, v)
        else:
            if mask is not None and not np.any(mask):
                return
            self.p.record(name, v)

    # ------------------------------------------------------------ statements
    def block(self, stmts: List[ast.stmt], mask: Any) -> None:
        for s in stmts:
            self.stmt(s, mask)

    def stmt(self, s: ast.stmt, mask: Any) -> None:
        np = _np()
        if isinstance(s, ast.AugAssign):
            load = _as_load(s.target)
            s = ast.Assign(targets=[s.target], value=ast.BinOp(left=load, op=s.op, right=s.value))
        if isinstance(s, ast.Assign):
            if len(s.targets) != 1:
                raise Unsupported("chained assignment")
            value = self.ev(s.value)
            if isinstance(s.targets[0], ast.Name) and _makes_list(s.value):
                name = s.targets[0].id
                arr = value if isinstance(value, _np().ndarray) else _new_list(list(value))
                self.env[name] = arr
                self.lists.add(name)
                self.rec(name, ListValue(arr))
                return
            self.assign(s.targets[0], value, mask)
            return
        if isinstance(s, ast.For):
            self.loop(s, mask)
            return
        if isinstance(s, ast.If):
            test = self.ev(s.test)
            if not isinstance(test, np.ndarray):
                if mask is None:
                    self.block(s.body if test else s.orelse, None)
                    return
                test = np.asarray(test)
            test = _lift(test, len(self.grid))
            yes = test if mask is None else np.logical_and(mask, test)
            no = np.logical_not(test) if mask is None else np.logical_and(mask, np.logical_not(test))
            if np.any(yes):
                self.block(s.body, yes)
            if s.orelse and np.any(no):
                self.block(s.orelse, no)
            return
        if isinstance(s, ast.Expr):
            if isinstance(s.value, ast.Constant):
                return
            self.ev(s.value)
            return
        if isinstance(s, ast.Pass):
            return
        if isinstance(s, ast.Return):
            raise _Returned(self.ev(s.value) if s.value is not None else None)
        raise Unsupported(type(s).__name__)

    def assign(self, t: ast.expr, value: Any, mask: Any) -> None:
        np = _np()
        if isinstance(t, ast.Name):
            if isinstance(value, list):
                raise Unsupported("a list value")
            if mask is not None and t.id in self.env:
                old = self.env[t.id]
                value = np.where(mask, value, _lift(old, len(self.grid)))
            self.env[t.id] = value
            self.rec(t.id, value, mask)
            return
        if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name):
            self.store(t, value, mask)
            return
        if isinstance(t, (ast.Tuple, ast.List)) and isinstance(value, tuple):
            for tt, v in zip(t.elts, value):
                self.assign(tt, v, mask)
            return
        raise Unsupported(ast.unparse(t))

    def store(self, t: ast.Subscript, value: Any, mask: Any) -> None:
        np = _np()
        name = t.value.id
        arr = self.env.get(name)
        if not isinstance(arr, np.ndarray):
            raise Unsupported(f"store into {name}")
        idx = _index(self.ev(t.slice))
        value = _arith(value) if not (isinstance(value, np.ndarray) and value.dtype.kind == "f") \
            else value
        nd = len(self.grid)
        if nd == 0 or not (isinstance(idx, np.ndarray) and idx.ndim):
            if isinstance(idx, np.ndarray) and idx.ndim == 0:
                idx = int(idx)
            if mask is not None and not np.any(mask):
                return
            if nd and math.prod(int(g[1].size) for g in self.grid) > 1:
                raise _Hazard()                       # (one element, every iteration: in order)
            arr = self._fit(name, arr, value)
            arr[idx] = value if not isinstance(value, np.ndarray) else value.reshape(())
            self.rec(name, value)
            return
        shape = np.broadcast_shapes(np.shape(_lift(idx, nd)), np.shape(_lift(value, nd)),
                                    np.shape(mask) if mask is not None else ())
        flat_idx = np.broadcast_to(_lift(idx, nd), shape).ravel()
        vals = np.broadcast_to(_lift(value, nd), shape).ravel() \
            if isinstance(value, np.ndarray) else value
        if mask is not None:
            m = np.broadcast_to(mask, shape).ravel()
            flat_idx = flat_idx[m]
            vals = vals[m] if isinstance(vals, np.ndarray) else vals
        if flat_idx.size and not self.p.injective(t, self.grid, flat_idx):
            raise _Hazard()
        arr = self._fit(name, arr, vals)
        arr[flat_idx] = vals
        self.rec(name, vals)

    def _fit(self, name: str, arr: Any, value: Any) -> Any:
        """The list able to hold value (ints made floats, int64 Python's integers)."""
        np = _np()
        kind = value.dtype.kind if isinstance(value, np.ndarray) else (
            "f" if isinstance(value, float) else "O" if isinstance(value, int)
            and abs(value) >= _BIG else "i")
        if kind == "f" and arr.dtype.kind in "iub":
            arr = arr.astype(float)
        elif kind == "O" and arr.dtype != object:
            arr = arr.astype(object)
        elif arr.dtype == bool and kind in "iu":
            arr = arr.astype(np.int64)
        else:
            return arr
        self.env[name] = arr
        return arr

    # ------------------------------------------------------------ loops
    def loop(self, s: ast.For, mask: Any) -> None:
        np = _np()
        values = self.ev(s.iter) if not _is_range(s.iter) else range(
            *[int(self.ev(a)) for a in s.iter.args])
        values = list(values)
        if not values:
            return
        info = self.p.loop_info(s)
        if info.parallel and len(values) > 1 and not self.p.serial(s):
            saved = {a: np.copy(self.env[a]) for a in info.written
                     if isinstance(self.env.get(a), np.ndarray)}
            saved_names = {k: self.env[k] for k in info.assigned if k in self.env}
            try:
                self.parallel(s, values, mask, info)
                return
            except _Hazard:
                self.env.update(saved)
                for k in info.assigned:
                    if k in saved_names:
                        self.env[k] = saved_names[k]
                    else:
                        self.env.pop(k, None)
                self.p.serialize(s)
        var = s.target.id if isinstance(s.target, ast.Name) else None
        if var is None:
            raise Unsupported("loop target")
        for v in values:
            self.env[var] = v
            self.rec(var, v, mask)
            self.block(s.body, mask)

    def parallel(self, s: ast.For, values: List[Any], mask: Any, info: _Loop) -> None:
        np = _np()
        var = s.target.id
        nd = len(self.grid)
        arr = np.asarray(values).reshape((1,) * nd + (len(values),))
        self.grid.append((s, arr))
        try:
            self.env[var] = arr
            self.rec(var, arr)
            m = _lift(mask, nd + 1) if mask is not None else None
            rounds = self.rounds(s, info) if info.straight else None
            if rounds is None:
                self.block(s.body, m)
            else:
                for r in rounds:
                    self.block(s.body, r if m is None else np.logical_and(m, r))
        finally:
            self.grid.pop()
        # After the loop: its names hold the last iteration's values (Python's)
        for k in list(info.assigned | {var}):
            v = self.env.get(k)
            if isinstance(v, np.ndarray) and v.ndim == nd + 1 and k not in self.lists:
                self.env[k] = v[..., -1]

    def rounds(self, s: ast.For, info: _Loop):
        """Masks of a straight loop whose stores' indices repeat: each round the k-th
        occurrences (None: they don't repeat)."""
        np = _np()
        nd = len(self.grid)
        masks = None
        for st in s.body:
            t = st.targets[0] if isinstance(st, ast.Assign) else getattr(st, "target", None)
            if not (isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)):
                continue
            if _names(t.slice, ast.Load) & (info.assigned | info.written):
                return None                            # (an index computed here: at the store)
            idx = _index(self.ev(t.slice))
            if not isinstance(idx, np.ndarray) or idx.ndim < nd:
                continue
            shape = tuple(int(g[1].shape[a]) for a, g in enumerate(self.grid))
            flat = np.broadcast_to(_lift(idx, nd), shape).ravel()
            if self.p.injective(t, self.grid, flat):
                continue
            # The occurrence rank of each iteration's index
            order = np.argsort(flat, kind="stable")
            sorted_idx = flat[order]
            starts = np.r_[0, np.flatnonzero(np.diff(sorted_idx)) + 1]
            run = np.repeat(starts, np.diff(np.r_[starts, flat.size]))
            rank = np.empty(flat.size, dtype=np.int64)
            rank[order] = np.arange(flat.size) - run
            if masks is not None:
                raise _Hazard()                        # (two repeating stores: in order)
            top = int(rank.max()) + 1
            if top > MAX_ROUNDS:
                raise _Hazard()
            rank = rank.reshape(shape)
            masks = [rank == r for r in range(top)]
        return masks


class _Returned(Exception):
    def __init__(self, value: Any) -> None:
        self.value = value


def _is_range(n: ast.expr) -> bool:
    return isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "range" \
        and not n.keywords


def _index(idx: Any) -> Any:
    np = _np()
    if isinstance(idx, np.ndarray):
        if idx.dtype.kind == "f":
            return idx.astype(np.int64)
        if idx.dtype == object:
            return idx.astype(np.int64)
        return idx
    if isinstance(idx, (bool, np.bool_)):
        return int(idx)
    return idx


def _as_load(t: ast.expr) -> ast.expr:
    import copy
    t = copy.deepcopy(t)
    for n in ast.walk(t):
        if hasattr(n, "ctx"):
            n.ctx = ast.Load()
    return t


def _new_list(values: List[Any]) -> Any:
    np = _np()
    arr = np.array(values)
    if arr.dtype.kind in "fiub" and arr.dtype != bool:
        return arr.astype(np.int64) if arr.dtype.kind in "iu" else arr
    if arr.dtype == bool:
        return arr.astype(np.int64)
    if all(isinstance(v, (int, np.integer)) for v in values):    # (past int64)
        return np.array([int(v) for v in values], dtype=object)
    return np.array(values, dtype=float)


def _to_python(v: Any) -> Any:
    np = _np()
    if isinstance(v, np.ndarray):
        return v.tolist() if v.ndim else v.item()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, tuple):
        return tuple(_to_python(x) for x in v)
    return v


# ==================================================================== a program

class Program:
    """A loop program (a function's AST and its namespace) run with numpy."""

    def __init__(self, fd: ast.FunctionDef, env: Dict[str, Any],
                 record: Optional[Callable[[str, Any], None]] = None,
                 table_keys: Tuple[str, ...] = ()) -> None:
        self.fd, self.genv = fd, env
        self.record = record
        self.table_keys = set(table_keys)
        self.params = [a.arg for a in fd.args.args]
        self._loops: Dict[int, _Loop] = {}
        self._serial: Set[int] = set()
        self._tables: Dict[int, Any] = {}
        self._inj: Dict[Tuple, bool] = {}
        _check_supported(fd)

    def loop_info(self, s: ast.For) -> _Loop:
        if id(s) not in self._loops:
            self._loops[id(s)] = _Loop(s)
        return self._loops[id(s)]

    def serial(self, s: ast.For) -> bool:
        return id(s) in self._serial

    def serialize(self, s: ast.For) -> None:
        self._serial.add(id(s))

    def globals_value(self, name: str) -> Any:
        if name in self.genv:
            return self.genv[name]
        import builtins
        if hasattr(builtins, name):
            return getattr(builtins, name)
        raise NameError(name)

    def attr_value(self, n: ast.expr) -> Any:
        return eval(compile(ast.Expression(n), "<attr>", "eval"), self.genv)

    def table(self, node: ast.expr, values: Any) -> Any:
        """A tuple or list constant as an array (once)."""
        key = id(values)
        if key not in self._tables:
            self._tables[key] = (values, _new_list(list(values)))
        return self._tables[key][1]

    def injective(self, t: ast.Subscript, grid, flat_idx) -> bool:
        """Do the store's indices over the grid not repeat? (Cached unless the index
        reads a list: then it depends on the data.)"""
        np = _np()
        key = (id(t), tuple(id(g[0]) for g in grid))
        reads_list = any(isinstance(n, ast.Subscript) for n in ast.walk(t.slice))
        if not reads_list and key in self._inj:
            return self._inj[key]
        ok = np.unique(flat_idx).size == flat_idx.size
        if not reads_list:
            self._inj[key] = ok
        return ok

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        np = _np()
        run = _Run(self, {})
        given = dict(zip(self.params, args))
        given.update(kwargs)
        originals: Dict[str, list] = {}
        for p in self.params:
            v = given[p]
            if isinstance(v, (list, tuple)):
                if isinstance(v, list):
                    originals[p] = v
                arr = _new_list(list(v))
                run.env[p] = arr
                run.lists.add(p)
                run.rec(p, ListValue(arr))
            else:
                run.env[p] = v
                run.rec(p, v)
        try:
            run.block(self.fd.body, None)
            result = None
        except _Returned as r:
            result = r.value
        for p, lst in originals.items():              # (lists changed in place, as Python's)
            arr = run.env[p]
            if isinstance(arr, np.ndarray):
                lst[:] = arr.tolist()
        return _to_python(result)


def _check_supported(fd: ast.FunctionDef) -> None:
    ok = (ast.Assign, ast.AugAssign, ast.For, ast.If, ast.Expr, ast.Return, ast.Pass)
    bad = (ast.Lambda, ast.ListComp, ast.GeneratorExp, ast.DictComp, ast.SetComp, ast.Starred,
           ast.Slice, ast.Dict, ast.Set, ast.JoinedStr, ast.Await, ast.Yield, ast.YieldFrom,
           ast.NamedExpr)

    def visit(stmts: List[ast.stmt], in_loop: bool) -> None:
        for s in stmts:
            if not isinstance(s, ok):
                raise Unsupported(type(s).__name__)
            if isinstance(s, ast.For):
                if not (isinstance(s.target, ast.Name) and _is_range(s.iter)):
                    raise Unsupported("a loop that isn't over a range")
                visit(s.body, True)
                continue
            if isinstance(s, ast.If):
                visit(s.body, in_loop)
                visit(s.orelse, in_loop)
                exprs = [s.test]
            else:
                exprs = [s]
            for e in exprs:
                if any(isinstance(n, bad) for n in ast.walk(e)):
                    raise Unsupported("an expression it doesn't know")
    visit(fd.body, False)


def program(fd: ast.FunctionDef, env: Dict[str, Any], fallback: Callable,
            record: Optional[Callable[[str, Any], None]] = None,
            table_keys: Tuple[str, ...] = ()) -> Callable:
    """The function (its AST, run in env) as a numpy program, or fallback (the same
    function as Python) if it uses something the runner doesn't know."""
    if not enabled():
        return fallback
    try:
        prog = Program(fd, env, record, table_keys)
    except Unsupported:
        return fallback
    import inspect
    prog.__signature__ = inspect.signature(fallback)          # (the function's own)
    prog.__name__ = getattr(fallback, "__name__", "program")
    prog.__wrapped__ = fallback
    return prog
