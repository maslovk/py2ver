# numpy_loops.py
"""
numpy code as loops: for designs too big to write out element by element.

numpy_fe.py writes every element of every array out as its own expression,
which suits a small network. A real one (a 64 -> 128 -> 64 -> 10 network is
17,000 multiply-accumulates per input) needs its loops kept, for the
sequencer (sequencer.py) to run them a step at a time. This translator runs
the method at translation time, the way Python would, keeping the arrays
symbolic:

  - Python values are real: the object, its lists of layers, isinstance(),
    loops over lists, ifs on them, the methods called (inlined, each with its
    own self)
  - an array is a view (shape, strides, offset) of a flat list in the program
    or of a constant table: an attribute read and never written (trained
    weights) is a table, an input a list argument; .T, np.atleast_2d and
    constant slices are views, so they cost nothing
  - each numpy operation becomes a loop nest writing a new list: @ (sums of
    products), + - * / ** and comparisons with broadcasting, np.exp / np.log /
    np.tanh / ... (tables), np.sqrt (exact), .sum / .max / .mean / .argmax
    with axis and keepdims, np.maximum / np.minimum, abs; dividing an array
    by a computed scalar multiplies by its reciprocal, and c ** n with a
    computed integer n multiplies the constants c ** 2 ** k for n's bits
  - an attribute the method reads and then changes (the weights and the
    optimiser's moments in a training step, in place or reassigned; a step
    counter) is state: a list argument of the program that it updates in
    place ('memory' in attr, its 'init' the object's values), kept between
    calls by the hardware
  - work whose result nothing uses is left out

The program is plain Python on flat lists, checked against the original in
float on the examples when it is made (on a copy of the object).
"""

from __future__ import annotations

import ast
import builtins
import copy
import inspect
import itertools
import linecache
import math
import textwrap
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .fixed import EXACT_SQRT
from .numpy_fe import TABLE_UFUNCS, NotTranslatable, Translation, _np

POW_BITS = 32                            # bits of an integer exponent (c ** n)


# ==================================================================== values

@dataclass
class Arr:
    """An array: a view of a flat list (or a constant table) in the program."""
    name: str
    shape: Tuple[int, ...]
    strides: Tuple[int, ...]
    offset: int = 0
    const: bool = False

    @property
    def size(self) -> int:
        return math.prod(self.shape)

    def at(self, idx: List[ast.expr]) -> ast.expr:
        """The element at index expressions idx (one per dimension)."""
        flat: Optional[ast.expr] = ast.Constant(self.offset) if self.offset else None
        for i, s in zip(idx, self.strides):
            if s == 0:
                continue
            term = i if s == 1 else ast.BinOp(left=i, op=ast.Mult(), right=ast.Constant(s))
            flat = term if flat is None else ast.BinOp(left=flat, op=ast.Add(), right=term)
        return ast.Subscript(value=ast.Name(id=self.name, ctx=ast.Load()),
                             slice=flat if flat is not None else ast.Constant(0), ctx=ast.Load())

    def view(self, shape, strides, offset=None) -> "Arr":
        return Arr(self.name, tuple(shape), tuple(strides),
                   self.offset if offset is None else offset, self.const)


class Lazy(Arr):
    """An elementwise result not written to a list yet: where it is used
    elementwise (or stored), its expression is computed in that loop, so a
    chain of elementwise operations is one loop. Anything else (a matmul, a
    reduction, a view) writes it to a list first, and so does a write to a
    list it reads (it holds the values from before)."""

    def __init__(self, it: "_Interp", shape: Tuple[int, ...], fn: Callable[..., ast.expr],
                 vals: Tuple[Any, ...]) -> None:
        self.it, self.shape, self.fn, self.vals = it, tuple(shape), fn, vals
        self.const = False
        self.real: Optional[Arr] = None

    def materialize(self) -> Arr:
        if self.real is None:
            out = self.it.new_list(self.shape)

            def body(idx):
                self.it.emit(ast.Assign(targets=[_store(out.at(idx))],
                                        value=self.it.elem(self, self.shape, idx)))
            self.it.loops(self.shape, body)
            self.real = out
        return self.real

    name = property(lambda self: self.materialize().name)
    strides = property(lambda self: self.materialize().strides)
    offset = property(lambda self: self.materialize().offset)

    def at(self, idx: List[ast.expr]) -> ast.expr:
        return self.materialize().at(idx)

    def view(self, shape, strides, offset=None) -> Arr:
        return self.materialize().view(shape, strides, offset)

    def reads(self) -> set:
        """The lists its value is read from (now)."""
        if self.real is not None:
            return {self.real.name}
        out = set()
        for v in self.vals:
            if isinstance(v, Lazy):
                out |= v.reads()
            elif isinstance(v, Arr):
                out.add(v.name)
            elif isinstance(v, Sym):
                out |= {n.value.id for n in ast.walk(v.expr)
                        if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)}
        return out


def _contiguous(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    strides, s = [], 1
    for d in reversed(shape):
        strides.insert(0, s)
        s *= d
    return tuple(strides)


@dataclass
class Sym:
    """A scalar in the program (an expression)."""
    expr: ast.expr


class _Return(Exception):
    def __init__(self, value: Any) -> None:
        self.value = value


def _bcast(*shapes: Tuple[int, ...]) -> Tuple[int, ...]:
    n = max(len(s) for s in shapes)
    out = []
    for d in range(n):
        sizes = {s[d - n + len(s)] for s in shapes if d - n + len(s) >= 0} - {1}
        if len(sizes) > 1:
            raise NotTranslatable(f"Shapes {shapes} don't broadcast")
        out.append(sizes.pop() if sizes else 1)
    return tuple(out)


_BINOPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
           ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b,
           ast.Pow: lambda a, b: a ** b, ast.FloorDiv: lambda a, b: a // b,
           ast.Mod: lambda a, b: a % b}
_CMPS = {ast.Gt: lambda a, b: a > b, ast.Lt: lambda a, b: a < b, ast.GtE: lambda a, b: a >= b,
         ast.LtE: lambda a, b: a <= b, ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b}


# ==================================================================== the interpreter

class _Interp:
    def __init__(self, ns: Dict[str, Any], state_keys: Optional[set] = None) -> None:
        self.ns = ns                           # the generated program's namespace
        # State: (object id, attribute) read and then changed. The first run
        # (state_keys None) finds them; the second builds them as lists
        self.state_keys = state_keys
        self.mutated: set = set()
        self.first: Dict[Tuple[int, str], str] = {}     # first access: 'r' (the object's) or 'w'
        self.key_of: Dict[str, Tuple[int, str]] = {}    # table / state list -> its attribute
        # (object id, attribute) -> (its list, its value, 'Class.attr', the object)
        self.state: Dict[Tuple[int, str], Tuple[str, Any, str, Any]] = {}
        self.pows: Dict[Any, Sym] = {}
        self.scalars: Dict[str, Sym] = {}               # top-level scalar expression -> its name
        self.memo: Dict[int, Dict[Any, str]] = {}       # loop body -> its elementwise results
        self.frames: List[Dict[str, Any]] = []          # the running functions' variables
        self.np = _np()
        self.code: List[List[ast.stmt]] = [[]]
        self.counter = itertools.count()
        self.shadow: Dict[int, Dict[str, Any]] = {}     # object -> attributes set while running
        self.tables: Dict[str, Any] = {}                # constant table name -> its values
        self.table_of: Dict[Tuple[int, str], Arr] = {}
        self.lists: Dict[str, int] = {}                 # list in the program -> its length

    # ------------------------------------------------------------ program text
    def fresh(self, base: str) -> str:
        return f"{base}{next(self.counter)}"

    def emit(self, s: ast.stmt) -> None:
        self.code[-1].append(s)

    def new_list(self, shape: Tuple[int, ...], base: str = "a") -> Arr:
        name = self.fresh(base)
        n = math.prod(shape)
        self.lists[name] = n
        self.emit(ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())],
                             value=ast.BinOp(left=ast.List(elts=[ast.Constant(0)], ctx=ast.Load()),
                                             op=ast.Mult(), right=ast.Constant(n))))
        return Arr(name, shape, _contiguous(shape))

    def loops(self, shape: Tuple[int, ...], body: Callable[[List[ast.expr]], None]) -> None:
        """for i0 in range(shape[0]): for i1 ...: body([i0, i1, ...])"""
        if not shape:
            body([])
            return
        names = [self.fresh("i") if size != 1 else None for size in shape]
        outer: List[ast.stmt] = []
        self.code.append(outer)
        self.code.append([])
        body([ast.Name(id=n, ctx=ast.Load()) if n else ast.Constant(0) for n in names])
        inner = self.code.pop()
        self.code.pop()
        for n, size in reversed(list(zip(names, shape))):
            if n is None:
                continue                              # (a dimension of 1: no loop)
            inner = [ast.For(target=ast.Name(id=n, ctx=ast.Store()),
                             iter=ast.Call(func=ast.Name(id="range", ctx=ast.Load()),
                                           args=[ast.Constant(size)], keywords=[]),
                             body=inner, orelse=[], type_comment=None)]
        for s in inner:
            self.emit(s)

    def scalar(self, value: ast.expr, base: str = "s") -> Sym:
        """A scalar expression kept in a variable (computed once)."""
        if isinstance(value, (ast.Name, ast.Constant)):
            return Sym(value)
        # The same expression of the same values at the top level: computed once
        # (its names are each assigned once there; a list element may change)
        key = ast.dump(value)
        top = len(self.code) == 1 and not any(isinstance(n, ast.Subscript) for n in ast.walk(value))
        if top and key in self.scalars:
            return self.scalars[key]
        name = self.fresh(base)
        self.emit(ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=value))
        out = Sym(ast.Name(id=name, ctx=ast.Load()))
        if top:
            self.scalars[key] = out
        return out

    # ------------------------------------------------------------ values
    def const_array(self, owner: Any, attr: str, value) -> Arr:
        key = (id(owner), attr)
        if key not in self.table_of:
            base = f"{type(owner).__name__}_{attr}" if owner is not None else attr
            name = f"T_{base}_{len(self.table_of)}"
            arr = self.np.asarray(value, dtype=float)
            self.tables[name] = tuple(float(v) for v in arr.ravel())
            self.ns[name] = self.tables[name]
            from .auto import WEIGHT_TABLES
            self.ns.setdefault(WEIGHT_TABLES, set()).add(name)
            self.table_of[key] = Arr(name, tuple(arr.shape), _contiguous(tuple(arr.shape)),
                                     const=True)
            self.key_of[name] = key
        return self.table_of[key]

    def state_value(self, obj: Any, attr: str, value: Any) -> Any:
        """An attribute that is state: its list (an array) or its value now (a number)."""
        key = (id(obj), attr)
        if key not in self.state:
            name = self.fresh(f"S_{type(obj).__name__}_{attr}_")
            if isinstance(value, self.np.ndarray):
                value = self.np.array(value, dtype=float)
                self.lists[name] = max(1, value.size)
            elif isinstance(value, (int, float, self.np.integer, self.np.floating)) \
                    and not isinstance(value, (bool, self.np.bool_)):
                value = int(value) if isinstance(value, (int, self.np.integer)) else float(value)
                self.lists[name] = 1
            else:
                raise NotTranslatable(f"{type(obj).__name__}.{attr} is changed, but it isn't a "
                                      "number or an array")
            self.state[key] = (name, value, f"{type(obj).__name__}.{attr}", obj)
            self.key_of[name] = key
        name, value, _, _ = self.state[key]
        if isinstance(value, self.np.ndarray):
            shape = tuple(value.shape)
            return Arr(name, shape, _contiguous(shape))
        return self.scalar(ast.Subscript(value=_load(name), slice=ast.Constant(0), ctx=ast.Load()),
                           "st")

    def written(self, target: Arr) -> None:
        """target (a constant table) is written: an attribute changed in place."""
        if self.state_keys is None and target.name in self.key_of:
            self.mutated.add(self.key_of[target.name])
            return
        raise NotTranslatable("A constant array is written")

    def elem(self, v: Any, a_shape: Tuple[int, ...], idx: List[ast.expr],
             emit: bool = True) -> ast.expr:
        """Operand v's element at index idx of a result of shape a_shape (broadcast)."""
        if isinstance(v, Lazy) and v.real is None:
            # Computed here, each operation's result in a variable of its own (a
            # format of its own, as each numpy result would have), once per loop body
            n = len(a_shape) - len(v.shape)
            mine = [idx[n + d] if v.shape[d] != 1 else ast.Constant(0) for d in range(len(v.shape))]
            value = v.fn(*[self.elem(o, v.shape, mine, emit) for o in v.vals])
            if not emit:
                return value
            memo = self.memo.setdefault(id(self.code[-1]), {})
            key = (id(v), tuple(ast.dump(i) for i in mine))
            if key not in memo:
                memo[key] = self.fresh("e")
                self.emit(_assign(memo[key], value))
            return _load(memo[key])
        if isinstance(v, Lazy):
            v = v.real
        if isinstance(v, Arr):
            n = len(a_shape) - len(v.shape)
            mine = [idx[n + d] if v.shape[d] != 1 else ast.Constant(0)
                    for d in range(len(v.shape))]
            return v.at(mine)
        if isinstance(v, Sym):
            return copy.deepcopy(v.expr)
        if isinstance(v, (bool, self.np.bool_)):
            return ast.Constant(int(v))
        if isinstance(v, (int, float, self.np.integer, self.np.floating)):
            return ast.Constant(float(v) if isinstance(v, (float, self.np.floating)) else int(v))
        raise NotTranslatable(f"Can't use {type(v).__name__} in an array expression")

    def to_arr(self, v: Any) -> Any:
        """A numpy array value (not an attribute) as a constant table, an elementwise
        result in a list; others as they are."""
        if isinstance(v, Lazy):
            return v.materialize()
        if isinstance(v, self.np.ndarray):
            return self.const_array(None, f"c{next(self.counter)}", v)
        return v

    def elementwise(self, fn: Callable[..., ast.expr], *vals: Any, base: str = "a") -> Any:
        vals = tuple(v if isinstance(v, Lazy) else self.to_arr(v) for v in vals)
        shapes = [v.shape for v in vals if isinstance(v, Arr)]
        if not shapes:
            return Sym(fn(*[self.elem(v, (), []) for v in vals]))
        return Lazy(self, _bcast(*shapes), fn, vals)

    def reduce(self, a: Arr, axis, keepdims: bool, kind: str) -> Any:
        a = self.to_arr(a)
        nd = len(a.shape)
        axes = tuple(range(nd)) if axis is None else (
            (axis % nd,) if isinstance(axis, int) else tuple(x % nd for x in axis))
        keep = [d for d in range(nd) if d not in axes]
        out_shape = tuple(a.shape[d] for d in keep)
        red_shape = tuple(a.shape[d] for d in axes)
        result = self.new_list(out_shape if out_shape else (1,), "r")

        def body(oidx):
            acc = self.fresh("acc")
            best = self.fresh("arg") if kind == "argmax" else None
            first = True

            def inner(ridx):
                full: List[ast.expr] = [None] * nd
                for d, i in zip(keep, oidx):
                    full[d] = i
                for d, i in zip(axes, ridx):
                    full[d] = i
                x = a.at(full)
                pos = ridx[0] if len(ridx) == 1 else None
                if kind == "sum" or kind == "mean":
                    self.emit(_assign(acc, ast.BinOp(left=_load(acc), op=ast.Add(), right=x)))
                elif kind == "max":
                    self.emit(_assign(acc, ast.Call(func=ast.Name(id="max", ctx=ast.Load()),
                                                    args=[_load(acc), x], keywords=[])))
                elif kind == "argmax":
                    if pos is None:
                        raise NotTranslatable("argmax over several axes is not supported")
                    self.emit(ast.If(test=ast.Compare(left=x, ops=[ast.Gt()],
                                                      comparators=[_load(acc)]),
                                     body=[_assign(acc, copy.deepcopy(x)),
                                           _assign(best, copy.deepcopy(pos))], orelse=[]))
            # The first element starts it (a max / argmax over all, a sum from 0)
            zero = [ast.Constant(0)] * len(axes)
            full0: List[ast.expr] = [None] * nd
            for d, i in zip(keep, oidx):
                full0[d] = i
            for d, i in zip(axes, zero):
                full0[d] = i
            self.emit(_assign(acc, ast.Constant(0) if kind in ("sum", "mean") else a.at(full0)))
            if best:
                self.emit(_assign(best, ast.Constant(0)))
            self.loops(red_shape, inner)
            value = _load(best) if best else _load(acc)
            if kind == "mean":
                value = ast.BinOp(left=value, op=ast.Div(), right=ast.Constant(math.prod(red_shape)))
            self.emit(ast.Assign(targets=[_store(result.at(oidx if out_shape else [ast.Constant(0)]))],
                                 value=value))
        self.loops(out_shape, body)
        if keepdims:
            shape = tuple(1 if d in axes else a.shape[d] for d in range(nd))
            strides = []
            k = 0
            for d in range(nd):
                if d in axes:
                    strides.append(0)
                else:
                    strides.append(result.strides[k])
                    k += 1
            return result.view(shape, strides)
        if not out_shape:
            return Sym(result.at([ast.Constant(0)]))
        return result

    def matmul(self, a: Any, b: Any) -> Arr:
        a, b = self.to_arr(a), self.to_arr(b)
        va = a if len(a.shape) == 2 else a.view((1,) + a.shape, (0,) + a.strides)
        vb = b if len(b.shape) == 2 else b.view(b.shape + (1,), b.strides + (0,))
        n, k = va.shape
        k2, m = vb.shape
        if k != k2:
            raise NotTranslatable(f"Shapes {a.shape} @ {b.shape} don't match")
        out = self.new_list((n, m), "mm")

        def body(idx):
            i, j = idx
            acc = self.fresh("acc")
            self.emit(_assign(acc, ast.Constant(0)))

            def inner(t):
                self.emit(_assign(acc, ast.BinOp(
                    left=_load(acc), op=ast.Add(),
                    right=ast.BinOp(left=va.at([copy.deepcopy(i), t[0]]), op=ast.Mult(),
                                    right=vb.at([t[0], copy.deepcopy(j)])))))
            self.loops((k,), inner)
            self.emit(ast.Assign(targets=[_store(out.at([i, j]))], value=_load(acc)))
        self.loops((n, m), body)
        shape = ((n,) if len(a.shape) == 2 else ()) + ((m,) if len(b.shape) == 2 else ())
        return out.view(shape, _contiguous(shape)) if shape != (n, m) else out

    # ------------------------------------------------------------ running Python
    def call(self, fn: Callable, args: List[Any], kwargs: Dict[str, Any], obj: Any = None) -> Any:
        src = textwrap.dedent(inspect.getsource(fn))
        fd = ast.parse(src).body[0]
        sig = inspect.signature(fn)
        bound = sig.bind(*([obj] if obj is not None else []), *args, **kwargs)
        bound.apply_defaults()
        frame = dict(bound.arguments)
        g = fn.__globals__
        self.frames.append(frame)
        try:
            self.block(fd.body, frame, g)
        except _Return as r:
            return r.value
        finally:
            self.frames.pop()
        return None

    def block(self, stmts, frame, g) -> None:
        for s in stmts:
            self.stmt(s, frame, g)

    def stmt(self, s: ast.stmt, frame, g) -> None:
        if isinstance(s, ast.Expr):
            if isinstance(s.value, ast.Constant):
                return
            self.expr(s.value, frame, g)
            return
        if isinstance(s, ast.Pass):
            return
        if isinstance(s, ast.Return):
            raise _Return(self.expr(s.value, frame, g) if s.value is not None else None)
        if isinstance(s, ast.Assign):
            value = self.expr(s.value, frame, g)
            for t in s.targets:
                self.store(t, value, frame, g)
            return
        if isinstance(s, ast.AugAssign):
            self.augassign(s, frame, g)
            return
        if isinstance(s, ast.For):
            items = self.expr(s.iter, frame, g)
            if isinstance(items, (Arr, Sym)):
                raise NotTranslatable(f"A loop over an array value (line {s.lineno})")
            for item in list(items):
                self.store(s.target, item, frame, g)
                self.block(s.body, frame, g)
            return
        if isinstance(s, ast.If):
            test = self.expr(s.test, frame, g)
            if isinstance(test, (Arr, Sym)):
                raise NotTranslatable(f"An if on a value computed in hardware (line {s.lineno})")
            self.block(s.body if test else s.orelse, frame, g)
            return
        raise NotTranslatable(f"Unsupported statement (line {s.lineno}): {ast.unparse(s)}")

    def store(self, t: ast.expr, value: Any, frame, g) -> None:
        if isinstance(t, ast.Name):
            if isinstance(value, Sym) and not isinstance(value.expr, (ast.Name, ast.Constant)):
                value = self.scalar(value.expr, t.id + "_")
            frame[t.id] = value
            return
        if isinstance(t, ast.Attribute):
            obj = self.expr(t.value, frame, g)
            key = (id(obj), t.attr)
            if self.first.setdefault(key, "w") == "r" and self.state_keys is None \
                    and t.attr not in self.shadow.get(id(obj), {}):
                self.mutated.add(key)                 # (read, then changed: state)
            if isinstance(value, Sym) and not isinstance(value.expr, (ast.Name, ast.Constant)):
                value = self.scalar(value.expr, t.attr + "_")
            self.shadow.setdefault(id(obj), {})[t.attr] = value
            return
        if isinstance(t, (ast.Tuple, ast.List)):
            vals = list(value)
            for tt, v in zip(t.elts, vals):
                self.store(tt, v, frame, g)
            return
        if isinstance(t, ast.Subscript):
            target = self.expr(t, frame, g)
            if isinstance(target, Arr):
                self.copy_into(target, value)
                return
        raise NotTranslatable(f"Unsupported assignment to {ast.unparse(t)}")

    def copy_into(self, target: Arr, value: Any) -> None:
        if target.const:
            self.written(target)
        if isinstance(target, Lazy):
            target = target.materialize()
        value = value if isinstance(value, Lazy) else self.to_arr(value)
        if isinstance(value, Arr) and not self.aligned(value, target):
            # (it reads the list elsewhere than where it writes: from a copy)
            value = value.materialize() if isinstance(value, Lazy) else self.method(value, "copy",
                                                                                     [], {})
        for lz in self.held():                      # (values from before the write)
            if lz.real is None and target.name in lz.reads():
                lz.materialize()

        def body(idx):
            self.emit(ast.Assign(targets=[_store(target.at(idx))],
                                 value=self.elem(value, target.shape, idx)))
        self.loops(target.shape, body)

    def held(self) -> List[Lazy]:
        """The elementwise results in the running functions' variables and the objects'
        attributes (the others are parts of an expression being computed)."""
        out: Dict[int, Lazy] = {}

        def visit(v, depth=0):
            if isinstance(v, Lazy):
                out[id(v)] = v
            elif isinstance(v, (list, tuple)) and depth < 4:
                for x in v:
                    visit(x, depth + 1)
            elif isinstance(v, dict) and depth < 4:
                for x in v.values():
                    visit(x, depth + 1)
            elif isinstance(v, _Method):
                visit(v.arr, depth + 1)
        for f in self.frames:
            visit(list(f.values()))
        for sh in self.shadow.values():
            visit(list(sh.values()))
        return list(out.values())

    def aligned(self, value: Arr, target: Arr) -> bool:
        """Does value read target's list only at the element written (target[idx] = value[idx])?"""
        if not (isinstance(value, Lazy) and value.real is None) and value.name != target.name:
            return True
        idx = [ast.Name(id=f"__i{d}", ctx=ast.Load()) for d in range(len(target.shape))]
        here = ast.dump(target.at(idx).slice)
        return all(ast.dump(n.slice) == here
                   for n in ast.walk(self.elem(value, target.shape, idx, emit=False))
                   if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
                   and n.value.id == target.name)

    def augassign(self, s: ast.AugAssign, frame, g) -> None:
        cur = self.expr(s.target, frame, g)
        rhs = self.expr(s.value, frame, g)
        op = _BINOPS.get(type(s.op))
        if op is None:
            raise NotTranslatable(f"Unsupported operator in {ast.unparse(s)}")
        if isinstance(cur, Arr):                  # in place, as numpy does
            if cur.const:
                self.written(cur)
            new = self.binop(s.op, cur, rhs)
            self.copy_into(cur, new)
            return
        self.store(s.target, self.binop(s.op, cur, rhs), frame, g)

    # ------------------------------------------------------------ expressions
    def expr(self, n: ast.expr, frame, g) -> Any:
        np = self.np
        if isinstance(n, ast.Constant):
            return n.value
        if isinstance(n, ast.Name):
            if n.id in frame:
                return frame[n.id]
            if n.id in g:
                return g[n.id]
            if hasattr(builtins, n.id):
                return getattr(builtins, n.id)
            raise NotTranslatable(f"Unknown name '{n.id}'")
        if isinstance(n, ast.Attribute):
            obj = self.expr(n.value, frame, g)
            if isinstance(obj, Arr):
                return self.arr_attr(obj, n.attr)
            sh = self.shadow.get(id(obj), {})
            if n.attr in sh:
                return sh[n.attr]
            key = (id(obj), n.attr)
            self.first.setdefault(key, "r")
            value = getattr(obj, n.attr)
            if self.state_keys is not None and key in self.state_keys:
                return self.state_value(obj, n.attr, value)
            if isinstance(value, np.ndarray) and value.dtype.kind in "fiub":
                return self.const_array(obj, n.attr, value)
            return value
        if isinstance(n, (ast.Tuple, ast.List)):
            vals = [self.expr(e, frame, g) for e in n.elts]
            return tuple(vals) if isinstance(n, ast.Tuple) else vals
        if isinstance(n, ast.BinOp):
            return self.binop(n.op, self.expr(n.left, frame, g), self.expr(n.right, frame, g))
        if isinstance(n, ast.UnaryOp):
            v = self.expr(n.operand, frame, g)
            if isinstance(v, (Arr, Sym)):
                if isinstance(n.op, ast.USub):
                    return self.elementwise(lambda x: ast.UnaryOp(op=ast.USub(), operand=x), v)
                raise NotTranslatable(f"Unsupported operator in {ast.unparse(n)}")
            return {ast.USub: lambda x: -x, ast.UAdd: lambda x: +x, ast.Not: lambda x: not x,
                    ast.Invert: lambda x: ~x}[type(n.op)](v)
        if isinstance(n, ast.Compare) and len(n.ops) == 1:
            a, b = self.expr(n.left, frame, g), self.expr(n.comparators[0], frame, g)
            op = n.ops[0]
            if isinstance(a, (Arr, Sym)) or isinstance(b, (Arr, Sym)):
                if type(op) not in _CMPS:
                    raise NotTranslatable(f"Unsupported comparison in {ast.unparse(n)}")
                return self.elementwise(lambda x, y: ast.Compare(left=x, ops=[type(op)()],
                                                                 comparators=[y]), a, b, base="m")
            return {ast.Is: lambda x, y: x is y, ast.IsNot: lambda x, y: x is not y,
                    ast.In: lambda x, y: x in y, ast.NotIn: lambda x, y: x not in y,
                    **_CMPS}[type(op)](a, b)
        if isinstance(n, ast.BoolOp):
            vals = [self.expr(v, frame, g) for v in n.values]
            if any(isinstance(v, (Arr, Sym)) for v in vals):
                raise NotTranslatable(f"and / or on values computed in hardware: {ast.unparse(n)}")
            return all(vals) if isinstance(n.op, ast.And) else any(vals)
        if isinstance(n, ast.IfExp):
            test = self.expr(n.test, frame, g)
            if isinstance(test, (Arr, Sym)):
                raise NotTranslatable(f"A conditional on a computed value: {ast.unparse(n)}")
            return self.expr(n.body if test else n.orelse, frame, g)
        if isinstance(n, ast.Subscript):
            base = self.expr(n.value, frame, g)
            if isinstance(base, Arr):
                return self.index(base, n.slice, frame, g)
            key = self.expr(n.slice, frame, g)
            return base[key]
        if isinstance(n, ast.Slice):
            return slice(*(self.expr(x, frame, g) if x is not None else None
                           for x in (n.lower, n.upper, n.step)))
        if isinstance(n, ast.Call):
            return self.call_expr(n, frame, g)
        if isinstance(n, ast.JoinedStr):
            return ""
        raise NotTranslatable(f"Unsupported expression: {ast.unparse(n)}")

    def arr_attr(self, a: Arr, attr: str) -> Any:
        if attr == "T":
            return a.view(a.shape[::-1], a.strides[::-1])
        if attr == "shape":
            return a.shape
        if attr == "size":
            return a.size
        if attr == "ndim":
            return len(a.shape)
        return _Method(a, attr)

    def index(self, a: Arr, sl: ast.expr, frame, g) -> Any:
        parts = sl.elts if isinstance(sl, ast.Tuple) else [sl]
        shape, strides, offset = [], [], a.offset
        d = 0
        idx_exprs: List[ast.expr] = []
        for p in parts:
            v = self.expr(p, frame, g)
            if isinstance(v, slice):
                start, stop, step = v.indices(a.shape[d])
                shape.append(len(range(start, stop, step)))
                strides.append(a.strides[d] * step)
                offset += start * a.strides[d]
            elif isinstance(v, int):
                offset += (v % a.shape[d]) * a.strides[d]
            else:
                raise NotTranslatable(f"Index with a value computed in hardware: {ast.unparse(sl)}")
            d += 1
        shape += list(a.shape[d:])
        strides += list(a.strides[d:])
        if not shape:
            return Sym(Arr(a.name, (1,), (1,), offset, a.const).at([ast.Constant(0)]))
        return Arr(a.name, tuple(shape), tuple(strides), offset, a.const)

    def binop(self, op: ast.operator, a: Any, b: Any) -> Any:
        if isinstance(op, ast.MatMult):
            return self.matmul(a, b)
        sym = any(isinstance(v, (Arr, Sym)) for v in (a, b))
        if not sym:
            if isinstance(op, ast.MatMult):
                return a @ b
            return _BINOPS[type(op)](a, b)
        if type(op) not in _BINOPS:
            raise NotTranslatable("Unsupported operator")
        if isinstance(op, ast.Pow) and isinstance(b, Sym):
            if not isinstance(a, (int, float, self.np.integer, self.np.floating)) \
                    or isinstance(a, bool):
                raise NotTranslatable("x ** n with a computed n needs a constant x")
            return self.power(float(a), b)
        if isinstance(op, ast.Div) and isinstance(b, Sym) and isinstance(self.to_arr(a), Arr) \
                and not isinstance(b.expr, ast.Constant):
            # An array by a computed scalar: by its reciprocal, computed once
            inv = self.scalar(ast.BinOp(left=ast.Constant(1.0), op=ast.Div(),
                                        right=copy.deepcopy(b.expr)), "inv")
            return self.elementwise(lambda x, y: ast.BinOp(left=x, op=ast.Mult(), right=y), a, inv)
        if isinstance(op, ast.Pow) and isinstance(b, int) and not isinstance(a, int):
            k = b                                    # x ** k: products
            if k < 1:
                raise NotTranslatable("x ** k with k < 1 is not supported")

            def power(x):
                out = x
                for _ in range(k - 1):
                    out = ast.BinOp(left=out, op=ast.Mult(), right=copy.deepcopy(x))
                return out
            return self.elementwise(power, a)
        return self.elementwise(lambda x, y: ast.BinOp(left=x, op=type(op)(), right=y), a, b)

    def power(self, c: float, n: Sym) -> Sym:
        """c ** n for an integer n computed in hardware (n >= 0, below 2 ** POW_BITS):
        the product of c ** 2 ** k over n's bits k (constants)."""
        if not -1.0 <= c <= 1.0:
            raise NotTranslatable(f"{c} ** n with a computed n: only |c| <= 1 is supported")
        key = (c, ast.dump(n.expr))
        if key in self.pows and len(self.code) == 1:
            return self.pows[key]
        table = self.const_array(None, f"pow{len(self.pows)}",
                                 self.np.array([c ** (2 ** k) for k in range(POW_BITS)]))
        p, k = self.fresh("pw"), self.fresh("k")
        self.emit(_assign(p, ast.Constant(1.0)))
        bit = ast.BinOp(left=ast.BinOp(left=copy.deepcopy(n.expr), op=ast.RShift(), right=_load(k)),
                        op=ast.BitAnd(), right=ast.Constant(1))
        self.emit(ast.For(target=ast.Name(id=k, ctx=ast.Store()),
                          iter=ast.Call(func=_load("range"), args=[ast.Constant(POW_BITS)],
                                        keywords=[]),
                          body=[ast.If(test=bit, body=[_assign(p, ast.BinOp(
                              left=_load(p), op=ast.Mult(), right=table.at([_load(k)])))],
                                       orelse=[])], orelse=[], type_comment=None))
        out = Sym(_load(p))
        if len(self.code) == 1:
            self.pows[key] = out
        return out

    def call_expr(self, n: ast.Call, frame, g) -> Any:
        np = self.np
        fn = self.expr(n.func, frame, g)
        args = [self.expr(a, frame, g) for a in n.args]
        kwargs = {k.arg: self.expr(k.value, frame, g) for k in n.keywords}
        if isinstance(fn, _Method):
            return self.method(fn.arr, fn.name, args, kwargs)
        sym = any(isinstance(v, (Arr, Sym)) for v in list(args) + list(kwargs.values()))
        if getattr(fn, "__module__", None) == "numpy" or (
                isinstance(fn, np.ufunc)):
            name = getattr(fn, "__name__", "")
            if not sym and name not in ("zeros", "ones", "zeros_like", "ones_like"):
                return fn(*args, **kwargs)
            return self.numpy(name, args, kwargs)
        if fn is isinstance:
            return isinstance(*args)
        if fn in (len,):
            a = args[0]
            return a.shape[0] if isinstance(a, Arr) else len(a)
        if fn in (abs,) and sym:
            return self.elementwise(lambda x: ast.Call(func=ast.Name(id="abs", ctx=ast.Load()),
                                                       args=[x], keywords=[]), args[0])
        if fn in (print,):
            return None
        if inspect.ismethod(fn) and not getattr(fn.__self__, "__module__", "").startswith("numpy"):
            return self.call(fn.__func__, args, kwargs, obj=fn.__self__)
        if inspect.isfunction(fn):
            return self.call(fn, args, kwargs)
        if sym:
            raise NotTranslatable(f"Unsupported call on values computed in hardware: "
                                  f"{ast.unparse(n)}")
        return fn(*args, **kwargs)                   # plain Python (range, reversed, ...)

    def method(self, a: Arr, name: str, args, kwargs) -> Any:
        if name in ("sum", "max", "mean", "argmax", "min"):
            axis = kwargs.get("axis", args[0] if args else None)
            keep = bool(kwargs.get("keepdims", False))
            if name == "min":
                neg = self.elementwise(lambda x: ast.UnaryOp(op=ast.USub(), operand=x), a)
                return self.elementwise(lambda x: ast.UnaryOp(op=ast.USub(), operand=x),
                                        self.reduce(neg, axis, keep, "max"))
            return self.reduce(a, axis, keep, name)
        if name == "reshape":
            shape = args[0] if len(args) == 1 and isinstance(args[0], tuple) else tuple(args)
            if a.strides != _contiguous(a.shape):
                raise NotTranslatable("reshape of a view that isn't contiguous")
            shape = tuple(a.size // -math.prod(shape) * -1 if s == -1 else s for s in shape)
            return a.view(shape, _contiguous(shape))
        if name == "copy":
            out = self.new_list(a.shape)
            self.copy_into(out, a)
            return out
        raise NotTranslatable(f"Unsupported array method .{name}()")

    def numpy(self, name: str, args, kwargs) -> Any:
        np = self.np
        if name in TABLE_UFUNCS:
            return self.elementwise(lambda x: ast.Call(
                func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr=name,
                                   ctx=ast.Load()), args=[x], keywords=[]), args[0])
        if name in ("maximum", "minimum"):
            fn = "max" if name == "maximum" else "min"
            return self.elementwise(lambda x, y: ast.Call(func=ast.Name(id=fn, ctx=ast.Load()),
                                                          args=[x, y], keywords=[]), *args[:2])
        if name in ("sum", "max", "mean", "argmax", "amax"):
            return self.method(self.to_arr(args[0]), "max" if name == "amax" else name,
                               list(args[1:]), kwargs)
        if name == "atleast_2d":
            a = self.to_arr(args[0])
            if isinstance(a, Arr) and len(a.shape) == 1:
                return a.view((1,) + a.shape, (0,) + a.strides)
            return a
        if name == "abs":
            return self.elementwise(lambda x: ast.Call(func=ast.Name(id="abs", ctx=ast.Load()),
                                                       args=[x], keywords=[]), args[0])
        if name in ("zeros", "ones", "zeros_like", "ones_like"):
            shape = args[0].shape if name.endswith("like") else (
                tuple(args[0]) if isinstance(args[0], (tuple, list)) else (args[0],))
            out = self.new_list(tuple(shape))
            if name.startswith("ones"):
                self.copy_into(out, 1.0)
            return out
        raise NotTranslatable(f"np.{name} is not supported")


class _Method:
    def __init__(self, arr: Arr, name: str) -> None:
        self.arr, self.name = arr, name


def _load(n: str) -> ast.Name:
    return ast.Name(id=n, ctx=ast.Load())


def _assign(n: str, v: ast.expr) -> ast.Assign:
    return ast.Assign(targets=[ast.Name(id=n, ctx=ast.Store())], value=v)


def _store(sub: ast.Subscript) -> ast.Subscript:
    sub = copy.deepcopy(sub)
    sub.ctx = ast.Store()
    return sub


# ==================================================================== translate

def translate(func: Callable, samples: List[Any]) -> Translation:
    """The method (or function) as a loop program on flat lists; see the module doc."""
    np = _np()
    if not samples:
        raise NotTranslatable("A numpy design needs example inputs (samples=...) for its shapes")
    obj = func.__self__ if inspect.ismethod(func) else None
    fn = func.__func__ if obj is not None else func
    sig = inspect.signature(func)
    first = samples[0] if isinstance(samples[0], tuple) else (samples[0],)
    names = list(sig.parameters)[:len(first)]
    shapes = {p: tuple(np.shape(v)) for p, v in zip(names, first)}

    def run(state_keys):
        ns: Dict[str, Any] = {"np": np, EXACT_SQRT: True}
        it = _Interp(ns, state_keys)
        args = []
        for p in names:
            if shapes[p] == ():
                args.append(Sym(ast.Name(id=p, ctx=ast.Load())))
            else:
                it.lists[p] = math.prod(shapes[p])
                args.append(Arr(p, shapes[p], _contiguous(shapes[p])))
        return it, it.call(fn, args, {}, obj=obj)

    # First to find the state (attributes read, then changed), then to build it
    it, _ = run(None)
    if it.mutated:
        it, result = run(set(it.mutated))
    else:
        it, result = run(set())
    ns = it.ns
    # The state's new values, where they were reassigned (not changed in place)
    for key, (name, value, _, _) in it.state.items():
        new = it.shadow.get(key[0], {}).get(key[1])
        if new is None:
            continue
        if isinstance(value, np.ndarray):
            arr = Arr(name, tuple(value.shape), _contiguous(tuple(value.shape)))
            if (isinstance(new, Lazy) and new.real is None) or not (
                    isinstance(new, Arr) and (new.name, new.shape, new.strides, new.offset)
                    == (arr.name, arr.shape, arr.strides, arr.offset)):
                it.copy_into(arr, new)
        else:
            it.emit(ast.Assign(targets=[ast.Subscript(value=_load(name), slice=ast.Constant(0),
                                                      ctx=ast.Store())],
                               value=it.elem(new, (), [])))
    # The results: arrays as flat lists, scalars as they are
    outs, ret = [], []
    for r in (result if isinstance(result, tuple) else (result,)):
        if isinstance(r, Arr):
            out = it.new_list(r.shape, "out")
            it.copy_into(out, r)
            outs.append((out.name, r.shape))
            ret.append(ast.Name(id=out.name, ctx=ast.Load()))
        elif isinstance(r, Sym):
            v = it.scalar(r.expr, "out")
            outs.append((ast.unparse(v.expr), ()))
            ret.append(v.expr)
        else:
            raise NotTranslatable(f"The result {r!r} isn't computed in hardware")
    state_lists = [name for name, _, _, _ in it.state.values()]
    roots = {n for n, _ in outs} | set(state_lists)
    body = _live(it.code[0], roots) + [ast.Return(value=ret[0] if len(ret) == 1 else
                                                  ast.Tuple(elts=ret, ctx=ast.Load()))]
    gen_name = f"{fn.__name__}_loops"
    params = names + state_lists
    fdef = ast.FunctionDef(name=gen_name, args=ast.arguments(
        posonlyargs=[], args=[ast.arg(arg=p) for p in params], vararg=None, kwonlyargs=[],
        kw_defaults=[], kwarg=None, defaults=[]), body=body, decorator_list=[], returns=None,
        type_comment=None, type_params=[])
    module = ast.fix_missing_locations(ast.Module(body=[fdef], type_ignores=[]))
    source = ast.unparse(module) + "\n"
    filename = f"<py2ver numpy loops {fn.__qualname__}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    exec(compile(source, filename, "exec"), ns)
    gen = ns[gen_name]
    attr = {p: {"length": math.prod(shapes[p])} for p in names if shapes[p] != ()}
    memories: Dict[str, Tuple[str, Any, Any]] = {}
    for name, value, label, owner in it.state.values():
        flat = (np.ravel(value).tolist() if isinstance(value, np.ndarray) else [value])
        attr[name] = {"length": len(flat), "memory": "ram", "init": flat}
        if not isinstance(value, np.ndarray) and isinstance(value, int):
            attr[name].update(signed=1, width=32)      # (a counter: an integer)
        memories[name] = (label, value, owner)
    tr = Translation(func=gen, source=source, attr=attr, params=names, shapes=shapes,
                     outputs=outs)
    tr.loops = True
    tr.tables = set()
    tr.const_tables = dict(it.tables)
    tr.memories = memories
    _check(tr, func, samples)
    return tr


def _stored(s: ast.stmt) -> set:
    out = set()
    for n in ast.walk(s):
        for t in (n.targets if isinstance(n, ast.Assign) else
                  [n.target] if isinstance(n, (ast.AugAssign, ast.For)) else []):
            for x in ast.walk(t):
                if isinstance(x, ast.Name):
                    out.add(x.id)
    return out


def _live(stmts: List[ast.stmt], roots: set) -> List[ast.stmt]:
    """The statements something needs: what the roots depend on (the rest, e.g. a
    gradient for a layer before the first, computed and never used, is left out)."""
    live, keep = set(roots), []
    for s in reversed(stmts):
        if _stored(s) & live:
            keep.append(s)
            live |= {n.id for n in ast.walk(s) if isinstance(n, ast.Name)}
    return keep[::-1]


def _check(tr: Translation, func: Callable, samples: List[Any]) -> None:
    """The program in float against the original on a few examples (with state:
    on a copy of the object, one example after another)."""
    np = _np()
    obj = func.__self__ if inspect.ismethod(func) else None
    memo: Dict[int, Any] = {}
    if obj is not None:
        func = getattr(copy.deepcopy(obj, memo), func.__name__)
    state = {name: list(tr.attr[name]["init"]) for name in tr.memories}
    for s in samples[:2 if tr.memories else 3]:
        args = s if isinstance(s, tuple) else (s,)
        want = np.ravel(np.asarray(func(*args), dtype=float))
        got = tr.func(*tr.adapt(args), *state.values())
        got = np.ravel(np.asarray([x for g in (got if isinstance(got, tuple) else (got,))
                                   for x in (g if isinstance(g, list) else [g])], dtype=float))
        if want.shape != got.shape or not np.allclose(want, got, rtol=1e-6, atol=1e-9):
            raise NotTranslatable(f"The loop translation of {getattr(func, '__name__', func)} "
                                  f"disagrees with it on an example: {got[:6]} vs {want[:6]}")
        for name, (label, _, owner) in tr.memories.items():
            value = getattr(memo.get(id(owner), owner), label.split(".")[1])
            if not np.allclose(np.ravel(np.asarray(value, dtype=float)), state[name],
                               rtol=1e-6, atol=1e-9):
                raise NotTranslatable(f"The loop translation of {func.__name__} changes "
                                      f"{label} differently from the original")
