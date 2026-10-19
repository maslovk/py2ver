# numpy_loops.py
"""
numpy code as loops: for designs too big to write out element by element.

numpy_fe.py writes every element of every array out as its own expression,
which suits a small network. A real one (a 64 -> 128 -> 64 -> 10 network is
17,000 multiply-accumulates per input) needs its loops kept, for the
sequencer (sequencer.py) to run them a step at a time. This translator runs
the method at translation time, the way Python would, keeping the arrays
symbolic:

  - Python values are real: the object (a copy of it), its lists of layers,
    isinstance(), hasattr(), loops over lists (break, continue), ifs on them,
    comprehensions, dicts, the methods called (inlined, each with its own self)
  - an array is a view (shape, strides, offset) of a flat list in the program
    or of a constant table: an attribute read and never written (trained
    weights) is a table, an input a list argument; .T, .transpose(),
    np.atleast_2d, sliding_window_view, constant slices and indices with None
    and ... are views, so they cost nothing; .reshape of a view in another
    order copies it
  - each numpy operation becomes a loop nest writing a new list: @ (sums of
    products), + - * / ** and comparisons with broadcasting, np.exp / np.log /
    np.tanh / ... (tables), np.sqrt (exact), .sum / .max / .mean / .argmax
    with axis (or axes) and keepdims, np.maximum / np.minimum, abs, np.pad
    (zeros), np.concatenate; dividing an array by a computed scalar multiplies
    by its reciprocal, and c ** n with a computed integer n multiplies the
    constants c ** 2 ** k for n's bits
  - indexing with arrays of integers (np.arange, labels given as an input)
    reads the elements they pick, and assigning to it (a[rows, labels] -= 1)
    writes them, one after another
  - an attribute the method reads and then changes (the weights and the
    optimiser's moments in a training step, in place or reassigned; a step
    counter) is state: a list argument of the program that it updates in
    place ('memory' in attr, its 'init' the object's values), kept between
    calls by the hardware. So is an array the object holds in a dict or a
    list that the method changes, and one it makes with np.zeros / np.ones
    and Python code it calls keeps in the object (Adam's moments by
    dict.setdefault on the first step: state from zeros)
  - a numpy Generator's .random(size) draws from a generator in hardware
    (xorshift32, its state kept between calls), not numpy's stream: the check
    runs the original with the same numbers in its place (_drawing)
  - work whose result nothing uses is left out

A function whose body loops over an argument given as batches (a training
epoch, train(model, loader, optimizer, ...)) is built as one step of that
loop (loop_step): its objects copied together, the loop's target the inputs,
the locals it changes state and results. PyTorch code (torch_ops.py,
autograd.py, torch_optim.py) runs in the same interpreter, with torch's
meanings for tensors.

The program is plain Python on flat lists, checked against the original in
float on the examples when it is made (on a copy of the object; a loop step
against the original loop over the first two batches).
"""

from __future__ import annotations

import ast
import builtins
import copy
import inspect
import itertools
import linecache
import math
import re
import textwrap
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .fixed import EXACT_SQRT
from .numpy_fe import TABLE_UFUNCS, NotTranslatable, Translation, _np

POW_BITS = 32                            # bits of an integer exponent (c ** n)
RANDOM_STATE = "random_state"            # a Generator's state in hardware (its attribute name)
RANDOM_BITS = 24                         # bits of a uniform number from Generator.random


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
                 vals: Tuple[Any, ...], extra: Tuple[Arr, ...] = (),
                 op: Optional[Tuple] = None) -> None:
        self.it, self.shape, self.fn, self.vals = it, tuple(shape), fn, vals
        self.op = op                           # (what fn computes, for autograd: ('add',) ...)
        self.extra = extra                     # (lists fn reads besides vals: a gather's)
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
            self.it.record("copy", out, self)
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
        for v in tuple(self.vals) + tuple(self.extra):
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
    op: Optional[Tuple] = None                 # (an elementwise result's: see Lazy)
    vals: Tuple[Any, ...] = ()
    arr: Optional["Arr"] = None                # (a list's element: that element, a view)


class _Return(Exception):
    def __init__(self, value: Any) -> None:
        self.value = value


class _Break(Exception):
    pass


class _Continue(Exception):
    pass


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
        self.generators: Dict[int, Any] = {}            # numpy Generators drawn from
        self.real_of: Dict[str, Any] = {}               # table / state list -> its ndarray
        self.fills: Dict[str, float] = {}               # list -> the value it was made with
        self.dirty: set = set()                         # lists written since they were made
        self.handed: Dict[int, Optional[float]] = {}    # Arr given to Python code -> its fill
        self.torch = False                              # tensors: torch's meanings
        # Autograd (see autograd.py): the operations on values that need gradients
        self.tape: List[Tuple[str, Any, Tuple[Any, ...], Dict[str, Any]]] = []
        self.req_lists: set = set()                     # lists whose values need gradients
        self.recording = True
        self.grads: Dict[str, Arr] = {}                 # parameter list -> its gradient (.grad)
        # numbers held in dicts that are state: key -> (the dict, the steps to them)
        self.item_paths: Dict[Tuple[int, str], Tuple[Any, Tuple]] = {}

    # ------------------------------------------------------------ program text
    def fresh(self, base: str) -> str:
        return f"{base}{next(self.counter)}"

    def emit(self, s: ast.stmt) -> None:
        self.code[-1].append(s)

    def new_list(self, shape: Tuple[int, ...], base: str = "a") -> Arr:
        name = self.fresh(base)
        n = math.prod(shape)
        self.lists[name] = n
        self.fills[name] = 0.0
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

    def rename(self, value: Sym, base: str) -> Sym:
        """A scalar expression kept in a variable (autograd: the same value)."""
        if not isinstance(value, Sym):
            return value                              # (a Python number)
        if value.arr is not None and isinstance(value.expr, ast.Subscript):
            return value                              # (a list's element: kept there)
        out = self.scalar(value.expr, base)
        if out is not value:
            out = Sym(out.expr)
            self.record("rename", out, value)
        return out

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
    def const_array(self, owner: Any, attr: str, value, hint: Optional[str] = None) -> Arr:
        key = (id(owner), attr)
        if key not in self.table_of:
            base = hint or (f"{type(owner).__name__}_{attr}" if owner is not None else attr)
            name = f"T_{base}_{len(self.table_of)}"
            arr = _values(value)
            self.tables[name] = tuple(float(v) for v in arr.ravel())
            self.ns[name] = self.tables[name]
            from .auto import WEIGHT_TABLES
            self.ns.setdefault(WEIGHT_TABLES, set()).add(name)
            self.table_of[key] = Arr(name, tuple(arr.shape), _contiguous(tuple(arr.shape)),
                                     const=True)
            self.key_of[name] = key
            self.real_of[name] = value
        return self.table_of[key]

    def state_value(self, obj: Any, attr: str, value: Any, hint: Optional[str] = None) -> Any:
        """An attribute that is state: its list (an array) or its value now (a number)."""
        key = (id(obj), attr)
        if key not in self.state:
            name = self.fresh(re.sub(r"\W+", "_", f"S_{hint or type(obj).__name__ + '_' + attr}_"))
            self.real_of[name] = value
            if _is_array(value):
                value = _values(value)
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

    def adopt(self, v: Any, hint: Optional[str] = None) -> Any:
        """An array reached in Python values (an attribute, a dict's or a list's item; a
        numpy array or a torch tensor): a table, or state if the method changes it.
        Arrays are known by their identity, however they are reached."""
        if _is_array(v):
            key = (id(v), "")
            if self.state_keys is not None and key in self.state_keys:
                out = self.state_value(v, "", v, hint)
            else:
                out = self.const_array(v, "", v, hint)
            if self.torch and getattr(v, "requires_grad", False):
                self.req_lists.add(out.name)          # (a parameter: autograd's leaf)
            return out
        return v

    def requires(self, v: Any) -> bool:
        """Does v depend on a value that needs gradients (a torch parameter)?"""
        if isinstance(v, Lazy):
            return getattr(v, "req", False) or (v.real is not None and v.real.name in self.req_lists)
        if isinstance(v, Arr):
            return v.name in self.req_lists
        if isinstance(v, Sym):
            return getattr(v, "req", False) or (v.arr is not None and v.arr.name in self.req_lists)
        return False

    def record(self, what: str, out: Any, *ins: Any, **info: Any) -> None:
        """An operation for autograd, if one of its inputs needs gradients."""
        if not self.recording or not any(self.requires(v) for v in ins):
            return
        if isinstance(out, (Lazy, Sym)):
            out.req = True
        if isinstance(out, Arr) and not (isinstance(out, Lazy) and out.real is None):
            self.req_lists.add(out.name)
        self.tape.append((what, out, ins, info))

    # ------------------------------------------------------------ numbers as state
    def _number_state(self, key, obj, attr: str, value: Any) -> Any:
        if self.state_keys is not None and key in self.state_keys:
            return self.state_value(obj, attr, value)
        return value

    def _set_number(self, key, expr: ast.expr) -> Sym:
        if self.state_keys is None:
            self.mutated.add(key)
        if key in self.state:
            name = self.state[key][0]
            self.emit(ast.Assign(targets=[ast.Subscript(value=_load(name), slice=ast.Constant(0),
                                                        ctx=ast.Store())], value=expr))
            return Sym(ast.Subscript(value=_load(name), slice=ast.Constant(0), ctx=ast.Load()))
        return Sym(expr)

    def attr_value(self, obj: Any, attr: str) -> Any:
        """A number attribute that Python code (a scheduler's step) changes: state."""
        return self._number_state((id(obj), attr), obj, attr, getattr(obj, attr))

    def set_attr(self, obj: Any, attr: str, expr: ast.expr) -> Sym:
        return self._set_number((id(obj), attr), expr)

    def int_value(self, t: Any) -> Any:
        """An integer held in a tensor (a step count): state, an integer."""
        value = int(t.item()) if _is_tensor(t) else int(t)
        return self._number_state((id(t), ""), t, "", value)

    def set_int(self, t: Any, expr: ast.expr) -> Sym:
        return self._set_number((id(t), ""), expr)

    def item_value(self, container: Any, key: Any, sub: Optional[int] = None) -> Any:
        """A number in a dict (an optimizer's group['lr'], ['betas'][0]): state if it
        is changed."""
        k = (id(container), f"[{key!r}]" + (f"[{sub}]" if sub is not None else ""))
        self.item_paths[k] = (container, (("item", key),) + ((("item", sub),) if sub is not None
                                                             else ()))
        value = container[key] if sub is None else container[key][sub]
        return self._number_state(k, container, k[1], float(value))

    def set_item(self, container: Any, key: Any, expr: ast.expr, sub: Optional[int] = None) -> Sym:
        k = (id(container), f"[{key!r}]" + (f"[{sub}]" if sub is not None else ""))
        self.item_paths[k] = (container, (("item", key),) + ((("item", sub),) if sub is not None
                                                             else ()))
        return self._set_number(k, expr)

    def power_of(self, base: Any, n: Any) -> Any:
        """base ** n for an integer n computed in hardware; base a constant or computed
        (then squared for n's bits)."""
        if not isinstance(n, Sym):
            return self.binop(ast.Pow(), base, n)
        if not isinstance(base, Sym):
            return self.power(float(base), n)
        p, b, k = self.fresh("pw"), self.fresh("pb"), self.fresh("k")
        self.emit(_assign(p, ast.Constant(1.0)))
        self.emit(_assign(b, copy.deepcopy(base.expr)))
        bit = ast.BinOp(left=ast.BinOp(left=copy.deepcopy(n.expr), op=ast.RShift(), right=_load(k)),
                        op=ast.BitAnd(), right=ast.Constant(1))
        self.emit(ast.For(target=ast.Name(id=k, ctx=ast.Store()),
                          iter=ast.Call(func=_load("range"), args=[ast.Constant(POW_BITS)],
                                        keywords=[]),
                          body=[ast.If(test=bit, body=[_assign(p, ast.BinOp(
                              left=_load(p), op=ast.Mult(), right=_load(b)))], orelse=[]),
                                _assign(b, ast.BinOp(left=_load(b), op=ast.Mult(), right=_load(b)))],
                          orelse=[], type_comment=None))
        return Sym(_load(p))

    def param(self, mod: Any, attr: str) -> Any:
        """A module's parameter or buffer (None if it has none)."""
        v = getattr(mod, attr)
        return None if v is None else self.adopt(v, f"{type(mod).__name__}_{attr}")

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

    def elementwise(self, fn: Callable[..., ast.expr], *vals: Any, base: str = "a",
                    op: Optional[Tuple] = None) -> Any:
        """fn of the operands' elements, broadcast: a Lazy (a Sym of scalars). op says
        what fn computes (for autograd)."""
        vals = tuple(v if isinstance(v, Lazy) else self.to_arr(v) for v in vals)
        shapes = [v.shape for v in vals if isinstance(v, Arr)]
        if not shapes:
            out: Any = Sym(fn(*[self.elem(v, (), []) for v in vals]), op, vals)
        else:
            out = Lazy(self, _bcast(*shapes), fn, vals, op=op)
        self.record("ew", out, *vals, op=op)
        return out

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
            out: Any = result.view(shape, strides)
        elif not out_shape:
            out = Sym(result.at([ast.Constant(0)]), arr=Arr(result.name, (), (), 0))
        else:
            out = result
        if kind != "argmax":
            self.record("reduce", out, a, axes=axes, kind=kind, result=result)
        return out

    def matmul(self, a: Any, b: Any) -> Arr:
        """a @ b with numpy's rules: 1-D operands as a row / a column, the dimensions
        before the last two broadcast (batches of products: attention's heads)."""
        a, b = self.to_arr(a), self.to_arr(b)
        a1, b1 = len(a.shape) == 1, len(b.shape) == 1
        va = a.view((1,) + a.shape, (0,) + a.strides) if a1 else a
        vb = b.view(b.shape + (1,), b.strides + (0,)) if b1 else b
        n, k = va.shape[-2:]
        k2, m = vb.shape[-2:]
        if k != k2:
            raise NotTranslatable(f"Shapes {a.shape} @ {b.shape} don't match")
        batch = _bcast(va.shape[:-2], vb.shape[:-2]) if len(va.shape) > 2 or len(vb.shape) > 2 \
            else ()
        out = self.new_list(batch + (n, m), "mm")

        def lead(v, bidx):
            nb = len(v.shape) - 2
            mine = bidx[len(bidx) - nb:] if nb else []
            return [ast.Constant(0) if v.shape[d] == 1 else copy.deepcopy(mine[d])
                    for d in range(nb)]

        def body(idx):
            bidx, i, j = idx[:-2], idx[-2], idx[-1]
            acc = self.fresh("acc")
            self.emit(_assign(acc, ast.Constant(0)))

            def inner(t):
                self.emit(_assign(acc, ast.BinOp(
                    left=_load(acc), op=ast.Add(),
                    right=ast.BinOp(left=va.at(lead(va, bidx) + [copy.deepcopy(i), t[0]]),
                                    op=ast.Mult(),
                                    right=vb.at(lead(vb, bidx) + [t[0], copy.deepcopy(j)])))))
            self.loops((k,), inner)
            self.emit(ast.Assign(targets=[_store(out.at(list(idx)))], value=_load(acc)))
        self.loops(batch + (n, m), body)
        self.record("matmul", out, va, vb)
        shape = batch + ((n,) if not a1 else ()) + ((m,) if not b1 else ())
        return out.view(shape, _contiguous(shape)) if shape != out.shape else out

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

    def loop_step(self, fn: Callable, loop, roots: Any, inputs: List[Any], holder: Any) -> Tuple:
        """One step of a function's loop over batches (see translate): the statements
        before the loop run as Python (model.train(), total = 0), the loop's body is
        the step, its target the inputs; the locals the body changes are state (kept
        from one step to the next) and the results."""
        np = self.np
        loader, fd, _, _ = loop
        g = fn.__globals__
        sig = inspect.signature(fn)
        frame: Dict[str, Any] = {}
        for p, par in sig.parameters.items():
            if p == loader:
                frame[p] = None
            elif hasattr(roots, p):
                frame[p] = getattr(roots, p)
            elif par.default is not inspect.Parameter.empty:
                frame[p] = par.default
        self.frames.append(frame)
        try:
            at = next(i for i, st in enumerate(fd.body) if isinstance(st, ast.For)
                      and isinstance(st.iter, ast.Name) and st.iter.id == loader)
            self.block(fd.body[:at], frame, g)
            lp = fd.body[at]
            assigned = {n.id for st in lp.body for n in ast.walk(st)
                        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)} | {
                st.target.id for st in ast.walk(lp) if isinstance(st, ast.AugAssign)
                and isinstance(st.target, ast.Name)}
            carried = [v for v in frame if v in assigned and isinstance(
                frame[v], (int, float, np.integer, np.floating)) and not isinstance(frame[v], bool)]
            for v in carried:
                key = (id(holder), v)
                if self.state_keys is None:
                    self.mutated.add(key)
                value = frame[v]
                value = int(value) if isinstance(value, (int, np.integer)) else float(value)
                frame[v] = (self.state_value(holder, v, value)
                            if self.state_keys is not None and key in self.state_keys else value)
            self.store(lp.target, tuple(inputs) if len(inputs) > 1 else inputs[0], frame, g)
            try:
                self.block(lp.body, frame, g)
            except _Continue:
                pass
            except _Break:
                raise NotTranslatable("break in the loop over batches is not supported")
            out = []
            for v in carried:
                key, new = (id(holder), v), frame[v]
                if key in self.state:
                    name = self.state[key][0]
                    self.emit(ast.Assign(targets=[ast.Subscript(value=_load(name),
                                                                slice=ast.Constant(0),
                                                                ctx=ast.Store())],
                                         value=self.elem(new, (), [])))
                out.append(new if isinstance(new, Sym) else Sym(ast.Constant(new)))
            if not out:
                raise NotTranslatable("The loop over batches computes nothing it keeps (its "
                                      "results are the locals it changes, like a total loss)")
            return tuple(out)
        finally:
            self.frames.pop()

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
                try:
                    self.block(s.body, frame, g)
                except _Continue:
                    continue
                except _Break:
                    break
            else:
                self.block(s.orelse, frame, g)
            return
        if isinstance(s, ast.Continue):
            raise _Continue()
        if isinstance(s, ast.Break):
            raise _Break()
        if isinstance(s, ast.If):
            test = self.expr(s.test, frame, g)
            if isinstance(test, (Arr, Sym)):
                raise NotTranslatable(f"An if on a value computed in hardware (line {s.lineno})")
            self.block(s.body if test else s.orelse, frame, g)
            return
        if isinstance(s, ast.Assert):                  # (checked as Python checks it)
            test = self.expr(s.test, frame, g)
            if isinstance(test, (Arr, Sym)):
                raise NotTranslatable(f"An assert on a value computed in hardware (line {s.lineno})")
            if not test:
                raise AssertionError(self.expr(s.msg, frame, g) if s.msg is not None else
                                     ast.unparse(s.test))
            return
        raise NotTranslatable(f"Unsupported statement (line {s.lineno}): {ast.unparse(s)}")

    def store(self, t: ast.expr, value: Any, frame, g) -> None:
        if isinstance(t, ast.Name):
            if isinstance(value, Sym) and not isinstance(value.expr, (ast.Name, ast.Constant)):
                value = self.rename(value, t.id + "_")
            frame[t.id] = value
            return
        if isinstance(t, ast.Attribute):
            obj = self.expr(t.value, frame, g)
            key = (id(obj), t.attr)
            if self.first.setdefault(key, "w") == "r" and self.state_keys is None \
                    and t.attr not in self.shadow.get(id(obj), {}):
                self.mutated.add(key)                 # (read, then changed: state)
            if isinstance(value, Sym) and not isinstance(value.expr, (ast.Name, ast.Constant)):
                value = self.rename(value, t.attr + "_")
            self.shadow.setdefault(id(obj), {})[t.attr] = value
            return
        if isinstance(t, (ast.Tuple, ast.List)):
            vals = list(value)
            for tt, v in zip(t.elts, vals):
                self.store(tt, v, frame, g)
            return
        if isinstance(t, ast.Subscript):
            target = self.target(t, frame, g)
            if isinstance(target, _Scatter):
                self.scatter(target, lambda old, new: new, value)
                return
            if isinstance(target, Arr):
                self.copy_into(target, value)
                return
        raise NotTranslatable(f"Unsupported assignment to {ast.unparse(t)}")

    def target(self, t: ast.Subscript, frame, g) -> Any:
        """The array a subscript assigns to (a view, or a _Scatter)."""
        base = self.expr(t.value, frame, g)
        if isinstance(base, Arr):
            return self.index(base, t.slice, frame, g, store=True)
        return self.expr(t, frame, g)

    def scatter(self, sc: "_Scatter", combine: Callable[[ast.expr, ast.expr], ast.expr],
                value: Any) -> None:
        """target[i0, i1, ...] = combine(its element, value), element by element, in order."""
        if sc.base.const:
            self.written(sc.base)
        self.dirty.add(sc.base.name)
        value = value if isinstance(value, Lazy) else self.to_arr(value)
        for lz in self.held():
            if lz.real is None and sc.base.name in lz.reads():
                lz.materialize()

        def body(idx):
            where = [_int_index(self.elem(o, sc.shape, idx)) for o in sc.ops]
            old = sc.base.at(where)
            self.emit(ast.Assign(targets=[_store(sc.base.at(copy.deepcopy(where)))],
                                 value=combine(old, self.elem(value, sc.shape, idx))))
        self.loops(sc.shape, body)

    def copy_into(self, target: Arr, value: Any) -> None:
        if target.const:
            self.written(target)
        self.dirty.add(target.name)
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
        self.record("copy", target, value)

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
        cur = (self.target(s.target, frame, g) if isinstance(s.target, ast.Subscript)
               else self.expr(s.target, frame, g))
        rhs = self.expr(s.value, frame, g)
        op = _BINOPS.get(type(s.op))
        if op is None:
            raise NotTranslatable(f"Unsupported operator in {ast.unparse(s)}")
        if isinstance(cur, _Scatter):
            self.scatter(cur, lambda old, new: ast.BinOp(left=old, op=type(s.op)(), right=new), rhs)
            return
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
                return self.adopt(frame[n.id])
            if n.id in g:
                return self.adopt(g[n.id])
            if hasattr(builtins, n.id):
                return getattr(builtins, n.id)
            raise NotTranslatable(f"Unknown name '{n.id}'")
        if isinstance(n, ast.Attribute):
            obj = self.expr(n.value, frame, g)
            if isinstance(obj, Arr):
                return self.arr_attr(obj, n.attr)
            if isinstance(obj, Sym) and self.torch:
                return _Method(obj, n.attr)
            sh = self.shadow.get(id(obj), {})
            if n.attr in sh:
                return sh[n.attr]
            key = (id(obj), n.attr)
            self.first.setdefault(key, "r")
            value = getattr(obj, n.attr)
            if self.state_keys is not None and key in self.state_keys:
                return self.state_value(obj, n.attr, value)       # (reassigned, or a number)
            return self.adopt(value, f"{type(obj).__name__}_{n.attr}")
        if isinstance(n, (ast.Tuple, ast.List)):
            vals = [self.expr(e, frame, g) for e in n.elts]
            return tuple(vals) if isinstance(n, ast.Tuple) else vals
        if isinstance(n, ast.BinOp):
            return self.binop(n.op, self.expr(n.left, frame, g), self.expr(n.right, frame, g))
        if isinstance(n, ast.UnaryOp):
            v = self.expr(n.operand, frame, g)
            if isinstance(v, (Arr, Sym)):
                if isinstance(n.op, ast.USub):
                    return self.elementwise(lambda x: ast.UnaryOp(op=ast.USub(), operand=x), v,
                                            op=("neg",))
                if isinstance(n.op, (ast.Invert, ast.Not)) and isinstance(v, Arr):
                    from .torch_ops import const_values
                    c = const_values(self, v)
                    if c is not None:                  # (a constant mask: ~ when translating)
                        return self.to_arr(self.np.logical_not(c != 0).astype(float))
                    return self.elementwise(lambda x: ast.BinOp(left=ast.Constant(1), op=ast.Sub(),
                                                                right=x), v, op=("cmp",))
                raise NotTranslatable(f"Unsupported operator in {ast.unparse(n)}")
            return {ast.USub: lambda x: -x, ast.UAdd: lambda x: +x, ast.Not: lambda x: not x,
                    ast.Invert: lambda x: ~x}[type(n.op)](v)
        if isinstance(n, ast.Compare) and len(n.ops) == 1:
            a, b = self.expr(n.left, frame, g), self.expr(n.comparators[0], frame, g)
            op = n.ops[0]
            if isinstance(op, (ast.Is, ast.IsNot)):        # (identity: Python's, x is None)
                return (a is b) if isinstance(op, ast.Is) else (a is not b)
            if isinstance(a, (Arr, Sym)) or isinstance(b, (Arr, Sym)):
                if type(op) not in _CMPS:
                    raise NotTranslatable(f"Unsupported comparison in {ast.unparse(n)}")
                return self.elementwise(lambda x, y: ast.Compare(left=x, ops=[type(op)()],
                                                                 comparators=[y]), a, b, base="m",
                                        op=("cmp",))
            return {ast.Is: lambda x, y: x is y, ast.IsNot: lambda x, y: x is not y,
                    ast.In: lambda x, y: x in y, ast.NotIn: lambda x, y: x not in y,
                    **_CMPS}[type(op)](a, b)
        if isinstance(n, ast.BoolOp):
            # (as Python: in order, the value that decides it - `caches and caches[0]`)
            for item in n.values:
                v = self.expr(item, frame, g)
                if isinstance(v, (Arr, Sym)):
                    raise NotTranslatable(f"and / or on values computed in hardware: "
                                          f"{ast.unparse(n)}")
                if (not v) if isinstance(n.op, ast.And) else v:
                    return v
            return v
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
            return self.adopt(base[key])
        if isinstance(n, ast.Slice):
            return slice(*(self.expr(x, frame, g) if x is not None else None
                           for x in (n.lower, n.upper, n.step)))
        if isinstance(n, ast.Call):
            return self.call_expr(n, frame, g)
        if isinstance(n, ast.JoinedStr):
            return ""
        if isinstance(n, (ast.ListComp, ast.GeneratorExp)):
            return self.comprehension(n, frame, g)
        raise NotTranslatable(f"Unsupported expression: {ast.unparse(n)}")

    def comprehension(self, n, frame, g) -> List[Any]:
        """[elt for ... in ... if ...] over Python values, as Python runs it."""
        out: List[Any] = []

        def run(k: int, fr) -> None:
            if k == len(n.generators):
                out.append(self.expr(n.elt, fr, g))
                return
            c = n.generators[k]
            items = self.expr(c.iter, fr, g)
            if isinstance(items, (Arr, Sym)):
                raise NotTranslatable(f"A loop over an array value: {ast.unparse(n)}")
            for item in list(items):
                inner = dict(fr)
                self.store(c.target, item, inner, g)
                tests = [self.expr(t, inner, g) for t in c.ifs]
                if any(isinstance(t, (Arr, Sym)) for t in tests):
                    raise NotTranslatable(f"An if on a computed value: {ast.unparse(n)}")
                if all(tests):
                    run(k + 1, inner)
        run(0, frame)
        return out

    def arr_attr(self, a: Arr, attr: str) -> Any:
        if self.torch and attr == "grad":
            g = self.grads.get(a.name) if not isinstance(a, Lazy) else None
            return None if g is None else Arr(g.name, a.shape, a.strides, a.offset)
        if self.torch and attr == "data":
            return a
        if self.torch and attr == "device":
            import torch
            return torch.device("cpu")
        if self.torch and attr == "dtype":
            import torch
            return torch.float32
        if attr == "T":
            return a.view(a.shape[::-1], a.strides[::-1])
        if attr == "shape":
            return a.shape
        if attr == "size" and not self.torch:        # (torch's is a method)
            return a.size
        if attr == "ndim":
            return len(a.shape)
        return _Method(a, attr)

    def index(self, a: Arr, sl: ast.expr, frame, g, store: bool = False) -> Any:
        parts = sl.elts if isinstance(sl, ast.Tuple) else [sl]
        vals = [self.expr(p, frame, g) for p in parts]
        lists = [k for k, v in enumerate(vals) if isinstance(v, list)
                 and v and all(isinstance(i, int) and not isinstance(i, bool) for i in v)]
        if len(lists) == 1 and not any(self.is_index_array(v) for v in vals):
            # (a list of indices among slices: evenly spaced, a slice keeping its
            #  dimension, x[:, [-1], :]; else, read, its elements' slices concatenated)
            k = lists[0]
            if Ellipsis in vals[:k]:                  # (the dimension it indexes)
                dim = len(a.shape) - sum(1 for v in vals[k:] if v is not None)
            else:
                dim = sum(1 for v in vals[:k] if v is not None)
            if not 0 <= dim < len(a.shape):
                raise NotTranslatable(f"Too many indices: {ast.unparse(sl)}")
            idx = [i % a.shape[dim] for i in vals[k]]
            step = idx[1] - idx[0] if len(idx) > 1 else 1
            if step > 0 and idx == list(range(idx[0], idx[0] + step * len(idx), step)):
                vals[k] = slice(idx[0], idx[0] + step * (len(idx) - 1) + 1, step)
            elif store:                               # (written: as numpy's integer array)
                vals[k] = self.np.asarray(idx, dtype=self.np.int64)
            else:
                shape = a.shape[:dim] + (1,) + a.shape[dim + 1:]
                pieces = [a.view(shape, a.strides, a.offset + i * a.strides[dim]) for i in idx]
                a = self.to_arr(self.numpy("concatenate", [pieces], {"axis": dim}))
                vals[k] = slice(None)
        if any(self.is_index_array(v) for v in vals):
            return self.fancy(a, vals, store)
        if any(v is Ellipsis for v in vals):          # (... : the dimensions not named)
            k = vals.index(Ellipsis)
            named = sum(1 for v in vals if v is not None and v is not Ellipsis)
            vals = vals[:k] + [slice(None)] * (len(a.shape) - named) + vals[k + 1:]
        shape, strides, offset = [], [], a.offset
        d = 0
        for v in vals:
            if v is None:                             # (a new axis of 1)
                shape.append(1)
                strides.append(0)
                continue
            if isinstance(v, slice):
                start, stop, step = v.indices(a.shape[d])
                shape.append(len(range(start, stop, step)))
                strides.append(a.strides[d] * step)
                offset += start * a.strides[d]
            elif isinstance(v, (int, self.np.integer)) and not isinstance(v, bool):
                offset += (int(v) % a.shape[d]) * a.strides[d]
            else:
                raise NotTranslatable(f"Index with a value computed in hardware: {ast.unparse(sl)}")
            d += 1
        shape += list(a.shape[d:])
        strides += list(a.strides[d:])
        if not shape:
            return Sym(Arr(a.name, (1,), (1,), offset, a.const).at([ast.Constant(0)]),
                       arr=Arr(a.name, (), (), offset, a.const))
        return a.view(tuple(shape), tuple(strides), offset)

    def is_index_array(self, v: Any) -> bool:
        return (isinstance(v, self.np.ndarray) and v.dtype.kind in "iu") or (
            isinstance(v, Arr) and not isinstance(v, slice)) or (
            _is_tensor(v) and not v.is_floating_point() and v.dtype != _torch_bool())

    def fancy(self, a: Arr, vals: List[Any], store: bool) -> Any:
        """a[i0, i1, ...] with arrays of indices (one per dimension, numpy's integer
        arrays or arrays computed in hardware, broadcast together): the elements
        they pick, read where used (a gather) or, as a target, a _Scatter."""
        np = self.np
        nadv = 0
        while nadv < len(vals) and not (isinstance(vals[nadv], slice) or vals[nadv] is None
                                        or vals[nadv] is Ellipsis):
            nadv += 1
        rest = vals[nadv:]
        if any(not (isinstance(v, slice) and v == slice(None)) for v in rest) \
                or len(vals) > len(a.shape):
            raise NotTranslatable("Indexing with arrays is supported with arrays (or integers) "
                                  "for the first dimensions, the others whole")
        if isinstance(a, Lazy):
            a = a.materialize()
        ops = [self.to_arr(v) if isinstance(v, np.ndarray) else self.adopt(v) if _is_tensor(v)
               else v for v in vals[:nadv]]
        adv = _bcast(*[o.shape for o in ops if isinstance(o, Arr)])
        trail = a.shape[nadv:]
        if trail:
            # (a[idx] with whole dimensions after: idx broadcast against their aranges)
            t = len(trail)
            ops = [o.view(o.shape + (1,) * t, o.strides + (0,) * t) if isinstance(o, Arr)
                   else o for o in ops]
            for d, size in enumerate(trail):
                r = self.to_arr(np.arange(size))
                sh = (1,) * len(adv) + (1,) * d + (size,) + (1,) * (t - 1 - d)
                st = (0,) * len(adv) + (0,) * d + (1,) + (0,) * (t - 1 - d)
                ops.append(r.view(sh, st))
        ops = tuple(ops)
        shape = _bcast(*[o.shape for o in ops if isinstance(o, Arr)])
        if store:
            return _Scatter(a, ops, shape)
        out = Lazy(self, shape, lambda *ix: a.at([_int_index(x) for x in ix]), ops, extra=(a,),
                   op=("gather",))
        self.record("gather", out, a, *ops)
        return out

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
            inv = self.rename(self.elementwise(
                lambda y: ast.BinOp(left=ast.Constant(1.0), op=ast.Div(), right=y), b,
                op=("recip",)), "inv")
            return self.elementwise(lambda x, y: ast.BinOp(left=x, op=ast.Mult(), right=y), a, inv,
                                    op=("mul",))
        if isinstance(op, ast.Pow) and isinstance(b, int) and not isinstance(a, int):
            k = b                                    # x ** k: products
            if k < 1:
                raise NotTranslatable("x ** k with k < 1 is not supported")

            def power(x):
                out = x
                for _ in range(k - 1):
                    out = ast.BinOp(left=out, op=ast.Mult(), right=copy.deepcopy(x))
                return out
            return self.elementwise(power, a, op=("pow", k))
        tag = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div"}.get(type(op), "int")
        return self.elementwise(lambda x, y: ast.BinOp(left=x, op=type(op)(), right=y), a, b,
                                op=(tag,))

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
        fn = self.expr(n.func, frame, g)
        args = [self.expr(a, frame, g) for a in n.args]
        kwargs = {k.arg: self.expr(k.value, frame, g) for k in n.keywords}
        return self.call_value(fn, args, kwargs, ast.unparse(n))

    def call_value(self, fn: Any, args: List[Any], kwargs: Dict[str, Any], text: str = "") -> Any:
        """Calling fn (a Python value) on interpreter values."""
        np = self.np
        if isinstance(fn, _Method):
            if self.torch:
                from .torch_ops import tensor_method
                out = tensor_method(self, fn.arr, fn.name, args, kwargs)
                if out is not NotImplemented:
                    return out
            return self.method(fn.arr, fn.name, args, kwargs)
        if isinstance(getattr(fn, "__self__", None), np.random.Generator):
            if getattr(fn, "__name__", "") != "random":
                raise NotTranslatable(f"Generator.{getattr(fn, '__name__', fn)} is not supported "
                                      "(only .random)")
            return self.random(fn.__self__, args, kwargs)
        sym = any(isinstance(v, (Arr, Sym)) or type(v).__name__ == "NegInf"
                  for v in list(args) + list(kwargs.values()))
        if self.torch:
            from . import torch_ops, torch_optim
            if torch_ops.is_module(fn):
                return torch_ops.module_call(self, fn, args, kwargs)
            if torch_ops.is_torch_function(fn):
                if sym or getattr(fn, "__name__", "") in torch_ops.UTILS or any(
                        isinstance(x, Arr) for v in args if isinstance(v, (list, tuple)) for x in v):
                    return torch_ops.function(self, fn, args, kwargs)
                return fn(*args, **kwargs)
            out = torch_optim.method_call(self, fn, args, kwargs)
            if out is not NotImplemented:
                return out
        if getattr(fn, "__module__", None) in ("numpy", "numpy.lib.stride_tricks") or (
                isinstance(fn, np.ufunc)):
            name = getattr(fn, "__name__", "")
            arrays = any(isinstance(v, Arr) for v in list(args) + list(kwargs.values())
                         + [x for v in args if isinstance(v, (list, tuple)) for x in v])
            if not sym and not arrays and name not in ("zeros", "ones", "zeros_like", "ones_like"):
                return fn(*args, **kwargs)
            return self.numpy(name, args, kwargs)
        if fn is isinstance:
            return isinstance(*args)
        if fn is id and len(args) == 1 and isinstance(args[0], Arr):
            a = args[0]
            real = self.real_of.get(a.name) if not isinstance(a, Lazy) else None
            if real is None or tuple(real.shape if hasattr(real, "shape") else ()) != a.shape:
                raise NotTranslatable("id() of an array computed in hardware")
            return id(real)
        if fn in (len,):
            a = args[0]
            return a.shape[0] if isinstance(a, Arr) else len(a)
        if fn in (abs,) and sym:
            return self.elementwise(lambda x: ast.Call(func=ast.Name(id="abs", ctx=ast.Load()),
                                                       args=[x], keywords=[]), args[0], op=("abs",))
        if fn in (print,):
            return None
        if inspect.ismethod(fn) and not getattr(fn.__self__, "__module__", "").startswith("numpy"):
            return self.call(fn.__func__, args, kwargs, obj=fn.__self__)
        if inspect.isfunction(fn):
            return self.call(fn, args, kwargs)
        if sym:
            raise NotTranslatable(f"Unsupported call on values computed in hardware: {text}")
        for a in _arrays_in(list(args) + list(kwargs.values())):    # (kept by Python code?)
            if not isinstance(a, Lazy):
                self.handed[id(a)] = (self.fills.get(a.name) if a.name not in self.dirty
                                      else None)
        return fn(*args, **kwargs)                   # plain Python (range, reversed, ...)

    def random(self, gen: Any, args, kwargs) -> Any:
        """gen.random(size): uniform numbers in [0, 1) from a generator in hardware, its
        state kept between calls (a memory): xorshift32, each number its top
        RANDOM_BITS bits (not numpy's stream: see _Draws)."""
        size = kwargs.get("size", args[0] if args else None)
        shape = () if size is None else (
            tuple(int(x) for x in size) if isinstance(size, (tuple, list)) else (int(size),))
        if gen is None:                          # (torch's: its default generator)
            gen = _torch_generator()
        key = (id(gen), RANDOM_STATE)
        self.generators[id(gen)] = gen
        if self.state_keys is None:
            self.mutated.add(key)
        if key not in self.state:
            self.state_value(gen, RANDOM_STATE, _seed(gen))
        state = self.state[key][0]
        out = self.new_list(shape or (1,), "rnd")

        def body(idx):
            x = self.fresh("x")
            self.emit(_assign(x, ast.Subscript(value=_load(state), slice=ast.Constant(0),
                                               ctx=ast.Load())))
            for op, k in _XORSHIFT:
                shifted = ast.BinOp(left=_load(x), op=op(), right=ast.Constant(k))
                if isinstance(shifted.op, ast.LShift):
                    shifted = ast.BinOp(left=shifted, op=ast.BitAnd(), right=ast.Constant(_MASK32))
                self.emit(_assign(x, ast.BinOp(left=_load(x), op=ast.BitXor(), right=shifted)))
            self.emit(ast.Assign(targets=[ast.Subscript(value=_load(state), slice=ast.Constant(0),
                                                        ctx=ast.Store())], value=_load(x)))
            self.emit(ast.Assign(targets=[_store(out.at(idx if shape else [ast.Constant(0)]))],
                                 value=ast.BinOp(left=ast.BinOp(
                                     left=_load(x), op=ast.RShift(),
                                     right=ast.Constant(32 - RANDOM_BITS)),
                                     op=ast.Mult(), right=ast.Constant(2.0 ** -RANDOM_BITS))))
        self.loops(shape, body)
        return out if shape else Sym(out.at([ast.Constant(0)]))

    def method(self, a: Arr, name: str, args, kwargs) -> Any:
        if name in ("sum", "max", "mean", "argmax", "min"):
            axis = kwargs.get("axis", args[0] if args else None)
            keep = bool(kwargs.get("keepdims", False))
            if name == "min":
                neg = self.elementwise(lambda x: ast.UnaryOp(op=ast.USub(), operand=x), a,
                                       op=("neg",))
                return self.elementwise(lambda x: ast.UnaryOp(op=ast.USub(), operand=x),
                                        self.reduce(neg, axis, keep, "max"), op=("neg",))
            return self.reduce(a, axis, keep, name)
        if name == "reshape":
            shape = args[0] if len(args) == 1 and isinstance(args[0], (tuple, list)) else tuple(args)
            shape = tuple(int(x) for x in shape)
            if -1 in shape:
                rest = math.prod(x for x in shape if x != -1)
                shape = tuple(a.size // rest if x == -1 else x for x in shape)
            if math.prod(shape) != a.size:
                raise NotTranslatable(f"Can't reshape {a.shape} to {shape}")
            strides = _reshaped(a, shape)
            if strides is not None:
                return a.view(shape, strides)
            if not _is_contiguous(a):
                a = self.method(a, "copy", [], {})      # (a view of another order: a copy)
            return a.view(shape, _contiguous(shape))
        if name == "transpose":
            axes = args[0] if len(args) == 1 and isinstance(args[0], (tuple, list)) else tuple(args)
            axes = tuple(axes) if axes else tuple(range(len(a.shape)))[::-1]
            return a.view(tuple(a.shape[x] for x in axes), tuple(a.strides[x] for x in axes))
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
                                   ctx=ast.Load()), args=[x], keywords=[]), args[0], op=(name,))
        if name in ("maximum", "minimum"):
            fn = "max" if name == "maximum" else "min"
            return self.elementwise(lambda x, y: ast.Call(func=ast.Name(id=fn, ctx=ast.Load()),
                                                          args=[x, y], keywords=[]), *args[:2],
                                    op=(fn,))
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
                                                       args=[x], keywords=[]), args[0], op=("abs",))
        if name == "transpose":
            return self.method(self.to_arr(args[0]), "transpose", list(args[1:2]), {})
        if name == "reshape":
            return self.method(self.to_arr(args[0]), "reshape", [args[1]], {})
        if name == "pad":
            a = args[0] if isinstance(args[0], Lazy) else self.to_arr(args[0])   # (computed in the copy)
            mode = kwargs.get("mode", args[2] if len(args) > 2 else "constant")
            if mode != "constant" or kwargs.get("constant_values", 0) != 0:
                raise NotTranslatable("np.pad: only zeros around (mode='constant') are supported")
            pads = np.broadcast_to(np.asarray(kwargs.get("pad_width", args[1] if len(args) > 1
                                                         else None)), (len(a.shape), 2))
            out = self.new_list(tuple(int(s + lo + hi) for s, (lo, hi) in zip(a.shape, pads)))
            inner = out.view(a.shape, out.strides, sum(int(lo) * st for (lo, _), st
                                                       in zip(pads, out.strides)))
            self.copy_into(inner, a)
            return out
        if name == "concatenate":
            parts = [self.to_arr(v) for v in args[0]]
            axis = kwargs.get("axis", args[1] if len(args) > 1 else 0)
            if len(parts) == 1:
                return parts[0]
            nd = len(parts[0].shape)
            axis %= nd
            shape = list(parts[0].shape)
            shape[axis] = sum(p.shape[axis] for p in parts)
            out = self.new_list(tuple(shape))
            at = 0
            for p in parts:
                self.copy_into(out.view(p.shape, out.strides, at * out.strides[axis]), p)
                at += p.shape[axis]
            return out
        if name == "sliding_window_view":
            a = self.to_arr(args[0])
            window = args[1] if len(args) > 1 else kwargs["window_shape"]
            window = tuple(window) if isinstance(window, (tuple, list)) else (window,)
            axis = kwargs.get("axis", args[2] if len(args) > 2 else None)
            axes = tuple(range(len(a.shape)))[-len(window):] if axis is None else (
                tuple(axis) if isinstance(axis, (tuple, list)) else (axis,))
            shape, strides = list(a.shape), list(a.strides)
            for x, w in zip(axes, window):
                shape[x % len(shape)] -= w - 1
            for x, w in zip(axes, window):
                shape.append(w)
                strides.append(a.strides[x % len(a.shape)])
            return a.view(tuple(shape), tuple(strides))
        if name in ("zeros", "ones", "zeros_like", "ones_like"):
            shape = args[0].shape if name.endswith("like") else (
                tuple(args[0]) if isinstance(args[0], (tuple, list)) else (args[0],))
            out = self.new_list(tuple(shape))
            if name.startswith("ones"):
                self.copy_into(out, 1.0)
                self.fills[out.name] = 1.0
                self.dirty.discard(out.name)
            return out
        raise NotTranslatable(f"np.{name} is not supported")


class _Method:
    def __init__(self, arr: Arr, name: str) -> None:
        self.arr, self.name = arr, name


def _is_tensor(v: Any) -> bool:
    return type(v).__module__.startswith("torch") and hasattr(v, "detach") \
        and hasattr(v, "is_floating_point")


def _is_array(v: Any) -> bool:
    """A numpy array of numbers or a torch tensor (an array the program holds)."""
    np = _np()
    if isinstance(v, np.ndarray):
        return v.dtype.kind in "fiub"
    return _is_tensor(v)


def _values(v: Any) -> Any:
    """An array's values as a float numpy array (a torch tensor's too)."""
    np = _np()
    if _is_tensor(v):
        v = v.detach().cpu().numpy()
    return np.array(v, dtype=float)


def _arrays_in(v: Any, depth: int = 0) -> List[Arr]:
    if isinstance(v, Arr):
        return [v]
    if isinstance(v, (list, tuple)) and depth < 4:
        return [a for x in v for a in _arrays_in(x, depth + 1)]
    if isinstance(v, dict) and depth < 4:
        return [a for x in v.values() for a in _arrays_in(x, depth + 1)]
    return []


_ATOMS = (int, float, complex, str, bytes, bool, type(None))


def _children(x: Any) -> List[Tuple[Tuple[str, Any], Any]]:
    """(step, value) for an object's attributes, a dict's or a list's items."""
    np = _np()
    if isinstance(x, dict):
        return [(("item", k), v) for k, v in x.items()]
    if isinstance(x, (list, tuple)):
        return [(("item", i), v) for i, v in enumerate(x)]
    if isinstance(x, (np.ndarray, type)) or inspect.ismodule(x) or inspect.isroutine(x) \
            or isinstance(x, (Arr, Sym)) or _is_tensor(x):
        return []
    d = getattr(x, "__dict__", None)
    return [(("attr", k), v) for k, v in d.items()] if isinstance(d, dict) else []


def _paths(root: Any, depth: int = 16) -> Dict[int, Tuple]:
    """id(x) -> the path from root to x (attributes, items), for what root holds (for
    _Roots, each object in turn: the first's paths first). A dict key that is id()
    of an array root holds, or such an object (a parameter), is written as its path."""
    out: Dict[int, Tuple] = {id(root): ()}
    starts = ([(v, (("attr", k),)) for k, v in vars(root).items()] if isinstance(root, _Roots)
              else [(root, ())])
    for start, p0 in starts:
        if isinstance(start, _ATOMS):
            continue
        out.setdefault(id(start), p0)
        queue = [(start, p0)]
        while queue:
            x, path = queue.pop(0)
            if len(path) >= depth:
                continue
            for step, v in _children(x):
                if isinstance(v, _ATOMS) or id(v) in out:
                    continue
                out[id(v)] = path + (step,)
                queue.append((v, path + (step,)))

    def symbolic(s):
        if s[0] != "item":
            return s
        k = s[1]
        if type(k) is int and k in out and k != id(None) and len(out[k]):
            return ("id", out[k])
        if not isinstance(k, _ATOMS) and id(k) in out and len(out[id(k)]):
            return ("key", out[id(k)])
        return s
    return {k: tuple(symbolic(s) for s in path) for k, path in out.items()}


def _resolve(root: Any, path: Tuple) -> Any:
    x = root
    for kind, k in path:
        if kind == "attr":
            x = getattr(x, k)
        elif kind == "item":
            x = x[k]
        elif kind == "key":
            x = x[_resolve(root, k)]
        else:
            x = x[id(_resolve(root, k))]
    return x


def loops_only(func: Callable, samples: List[Any]) -> bool:
    """Designs only this translator takes: torch code, or a loop over batches."""
    if not samples:
        return False
    first = samples[0] if isinstance(samples[0], tuple) else (samples[0],)
    owner = func.__self__ if inspect.ismethod(func) else None
    if any(_is_tensor(v) for v in first) or (owner is not None and _torchy(owner)):
        return True
    return owner is None and _training_loop(func, first) is not None


class _Roots:
    """The objects a function is given (the model, the optimizer, ...), by name."""

    def __init__(self, objs: Dict[str, Any]) -> None:
        self.__dict__.update(objs)


def _torchy(obj: Any) -> bool:
    """Is obj (or what it holds, for _Roots) one of torch's objects?"""
    objs = list(vars(obj).values()) if isinstance(obj, _Roots) else [obj]
    return any(any(c.__module__.startswith("torch") for c in type(o).__mro__) for o in objs)


def _training_loop(fn: Callable, first: Tuple[Any, ...]):
    """A function whose body loops over an argument given as batches (a list of
    tuples of arrays, a training epoch): (that argument, the function's AST, its
    position, the loop's target names); else None."""
    try:
        fd = ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]
    except (OSError, TypeError, SyntaxError):
        return None
    params = [a.arg for a in fd.args.args]
    for st in fd.body:
        if isinstance(st, ast.For) and isinstance(st.iter, ast.Name) and st.iter.id in params:
            k = params.index(st.iter.id)
            if k >= len(first):
                return None
            batches = first[k]
            if not isinstance(batches, (list, tuple)) or not batches:
                return None
            b0 = batches[0]
            items = b0 if isinstance(b0, (tuple, list)) else (b0,)
            if not all(_is_array(x) for x in items):
                return None
            t = st.target
            names = ([e.id for e in t.elts] if isinstance(t, (ast.Tuple, ast.List))
                     and all(isinstance(e, ast.Name) for e in t.elts)
                     else [t.id] if isinstance(t, ast.Name) else None)
            if names is None or len(names) != len(items):
                raise NotTranslatable("The loop over batches must unpack each into names")
            return st.iter.id, fd, k, names
    return None


def _callable(func: Callable) -> Callable:
    """A torch module stands for its forward method."""
    import sys
    if "torch" in sys.modules:
        from .torch_ops import is_module
        if is_module(func):
            return func.forward
    return func


def _holder(path: Tuple) -> Optional[Tuple[Tuple, str]]:
    """(the path of the object, the attribute) if path ends at an attribute (a torch
    module's parameter or buffer too); else None."""
    if path and path[-1][0] == "attr":
        return path[:-1], path[-1][1]
    if len(path) >= 2 and path[-2] in (("attr", "_parameters"), ("attr", "_buffers")) \
            and path[-1][0] == "item":
        return path[:-2], path[-1][1]
    return None


def _path_text(path: Tuple) -> str:
    out = ""
    path = list(path)
    for i, (kind, k) in enumerate(path):
        # (a torch module's children and parameters as you would write them)
        if kind == "attr" and k in ("_modules", "_parameters", "_buffers") \
                and i + 1 < len(path) and path[i + 1][0] == "item":
            name = path[i + 1][1]
            path[i + 1] = ("attr" if str(name).isidentifier() else "item",
                           name if str(name).isidentifier() else
                           (int(name) if str(name).isdigit() else name))
            continue
        out += (f".{k}" if kind == "attr" else f"[{k!r}]" if kind == "item"
                else f"[{_path_text(k).lstrip('.')}]" if kind == "key"
                else f"[id({_path_text(k)})]")
    return out


_MASK32 = (1 << 32) - 1
_XORSHIFT = ((ast.LShift, 13), (ast.RShift, 17), (ast.LShift, 5))


def _seed(gen: Any) -> int:
    """A generator's starting state in hardware, from its state (not drawing)."""
    import zlib
    if hasattr(gen, "bit_generator"):
        return zlib.crc32(repr(gen.bit_generator.state).encode()) or 1
    return zlib.crc32(bytes(gen.get_state().tolist())) or 1          # (torch's)


def _torch_bool() -> Any:
    import torch
    return torch.bool


def _torch_generator() -> Any:
    import torch
    return torch.default_generator


def _xorshift(x: int) -> int:
    for op, k in _XORSHIFT:
        x ^= ((x << k) & _MASK32) if op is ast.LShift else x >> k
    return x


class _Draws:
    """Stands in for a numpy Generator while the original is checked against the
    program: .random() gives the hardware's numbers."""

    def __init__(self, seed: int) -> None:
        self.x = seed

    def random(self, size=None, dtype=None, out=None):
        np = _np()
        shape = () if size is None else (tuple(size) if isinstance(size, (tuple, list))
                                         else (int(size),))
        vals = []
        for _ in range(math.prod(shape)):
            self.x = _xorshift(self.x)
            vals.append((self.x >> (32 - RANDOM_BITS)) * 2.0 ** -RANDOM_BITS)
        return np.array(vals).reshape(shape) if shape else vals[0]


@dataclass
class _Scatter:
    """The elements of base that arrays of indices pick, as an assignment's target."""
    base: Arr
    ops: Tuple[Any, ...]
    shape: Tuple[int, ...]


def _is_contiguous(a: Arr) -> bool:
    """Is the view its list's elements in order, from its offset (dimensions of 1 aside)?"""
    want = _contiguous(a.shape)
    return isinstance(a, Lazy) or all(s == w for n, s, w in zip(a.shape, a.strides, want) if n != 1)


def _reshaped(a: Arr, shape: Tuple[int, ...]) -> Optional[Tuple[int, ...]]:
    """The strides of a's view reshaped to shape without a copy (dimensions split, or
    merged where they are contiguous with each other), or None."""
    if isinstance(a, Lazy):
        return None
    old = [(n, st) for n, st in zip(a.shape, a.strides) if n != 1]
    new = [n for n in shape if n != 1]
    out: List[int] = []
    i = j = 0
    while i < len(old) and j < len(new):
        # Take dimensions from both sides until their sizes agree
        ni, nj = old[i][0], new[j]
        i2, j2 = i + 1, j + 1
        while ni != nj:
            if ni < nj:
                if i2 >= len(old):
                    return None
                ni *= old[i2][0]
                i2 += 1
            else:
                if j2 >= len(new):
                    return None
                nj *= new[j2]
                j2 += 1
        for k in range(i, i2 - 1):                      # (merged: contiguous with each other)
            if old[k][1] != old[k + 1][1] * old[k + 1][0]:
                return None
        st = old[i2 - 1][1]
        part = []
        for k in range(j2 - 1, j - 1, -1):              # (split: from the last)
            part.insert(0, st)
            st *= new[k]
        out += part
        i, j = i2, j2
    if i != len(old) or j != len(new):
        return None
    it = iter(out)
    return tuple(next(it) if n != 1 else 0 for n in shape)


def _int_index(e: ast.expr) -> ast.expr:
    """An index read from a list: an integer (its list holds integers as floats in the
    float program)."""
    if isinstance(e, ast.Constant):
        return ast.Constant(int(e.value))
    return ast.Call(func=_load("int"), args=[e], keywords=[])


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
    func = _callable(func)
    original = func.__self__ if inspect.ismethod(func) else None
    fn = func.__func__ if original is not None else func
    sig = inspect.signature(func)
    first = samples[0] if isinstance(samples[0], tuple) else (samples[0],)
    loop = _training_loop(fn, first) if original is None else None
    # (run on a copy: Python code the method calls, dict.setdefault, may change the object)
    memo: Dict[int, Any] = {}
    steps: Optional[List[Any]] = None
    if loop is not None:
        # A function with a loop over batches: one step of the loop, its objects copied
        params, (loader, fd, k, targets) = list(sig.parameters), loop
        steps = [b for smp in samples for b in (smp if isinstance(smp, tuple) else (smp,))[k]]
        steps = [tuple(b) if isinstance(b, (tuple, list)) else (b,) for b in steps]
        original = _Roots({p: v for p, v in zip(params, first) if p != loader})
        names = targets
        first = steps[0]
    else:
        names = list(sig.parameters)[:len(first)]
    obj = copy.deepcopy(original, memo) if original is not None else None
    back = {id(memo[id(o)]): o for o in memo.get(id(memo), []) if id(o) in memo}
    shapes = {p: tuple(np.shape(v)) for p, v in zip(names, first)}

    torch_mode = any(_is_tensor(v) for v in first) or _torchy(original)
    carried_holder = type(fn.__name__, (), {})()               # (the loop's carried locals)

    def run(state_keys):
        ns: Dict[str, Any] = {"np": np, EXACT_SQRT: True}
        it = _Interp(ns, state_keys)
        it.torch = torch_mode
        args = []
        for p in names:
            if shapes[p] == ():
                args.append(Sym(ast.Name(id=p, ctx=ast.Load())))
            else:
                it.lists[p] = math.prod(shapes[p])
                args.append(Arr(p, shapes[p], _contiguous(shapes[p])))
        if loop is not None:
            return it, it.loop_step(fn, loop, obj, args, carried_holder)
        return it, it.call(fn, args, {}, obj=obj)

    # First to find the state (attributes read, then changed), then to build it
    it, _ = run(None)
    mutated = set(it.mutated) | _kept(it, obj)
    it, result = run(mutated)
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
        if r is None and isinstance(result, tuple) and any(x is not None for x in result):
            continue                                  # (a result not computed: logits, None)
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
    body = _live(it.code[0], roots)
    # (identical layers, one after the other: a loop over them)
    from .repeats import fold_repeats
    body, merged = fold_repeats(body, ns, set(names))
    it.tables.update(merged)
    body += [ast.Return(value=ret[0] if len(ret) == 1 else ast.Tuple(elts=ret, ctx=ast.Load()))]
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
    # (run with numpy where it can: the check, the formats' run, the bit-exact model)
    from . import vectorize
    gen.__py2ver_loops__ = fdef
    ns[vectorize.VECTOR_KEY] = True
    attr = {p: {"length": math.prod(shapes[p])} for p in names if shapes[p] != ()}
    memories: Dict[str, Tuple[str, Any, Any]] = {}
    paths = _paths(obj) if obj is not None else {}
    tr_paths: Dict[str, Tuple] = {}
    root_name = "" if isinstance(original, _Roots) else type(original).__name__

    def where(path):
        text = _path_text(path)
        return text[1:] if not root_name and text.startswith(".") else root_name + text
    for key, (name, value, label, owner) in list(it.state.items()):
        if key in it.item_paths:               # (a number in a dict: its path)
            container, steps_to = it.item_paths[key]
            if id(container) not in paths:
                raise NotTranslatable("A number the method changes isn't held by the object")
            path = paths[id(container)] + steps_to
            tr_paths[name] = path
            label, owner = where(path), original
        elif key[1] == "":                     # (an array: where the object holds it)
            if key[0] not in paths:
                raise NotTranslatable("An array the method changes isn't held by the object")
            path = paths[key[0]]
            held = _holder(path)
            if held is not None:               # (an attribute: of its object)
                holder = _resolve(obj, held[0])
                label, owner = f"{type(holder).__name__}.{held[1]}", back.get(id(holder), holder)
            else:                              # (in a dict or a list: its path)
                tr_paths[name] = path
                label, owner = where(path), original
        else:
            owner = back.get(id(owner), owner)
        it.state[key] = (name, value, label, owner)
    for name, value, label, owner in it.state.values():
        flat = (np.ravel(value).tolist() if isinstance(value, np.ndarray) else [value])
        attr[name] = {"length": len(flat), "memory": "ram", "init": flat}
        if not isinstance(value, np.ndarray) and isinstance(value, int):
            attr[name].update(signed=1, width=32)      # (a counter: an integer)
        if label.endswith("." + RANDOM_STATE):
            attr[name].update(signed=0, width=32)      # (a generator's: 32 bits)
        memories[name] = (label, value, owner)
    tr = Translation(func=gen, source=source, attr=attr, params=names, shapes=shapes,
                     outputs=outs)
    tr.loops = True
    tr.tables = set()
    tr.const_tables = dict(it.tables)
    tr.memories = memories
    tr.generators = {id(gen): _seed(gen) for gen in it.generators.values()}
    tr.paths = tr_paths
    tr.torch = torch_mode
    tr.fast = vectorize.program(fdef, ns, gen)
    tr.steps = steps
    if loop is not None:
        tr.carried = type(carried_holder).__name__
        _check_loop(tr, fn, original, loop, steps)
    else:
        _check(tr, func, samples)
    return tr


def _kept(it: _Interp, obj: Any) -> set:
    """Arrays the method made (np.zeros_like ...) that Python code it called kept in
    the object (Adam's moments by dict.setdefault on its first step): put in as
    arrays of the values they were made with, their keys (state, by identity)."""
    np = _np()
    keys: set = set()
    if obj is None:
        return keys

    def real(v):
        if isinstance(v, (Lazy, Sym)):
            raise NotTranslatable("A value computed in hardware is kept in a Python container")
        if isinstance(v, Arr):
            fill = it.handed.get(id(v))
            if fill is None:
                raise NotTranslatable("An array kept in a Python container isn't made with "
                                      "np.zeros / np.ones (its starting value)")
            out = np.full(v.shape, float(fill))
            keys.add((id(out), ""))
            return out
        if isinstance(v, tuple):
            return tuple(real(x) for x in v)
        return v

    seen, queue = {id(obj)}, [obj]
    while queue:
        x = queue.pop()
        if isinstance(x, dict):
            for k, v in list(x.items()):
                if _arrays_in(v):
                    x[k] = real(v)
        elif isinstance(x, list):
            for i, v in enumerate(x):
                if _arrays_in(v):
                    x[i] = real(v)
        for _, v in _children(x):
            if not isinstance(v, _ATOMS) and id(v) not in seen:
                seen.add(id(v))
                queue.append(v)
    return keys


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


class _drawing:
    """The modules' numpy Generators (by id) replaced by _Draws while the original runs."""

    def __init__(self, generators: Dict[int, int]) -> None:
        self.generators, self.saved = generators, []

    def __enter__(self):
        import sys
        draws = {g: _Draws(seed) for g, seed in self.generators.items()}
        torch_gen = id(_torch_generator()) if "torch" in sys.modules else None
        for mod in list(sys.modules.values()):
            d = getattr(mod, "__dict__", None)
            if not isinstance(d, dict) or getattr(mod, "__name__", "").startswith("torch"):
                continue
            for k, v in list(d.items()):
                if id(v) in self.generators:
                    self.saved.append((d, k, draws[id(v)], v))
        if torch_gen in draws:                    # (torch's dropout: these numbers)
            import torch.nn.functional as F
            self.saved.append((F.__dict__, "dropout", _dropout(draws[torch_gen]), F.dropout))
            # (attention's dropout_p too: its reference, with that dropout)
            self.saved.append((F.__dict__, "scaled_dot_product_attention",
                               _attention(_dropout(draws[torch_gen])),
                               F.scaled_dot_product_attention))
        for d, k, new, _ in self.saved:
            d[k] = new
        return self

    def __exit__(self, *exc):
        for d, k, _, old in self.saved:
            d[k] = old
        return False


def _dropout(draws: "_Draws") -> Callable:
    """torch.nn.functional.dropout with the hardware's numbers (as torch_ops.dropout)."""
    import torch

    def dropout(input, p=0.5, training=True, inplace=False):
        if not training or p == 0:
            return input
        u = torch.tensor(draws.random(tuple(input.shape)), dtype=input.dtype)
        return input * ((u > p).to(input.dtype) * (1.0 / (1.0 - p)))
    return dropout


def _attention(dropout: Callable) -> Callable:
    """torch.nn.functional.scaled_dot_product_attention as its documentation's
    reference computes it (as torch_ops builds it), its dropout `dropout`."""
    import math
    import torch

    def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
                                     is_causal=False, scale=None, enable_gqa=False):
        if attn_mask is not None or enable_gqa:
            raise NotTranslatable("scaled_dot_product_attention: attn_mask / enable_gqa are not "
                                  "supported")
        L, S = query.size(-2), key.size(-2)
        w = query @ key.transpose(-2, -1) * (scale if scale is not None
                                             else 1.0 / math.sqrt(query.size(-1)))
        if is_causal:
            w = w.masked_fill(~torch.ones(L, S, dtype=torch.bool).tril(), float("-inf"))
        w = dropout(torch.softmax(w, dim=-1), dropout_p, True)
        return w @ value
    return scaled_dot_product_attention


def _flat(result: Any) -> Any:
    """A result (arrays, tensors, numbers, tuples of them) as a flat float array."""
    np = _np()
    parts = result if isinstance(result, (tuple, list)) else (result,)
    parts = [x for x in parts if x is not None] if len(parts) > 1 else parts   # (logits, None)
    return np.concatenate([np.ravel(_values(x)) if _is_array(x) or isinstance(x, np.ndarray)
                           else np.ravel(np.asarray(x, dtype=float)) for x in parts]) \
        if parts else np.zeros(0)


def _check_loop(tr: Translation, fn: Callable, original: Any, loop, steps: List[Any]) -> None:
    """A loop step against the original loop: the original runs over the first two
    batches (on a copy of its objects), the program twice; the objects' state must
    agree."""
    np = _np()
    loader = loop[0]
    memo: Dict[int, Any] = {}
    mine = copy.deepcopy(original, memo)
    sig = inspect.signature(fn)
    args = [steps[:2] if p == loader else getattr(mine, p)
            for p in sig.parameters if p == loader or hasattr(mine, p)]
    import warnings
    with _drawing(getattr(tr, "generators", {})), warnings.catch_warnings():
        warnings.simplefilter("ignore")                 # (torch's, about the copies)
        fn(*args)
    state = {name: list(tr.attr[name]["init"]) for name in tr.memories}
    for s in steps[:2]:
        tr.fast(*tr.adapt(s), *state.values())
    tol = dict(rtol=1e-3, atol=1e-4) if tr.torch else dict(rtol=1e-6, atol=1e-9)
    for name, (label, _, owner) in tr.memories.items():
        if label.endswith("." + RANDOM_STATE) or label.startswith(tr.carried + "."):
            continue
        path = tr.paths.get(name)
        value = (_resolve(mine, path) if path is not None
                 else getattr(memo.get(id(owner), owner), label.split(".", 1)[1]))
        want, got = np.ravel(_values(value)), np.asarray(state[name], dtype=float)
        if not np.allclose(want, got, **tol):
            bad = ~np.isclose(want, got, **tol)
            raise NotTranslatable(
                f"The loop translation of {fn.__name__} changes {label} differently from the "
                f"original: {bad.sum()} of {bad.size} values, by up to "
                f"{np.abs(want - got).max():.3g} (e.g. {got[bad][:3]} vs {want[bad][:3]})")


def _check(tr: Translation, func: Callable, samples: List[Any]) -> None:
    """The program in float against the original on a few examples (with state:
    on a copy of the object, one example after another)."""
    np = _np()
    obj = func.__self__ if inspect.ismethod(func) else None
    memo: Dict[int, Any] = {}
    if obj is not None:
        func = getattr(copy.deepcopy(obj, memo), func.__name__)
    state = {name: list(tr.attr[name]["init"]) for name in tr.memories}
    with _drawing(getattr(tr, "generators", {})):
        _check_runs(tr, func, samples, memo, state)


def _check_runs(tr, func, samples, memo, state) -> None:
    np = _np()
    # (torch computes in float32)
    tol = dict(rtol=1e-3, atol=1e-4) if getattr(tr, "torch", False) else dict(rtol=1e-6, atol=1e-9)
    for s in samples[:2 if tr.memories else 3]:
        args = s if isinstance(s, tuple) else (s,)
        want = _flat(func(*args))
        got = tr.fast(*tr.adapt(args), *state.values())
        got = np.ravel(np.asarray([x for g in (got if isinstance(got, tuple) else (got,))
                                   for x in (g if isinstance(g, list) else [g])], dtype=float))
        if want.shape != got.shape or not np.allclose(want, got, **tol):
            raise NotTranslatable(f"The loop translation of {getattr(func, '__name__', func)} "
                                  f"disagrees with it on an example: {got[:6]} vs {want[:6]}")
        for name, (label, _, owner) in tr.memories.items():
            if label.endswith("." + RANDOM_STATE):
                continue
            mine = memo.get(id(owner), owner)
            path = getattr(tr, "paths", {}).get(name)
            value = (_resolve(mine, path) if path is not None
                     else getattr(mine, label.split(".", 1)[1]))
            if not np.allclose(np.ravel(_values(value)), state[name], **tol):
                raise NotTranslatable(f"The loop translation of {func.__name__} changes "
                                      f"{label} differently from the original")
