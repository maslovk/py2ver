# numpy_fe.py
"""
Python with numpy arrays and methods, as plain scalar Python for py2ver.

A function using numpy (or a method of an object, e.g. net.predict) is
translated into an ordinary scalar function that the rest of py2ver builds:

  - calls of the object's methods (self.forward(x)) and of your functions
    are inlined; a function of one value that uses np.exp / np.log / ... or
    divides by a variable (sigmoid) is kept as a table of the whole function
  - the object's attributes: read and never written -> constants (trained
    weights); read, then written (self.W1 -= ...) -> state, starting from
    the object's values; written before read (self.h = ...) -> variables
  - arrays have the shapes of the example inputs, so every numpy operation
    is written out element by element: @ (sums of products), + - * / and **
    with broadcasting, .T, .sum(axis, keepdims), np.mean, np.sum, len,
    np.atleast_2d, np.zeros / np.ones, np.maximum / np.minimum, abs, and
    elementwise np.tanh / np.exp / np.log / ... (tables)
  - a method whose body is a loop with print / break on a condition (a
    training loop) is built as one step of the loop: the step's variables
    used by the condition (the loss) are its results, and the loop itself
    stays in Python

The generated function is checked against the original on the example
inputs (in float) when it is made.
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
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# numpy functions applied elementwise as tables
TABLE_UFUNCS = {"tanh", "exp", "log", "log2", "log10", "sqrt", "sin", "cos", "tan", "arctan",
                "arcsin", "arccos", "sinh", "cosh", "exp2", "expm1", "log1p"}
TABLES_KEY = "__py2ver_tables__"          # names of functions built as tables (in a namespace)


class NotTranslatable(NotImplementedError):
    """Code the numpy front end can't write out as scalars."""


def _err(node: Any, msg: str):
    line = getattr(node, "lineno", None)
    raise NotTranslatable(f"{msg} (line {line})" if line else msg)


def _np():
    import numpy
    return numpy


def needs_frontend(func: Callable) -> bool:
    """A method, or a function using numpy (or torch)."""
    if inspect.ismethod(func):
        return True
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    except (OSError, TypeError, SyntaxError):
        return False
    ns = getattr(func, "__globals__", {})
    np_names = {k for k, v in ns.items() if inspect.ismodule(v) and (
        v.__name__ == "numpy" or v.__name__ == "torch" or v.__name__.startswith("torch."))}
    return any(isinstance(n, ast.Name) and n.id in np_names for n in ast.walk(tree))


def _parse(fn: Callable) -> ast.FunctionDef:
    fd = ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]
    if not isinstance(fd, ast.FunctionDef):
        raise NotTranslatable(f"{fn} is not a plain function or method")
    fd.decorator_list = []
    return fd


def _is_print(s: ast.stmt) -> bool:
    return (isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)
            and isinstance(s.value.func, ast.Name) and s.value.func.id == "print")


def _is_control(s: ast.stmt) -> bool:
    """print / break / continue / pass, or an if of only those."""
    if _is_print(s) or isinstance(s, (ast.Break, ast.Continue, ast.Pass)):
        return True
    return isinstance(s, ast.If) and all(_is_control(x) for x in s.body + s.orelse)


def _names(node: ast.AST, ctx=ast.Load) -> set:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ctx)}


# ----------------------------- inlining methods and functions -----------------------------

class _Inliner:
    def __init__(self, obj: Any, ns: Dict[str, Any]) -> None:
        self.obj = obj
        self.ns = ns
        self.counter = itertools.count()
        self.stack: List[str] = []
        self.tables: set = set()

    def _target(self, call: ast.Call) -> Optional[Callable]:
        """The function a call runs, if it is to be inlined (or None)."""
        f = call.func
        if (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "self"
                and self.obj is not None):
            m = getattr(self.obj, f.attr, None)
            return m if inspect.ismethod(m) else None
        if isinstance(f, ast.Name):
            fn = self.ns.get(f.id)
            if inspect.isfunction(fn) and not fn.__module__.startswith("numpy"):
                if self._is_table(fn):
                    self.tables.add(f.id)
                    return None
                return fn
        return None

    def _is_table(self, fn: Callable) -> bool:
        """A function of one value using np.exp & co. or dividing by a variable."""
        fd = _parse(fn)
        if len(fd.args.args) != 1:
            return False
        for n in ast.walk(fd):
            if isinstance(n, ast.Call):
                g = _dotted(n.func, self.ns)
                if getattr(g, "__name__", None) in TABLE_UFUNCS or \
                        getattr(g, "__module__", None) == "math":
                    return True
            if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div) and \
                    not isinstance(n.right, ast.Constant):
                return True
        return False

    def block(self, stmts: List[ast.stmt]) -> List[ast.stmt]:
        out: List[ast.stmt] = []
        for s in stmts:
            if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call) and self._target(s.value):
                out += self.inline(s.value)[0]                  # self.backward(Y, lr)
                continue
            pre: List[ast.stmt] = []
            inl = self

            class _Calls(ast.NodeTransformer):
                def visit_Call(self, call):
                    self.generic_visit(call)
                    if inl._target(call) is None:
                        return call
                    stmts_, value = inl.inline(call)
                    pre.extend(stmts_)
                    if value is None:
                        _err(call, "A function without a return value is used as a value")
                    return value

            for field_ in ("value", "test", "iter"):
                if getattr(s, field_, None) is not None:
                    setattr(s, field_, _Calls().visit(getattr(s, field_)))
            for field_ in ("body", "orelse"):
                if isinstance(getattr(s, field_, None), list):
                    setattr(s, field_, self.block(getattr(s, field_)))
            out += pre + [s]
        return out

    def inline(self, call: ast.Call) -> Tuple[List[ast.stmt], Optional[ast.expr]]:
        fn = self._target(call)
        name = fn.__name__
        if name in self.stack:
            _err(call, f"Recursive call to {name}")
        fd = _parse(fn)
        params = [a.arg for a in fd.args.args]
        if inspect.ismethod(fn):
            params = params[1:]                                  # self stays self
        sig = inspect.signature(fn)
        try:
            bound = sig.bind(*call.args, **{k.arg: k.value for k in call.keywords})
        except TypeError as e:
            _err(call, f"Call of {name}: {e}")
        args = {}
        for p in params:
            if p in bound.arguments:
                args[p] = bound.arguments[p]
            else:
                default = sig.parameters[p].default
                if not isinstance(default, (int, float)):
                    _err(call, f"Call of {name}: give '{p}'")
                args[p] = ast.Constant(default)
        k = next(self.counter)
        prefix = f"{name}{k}_"
        body = [s for s in fd.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
        local = set(params) | {n for s in body for n in _names(s, ast.Store)}
        local.discard("self")

        class _Rename(ast.NodeTransformer):
            def visit_Name(self, node):
                if node.id in local:
                    return ast.copy_location(ast.Name(id=prefix + node.id, ctx=node.ctx), node)
                return node

        body = [_Rename().visit(copy.deepcopy(s)) for s in body]
        if any(isinstance(n, ast.Return) for s in body[:-1] for n in ast.walk(s)):
            _err(call, f"{name} may only return at its end")
        out = [ast.Assign(targets=[ast.Name(id=prefix + p, ctx=ast.Store())], value=args[p])
               for p in params]
        ret = body[-1] if body and isinstance(body[-1], ast.Return) else None
        self.stack.append(name)
        try:
            out += self.block(body[:-1] if ret else body)
            value = None
            if ret is not None and ret.value is not None:
                out += self.block([ast.Assign(targets=[ast.Name(id=prefix + "ret", ctx=ast.Store())],
                                              value=ret.value)])
                value = ast.Name(id=prefix + "ret", ctx=ast.Load())
        finally:
            self.stack.pop()
        return out, value


def _dotted(node: ast.expr, ns: Dict[str, Any]) -> Any:
    """The value of a dotted global name (np.tanh), or None."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    if node.id in ns:
        v = ns[node.id]
    elif hasattr(builtins, node.id):                 # len, abs, float, ...
        v = getattr(builtins, node.id)
    else:
        return None
    for p in reversed(parts):
        v = getattr(v, p, None)
    return v


# ----------------------------- arrays as elements -----------------------------

@dataclass
class Arr:
    shape: Tuple[int, ...]
    elems: List[ast.expr]          # row-major

    def at(self, idx: Tuple[int, ...]) -> ast.expr:
        flat = 0
        for i, n in zip(idx, self.shape):
            flat = flat * n + i
        return copy.deepcopy(self.elems[flat])


def _indices(shape: Tuple[int, ...]):
    return itertools.product(*[range(n) for n in shape]) if shape else [()]


def _const_value(e: ast.expr) -> Optional[float]:
    if isinstance(e, ast.Constant) and isinstance(e.value, (int, float)) and not isinstance(e.value, bool):
        return e.value
    if isinstance(e, ast.UnaryOp) and isinstance(e.op, ast.USub):
        v = _const_value(e.operand)
        return None if v is None else -v
    return None


def _num(v: float) -> ast.expr:
    v = int(v) if float(v).is_integer() and abs(v) < 2 ** 53 else float(v)
    return ast.UnaryOp(op=ast.USub(), operand=ast.Constant(-v)) if v < 0 else ast.Constant(v)


def _bin(a: ast.expr, op: ast.operator, b: ast.expr) -> ast.expr:
    """a op b, simplified when a side is the constant 0 or 1."""
    ca, cb = _const_value(a), _const_value(b)
    if ca is not None and cb is not None and not isinstance(op, (ast.Div, ast.Pow)):
        return _num({ast.Add: ca + cb, ast.Sub: ca - cb, ast.Mult: ca * cb}[type(op)])
    if isinstance(op, ast.Add):
        if ca == 0:
            return b
        if cb == 0:
            return a
    if isinstance(op, ast.Sub) and cb == 0:
        return a
    if isinstance(op, ast.Mult):
        if ca == 0 or cb == 0:
            return ast.Constant(0)
        if ca == 1:
            return b
        if cb == 1:
            return a
    return ast.BinOp(left=a, op=op, right=b)


def _broadcast(a: Tuple[int, ...], b: Tuple[int, ...], node) -> Tuple[int, ...]:
    n = max(len(a), len(b))
    a, b = (1,) * (n - len(a)) + a, (1,) * (n - len(b)) + b
    out = []
    for x, y in zip(a, b):
        if x != y and 1 not in (x, y):
            _err(node, f"Shapes {a} and {b} don't broadcast")
        out.append(max(x, y))
    return tuple(out)


def _pick(arr: Arr, out_shape: Tuple[int, ...], idx: Tuple[int, ...]) -> ast.expr:
    """arr's element for index idx of a broadcast result."""
    s = arr.shape
    idx = idx[len(idx) - len(s):] if s else ()
    return arr.at(tuple(0 if n == 1 else i for i, n in zip(idx, s)))


class _Scalar:
    """Write a straight-line numpy body out as scalar statements."""

    def __init__(self, ns: Dict[str, Any], shapes: Dict[str, Tuple[int, ...]]) -> None:
        self.ns = ns
        self.shape = dict(shapes)          # runtime variable -> shape (() for a number)
        self.const: Dict[str, Any] = {}    # translation-time numbers
        self.alias: Dict[str, Arr] = {}    # plain copies: name -> the copied elements
        self.out: List[ast.stmt] = []
        self.n_tmp = 0
        self.keep: set = set()             # names that must exist (state)

    # ---- values ----
    def var(self, name: str) -> Arr:
        shape = self.shape[name]
        if shape == ():
            return Arr((), [ast.Name(id=name, ctx=ast.Load())])
        return Arr(shape, [ast.Subscript(value=ast.Name(id=name, ctx=ast.Load()), slice=ast.Constant(k),
                                         ctx=ast.Load()) for k in range(math.prod(shape))])

    def constant(self, v: Any, node) -> Arr:
        np = _np()
        if isinstance(v, (bool, np.bool_)):
            _err(node, "Booleans are not supported here")
        if isinstance(v, (int, float, np.integer, np.floating)):
            return Arr((), [_num(float(v) if isinstance(v, (float, np.floating)) else int(v))])
        if isinstance(v, (list, tuple, np.ndarray)):
            a = np.asarray(v, dtype=float)
            return Arr(tuple(a.shape), [_num(float(x)) for x in a.ravel()])
        _err(node, f"Can't use {type(v).__name__} values in hardware")

    def materialize(self, a: Arr) -> Arr:
        """a held in a temporary, so its elements are names (to use them more than once)."""
        if all(isinstance(e, (ast.Name, ast.Subscript, ast.Constant)) for e in a.elems):
            return a
        name = f"np_t{self.n_tmp}"
        self.n_tmp += 1
        self.assign(name, a)
        return self.var(name)

    def value(self, n: ast.expr) -> Arr:
        np = _np()
        if isinstance(n, ast.Constant):
            return self.constant(n.value, n)
        if isinstance(n, ast.Name):
            if n.id in self.const:
                return self.constant(self.const[n.id], n)
            if n.id in self.alias:
                a = self.alias[n.id]
                return Arr(a.shape, [copy.deepcopy(e) for e in a.elems])
            if n.id in self.shape:
                return self.var(n.id)
            if n.id in self.ns:
                return self.constant(self.ns[n.id], n)
            _err(n, f"'{n.id}' has no value here")
        if isinstance(n, ast.Attribute):
            if n.attr == "T":
                a = self.value(n.value)
                if len(a.shape) < 2:
                    return a
                if len(a.shape) != 2:
                    _err(n, ".T of an array of more than 2 dimensions")
                r, c = a.shape
                return Arr((c, r), [a.at((i, j)) for j in range(c) for i in range(r)])
            v = _dotted(n, self.ns)
            if v is None:
                _err(n, f"Unsupported attribute .{n.attr}")
            return self.constant(v, n)
        if isinstance(n, ast.UnaryOp):
            a = self.value(n.operand)
            if isinstance(n.op, ast.USub):
                return Arr(a.shape, [_num(-c) if (c := _const_value(e)) is not None
                                     else ast.UnaryOp(op=ast.USub(), operand=e) for e in a.elems])
            if isinstance(n.op, ast.UAdd):
                return a
            _err(n, f"Unsupported {type(n.op).__name__}")
        if isinstance(n, ast.BinOp):
            return self.binop(n)
        if isinstance(n, ast.Call):
            return self.call(n)
        if isinstance(n, ast.Subscript):
            a = self.value(n.value)
            parts = n.slice.elts if isinstance(n.slice, ast.Tuple) else [n.slice]
            if len(parts) > len(a.shape):
                _err(n, "Too many indices")
            picks = []                          # per dimension: an int (dropped) or a range
            for d, part in enumerate(parts):
                if isinstance(part, ast.Slice):
                    bounds = [None if b is None else self.number(b)
                              for b in (part.lower, part.upper, part.step)]
                    if not all(b is None or isinstance(b, int) for b in bounds):
                        _err(n, "Only constant integer indices and slices are supported")
                    picks.append(range(a.shape[d])[slice(*bounds)])
                else:
                    i = self.number(part)
                    if not isinstance(i, int):
                        _err(n, "Only constant integer indices and slices are supported")
                    picks.append(i % a.shape[d])
            picks += [range(m) for m in a.shape[len(parts):]]
            shape = tuple(len(p) for p in picks if isinstance(p, range))
            elems = []
            for combo in itertools.product(*[p if isinstance(p, range) else [p] for p in picks]):
                elems.append(a.at(tuple(combo)))
            return Arr(shape, elems)
        if isinstance(n, (ast.List, ast.Tuple)):
            parts = [self.value(e) for e in n.elts]
            if any(p.shape != parts[0].shape for p in parts):
                _err(n, "Ragged lists are not supported")
            return Arr((len(parts),) + (parts[0].shape if parts else ()),
                       [e for p in parts for e in p.elems])
        _err(n, f"Unsupported expression: {ast.unparse(n)}")

    def number(self, n: ast.expr) -> Any:
        """A value known at translation time."""
        if isinstance(n, ast.Constant):
            return n.value
        a = self.value(n) if not isinstance(n, ast.Tuple) else None
        if isinstance(n, ast.Tuple):
            return tuple(self.number(e) for e in n.elts)
        if a.shape != () or _const_value(a.elems[0]) is None:
            _err(n, f"'{ast.unparse(n)}' must be known at translation time")
        return _const_value(a.elems[0])

    def binop(self, n: ast.BinOp) -> Arr:
        a, b = self.value(n.left), self.value(n.right)
        op = n.op
        if isinstance(op, ast.MatMult):
            return self.matmul(a, b, n)
        if isinstance(op, ast.Pow):
            k = _const_value(b.elems[0]) if b.shape == () else None
            if k is None or k != int(k) or k < 1:
                _err(n, "** needs a positive integer constant exponent")
            a = self.materialize(a)
            elems = []
            for e in a.elems:
                x = copy.deepcopy(e)
                for _ in range(int(k) - 1):
                    x = _bin(x, ast.Mult(), copy.deepcopy(e))
                elems.append(x)
            return Arr(a.shape, elems)
        if not isinstance(op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            _err(n, f"Unsupported operator {type(op).__name__}")
        shape = _broadcast(a.shape, b.shape, n)
        return Arr(shape, [_bin(_pick(a, shape, i), copy.deepcopy(op), _pick(b, shape, i))
                           for i in _indices(shape)])

    def matmul(self, a: Arr, b: Arr, node) -> Arr:
        sa, sb = a.shape, b.shape
        if len(sa) == 1:
            a, sa = Arr((1,) + sa, a.elems), (1,) + sa
            squeeze_a = True
        else:
            squeeze_a = False
        if len(sb) == 1:
            b, sb = Arr(sb + (1,), b.elems), sb + (1,)
            squeeze_b = True
        else:
            squeeze_b = False
        if len(sa) != 2 or len(sb) != 2 or sa[1] != sb[0]:
            _err(node, f"@ of shapes {a.shape} and {b.shape}")
        m, k, nn = sa[0], sa[1], sb[1]
        elems = []
        for i in range(m):
            for j in range(nn):
                acc: ast.expr = ast.Constant(0)
                for t in range(k):
                    acc = _bin(acc, ast.Add(), _bin(a.at((i, t)), ast.Mult(), b.at((t, j))))
                elems.append(acc)
        shape = ((m,) if not squeeze_a else ()) + ((nn,) if not squeeze_b else ())
        return Arr(shape, elems)

    def reduce(self, a: Arr, axis: Any, keepdims: bool, mean: bool, node) -> Arr:
        if axis is None:
            total: ast.expr = ast.Constant(0)
            for e in a.elems:
                total = _bin(total, ast.Add(), copy.deepcopy(e))
            if mean:
                total = ast.BinOp(left=total, op=ast.Div(), right=ast.Constant(len(a.elems)))
            shape = (1,) * len(a.shape) if keepdims else ()
            return Arr(shape, [total])
        axis = axis % len(a.shape)
        shape = tuple(1 if d == axis else n for d, n in enumerate(a.shape))
        elems = []
        for idx in _indices(shape):
            total = ast.Constant(0)
            for t in range(a.shape[axis]):
                total = _bin(total, ast.Add(), a.at(idx[:axis] + (t,) + idx[axis + 1:]))
            if mean:
                total = ast.BinOp(left=total, op=ast.Div(), right=ast.Constant(a.shape[axis]))
            elems.append(total)
        if not keepdims:
            shape = tuple(n for d, n in enumerate(shape) if d != axis)
        return Arr(shape, elems)

    def call(self, n: ast.Call) -> Arr:
        np = _np()
        f = n.func
        kw = {k.arg: self.number(k.value) for k in n.keywords if k.arg in ("axis", "keepdims")}
        if any(k.arg not in ("axis", "keepdims") for k in n.keywords):
            _err(n, f"Unsupported keyword in {ast.unparse(f)}")
        # Methods of arrays: a.sum(...), a.mean()
        if isinstance(f, ast.Attribute) and f.attr in ("sum", "mean") and _dotted(f, self.ns) is None:
            return self.reduce(self.value(f.value), kw.get("axis"), bool(kw.get("keepdims")),
                               f.attr == "mean", n)
        fn = _dotted(f, self.ns)
        name = getattr(fn, "__name__", None)
        if isinstance(f, ast.Name) and f.id in self.ns.get(TABLES_KEY, ()):
            a = self.value(n.args[0])                        # your function, as a table
            return Arr(a.shape, [ast.Call(func=ast.Name(id=f.id, ctx=ast.Load()), args=[e], keywords=[])
                                 for e in a.elems])
        if fn is len:
            return Arr((), [ast.Constant(self.value(n.args[0]).shape[0])])
        if fn is abs or fn is getattr(np, "abs", None) or fn is getattr(np, "absolute", None):
            a = self.value(n.args[0])
            return Arr(a.shape, [ast.Call(func=ast.Name(id="abs", ctx=ast.Load()), args=[e], keywords=[])
                                 for e in a.elems])
        if isinstance(fn, np.ufunc) and name in TABLE_UFUNCS:
            a = self.value(n.args[0])
            return Arr(a.shape, [ast.Call(func=copy.deepcopy(f), args=[e], keywords=[]) for e in a.elems])
        if fn in (np.maximum, np.minimum):
            a, b = self.value(n.args[0]), self.value(n.args[1])
            shape = _broadcast(a.shape, b.shape, n)
            g = "max" if fn is np.maximum else "min"
            return Arr(shape, [ast.Call(func=ast.Name(id=g, ctx=ast.Load()),
                                        args=[_pick(a, shape, i), _pick(b, shape, i)], keywords=[])
                               for i in _indices(shape)])
        if fn in (np.sum, np.mean):
            return self.reduce(self.value(n.args[0]), kw.get("axis"), bool(kw.get("keepdims")),
                               fn is np.mean, n)
        if fn is np.dot or fn is np.matmul:
            return self.matmul(self.value(n.args[0]), self.value(n.args[1]), n)
        if fn is np.atleast_2d:
            a = self.value(n.args[0])
            return Arr((1,) * (2 - len(a.shape)) + a.shape if len(a.shape) < 2 else a.shape, a.elems)
        if fn in (np.zeros, np.ones):
            shape = self.number(n.args[0])
            shape = tuple(shape) if isinstance(shape, tuple) else (int(shape),)
            v = 0 if fn is np.zeros else 1
            return Arr(shape, [ast.Constant(v) for _ in range(math.prod(shape))])
        if fn in (np.array, np.asarray, float):
            return self.value(n.args[0])
        _err(n, f"Unsupported call: {ast.unparse(f)}")

    # ---- statements ----
    def _is_copy(self, a: Arr) -> bool:
        """Every element a variable's element or a variable: a plain copy."""
        return all(isinstance(e, ast.Name) or (isinstance(e, ast.Subscript)
                                                and isinstance(e.value, ast.Name)
                                                and isinstance(e.slice, ast.Constant))
                   for e in a.elems)

    def assign(self, name: str, a: Arr) -> None:
        # Copies that read this variable are written out before it changes
        for other, al in list(self.alias.items()):
            if other != name and any(isinstance(x, ast.Name) and x.id == name
                                     for e in al.elems for x in ast.walk(e)):
                del self.alias[other]
                self._emit(other, al)
        self.alias.pop(name, None)
        if a.shape == () and _const_value(a.elems[0]) is not None:
            self.const[name] = _const_value(a.elems[0])           # n = len(y), eps = 1e-9
            self.shape.pop(name, None)
            return
        self.const.pop(name, None)
        if self._is_copy(a) and name not in self.keep:           # x2 = x: no new signal
            self.alias[name] = a
            self.shape.pop(name, None)
            return
        self._emit(name, a)

    def _emit(self, name: str, a: Arr) -> None:
        value = a.elems[0] if a.shape == () else ast.List(elts=a.elems, ctx=ast.Load())
        self.out.append(ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=value))
        self.shape[name] = a.shape

    def finish(self) -> None:
        """Copies still needed under their own names (state written at the end)."""
        for name in list(self.alias):
            if name in self.keep:
                self._emit(name, self.alias.pop(name))

    def stmt(self, s: ast.stmt) -> None:
        if isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name):
            self.assign(s.targets[0].id, self.value(s.value))
        elif isinstance(s, ast.AugAssign) and isinstance(s.target, ast.Name):
            load = ast.Name(id=s.target.id, ctx=ast.Load())
            self.assign(s.target.id, self.value(ast.BinOp(left=load, op=s.op, right=s.value)))
        elif isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant):
            pass
        elif isinstance(s, ast.Pass):
            pass
        else:
            _err(s, f"Unsupported statement here: {ast.unparse(s).splitlines()[0]}")


# ----------------------------- the translation -----------------------------

@dataclass
class Translation:
    func: Callable                       # the generated scalar function
    source: str
    attr: Dict[str, Dict[str, Any]]      # lists, state
    params: List[str]                    # the original's arguments that are inputs, in order
    shapes: Dict[str, Tuple[int, ...]]
    outputs: List[Tuple[str, Tuple[int, ...]]]
    state: Dict[str, Any] = field(default_factory=dict)      # attribute -> its initial value
    loop: bool = False
    tables: set = field(default_factory=set)

    def adapt(self, args: Any) -> tuple:
        """Arguments as the original takes them -> as the generated function does."""
        np = _np()
        if isinstance(args, list) and len(self.params) > 1 and len(args) == len(self.params):
            args = tuple(args)                   # (a batch from a DataLoader: a list)
        if not isinstance(args, tuple):
            args = (args,)
        out = []
        for name, v in zip(self.params, args):
            a = np.asarray(v, dtype=float)
            out.append(a.ravel().tolist() if self.shapes[name] != () else float(a))
        return tuple(out)


def translate(func: Callable, samples: List[Any]) -> Translation:
    np = _np()
    if not samples:
        raise NotTranslatable("A numpy design needs example inputs (samples=...) for its shapes")
    obj = func.__self__ if inspect.ismethod(func) else None
    fn = func.__func__ if obj is not None else func
    ns = dict(fn.__globals__)
    fd = _parse(fn)
    params = [a.arg for a in fd.args.args][1 if obj is not None else 0:]
    first = samples[0] if isinstance(samples[0], tuple) else (samples[0],)
    given = params[:len(first)]
    sig = inspect.signature(func)
    body = [s for s in fd.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]

    # A training loop: one step of it, its results the variables its control uses
    loop = None
    for s in body:
        if isinstance(s, ast.For) and any(isinstance(n, ast.Break) for n in ast.walk(s)):
            loop = s
    results: List[str] = []
    if loop is not None:
        if body.index(loop) != 0 or not all(isinstance(s, ast.Return) for s in body[1:]):
            _err(loop, "A training loop must be the whole method (then a return)")
        control = [s for s in loop.body if _is_control(s)]
        step = [s for s in loop.body if not _is_control(s)]
        if isinstance(loop.target, ast.Name) and loop.target.id in {n for s in step for n in _names(s)}:
            _err(loop, f"The loop variable '{loop.target.id}' is used by the step itself")
        assigned = {n for s in step for n in _names(s, ast.Store)}
        used_after = {n for s in control + body[1:] for n in _names(s)}
        results = sorted(assigned & used_after)
        body = step + [ast.Return(value=ast.Tuple(elts=[ast.Name(id=r, ctx=ast.Load()) for r in results],
                                                   ctx=ast.Load()))]
    # Inputs: the given arguments the body uses; the others are their defaults
    used = {n for s in body for n in _names(s)}
    inputs = [p for p in given if p in used or loop is None]
    consts = {}
    for p in params:
        if p not in given:
            d = sig.parameters[p].default
            if d is inspect.Parameter.empty:
                raise NotTranslatable(f"Give '{p}' in the samples (it has no default)")
            consts[p] = d

    inl = _Inliner(obj, ns)
    body = inl.block(copy.deepcopy(body))      # (the original stays as it is, for the check)
    ns[TABLES_KEY] = set(inl.tables)

    # self.<a>: constants, state or variables
    first_access: Dict[str, str] = {}
    written: set = set()

    class _Attrs(ast.NodeVisitor):
        def visit_Attribute(self, node):
            if isinstance(node.value, ast.Name) and node.value.id == "self":
                kind = "w" if isinstance(node.ctx, ast.Store) else "r"
                first_access.setdefault(node.attr, kind)
                if kind == "w":
                    written.add(node.attr)
            self.generic_visit(node)

    for s in body:                     # loads of a statement come before its stores
        if isinstance(s, (ast.Assign, ast.AugAssign)):
            _Attrs().visit(s.value)
            if isinstance(s, ast.AugAssign):
                _Attrs().visit(ast.Attribute(value=s.target.value, attr=s.target.attr, ctx=ast.Load())
                               if isinstance(s.target, ast.Attribute) else s.target)
            for t in (s.targets if isinstance(s, ast.Assign) else [s.target]):
                _Attrs().visit(t)
        else:
            _Attrs().visit(s)
    state = {a: np.array(getattr(obj, a), dtype=float) for a, k in first_access.items()
             if k == "r" and a in written}
    for a, k in first_access.items():
        if k == "r" and a not in written:
            if obj is None or not hasattr(obj, a):
                raise NotTranslatable(f"self.{a} is read but has no value")
            ns[f"self_{a}"] = getattr(obj, a)                  # a constant (weights)

    class _SelfNames(ast.NodeTransformer):
        def visit_Attribute(self, node):
            self.generic_visit(node)
            if isinstance(node.value, ast.Name) and node.value.id == "self":
                return ast.copy_location(ast.Name(id=f"self_{node.attr}", ctx=node.ctx), node)
            return node

    body = [_SelfNames().visit(s) for s in body]
    if any(isinstance(n, ast.Name) and n.id == "self" for s in body for n in ast.walk(s)):
        raise NotTranslatable("self is used other than as self.<attribute>")

    # Shapes of the inputs (from the first sample) and the state
    shapes = {p: tuple(np.asarray(v, dtype=float).shape) for p, v in zip(given, first)}
    for a, v in state.items():
        shapes[f"self_{a}"] = tuple(v.shape)
    sc = _Scalar(ns, {p: shapes[p] for p in inputs} | {f"self_{a}": shapes[f"self_{a}"] for a in state})
    sc.const.update(consts)
    sc.keep = {f"self_{a}" for a in state}
    ret = body[-1] if body and isinstance(body[-1], ast.Return) else None
    for s in (body[:-1] if ret is not None else body):
        sc.stmt(s)
    outs: List[Tuple[str, Tuple[int, ...]]] = []
    if ret is not None and ret.value is not None:
        vals = ret.value.elts if isinstance(ret.value, ast.Tuple) else [ret.value]
        for i, v in enumerate(vals):
            name = results[i] if loop is not None else ("out" if len(vals) == 1 else f"out{i}")
            a = sc.value(v)
            out_name = f"r_{name}" if loop is not None else name
            sc.out.append(ast.Assign(targets=[ast.Name(id=out_name, ctx=ast.Store())],
                                     value=a.elems[0] if a.shape == () else
                                     ast.List(elts=a.elems, ctx=ast.Load())))
            outs.append((out_name, a.shape))
    for a in state:                                           # the state's next values
        outs.append((f"self_{a}", shapes[f"self_{a}"]))

    gen_params = list(inputs) + [f"self_{a}" for a in state]
    gen_name = f"{fn.__name__}_step" if loop is not None else fn.__name__
    fdef = ast.FunctionDef(
        name=gen_name, args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=p) for p in gen_params],
                                          kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=sc.out + [ast.Return(value=ast.Tuple(elts=[ast.Name(id=n, ctx=ast.Load()) for n, _ in outs],
                                                  ctx=ast.Load()) if len(outs) > 1
                                  else ast.Name(id=outs[0][0], ctx=ast.Load()))],
        decorator_list=[], returns=None, type_comment=None)
    module = ast.fix_missing_locations(ast.Module(body=[fdef], type_ignores=[]))
    source = ast.unparse(module) + "\n"
    filename = f"<py2ver numpy {fn.__qualname__}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    exec(compile(source, filename, "exec"), ns)
    gen = ns[gen_name]

    attr: Dict[str, Dict[str, Any]] = {}
    for p in inputs:
        if shapes[p] != ():
            attr[p] = {"length": math.prod(shapes[p])}
    for a, v in state.items():
        m = {"next": f"self_{a}", "init": v.ravel().tolist() if v.shape else float(v)}
        if v.shape:
            m["length"] = v.size
        attr[f"self_{a}"] = m
    tr = Translation(gen, source, attr, inputs, {p: shapes[p] for p in inputs}, outs, state,
                     loop is not None, set(inl.tables))
    _self_check(func, tr, samples, obj, loop, results, ns, fd, given)
    return tr


def _self_check(func, tr: Translation, samples, obj, loop, results, ns, fd, given) -> None:
    """The generated function in float must compute what the original does."""
    np = _np()
    state = {a: v.ravel().tolist() if v.shape else float(v) for a, v in tr.state.items()}
    if loop is not None:
        # The original step: the loop's body without its control, on a copy of the object
        step = [s for s in loop.body if not _is_control(s)]
        src_fd = ast.FunctionDef(
            name="__py2ver_step", args=ast.arguments(
                posonlyargs=[], args=[ast.arg(arg=a) for a in ["self"] + list(given)],
                kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=copy.deepcopy(step) + [ast.Return(value=ast.Tuple(
                elts=[ast.Name(id=r, ctx=ast.Load()) for r in results], ctx=ast.Load()))],
            decorator_list=[], returns=None, type_comment=None)
        env = dict(ns)
        params = inspect.signature(func).parameters
        for p in params:
            if p not in given and params[p].default is not inspect.Parameter.empty:
                env[p] = params[p].default
        exec(compile(ast.fix_missing_locations(ast.Module(body=[src_fd], type_ignores=[])),
                     "<py2ver step>", "exec"), env)
        original = env["__py2ver_step"]
        twin = copy.deepcopy(obj)
    for k, sample in enumerate(samples[:8]):
        args = sample if isinstance(sample, tuple) else (sample,)
        gargs = tr.adapt(args)
        if loop is not None:
            want = list(np.ravel(np.asarray(original(twin, *args), dtype=float)))
            want += [x for a in tr.state for x in np.ravel(getattr(twin, a))]
        else:
            twin = copy.deepcopy(obj) if obj is not None else None
            r = getattr(twin, func.__name__)(*args) if obj is not None else func(*args)
            r = r if isinstance(r, tuple) else (r,)
            want = [x for v in r for x in np.ravel(np.asarray(v, dtype=float))]
            want += [x for a in tr.state for x in np.ravel(getattr(twin, a))]
        got = tr.func(*gargs, *[state[a] for a in tr.state])
        got = got if isinstance(got, tuple) else (got,)
        flat = [x for v in got for x in (v if isinstance(v, list) else [v])]
        if len(flat) != len(want) or not np.allclose(flat, want, rtol=1e-9, atol=1e-12):
            raise NotTranslatable(
                f"The scalar translation of {func.__qualname__} disagrees with the original on "
                f"sample {k}: {flat[:6]}... vs {want[:6]}...")
        if tr.state:                                    # the next sample starts from the new state
            n_out = len(tr.outputs) - len(tr.state)
            for a, v in zip(tr.state, got[n_out:]):
                state[a] = v
