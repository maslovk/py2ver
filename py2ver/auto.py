# auto.py
"""
Choose fixed-point formats for a float model automatically.

auto_fixed() runs the function as written on example inputs, records the
range of every parameter and variable, and gives each one that holds floats
a format in attr, with enough integer bits for its range (plus `margin`):

  - width=W: every value gets W bits; what the integer part leaves over is
    fraction
  - max_error=E: the fraction bits are searched: the fewest bits per value
    that keep every output of the bit-exact model within E of the float
    function (quantization_error) on the same inputs. Inputs keep the
    formats you give in attr, or `width` bits.

The result is an ordinary attr dict to read, adjust and pass to Py2ver.
Ranges come only from the example inputs: values they don't reach can still
overflow, so give inputs that cover the real range (or more margin, or
'saturate').
"""

from __future__ import annotations

import ast
import copy
import inspect
import logging
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .fixed import DEFAULT_CONST_BITS, DEFAULT_TABLE_BITS, quantize, wrap
from .hwtypes import _namespace, normalize_attr, read_annotations

log = logging.getLogger(__name__)

MAX_SEARCH_FRAC = 64
MAX_TABLE_BITS = 14


class _Range:
    def __init__(self) -> None:
        self.lo = math.inf
        self.hi = -math.inf
        self.is_float = False
        self.length = 0          # for a list: its longest length seen
        self.is_list = False
        self.is_complex = False
        self.shape: Optional[Tuple[int, ...]] = None    # a list of lists

    def add(self, v: Any) -> None:
        if isinstance(v, (list, tuple)):
            self.is_list = True
            self.length = max(self.length, len(v))
            if v and all(isinstance(x, (list, tuple)) for x in v):    # a list of lists
                shape = _shape(v)
                if shape is not None:
                    self.shape = shape
            for x in v:
                self.add(x) if isinstance(x, (list, tuple)) else self._add(x)
        else:
            self._add(v)

    def _add(self, x: Any) -> None:
        if isinstance(x, complex):
            self.is_complex = True
            self._add(x.real)
            self._add(x.imag)
            return
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return
        self.is_float |= isinstance(x, float)
        self.lo, self.hi = min(self.lo, x), max(self.hi, x)

    def add_array(self, v: Any) -> None:
        """Values from a numpy run (vectorize.py): an array of values, or a whole
        list (ListValue)."""
        import numpy as np
        if type(v).__name__ == "ListValue":
            v = v.arr
            self.is_list = True
            self.length = max(self.length, int(v.size))
        if not v.size or v.dtype == bool:
            return
        if v.dtype == object:
            for x in v.flat:
                self._add(x)
            return
        if v.dtype.kind == "f":
            self.is_float = True
        elif v.dtype.kind not in "iu":
            return
        lo, hi = v.min(), v.max()
        self.lo = min(self.lo, float(lo) if v.dtype.kind == "f" else int(lo))
        self.hi = max(self.hi, float(hi) if v.dtype.kind == "f" else int(hi))

    def merge(self, other: "_Range") -> None:
        self.lo, self.hi = min(self.lo, other.lo), max(self.hi, other.hi)
        self.is_float |= other.is_float
        self.length = max(self.length, other.length)
        self.shape = self.shape or other.shape
        self.is_list |= other.is_list
        self.is_complex |= other.is_complex

    def int_bits(self) -> int:
        """Smallest I with -2**I <= lo and hi < 2**I (0 for a value that is always 0)."""
        if self.lo > self.hi or (self.lo == 0 and self.hi == 0):
            return 0
        i = math.frexp(max(abs(self.lo), abs(self.hi)))[1]     # m < 2**i
        while i > -1000 and -(2.0 ** (i - 1)) <= self.lo and self.hi < 2.0 ** (i - 1):
            i -= 1
        while not (-(2.0 ** i) <= self.lo and self.hi < 2.0 ** i):
            i += 1
        return i


def _returned_names(fd: ast.FunctionDef) -> List[str]:
    ret = next((s for s in reversed(fd.body) if isinstance(s, ast.Return)), None)
    if ret is None or ret.value is None:
        return []
    vals = ret.value.elts if isinstance(ret.value, ast.Tuple) else [ret.value]
    return [v.id if isinstance(v, ast.Name) else (f"ret{i}" if len(vals) > 1 else "ret")
            for i, v in enumerate(vals)]


_TRACED: Dict[Any, Any] = {}
_TABLE_KEYS: Dict[Any, Tuple[str, ...]] = {}      # (func, source) -> its table functions' keys


def _traced(func: Callable, record: Callable[[str, Any], None],
            record_elem: Optional[Callable[[str, Any, Any], None]] = None):
    """The function, recording every value assigned to a name (and its parameters);
    an element stored in a list, record_elem(list, the list, the element) (default:
    record(list, element))."""
    source = inspect.getsource(func)
    key = (func, source)
    if key not in _TRACED:                     # (instrumented once, run many times)
        if len(_TRACED) >= 32:
            _TRACED.pop(next(iter(_TRACED)))
        _TRACED[key] = _traced_code(func, source)
    code, name, names, returned, ann = _TRACED[key]
    ns = _namespace(func)

    def call(key: str, fn: Callable, *args: Any) -> Any:
        y = fn(*args)
        record(f"{key}{_TABLE_IN}", args[0])
        record(key, y)
        return y

    ns["__py2ver_trace"] = record
    ns["__py2ver_trace_elem"] = record_elem or (lambda name, xs, x: record(name, x))
    ns["__py2ver_call"] = call
    exec(code, ns)
    return ns[name], list(names), list(returned), copy.deepcopy(ann)


def _traced_code(func: Callable, source: str):
    """The instrumented function, compiled: (code, name, names in order of first
    assignment, returned names, annotations)."""
    tree = ast.parse(source)
    tree, ann = read_annotations(tree, func)
    # Traced with its helpers inlined, as it is built (math calls in helpers
    # included); the helpers' own variables (__<helper><k>_...) get no format
    from .inline import inline_helpers
    tree = inline_helpers(tree, func.__globals__)
    fd = tree.body[0]
    order: List[str] = [a.arg for a in fd.args.args]

    def trace(names: List[str]) -> List[ast.stmt]:
        for n in names:
            if n not in order and not n.startswith("__"):
                order.append(n)
        return [ast.Expr(ast.Call(func=ast.Name(id="__py2ver_trace", ctx=ast.Load()), keywords=[],
                                  args=[ast.Constant(n), ast.Name(id=n, ctx=ast.Load())]))
                for n in names]

    def targets(t: ast.expr) -> List[str]:
        if isinstance(t, ast.Name):
            return [t.id]
        if isinstance(t, (ast.Tuple, ast.List)):
            return [n for e in t.elts for n in targets(e)]
        if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name):
            return [t.value.id]
        return []

    def stored(ts: List[ast.expr]) -> List[ast.stmt]:
        """Traces of assignment targets: an element stored in a list as that element
        (not the whole list again: a loop filling a long list stays linear)."""
        out, names = [], []
        for t in ts:
            if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) \
                    and not isinstance(t.slice, ast.Slice):
                names.append(t.value.id)
                load = copy.deepcopy(t)
                for x in ast.walk(load):
                    if hasattr(x, "ctx"):
                        x.ctx = ast.Load()
                out.append(ast.Expr(ast.Call(
                    func=ast.Name(id="__py2ver_trace_elem", ctx=ast.Load()), keywords=[],
                    args=[ast.Constant(t.value.id), ast.Name(id=t.value.id, ctx=ast.Load()), load])))
            else:
                out += trace(targets(t))
        trace(names)                               # (in the order of names)
        return out

    class _Trace(ast.NodeTransformer):
        def visit_Assign(self, node):
            self.generic_visit(node)
            return [node] + stored(node.targets)

        def visit_AugAssign(self, node):
            self.generic_visit(node)
            return [node] + stored([node.target])

        def visit_AnnAssign(self, node):
            self.generic_visit(node)
            return [node] + (trace(targets(node.target)) if node.value is not None else [])

        def visit_For(self, node):
            self.generic_visit(node)
            node.body = trace(targets(node.target)) + node.body
            return node

        def visit_While(self, node):
            # Counts its iterations (the largest count becomes attr['while'])
            self.generic_visit(node)
            k = ast.Name(id=f"__py2ver_w{id(node)}", ctx=ast.Load())
            store = ast.Name(id=k.id, ctx=ast.Store())
            node.body = [ast.AugAssign(target=store, op=ast.Add(), value=ast.Constant(1)),
                         ast.Expr(ast.Call(func=ast.Name(id="__py2ver_trace", ctx=ast.Load()),
                                           keywords=[], args=[ast.Constant(WHILE_KEY), k]))
                         ] + node.body
            return [ast.Assign(targets=[copy.deepcopy(store)], value=ast.Constant(0)), node]

        def visit_Call(self, node):
            self.generic_visit(node)
            key = ast.unparse(node.func)
            if key in tables or (len(node.args) == 1 and not node.keywords
                                 and _math_function(node.func, ns) is not None):
                tables.setdefault(key, None)
                return ast.Call(func=ast.Name(id="__py2ver_call", ctx=ast.Load()), keywords=[],
                                args=[ast.Constant(key), node.func] + node.args)
            return node

    returned = _returned_names(fd)
    ns = _namespace(func)
    tables: Dict[str, None] = {}
    fd.body = trace([a.arg for a in fd.args.args]) + [_Trace().visit(s) for s in fd.body]
    fd.body = [s for item in fd.body for s in (item if isinstance(item, list) else [item])]
    fd.decorator_list = []

    code = compile(ast.fix_missing_locations(tree), f"<traced {fd.name}>", "exec")
    _TABLE_KEYS[(func, source)] = tuple(tables)
    return code, fd.name, order + list(tables), returned, ann


_TABLE_IN = ":in"               # range key of a table function's argument
WHILE_KEY = "__while__"         # iterations of while loops, as traced
MAX_WIDER = 16          # a repeated layer's value: at most this many bits wider (repeats.py)
WEIGHT_TABLES = "__py2ver_weight_tables__"   # (in a namespace) tables that get formats
_NO_TABLE = {"floor", "ceil"}   # math functions computed exactly


def _math_function(f: ast.expr, ns: Dict[str, Any]) -> Optional[Callable]:
    node = f
    while isinstance(node, ast.Attribute):
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    try:
        fn = eval(compile(ast.Expression(copy.deepcopy(f)), "<f>", "eval"), dict(ns))
    except Exception:
        return None
    from .fixed import EXACT_SQRT, is_sqrt
    if ns.get(EXACT_SQRT) and is_sqrt(fn):
        return None                                 # (computed exactly)
    if getattr(fn, "__module__", None) == "math" and callable(fn) \
            and getattr(fn, "__name__", "") not in _NO_TABLE:
        return fn
    from .numpy_fe import TABLE_UFUNCS, TABLES_KEY
    if getattr(fn, "__module__", None) == "numpy" and getattr(fn, "__name__", "") in TABLE_UFUNCS:
        return fn                                   # np.tanh, np.exp, ...
    if isinstance(f, ast.Name) and f.id in ns.get(TABLES_KEY, ()):
        return fn                                   # your function, built as a table
    return None


def run_float(func: Callable, samples: List[Any], attr: Dict[str, Dict[str, Any]],
              record: Optional[Callable[[str, Any], None]] = None):
    """
    Run the function as written on the samples, feeding state back ('next')
    and keeping memories ('memory') between samples. Returns (outputs per
    sample, flattened like the ports; names in order of first assignment;
    returned names). record(name, value) sees every assigned value.
    """
    from .reference import _flatten_outputs
    last: Dict[str, Any] = {}

    def rec(name: str, value: Any) -> None:
        last[name] = list(value) if isinstance(value, list) else value
        if record is not None:
            record(name, value)

    def rec_elem(name: str, xs: Any, x: Any) -> None:
        last[name] = xs                        # (the list itself: its values at the end)
        if record is not None:
            record(name, x)

    traced, order, returned, _ = _traced(func, rec, rec_elem)
    if getattr(func, "__py2ver_loops__", None) is not None:      # (a loop program: numpy)
        from . import vectorize
        keys = _TABLE_KEYS.get((func, inspect.getsource(func)), ())
        traced = vectorize.program(func.__py2ver_loops__, func.__globals__, traced, rec, keys)
    params = list(inspect.signature(func).parameters)
    state = [p for p in params if "next" in attr.get(p, {}) or "memory" in attr.get(p, {})]
    inputs = [p for p in params if p not in state]
    values: Dict[str, Any] = {}
    for p in state:
        m = attr[p]
        init = m.get("init", 0)
        if "length" in m:
            values[p] = list(init) if isinstance(init, (list, tuple)) else [init] * int(m["length"])
        else:
            values[p] = init
    outputs = []
    for sample in samples:
        args = sample if isinstance(sample, tuple) else (sample,)
        if not state and len(args) != len(params):
            raise ValueError(f"{func.__name__} takes {len(params)} arguments, a sample has {len(args)}")
        last.clear()
        ret = traced(**dict(zip(inputs, args)), **values)
        rets = ret if isinstance(ret, tuple) and len(returned) > 1 else (ret,)
        for name, v in zip(returned, rets):
            rec(name, v)
        outputs.append(_flatten_outputs(ret))
        for p in state:
            nxt = attr[p].get("next")
            # The last value given to the next state (a state reassigned under its
            # own name too; a list updated in place keeps its object otherwise)
            if nxt is not None:
                v = last.get(nxt, values[p])
                values[p] = list(v) if isinstance(v, list) else v
    return outputs, order, returned


def float_reference(func: Callable, attr: Dict[str, Dict[str, Any]],
                    samples: List[Any]) -> List[Tuple[Any, ...]]:
    """
    PY_float(): the function as written on the samples, with the inputs (and
    state / memory 'init') quantized to their formats in attr first, so only
    the quantization inside the design is measured against it.
    """
    params = list(inspect.signature(func).parameters)
    inputs = [p for p in params if not {"next", "memory"} & set(attr.get(p, {}))]

    def number(name: str, v: Any) -> Any:
        m = attr.get(name, {})
        if isinstance(v, (list, tuple)):
            return [number(name, x) for x in v]

        def real(x: Any) -> Any:
            if m.get("frac") is None:
                return x
            return quantize(x, m["frac"], m["width"], bool(m.get("signed")), bool(m.get("round")),
                            bool(m.get("saturate"))) / (1 << m["frac"])

        if isinstance(v, complex) or m.get("complex"):    # complex in hardware: complex here too
            v = complex(v)
            return complex(real(v.real), real(v.imag))
        return real(v)

    def sample(v: Any) -> Any:
        args = v if isinstance(v, tuple) else (v,)
        return tuple(number(p, x) for p, x in zip(inputs, args)) + tuple(args[len(inputs):])

    attr = {k: (dict(m, init=number(k, m["init"])) if isinstance(m, dict) and "init" in m else m)
            for k, m in attr.items()}
    return run_float(func, [sample(v) for v in samples], attr)[0]


def model_outputs(fe: Any, samples: List[Any]) -> Tuple[List[str], List[Tuple[float, ...]]]:
    """
    The bit-exact model of a front end (core.front_end) on the samples, with
    state fed back and memories kept, like PY_batch / PY_stream but without
    building the hardware. Returns (output names, values per sample).
    """
    from .reference import expand_args
    model, attr = fe.model, fe.attr
    params = list(inspect.signature(model).parameters)
    ret = next(s for s in reversed(fe.tree.body[0].body) if isinstance(s, ast.Return))
    names = [e.id for e in (ret.value.elts if isinstance(ret.value, ast.Tuple) else [ret.value])]
    state = [p for p in params if {"next", "memory"} & set(attr.get(p, {}))]
    inputs = [p for p in params if p not in state]

    def fit(name: str, v: Any) -> Any:
        m = attr.get(name, {})
        if "width" not in m:
            return v
        if isinstance(v, list):
            return [wrap(int(x), int(m["width"]), bool(m.get("signed"))) for x in v]
        return wrap(int(v), int(m["width"]), bool(m.get("signed")))

    def raw(name: str, v: Any) -> Any:
        m = attr.get(name, {})
        if m.get("frac") is None:
            return v
        q = lambda x: quantize(x, m["frac"], m["width"], bool(m.get("signed")),  # noqa: E731
                               bool(m.get("round")), bool(m.get("saturate")))
        return [q(x) for x in v] if isinstance(v, (list, tuple)) else q(v)

    def number(name: str, v: Any) -> float:
        t = fe.fx_types.get(name)
        t = t[1] if t is not None and t[0] == "l" else t
        return v / (1 << t[1]) if t is not None and t[0] == "f" else v

    values: Dict[str, Any] = {}
    for p in state:
        m = attr[p]
        init = m.get("init", 0)
        if "length" in m:
            init = list(init) if isinstance(init, (list, tuple)) else [init] * int(m["length"])
        values[p] = fit(p, init)
    ports: List[str] = []
    rows: List[Tuple[float, ...]] = []
    for sample in samples:
        args = expand_args(fe.call_inputs, fe.cplx_params, sample if isinstance(sample, tuple) else (sample,))
        out = model(**{p: raw(p, v) for p, v in zip(inputs, args)}, **values)
        outs = out if isinstance(out, tuple) and len(names) > 1 else (out,)
        row = dict(zip(names, outs))
        flat, ports = [], []
        for n in names:
            v = row[n]
            if isinstance(v, list):
                flat += [number(n, x) for x in v]
                ports += [f"{n}_{k}" for k in range(len(v))]
            else:
                flat.append(number(n, v))
                ports.append(n)
        rows.append(tuple(flat))
        for p in state:
            nxt = attr[p].get("next")
            # (a memory, or a list updated in place and not returned, keeps its object)
            values[p] = fit(p, row[nxt] if nxt in row else values[p])
    return [fe.py_name.get(n, n) for n in ports], rows


def _profile(func: Callable, samples: List[Any], attr: Dict[str, Dict[str, Any]]):
    """Run the float function on the samples (feeding state back like PY_float)
    and return ({name: _Range}, names in order of first assignment, returned names)."""
    ranges: Dict[str, _Range] = {}
    # (a loop over a model's repeated layers, repeats.py: its values also per turn,
    # as name@turn)
    from .repeats import REPEATS_KEY
    folded = _namespace(func).get(REPEATS_KEY, {})
    turn_of = {n: v for v, names in folded.items() for n in names}
    turn: Dict[str, Any] = {}

    def record(name: str, value: Any) -> None:
        if name in folded:
            turn[name] = int(value)
        elif name in turn_of and turn_of[name] in turn:
            keep(f"{name}@{turn[turn_of[name]]}", value)
        keep(name, value)

    def keep(name: str, value: Any) -> None:
        r = ranges.get(name)
        if r is None:
            r = ranges[name] = _Range()
        if type(value) is float:                  # (the common case, fast)
            r.is_float = True
            if value < r.lo:
                r.lo = value
            if value > r.hi:
                r.hi = value
        elif type(value).__name__ in ("ndarray", "ListValue"):    # (from vectorize.py)
            r.add_array(value)
        else:
            r.add(value)

    _, order, returned = run_float(func, samples, attr, record)
    params = list(inspect.signature(func).parameters)
    state = [p for p in params if "next" in attr.get(p, {}) or "memory" in attr.get(p, {})]
    # A state also takes the values of its next state
    for p in state:
        nxt = attr[p].get("next")
        if nxt in ranges and p in ranges:
            ranges[p].merge(ranges[nxt])
    return ranges, order, returned


def _shape(v: Any) -> Optional[Tuple[int, ...]]:
    if not isinstance(v, (list, tuple)):
        return ()
    subs = {_shape(x) for x in v}
    return (len(v),) + subs.pop() if len(subs) == 1 and None not in subs else None


def _fmt(r: _Range, int_bits: int, frac: int) -> Dict[str, Any]:
    out = {"signed": 1, "width": 1 + int_bits + frac, "frac": frac}
    if r.shape is not None and len(r.shape) > 1:
        out["shape"] = r.shape
    elif r.is_list:
        out["length"] = r.length
    return out


def _table_range(r: _Range, margin: int = 0) -> Tuple[float, float]:
    """A table's input range: the arguments seen, each end's magnitude 2 ** margin
    times as large (as a value's margin integer bits), rounded outward to 1/16 of
    the span."""
    lo, hi = r.lo, r.hi
    lo, hi = (lo * 2 ** margin if lo < 0 else lo), (hi * 2 ** margin if hi > 0 else hi)
    if not lo < hi:
        lo, hi = lo - 0.5, hi + 0.5
    q = 2.0 ** (math.floor(math.log2(hi - lo)) - 4)
    return math.floor(lo / q) * q, math.ceil(hi / q) * q


def auto_fixed(func: Callable, samples: List[Any], attr: Optional[Dict[str, Any]] = None, *,
               width: Optional[int] = None,
               max_error: Union[None, float, Dict[str, float]] = None,
               margin: int = 1, round: bool = False, saturate: bool = False,
               const_bits: int = DEFAULT_CONST_BITS) -> Dict[str, Dict[str, Any]]:
    """
    Fixed-point formats for a float model, as an attr dict for Py2ver.

    samples: example inputs, like PY_batch() vectors (or PY_stream() samples
        for a design with state). Ranges are measured on them.
    attr: what you already know: formats you want kept as they are (any
        entry with a 'width'), and state / memory / list details ('next',
        'init', 'length', 'memory'), which are kept.
    width: bits for every value that holds floats (and for inputs in
        max_error mode, unless attr gives them).
    max_error: search the fraction bits instead: the largest allowed
        absolute error of each output against the float function, one
        number for all outputs or {output: error} (other outputs are free).
    margin: extra integer bits above the measured range.
    round / saturate: added to every chosen format.
    """
    if width is None and max_error is None:
        raise ValueError("Give width=<bits> or max_error=<error> (or both)")
    attr = dict(attr or {})
    while_n = attr.pop("while", None)          # iterations of the while loops (else measured)
    base = {k: dict(v) for k, v in normalize_attr(attr).items()}
    samples = list(samples)
    if not samples:
        raise ValueError("auto_fixed needs example inputs")
    _, _, _, ann = _traced(func, lambda *_: None)
    base = {k: dict(ann.get(k, {}), **base.get(k, {})) for k in dict.fromkeys([*ann, *base])}

    ranges, order, returned = _profile(func, samples, base)
    # The weight tables of a numpy loop program (numpy_loops.py): a format
    # each, from their values
    ns0 = _namespace(func)
    for name in sorted(ns0.get(WEIGHT_TABLES, ())):
        v = ns0[name]
        if name not in ranges:
            r = _Range()
            r.add(list(v))
            ranges[name] = r
            order.append(name)
    if while_n is None and WHILE_KEY in ranges:
        while_n = int(ranges.pop(WHILE_KEY).hi)
    ranges.pop(WHILE_KEY, None)
    params = list(inspect.signature(func).parameters)
    state = [p for p in params if "next" in base.get(p, {}) or "memory" in base.get(p, {})]
    nexts = {base[p]["next"] for p in state
             if base[p].get("next") not in (None, p) and base[p]["next"] not in params}
    # A state loaded from another parameter (d2 <- d1 <- x) shares its format:
    # group them (union-find)
    parent = {p: p for p in params}

    def find(p: str) -> str:
        while parent[p] != p:
            p = parent[p]
        return p

    for p in state:
        nxt = base[p].get("next")
        if nxt in params and nxt != p:
            parent[find(p)] = find(nxt)
    groups: Dict[str, List[str]] = {}
    for p in params:
        groups.setdefault(find(p), []).append(p)
    group_of = {p: g for g in groups.values() if len(g) > 1 for p in g}
    extra = {"round": True} if round else {}
    if saturate:
        extra["saturate"] = True

    kept = {k for k, m in base.items() if "width" in m}
    # A loop variable over a list takes the list's elements: no format of its own
    fd = ast.parse(inspect.getsource(func)).body[0]
    kept |= {s.target.id for s in ast.walk(fd)
             if isinstance(s, ast.For) and isinstance(s.target, ast.Name)
             and not (isinstance(s.iter, ast.Call) and isinstance(s.iter.func, ast.Name)
                      and s.iter.func.id == "range")}
    result: Dict[str, Dict[str, Any]] = {}
    auto: Dict[str, Tuple[_Range, int]] = {}       # name -> (range, integer bits)
    tables = {n for n in order if f"{n}{_TABLE_IN}" in ranges}
    for name in order + [n for n in returned if n not in order]:
        r = ranges.get(name)
        if name in kept or name in nexts or r is None or name in result:
            continue
        if name in tables and not r.is_float:
            r.is_float = True                        # a table's entries are fixed point
        if r.lo > r.hi and not r.is_float:
            continue                                 # never held a number (e.g. complex)
        if not r.is_float:
            if name in params:                       # an integer input: just wide enough
                signed = r.lo < 0
                bits = (r.int_bits() + 1 if signed else max(1, int(r.hi).bit_length())) + margin
                result[name] = dict(base.get(name, {}), signed=int(signed), width=bits)
                if r.shape is not None and len(r.shape) > 1:
                    result[name].setdefault("shape", r.shape)
                elif r.is_list:
                    result[name].setdefault("length", r.length)
            continue                                 # integer locals keep exact widths
        auto[name] = (r, r.int_bits() + margin)

    for g in {tuple(g) for g in group_of.values()}:      # one integer part per group
        members = [m for m in g if m in auto]
        if members:
            ib = max(auto[m][1] for m in members)
            for m in members:
                auto[m] = (auto[m][0], ib)
    # A value of a loop over repeated layers (repeats.py): the integer bits of all
    # of them, the fraction bits of the one with the smallest values (as precise as
    # each layer's own, written out, would be): this many bits wider
    wider: Dict[str, int] = {}
    for name, (r, ib) in auto.items():
        per = [ranges[k].int_bits() + margin for k in ranges
               if k.startswith(f"{name}@") and ranges[k].lo <= ranges[k].hi]
        if per and min(per) < ib:
            wider[name] = min(ib - min(per), MAX_WIDER)
    bits = {n: int(base.get(n, {}).get("bits", DEFAULT_TABLE_BITS)) for n in auto if n in tables}

    def formats(fracs: Dict[str, int]) -> Dict[str, Dict[str, Any]]:
        out = {k: dict(base[k]) for k in params if k in base}
        out.update(result)
        for name, (r, ib) in auto.items():
            f = max(0, fracs[name])
            fmt = _fmt(r, ib, f)
            if name in tables:                       # 'math.sin': {'range', 'bits', format}
                given = base.get(name, {}).get("range")   # (yours, if you gave one)
                fmt = dict(range=tuple(given) if given is not None
                           else _table_range(ranges[f"{name}{_TABLE_IN}"], margin),
                           bits=bits[name], **fmt)
            elif name in params and r.is_complex:
                fmt["complex"] = True
            out[name] = dict(base.get(name, {}), **fmt, **extra)
            if f + ib + 1 < 1:
                out[name]["width"] = 1
        for k, m in base.items():
            out.setdefault(k, m)
        if while_n is not None:
            out["while"] = while_n
        # Parameters first, in order, then the rest as they were first assigned
        return {k: out[k] for k in sorted(out, key=lambda k: (k not in params,
                                                                params.index(k) if k in params else 0))}

    if max_error is None:
        fracs = {n: width - 1 - ib + wider.get(n, 0) for n, (_, ib) in auto.items()}
        return formats(fracs)

    # max_error: inputs are interface formats, not searched
    for p in params:
        if p in auto and p not in state:
            if width is None:
                raise ValueError(f"Give input '{p}' a format in attr, or width=<bits> for inputs")
            r, ib = auto.pop(p)
            result[p] = dict(base.get(p, {}), **_fmt(r, ib, max(0, width - 1 - ib)))
            if r.is_complex:
                result[p]["complex"] = True
            # State loaded from this input takes its format too
            for m in group_of.get(p, []):
                if m in auto:
                    rm, _ = auto.pop(m)
                    result[m] = dict(base.get(m, {}), **{k: v for k, v in result[p].items()
                                                        if k in ("signed", "width", "frac")})
                    if rm.is_complex:
                        result[m]["complex"] = True

    def target(port: str) -> Optional[float]:
        if not isinstance(max_error, dict):
            return float(max_error)
        for name in (port, port.rsplit("_", 1)[0], port[:-4] if port.endswith("_out") else None):
            if name in max_error:
                return float(max_error[name])
        return None

    from .core import front_end
    n_evals = 0
    ideals: Dict[str, Any] = {}

    def error_ok(fracs: Dict[str, int]) -> bool:
        """Does the design with these formats meet max_error on the samples?
        (The bit-exact model against the float function, without building hardware.)"""
        nonlocal n_evals
        n_evals += 1
        attr = formats(fracs)
        ports, exact = model_outputs(front_end(func, attr, const_bits), samples)
        # The float reference depends only on the inputs' formats (not searched)
        key = repr([attr.get(p) for p in params])
        if key not in ideals:
            ideals[key] = float_reference(func, attr, samples)
        ideal = ideals[key]
        for i, port in enumerate(ports):
            t = target(port)
            if t is not None and any(abs(float(e[i]) - float(r[i])) > t
                                     for e, r in zip(exact, ideal)):
                return False
        return True

    tightest = min(v for v in ([max_error] if not isinstance(max_error, dict)
                               else max_error.values()))
    start = max(4, math.ceil(-math.log2(tightest)) + 4) if tightest > 0 else MAX_SEARCH_FRAC
    uniform = lambda f: {n: f for n in auto}  # noqa: E731
    hi = start
    while not error_ok(uniform(hi)):
        if hi < MAX_SEARCH_FRAC:
            hi = min(MAX_SEARCH_FRAC, hi + 8)
        elif bits and min(bits.values()) < MAX_TABLE_BITS:     # finer tables, then
            for name in bits:
                bits[name] = min(MAX_TABLE_BITS, bits[name] + 2)
        else:
            raise ValueError(
                f"max_error {max_error} can't be met with up to {MAX_SEARCH_FRAC} fraction bits"
                + (f" and {MAX_TABLE_BITS}-bit tables" if bits else "")
                + "; check the input formats and const_bits")
    # The smallest fraction for all values together, then for each value on its own
    lo = 0
    while lo < hi:
        mid = (lo + hi) // 2
        if error_ok(uniform(mid)):
            hi = mid
        else:
            lo = mid + 1
    fracs = uniform(hi)
    for _ in range(3):
        changed = False
        for name in auto:
            same = [m for m in group_of.get(name, [name]) if m in auto] or [name]
            if name != same[0]:
                continue                             # a group is searched as one
            lo, hi = 0, fracs[name]
            while lo < hi:
                mid = (lo + hi) // 2
                if error_ok(dict(fracs, **{m: mid for m in same})):
                    hi = mid
                else:
                    lo = mid + 1
            if hi != fracs[name]:
                fracs.update({m: hi for m in same})
                changed = True
        if not changed:
            break
    # Then the fewest entries for each table
    for name in bits:
        lo, hi = 2, bits[name]
        while lo < hi:
            mid = (lo + hi) // 2
            bits[name] = mid
            if error_ok(fracs):
                hi = mid
            else:
                lo = mid + 1
        bits[name] = hi
    log.info("auto_fixed: %d trial designs", n_evals)
    return formats(fracs)
