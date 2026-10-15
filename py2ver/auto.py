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
import contextlib
import copy
import inspect
import logging
import math
import os
import tempfile
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

from .fixed import DEFAULT_CONST_BITS
from .hwtypes import _namespace, normalize_attr, read_annotations

log = logging.getLogger(__name__)

MAX_SEARCH_FRAC = 64


class _Range:
    def __init__(self) -> None:
        self.lo = math.inf
        self.hi = -math.inf
        self.is_float = False
        self.length = 0          # for a list: its longest length seen
        self.is_list = False

    def add(self, v: Any) -> None:
        if isinstance(v, (list, tuple)):
            self.is_list = True
            self.length = max(self.length, len(v))
            for x in v:
                self._add(x)
        else:
            self._add(v)

    def _add(self, x: Any) -> None:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return
        self.is_float |= isinstance(x, float)
        self.lo, self.hi = min(self.lo, x), max(self.hi, x)

    def merge(self, other: "_Range") -> None:
        self.lo, self.hi = min(self.lo, other.lo), max(self.hi, other.hi)
        self.is_float |= other.is_float
        self.length = max(self.length, other.length)
        self.is_list |= other.is_list

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


def _traced(func: Callable, record: Callable[[str, Any], None]):
    """The function, recording every value assigned to a name (and its parameters)."""
    tree = ast.parse(inspect.getsource(func))
    tree, ann = read_annotations(tree, func)
    fd = tree.body[0]
    order: List[str] = [a.arg for a in fd.args.args]

    def trace(names: List[str]) -> List[ast.stmt]:
        for n in names:
            if n not in order:
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

    class _Trace(ast.NodeTransformer):
        def visit_Assign(self, node):
            self.generic_visit(node)
            return [node] + trace([n for t in node.targets for n in targets(t)])

        def visit_AugAssign(self, node):
            self.generic_visit(node)
            return [node] + trace(targets(node.target))

        def visit_AnnAssign(self, node):
            self.generic_visit(node)
            return [node] + (trace(targets(node.target)) if node.value is not None else [])

        def visit_For(self, node):
            self.generic_visit(node)
            node.body = trace(targets(node.target)) + node.body
            return node

    returned = _returned_names(fd)
    fd.body = trace([a.arg for a in fd.args.args]) + [_Trace().visit(s) for s in fd.body]
    fd.body = [s for item in fd.body for s in (item if isinstance(item, list) else [item])]
    fd.decorator_list = []
    ns = _namespace(func)
    ns["__py2ver_trace"] = record
    exec(compile(ast.fix_missing_locations(tree), f"<traced {fd.name}>", "exec"), ns)
    return ns[fd.name], order, returned, ann


def _profile(func: Callable, samples: List[Any], attr: Dict[str, Dict[str, Any]]):
    """Run the float function on the samples (feeding state back like PY_float)
    and return ({name: _Range}, names in order of first assignment, returned names)."""
    ranges: Dict[str, _Range] = {}
    last: Dict[str, Any] = {}

    def record(name: str, value: Any) -> None:
        ranges.setdefault(name, _Range()).add(value)
        last[name] = list(value) if isinstance(value, list) else value

    traced, order, returned, _ = _traced(func, record)
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
    for sample in samples:
        args = sample if isinstance(sample, tuple) else (sample,)
        if not state and len(args) != len(params):
            raise ValueError(f"{func.__name__} takes {len(params)} arguments, a sample has {len(args)}")
        last.clear()
        ret = traced(**dict(zip(inputs, args)), **values)
        rets = ret if isinstance(ret, tuple) and len(returned) > 1 else (ret,)
        for name, v in zip(returned, rets):
            record(name, v)
        for p in state:
            nxt = attr[p].get("next")
            if nxt is not None and nxt != p:
                values[p] = last.get(nxt, values[p])
    # A state also takes the values of its next state
    for p in state:
        nxt = attr[p].get("next")
        if nxt in ranges and p in ranges:
            ranges[p].merge(ranges[nxt])
    return ranges, order, returned


def _fmt(r: _Range, int_bits: int, frac: int) -> Dict[str, Any]:
    out = {"signed": 1, "width": 1 + int_bits + frac, "frac": frac}
    if r.is_list:
        out["length"] = r.length
    return out


@contextlib.contextmanager
def _scratch_dir() -> Iterator[None]:
    """Build trial designs out of the way (Py2ver writes hdl/ to the working directory)."""
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="py2ver_auto_") as tmp:
        os.chdir(tmp)
        try:
            yield
        finally:
            os.chdir(cwd)


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
    base = {k: dict(v) for k, v in normalize_attr(attr).items()}
    samples = list(samples)
    if not samples:
        raise ValueError("auto_fixed needs example inputs")
    _, _, _, ann = _traced(func, lambda *_: None)
    base = {k: dict(ann.get(k, {}), **base.get(k, {})) for k in dict.fromkeys([*ann, *base])}

    ranges, order, returned = _profile(func, samples, base)
    params = list(inspect.signature(func).parameters)
    state = [p for p in params if "next" in base.get(p, {}) or "memory" in base.get(p, {})]
    nexts = {base[p]["next"] for p in state if base[p].get("next") not in (None, p)}
    extra = {"round": True} if round else {}
    if saturate:
        extra["saturate"] = True

    kept = {k for k, m in base.items() if "width" in m}
    result: Dict[str, Dict[str, Any]] = {}
    auto: Dict[str, Tuple[_Range, int]] = {}       # name -> (range, integer bits)
    for name in order + [n for n in returned if n not in order]:
        r = ranges.get(name)
        if name in kept or name in nexts or r is None or name in result:
            continue
        if r.lo > r.hi and not r.is_float:
            continue                                 # never held a number (e.g. complex)
        if not r.is_float:
            if name in params:                       # an integer input: just wide enough
                signed = r.lo < 0
                bits = (r.int_bits() + 1 if signed else max(1, int(r.hi).bit_length())) + margin
                result[name] = dict(base.get(name, {}), signed=int(signed), width=bits)
                if r.is_list:
                    result[name].setdefault("length", r.length)
            continue                                 # integer locals keep exact widths
        auto[name] = (r, r.int_bits() + margin)

    def formats(fracs: Dict[str, int]) -> Dict[str, Dict[str, Any]]:
        out = {k: dict(base[k]) for k in params if k in base}
        out.update(result)
        for name, (r, ib) in auto.items():
            f = max(0, fracs[name])
            out[name] = dict(base.get(name, {}), **_fmt(r, ib, f), **extra)
            if f + ib + 1 < 1:
                out[name]["width"] = 1
        for k, m in base.items():
            out.setdefault(k, m)
        # Parameters first, in order, then the rest as they were first assigned
        return {k: out[k] for k in sorted(out, key=lambda k: (k not in params,
                                                                params.index(k) if k in params else 0))}

    if max_error is None:
        fracs = {n: width - 1 - ib for n, (_, ib) in auto.items()}
        return formats(fracs)

    # max_error: inputs are interface formats, not searched
    for p in params:
        if p in auto and p not in state:
            if width is None:
                raise ValueError(f"Give input '{p}' a format in attr, or width=<bits> for inputs")
            r, ib = auto.pop(p)
            result[p] = dict(base.get(p, {}), **_fmt(r, ib, max(0, width - 1 - ib)))

    def target(port: str) -> Optional[float]:
        if not isinstance(max_error, dict):
            return float(max_error)
        for name in (port, port.rsplit("_", 1)[0], port[:-4] if port.endswith("_out") else None):
            if name in max_error:
                return float(max_error[name])
        return None

    from .core import Py2ver                      # noqa: E402  (core imports this module's peers)
    n_evals = 0

    def error_ok(fracs: Dict[str, int]) -> bool:
        nonlocal n_evals
        n_evals += 1
        with _scratch_dir():
            p = Py2ver(func, formats(fracs), const_bits=const_bits)
        err = p.quantization_error(samples)
        return all(t is None or worst <= t for port, (worst, _) in err.items()
                   for t in [target(port)])

    tightest = min(v for v in ([max_error] if not isinstance(max_error, dict)
                               else max_error.values()))
    start = max(4, math.ceil(-math.log2(tightest)) + 4) if tightest > 0 else MAX_SEARCH_FRAC
    uniform = lambda f: {n: f for n in auto}  # noqa: E731
    hi = start
    while not error_ok(uniform(hi)):
        if hi >= MAX_SEARCH_FRAC:
            raise ValueError(
                f"max_error {max_error} can't be met with up to {MAX_SEARCH_FRAC} fraction bits; "
                "check the input formats and const_bits")
        hi = min(MAX_SEARCH_FRAC, hi + 8)
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
            lo, hi = 0, fracs[name]
            while lo < hi:
                mid = (lo + hi) // 2
                if error_ok(dict(fracs, **{name: mid})):
                    hi = mid
                else:
                    lo = mid + 1
            if hi != fracs[name]:
                fracs[name], changed = hi, True
        if not changed:
            break
    log.info("auto_fixed: %d trial designs", n_evals)
    return formats(fracs)
