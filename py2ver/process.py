"""
Processes: hardware that follows a protocol, written as a Python generator in
which a yield is a clock.

A protocol is usually specified as a sequence in time: the line low for a bit
time, then eight bits, least significant first, then high. A generator says
just that, and py2ver builds it as a state machine, clock for clock:

    BIT = 434                                  # 50 MHz / 115200 baud

    def uart_tx(port):
        while True:
            port.busy = 0
            while not port.start:              # wait for a byte to send
                yield
            data = port.data
            port.busy = 1
            port.tx = 0                        # start bit
            yield BIT
            for i in range(8):
                port.tx = (data >> i) & 1      # least significant bit first
                yield BIT
            port.tx = 1                        # stop bit
            yield BIT

The generator's one argument is its port: an attribute it reads and never
writes is an input pin (sampled at each clock), one it writes is an output, a
register (what it writes is on the pin from the next clock: outputs don't
change between clocks). `yield` lets a clock pass, `yield n` n clocks (n an
expression: a counter loaded with it). The code between two yields runs in one
clock. Locals keep their values from clock to clock (registers), and a loop
or an `if` that yields becomes states of the machine; one that doesn't is
logic within a clock (a loop of constant bounds, written out).

attr gives the pins their widths ('width', 'signed', an output's 'init': its
value after reset) and, if their values can't be followed, the locals' too.

Parameters after the port are constants: their defaults, or a partial's
(functools.partial(uart_tx, bit=217): the same transmitter, twice the baud).
A step of a protocol written once is a generator of its own, used with
`yield from` (its body in the process where it's used; its value, `return`ed,
with `x = yield from ...`):

    def wait_until(port, name, value=1):
        while getattr(port, name) != value:
            yield

    yield from wait_until(port, "start")

connect() puts processes together as one design, an input of one wired to an
output of another.

The generator itself is the reference: simulate() runs it a clock at a time,
the inputs of each clock given, its outputs read; the machine built from it
(compile_process: a function of one clock with its state, see "State
registers" in the README) computes the same, clock for clock.
"""
from __future__ import annotations

import ast
import copy
import functools
import inspect
import linecache
import os
import textwrap
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

PC, WAIT = "__pc", "__wait"
MAX_BITS = 64                 # a local wider than this: its width from attr


class ProcessError(ValueError):
    """A generator py2ver can't build as a process."""


@dataclass
class System:
    """Processes put together (connect()), built as one design."""
    name: str
    processes: Dict[str, Callable]
    wires: Dict[Tuple[str, str], Tuple[str, str]]     # (process, input): (process, output)
    attrs: Dict[str, Dict[str, Any]] = field(default_factory=dict)   # (each one's own)


def is_process(func: Any) -> bool:
    return isinstance(func, System) or inspect.isgeneratorfunction(func)


@dataclass
class Process:
    """A process built: its clock's function and attr, its pins, its registers."""
    func: Any                            # the generator (or the System)
    step: Callable                       # one clock of it
    attr: Dict[str, Dict[str, Any]]
    inputs: List[str]
    outputs: List[str]
    locals: List[str]
    states: int
    source: str = ""
    widths: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    tree: Optional[ast.FunctionDef] = None              # (the clock's function)
    where: List[str] = field(default_factory=list)      # (each state: where in the source)
    parts: Dict[str, "Process"] = field(default_factory=dict)  # (a System's processes)

    def explain(self) -> str:
        """Its states, each where in the generator it waits, and its registers."""
        if self.parts:
            return "\n\n".join(f"{p}: " + pr.explain() for p, pr in self.parts.items())
        name = getattr(_unwrap(self.func)[0], "__name__", "?")
        lines = [f"{name}: {self.states} states (a state waits at its yield: a clock, or "
                 f"yield n's n)"]
        width = len(str(len(self.where)))
        lines += [f"  {k:>{width}}  {w}" for k, w in enumerate(self.where)]

        def bits(names):
            return ", ".join(f"{n} {self.attr[n]['width']}" for n in names)
        lines += ["bits (the outputs, the locals and the state: registers):",
                  f"  inputs   {bits(self.inputs)}" if self.inputs else "  inputs   -",
                  f"  outputs  {bits(self.outputs)}",
                  f"  locals   {bits(self.locals)}" if self.locals else "  locals   -",
                  f"  state    {bits([PC, WAIT])}"]
        return "\n".join(lines)


# ==================================================================== the reference

class _Runner:
    """A generator run a clock at a time on its port."""

    def __init__(self, func: Callable, attr: Dict[str, Any]) -> None:
        self.ins, self.outs = _pins(func)
        init = {o: attr.get(o) if isinstance(attr.get(o), dict) else {} for o in self.outs}
        self.port = SimpleNamespace(**{n: 0 for n in self.ins},
                                    **{o: int(init[o].get("init", 0)) for o in self.outs})
        self.gen = func(self.port)
        self.wait, self.done = 0, False

    def clock(self, k: int) -> None:
        if self.done:
            return
        if self.wait:
            self.wait -= 1
            return
        try:
            n = next(self.gen)
        except StopIteration:
            self.done = True
            return
        self.wait = (1 if n is None else int(n)) - 1
        if self.wait < 0:
            raise ProcessError(f"yield {n} at clock {k}: a yield is a clock at least")


def simulate(func: Any, inputs: Sequence[Any], attr: Optional[Dict[str, Any]] = None,
             outputs: Optional[Sequence[str]] = None) -> List[Dict[str, int]]:
    """Run the generator a clock at a time: inputs, a clock's input pins each (a
    dict, an input not in it 0; or a tuple in the port's order of inputs);
    returns each clock's outputs (a dict), the values its pins have during that
    clock (written before it). A System: its processes so, together."""
    from .hwtypes import normalize_attr
    attr = normalize_attr(attr)
    if isinstance(func, System):
        return _simulate_system(func, inputs, attr, outputs)
    r = _Runner(func, attr)
    outputs = list(outputs or r.outs)
    rows = []
    for k, vec in enumerate(inputs):
        rows.append({o: getattr(r.port, o) for o in outputs})
        values = vec if isinstance(vec, dict) else dict(zip(r.ins, vec))
        for name in r.ins:
            setattr(r.port, name, values.get(name, 0))
        r.clock(k)
    return rows


class ProcessMixin:
    """Py2ver for a process: its references and simulation a clock at a time,
    each clock's inputs a dict, each clock's outputs (its pins) a dict."""

    process: Optional[Process] = None

    def _clocks(self, run_stream: Callable, vcd: Optional[str]) -> Callable[[Sequence[Dict[str, int]]], List[Dict[str, int]]]:
        pr = self.process
        if pr is None:
            raise ValueError("Not a process (a generator function): use PY_stream / TB_stream")

        def run(clocks: Sequence[Dict[str, int]]) -> List[Dict[str, int]]:
            rows = run_stream([tuple(int(c.get(p, 0)) for p in pr.inputs) for c in clocks])
            rows = [dict(zip(pr.outputs, r[:len(pr.outputs)])) for r in rows]
            if vcd:
                write_vcd(vcd, pr, clocks, rows)
            return rows
        return run

    def PY_process(self, vcd: Optional[str] = None) -> Callable[[Sequence[Dict[str, int]]], List[Dict[str, int]]]:
        """The machine's bit-exact model, a clock at a time (PY_stream's); vcd: a
        file for the pins' waveforms."""
        return self._clocks(self.PY_stream(), vcd)

    def TB_process(self, vcd: Optional[str] = None) -> Callable[[Sequence[Dict[str, int]]], List[Dict[str, int]]]:
        """The machine simulated (Verilog), a clock at a time (TB_stream's)."""
        return self._clocks(self.TB_stream(), vcd)

    def GEN_process(self, vcd: Optional[str] = None) -> Callable[[Sequence[Dict[str, int]]], List[Dict[str, int]]]:
        """The generator itself, run a clock at a time (simulate)."""
        pr = self.process

        def run(clocks: Sequence[Dict[str, int]]) -> List[Dict[str, int]]:
            rows = simulate(pr.func, clocks, self._process_attr, pr.outputs)
            if vcd:
                write_vcd(vcd, pr, clocks, rows)
            return rows
        return run

    def explain(self) -> str:
        """The process's states, each where in the generator it waits, and its
        registers' widths."""
        if self.process is None:
            raise ValueError("Not a process (a generator function)")
        return self.process.explain()


def difference(a: Sequence[Dict[str, int]], b: Sequence[Dict[str, int]]) -> Optional[str]:
    """Where two runs a clock at a time (GEN_process()'s, PY_process()'s, ...)
    first differ: 'clock 2,317: tx 1 / 0'; None if they don't."""
    for k, (x, y) in enumerate(zip(a, b)):
        if x != y:
            pins = [n for n in x if x.get(n) != y.get(n)] + [n for n in y if n not in x]
            return f"clock {k:,}: " + ", ".join(f"{n} {x.get(n)} / {y.get(n)}" for n in pins)
    if len(a) != len(b):
        return f"{len(a):,} clocks / {len(b):,}"
    return None


def write_vcd(path: str, pr: Process, clocks: Sequence[Any], rows: Sequence[Dict[str, int]],
              period_ns: int = 20) -> None:
    """The pins' waveforms (VCD, for GTKWave or Surfer): clk, the inputs, the
    outputs; a clock's values from its rising edge."""
    name = getattr(pr.func, "name", None) or getattr(_unwrap(pr.func)[0], "__name__", "process")
    sigs = [(n, int(pr.attr[n]["width"])) for n in pr.inputs + pr.outputs]
    ids = {n: _vcd_id(k + 1) for k, (n, _) in enumerate(sigs)}
    clk = _vcd_id(0)
    out = ["$timescale 1ns $end", f"$scope module {name} $end", f"$var wire 1 {clk} clk $end"]
    out += [f"$var wire {w} {ids[n]} {n} $end" for n, w in sigs]
    out += ["$upscope $end", "$enddefinitions $end"]
    last: Dict[str, int] = {}
    for k, (c, r) in enumerate(zip(clocks, rows)):
        values = dict(c if isinstance(c, dict) else zip(pr.inputs, c), **r)
        out += [f"#{k * period_ns}", f"1{clk}"]
        for n, w in sigs:
            v = int(values.get(n, 0)) & ((1 << w) - 1)
            if last.get(n) != v:
                last[n] = v
                out.append(f"{v}{ids[n]}" if w == 1 else f"b{v:b} {ids[n]}")
        out += [f"#{k * period_ns + period_ns // 2}", f"0{clk}"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


def _vcd_id(k: int) -> str:
    s = ""
    while True:
        s += chr(33 + k % 94)
        k //= 94
        if not k:
            return s


# ==================================================================== the pins

def _source(func: Callable) -> ast.FunctionDef:
    """func's definition, each node with its file (_file) and its line there."""
    lines, start = inspect.getsourcelines(func)
    fd = ast.parse(textwrap.dedent("".join(lines))).body[0]
    if not isinstance(fd, ast.FunctionDef):
        raise ProcessError(f"{func.__name__} isn't a function")
    ast.increment_lineno(fd, start - 1)
    file = inspect.getsourcefile(func)
    for n in ast.walk(fd):
        if hasattr(n, "lineno"):
            n._file = file
    return fd


def _line(n: ast.AST) -> str:
    """n's line of source, as written."""
    f = getattr(n, "_file", None)
    return linecache.getline(f, n.lineno).strip() if f else ""


def _at(n: ast.AST) -> str:
    """Where n is: file:line (the file relative to here if it's below)."""
    f = getattr(n, "_file", None)
    if f is None:
        return f"line {getattr(n, 'lineno', '?')}"
    try:
        rel = os.path.relpath(f)
        f = rel if not rel.startswith("..") else f
    except ValueError:
        pass
    return f"{f}:{n.lineno}"


def _unwrap(func: Any) -> Tuple[Callable, inspect.Signature]:
    """(the generator function, its signature with a partial's values as defaults)."""
    base = func
    while isinstance(base, functools.partial):
        if base.args:
            raise ProcessError(f"{getattr(base.func, '__name__', base)}: give a partial's "
                               f"values by name")
        base = base.func
    if not inspect.isgeneratorfunction(base):
        raise ProcessError(f"{getattr(base, '__name__', base)} isn't a generator (a process, or a "
                           f"step of one, yields)")
    return base, inspect.signature(func)


def _env(base: Callable) -> Dict[str, Any]:
    """The names a function sees: its module's, its closure's."""
    env = dict(base.__globals__)
    try:
        env.update(inspect.getclosurevars(base).nonlocals)
    except (TypeError, ValueError):
        pass
    return env


def _stored(stmts: List[ast.stmt]) -> Set[str]:
    return {n.id for s in stmts for n in ast.walk(s)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}


class _Inline(ast.stmt):
    """A step's generator in the process where it's used (yield from): its
    returns leave it, the value to target (a Name, or None)."""
    _fields = ("body", "target")


_P = inspect.Parameter


def _expand(func: Any) -> Tuple[ast.FunctionDef, str]:
    """The generator's definition as it's built, and its port's name: its
    parameters' values and the module's constants in it (as values), the
    generators it uses with yield from inlined."""
    base, sig = _unwrap(func)
    params = list(sig.parameters.values())
    if not params or params[0].default is not _P.empty \
            or params[0].kind not in (_P.POSITIONAL_ONLY, _P.POSITIONAL_OR_KEYWORD):
        raise ProcessError(f"{base.__name__}: a process takes its port, then parameters with "
                           f"values")
    port = params[0].name
    values = {}
    for p in params[1:]:
        if p.kind in (_P.VAR_POSITIONAL, _P.VAR_KEYWORD) or p.default is _P.empty:
            raise ProcessError(f"{base.__name__}: the parameter '{p.name}' has no value (give it "
                               f"a default, or functools.partial({base.__name__}, "
                               f"{p.name}=...))")
        values[p.name] = p.default
    fd = _source(base)
    ex = _Expander(port, {n.id for n in ast.walk(fd) if isinstance(n, ast.Name)})
    fd.body = ex.body(base, fd, {port: port}, values)
    return fd, port


class _Expander:
    """Inlines the steps a process uses (yield from): each step's parameters as
    the values given it (the port: the process's; a constant: the value; else
    a local, set where it's used), its locals named after it (<step>_<name>,
    one set of them however many places use it: only one runs at a time)."""

    def __init__(self, port: str, used: Set[str]) -> None:
        self.port = port
        self.used = set(used)
        self.stack: List[Callable] = []
        self.names: Dict[Callable, Dict[str, str]] = {}

    def body(self, base: Callable, fd: ast.FunctionDef, rename: Dict[str, str],
             values: Dict[str, Any]) -> List[ast.stmt]:
        """fd's body, rename's names renamed, values' names (constants) as values."""
        stored = _stored(fd.body)
        pre: List[ast.stmt] = []
        consts: Dict[str, Any] = {}
        for name, v in values.items():
            if not isinstance(v, (int, str)):
                raise ProcessError(f"{fd.name}: the parameter '{name}' is {v!r}: a constant (an "
                                   f"int) or the port")
            if name in stored:                       # (a parameter changed: a local)
                pre.append(_assign(rename.get(name, name), ast.Constant(v)))
            else:
                consts[name] = v
        env = _env(base)
        ex = self

        class _T(ast.NodeTransformer):
            def visit_Name(self, n: ast.Name):
                if n.id in rename:
                    n.id = rename[n.id]
                elif isinstance(n.ctx, ast.Load):
                    if n.id in consts:
                        return ast.copy_location(ast.Constant(consts[n.id]), n)
                    v = env.get(n.id)
                    if n.id not in stored and isinstance(v, int):
                        return ast.copy_location(ast.Constant(v), n)
                return n

            def visit_Call(self, n: ast.Call):
                self.generic_visit(n)
                # (getattr(port, 'start'): port.start)
                if isinstance(n.func, ast.Name) and n.func.id == "getattr" and len(n.args) == 2 \
                        and _is_port(n.args[0], ex.port) and _is_str(n.args[1]):
                    return ast.copy_location(ast.Attribute(
                        value=n.args[0], attr=n.args[1].value, ctx=ast.Load()), n)
                return n

            def visit_Expr(self, n: ast.Expr):
                self.generic_visit(n)
                v = n.value
                if isinstance(v, ast.YieldFrom):
                    return ex.inline(v, None, env, n)
                # (setattr(port, 'tx', v): port.tx = v)
                if isinstance(v, ast.Call) and isinstance(v.func, ast.Name) \
                        and v.func.id == "setattr" and len(v.args) == 3 \
                        and _is_port(v.args[0], ex.port) and _is_str(v.args[1]):
                    return ast.copy_location(ast.Assign(targets=[ast.Attribute(
                        value=v.args[0], attr=v.args[1].value, ctx=ast.Store())],
                        value=v.args[2]), n)
                return n

            def visit_Assign(self, n: ast.Assign):
                self.generic_visit(n)
                if isinstance(n.value, ast.YieldFrom):
                    if len(n.targets) != 1 or not isinstance(n.targets[0], ast.Name):
                        raise ProcessError(f"{_at(n)}: x = yield from ..., x a name")
                    return ex.inline(n.value, n.targets[0], env, n)
                return n
        out = [_T().visit(s) for s in copy.deepcopy(fd.body)]
        return pre + out

    def inline(self, y: ast.YieldFrom, target: Optional[ast.Name], env: Dict[str, Any],
               at: ast.stmt) -> _Inline:
        call = y.value
        fn = env.get(call.func.id) if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) \
            else None
        if fn is None or not is_process(fn) or isinstance(fn, System):
            raise ProcessError(f"{_at(at)}: yield from a generator function's call "
                               f"(yield from wait_until(port, 'start'))")
        base, sig = _unwrap(fn)
        if base in self.stack:
            raise ProcessError(f"{_at(at)}: {base.__name__} uses itself (yield from): a step "
                               f"can't")
        if any(isinstance(a, ast.Starred) for a in call.args) or any(k.arg is None
                                                                     for k in call.keywords):
            raise ProcessError(f"{_at(at)}: {base.__name__}'s arguments one by one")
        try:
            bound = sig.bind(*call.args, **{k.arg: k.value for k in call.keywords})
        except TypeError as e:
            raise ProcessError(f"{_at(at)}: {base.__name__}: {e}") from None
        bound.apply_defaults()
        fd = _source(base)
        names = self.names.get(base)
        if names is None:
            names = {}
            for n in list(sig.parameters) + sorted(_stored(fd.body)):
                name, k = f"{base.__name__}_{n}", 1
                while name in self.used:
                    k += 1
                    name = f"{base.__name__}{k}_{n}"
                self.used.add(name)
                names[n] = name
            self.names[base] = names
        stored = _stored(fd.body)
        rename = {n: names[n] for n in stored}
        values: Dict[str, Any] = {}
        pre: List[ast.stmt] = []
        for name, v in bound.arguments.items():
            if not isinstance(v, ast.AST):
                values[name] = v                         # (a default)
            elif _is_port(v, self.port) and name not in stored:
                rename[name] = self.port
            elif isinstance(v, ast.Constant) and isinstance(v.value, (int, str)):
                values[name] = v.value
            else:                                        # (an expression: a local)
                rename[name] = names[name]
                pre.append(_assign(names[name], v))
        for name in values:
            if name in stored:
                rename[name] = names[name]
        # (a step that returns one local: that local the target itself, no copy of it)
        rets = [n.value for n in ast.walk(fd) if isinstance(n, ast.Return)]
        if target is not None and rets and all(isinstance(r, ast.Name) and r.id == rets[0].id
                                               for r in rets) \
                and rets[0].id in stored and rets[0].id not in sig.parameters:
            rename[rets[0].id] = target.id
        self.stack.append(base)
        body = self.body(base, fd, rename, values)
        self.stack.pop()
        for s in body:                                   # (where each yield is: in which step)
            for n in ast.walk(s):
                if not hasattr(n, "_in"):
                    n._in = base.__name__
        node = ast.copy_location(_Inline(body=pre + body, target=target), at)
        node._file = getattr(at, "_file", None)
        return node


def _is_port(e: ast.expr, port: str) -> bool:
    return isinstance(e, ast.Name) and e.id == port


def _is_str(e: ast.expr) -> bool:
    return isinstance(e, ast.Constant) and isinstance(e.value, str)


def _pins(func: Any) -> Tuple[List[str], List[str]]:
    """(inputs, outputs) of the generator's port: its attributes read and never
    written, and written, in the order they first appear."""
    return _pins_of(*_expand(func))


def _pins_of(fd: ast.FunctionDef, port: str) -> Tuple[List[str], List[str]]:
    read: Dict[str, None] = {}
    written: Dict[str, None] = {}
    for n in ast.walk(fd):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == port:
            (written if isinstance(n.ctx, ast.Store) else read)[n.attr] = None
    return [a for a in read if a not in written], list(written)


# ==================================================================== the machine

class _Code:
    """The generator's body as instructions: statements, jumps, yields."""

    def __init__(self) -> None:
        self.ins: List[Tuple] = []

    def emit(self, *x) -> int:
        self.ins.append(x)
        return len(self.ins) - 1

    def here(self) -> int:
        return len(self.ins)

    def patch(self, at: int, target: int) -> None:
        kind, *rest = self.ins[at]
        if kind == "jump":
            self.ins[at] = ("jump", target)
        else:                                          # ("jump_unless", test, target)
            self.ins[at] = ("jump_unless", rest[0], target)


def _yields(nodes) -> bool:
    return any(isinstance(n, (ast.Yield, ast.YieldFrom)) for x in nodes for n in ast.walk(x))


class _Lower:
    """Structured code with yields as instructions (loops and ifs that yield as
    jumps; the rest as statements)."""

    def __init__(self, fresh: Callable[[str], str]) -> None:
        self.code = _Code()
        self.loops: List[Tuple[List[int], List[int]]] = []   # (breaks, continues) to patch
        self.steps: List[Tuple[Optional[str], List[int]]] = []  # (an inlined step's target, its returns)
        self.fresh = fresh
        self.ranges: Dict[str, Tuple[ast.expr, ast.expr]] = {}  # a for loop's var: (start, stop)
        # (a for loop's body: its var, its instructions [first, end), start, stop, step)
        self.bodies: List[Tuple[str, int, int, ast.expr, ast.expr, int]] = []

    def body(self, stmts: List[ast.stmt]) -> None:
        for s in stmts:
            self.stmt(s)

    def stmt(self, s: ast.stmt) -> None:
        c = self.code
        if isinstance(s, ast.Expr) and isinstance(s.value, ast.Yield):
            c.emit("yield", s.value.value, s)
            return
        if isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant):
            return                                              # (a docstring)
        if isinstance(s, _Inline):                              # (a step: yield from)
            self.steps.append((s.target.id if s.target is not None else None, []))
            self.body(s.body)
            _, returns = self.steps.pop()
            for at in returns:
                c.patch(at, c.here())
            return
        if isinstance(s, ast.Return) and self.steps:            # (out of the step)
            target, returns = self.steps[-1]
            if target is not None and s.value is not None \
                    and not (isinstance(s.value, ast.Name) and s.value.id == target):
                c.emit("stmt", _assign(target, s.value))
            returns.append(c.emit("jump", -1))
            return
        if isinstance(s, ast.Return):
            if s.value is not None:
                raise ProcessError(f"{_at(s)}: a process returns nothing")
            c.emit("end")
            return
        if isinstance(s, (ast.Break, ast.Continue)) and self.loops:
            at = c.emit("jump", -1)
            self.loops[-1][0 if isinstance(s, ast.Break) else 1].append(at)
            return
        if not _yields([s]) and not _escapes(s):
            if any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in ast.walk(s)):
                raise ProcessError(f"{_at(s)}: yield is a statement of its own (and yield "
                                   f"from: x = yield from ...)")
            if any(isinstance(n, ast.While) for n in ast.walk(s)):
                raise ProcessError(f"{_at(s)}: a while loop that doesn't yield (it would "
                                   f"wait without a clock passing)")
            c.emit("stmt", s)
            return
        if isinstance(s, ast.If):
            skip = c.emit("jump_unless", s.test, -1)
            self.body(s.body)
            if s.orelse:
                out = c.emit("jump", -1)
                c.patch(skip, c.here())
                self.body(s.orelse)
                c.patch(out, c.here())
            else:
                c.patch(skip, c.here())
            return
        if isinstance(s, ast.While) and not s.orelse:
            top = c.here()
            always = isinstance(s.test, ast.Constant) and bool(s.test.value)
            skip = None if always else c.emit("jump_unless", s.test, -1)
            self.loops.append(([], []))
            self.body(s.body)
            breaks, continues = self.loops.pop()
            for at in continues:
                c.patch(at, top)
            c.emit("jump", top)
            for at in ([skip] if skip is not None else []) + breaks:
                c.patch(at, c.here())
            return
        if isinstance(s, ast.For) and not s.orelse and isinstance(s.target, ast.Name) \
                and isinstance(s.iter, ast.Call) and isinstance(s.iter.func, ast.Name) \
                and s.iter.func.id == "range" and 1 <= len(s.iter.args) <= 3 and not s.iter.keywords:
            args = list(s.iter.args)
            start, stop = (ast.Constant(0), args[0]) if len(args) == 1 else (args[0], args[1])
            step = ast.literal_eval(args[2]) if len(args) == 3 else 1
            if not isinstance(step, int) or step == 0:
                raise ProcessError(f"{_at(s)}: a range's step is a constant, not 0")
            v = s.target.id
            self.ranges[v] = (start, stop)
            c.emit("stmt", _assign(v, start))
            top = c.here()
            skip = c.emit("jump_unless", ast.Compare(
                left=ast.Name(id=v, ctx=ast.Load()), ops=[ast.Lt() if step > 0 else ast.Gt()],
                comparators=[copy.deepcopy(stop)]), -1)
            self.loops.append(([], []))
            first = c.here()
            self.body(s.body)
            breaks, continues = self.loops.pop()
            nxt = c.here()
            self.bodies.append((v, first, nxt, start, stop, step))
            for at in continues:
                c.patch(at, nxt)
            c.emit("stmt", _assign(v, ast.BinOp(left=ast.Name(id=v, ctx=ast.Load()), op=ast.Add(),
                                                right=ast.Constant(step))))
            c.emit("jump", top)
            c.patch(skip, c.here())
            for at in breaks:
                c.patch(at, c.here())
            return
        raise ProcessError(f"{_at(s)}: {type(s).__name__} with a yield in it isn't "
                           f"supported (if, while, for ... in range)")

def _escapes(s: ast.stmt) -> bool:
    """Has s a break or continue of a loop around it (not one of its own), or a
    return?"""
    def walk(n: ast.AST, inside: bool) -> bool:
        if isinstance(n, ast.Return) or (isinstance(n, (ast.Break, ast.Continue)) and not inside):
            return True
        loop = isinstance(n, (ast.For, ast.While))
        for f, v in ast.iter_fields(n):
            vals = v if isinstance(v, list) else [v]
            for x in vals:
                if isinstance(x, ast.AST):
                    # (a loop's own body: its breaks are its own; its else isn't)
                    if walk(x, inside or (loop and f == "body")):
                        return True
        return False
    return walk(s, False)


def _assign(name: str, value: ast.expr) -> ast.Assign:
    return ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=copy.deepcopy(value))


def _desugar(fd: ast.FunctionDef, port: str, inputs: Set[str], outputs: Set[str]) -> None:
    """In place: port.x as the pin's name (an output's working value: x__n),
    augmented assignments written out."""
    class _T(ast.NodeTransformer):
        def visit_Attribute(self, n: ast.Attribute):
            self.generic_visit(n)
            if isinstance(n.value, ast.Name) and n.value.id == port:
                name = n.attr if n.attr in inputs else f"{n.attr}__n"
                return ast.Name(id=name, ctx=n.ctx)
            return n

        def visit_AugAssign(self, n: ast.AugAssign):
            self.generic_visit(n)
            load = copy.deepcopy(n.target)
            load.ctx = ast.Load()
            for x in ast.walk(load):
                if hasattr(x, "ctx"):
                    x.ctx = ast.Load()
            out = ast.copy_location(ast.Assign(targets=[n.target], value=ast.BinOp(
                left=load, op=n.op, right=n.value)), n)
            out._file = getattr(n, "_file", None)
            return out
    _T().visit(fd)
    ast.fix_missing_locations(fd)


def compile_process(func: Callable, attr: Optional[Dict[str, Any]] = None) -> Process:
    """The generator as a function of one clock: its inputs, then its state (the
    state's number __pc, the clocks left to wait __wait, the locals, the
    outputs' registers) as 'next' registers; it returns the outputs (their
    registers: what was written before this clock), then the state's next
    values. With the attr for it. A System: its processes' functions as one."""
    from .hwtypes import normalize_attr
    attr = {k: dict(v) for k, v in normalize_attr(attr).items() if isinstance(v, dict)}
    if isinstance(func, System):
        return _compile_system(func, attr)
    base, _ = _unwrap(func)
    fd, port = _expand(func)
    inputs, outputs = _pins_of(fd, port)
    for pin in inputs:                     # (an output's: followed from what's written)
        if "width" not in attr.get(pin, {}):
            raise ProcessError(f"{_at(fd)}: {fd.name}: give the input '{pin}' its width in attr")
    used = {n.id for n in ast.walk(fd) if isinstance(n, ast.Name)} | set(inputs) | set(outputs)
    # (a local named as a pin, data = port.data: its own name, the pins theirs)
    for n in ast.walk(fd):
        if isinstance(n, ast.Name) and n.id in set(inputs) | set(outputs):
            n.id = f"{n.id}_v"
    used |= {n.id for n in ast.walk(fd) if isinstance(n, ast.Name)}
    _desugar(fd, port, set(inputs), set(outputs))
    counter = [0]

    def fresh(base: str) -> str:
        while True:
            counter[0] += 1
            name = f"{base}{counter[0]}"
            if name not in used:
                used.add(name)
                return name
    low = _Lower(fresh)
    low.body(fd.body)
    low.code.emit("end")
    code = low.code.ins

    # The states: the start, and the instruction after each yield
    states: Dict[int, int] = {0: 0}
    order = [0]

    def state_of(i: int) -> int:
        if i not in states:
            states[i] = len(order)
            order.append(i)
        return states[i]
    waits: List[ast.expr] = []
    origin: Dict[int, ast.stmt] = {}                  # (a state: the yield it comes after)

    def segment(i: int, seen: Set[int]) -> List[ast.stmt]:
        out: List[ast.stmt] = []
        while True:
            if i in seen:
                raise ProcessError(f"{_at(fd)}: {fd.name}: a loop that doesn't yield (a clock "
                                   f"without end)")
            seen = seen | {i}
            kind, *rest = code[i]
            if kind == "stmt":
                out.append(copy.deepcopy(rest[0]))
                i += 1
            elif kind == "jump":
                i = rest[0]
            elif kind == "jump_unless":
                test, target = rest
                out.append(ast.If(test=copy.deepcopy(test), body=segment(i + 1, seen) or [ast.Pass()],
                                  orelse=segment(target, seen) or [ast.Pass()]))
                return out
            elif kind == "yield":
                n = rest[0]
                if n is None or (isinstance(n, ast.Constant) and n.value == 1):
                    wait: ast.expr = ast.Constant(0)
                else:
                    waits.append(n)
                    wait = ast.BinOp(left=copy.deepcopy(n), op=ast.Sub(), right=ast.Constant(1))
                st = state_of(i + 1)
                origin.setdefault(st, rest[1])
                out += [_assign(WAIT, wait), _assign(PC, ast.Constant(st))]
                return out
            else:                                     # end: the last state, forever
                out.append(_assign(PC, ast.Constant(DONE)))
                return out
    DONE = -1
    segments: List[List[ast.stmt]] = []
    k = 0
    while k < len(order):                             # (states found as segments are built)
        segments.append(segment(order[k], set()))
        k += 1
    done = len(order)
    for seg in segments:                              # (the end: the state after the last)
        for n in ast.walk(ast.Module(body=seg, type_ignores=[])):
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) \
                    and n.targets[0].id == PC and isinstance(n.value, ast.Constant) \
                    and n.value.value == DONE:
                n.value = ast.Constant(done)

    # The locals: every name assigned, but the pins'
    pins = set(inputs) | {f"{o}__n" for o in outputs}
    names: Dict[str, None] = {}
    for x in code:
        if x[0] == "stmt":
            for n in ast.walk(x[1]):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and n.id not in pins:
                    names[n.id] = None
    local_names = list(names)

    # The clock's function
    # (an output's register o__q: on the pin o, the port's name; its working value
    # o__n from it)
    body: List[ast.stmt] = [_assign(o, ast.Name(id=f"{o}__q", ctx=ast.Load())) for o in outputs]
    body += [_assign(f"{o}__n", ast.Name(id=o, ctx=ast.Load())) for o in outputs]
    chain: List[ast.stmt] = [ast.Pass()]
    for s in reversed(range(len(segments) + 1)):
        seg = segments[s] if s < len(segments) else [ast.Pass()]
        chain = [ast.If(test=ast.Compare(left=ast.Name(id=PC, ctx=ast.Load()), ops=[ast.Eq()],
                                         comparators=[ast.Constant(s)]),
                        body=seg or [ast.Pass()], orelse=chain)]
    body.append(ast.If(test=ast.Compare(left=ast.Name(id=WAIT, ctx=ast.Load()), ops=[ast.NotEq()],
                                        comparators=[ast.Constant(0)]),
                       body=[_assign(WAIT, ast.BinOp(left=ast.Name(id=WAIT, ctx=ast.Load()),
                                                     op=ast.Sub(), right=ast.Constant(1)))],
                       orelse=chain))
    returned = outputs + [PC, WAIT] + local_names + [f"{o}__n" for o in outputs]
    params = inputs + [PC, WAIT] + local_names + [f"{o}__q" for o in outputs]
    ns = dict(base.__globals__)
    # (the module's name: the generator's)
    step, step_fd, source = _function(fd.name, params, body, returned, ns, base.__qualname__)

    # Its attr: the pins as given, the outputs registers (their working values the
    # same format; without a width, followed from what's written), the locals'
    # widths followed from what's assigned to them
    widths = _local_widths(code, local_names, low.ranges, inputs, outputs, attr, ns, fd.name,
                           low.bodies)
    out_attr: Dict[str, Dict[str, Any]] = {}
    for p in inputs:
        out_attr[p] = dict(attr[p])
    # (the state's next values: returned for their registers, 'internal': no ports)
    for o in outputs:
        given = attr.get(o, {})
        m = {k: v for k, v in given.items() if k != "init"}
        if "width" not in given:
            m.update(widths[o])
        out_attr[f"{o}__q"] = dict(m, next=f"{o}__n", init=int(given.get("init", 0)))
        out_attr[o] = dict(m)
        out_attr[f"{o}__n"] = dict(m, internal=True)
    for v in local_names:
        out_attr[v] = dict(widths[v], next=v, internal=True)
    n_states = done + 1
    out_attr[PC] = {"signed": 0, "width": max(1, (n_states - 1).bit_length()), "next": PC,
                    "internal": True}
    longest = max([_bound(n, widths, attr, ns)[1] for n in waits] + [1])
    out_attr[WAIT] = {"signed": 0, "width": max(1, (longest - 1).bit_length()), "next": WAIT,
                      "internal": True}
    where = [f"{_at(fd)}  {_line(fd)}  (the start)"]
    for st in range(1, done):
        y = origin[st]
        step_in = f"  (in {y._in})" if getattr(y, "_in", None) else ""
        where.append(f"{_at(y)}  {_line(y)}{step_in}")
    where.append("(the end: the generator returned)")
    return Process(func=func, step=step, attr=out_attr, inputs=inputs, outputs=outputs,
                   locals=local_names, states=n_states, source=source, widths=widths,
                   tree=step_fd, where=where)


def _function(name: str, params: List[str], body: List[ast.stmt], returned: List[str],
              ns: Dict[str, Any], label: str) -> Tuple[Callable, ast.FunctionDef, str]:
    """The clock's function: (it, its definition, its source)."""
    body = body + [ast.Return(value=ast.Tuple(
        elts=[ast.Name(id=n, ctx=ast.Load()) for n in returned], ctx=ast.Load()))]
    fd = ast.FunctionDef(name=name, args=ast.arguments(
        posonlyargs=[], args=[ast.arg(arg=p) for p in params], vararg=None, kwonlyargs=[],
        kw_defaults=[], kwarg=None, defaults=[]), body=body, decorator_list=[], returns=None,
        type_comment=None, type_params=[])
    module = ast.fix_missing_locations(ast.Module(body=[fd], type_ignores=[]))
    source = ast.unparse(module) + "\n"
    filename = f"<py2ver process {label}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    exec(compile(source, filename, "exec"), ns)
    return ns[name], fd, source


def _bound(e: ast.expr, widths, attr, ns) -> Tuple[int, int]:
    from .share import NotShareable, _fmt_range, _interval
    rng = {k: _fmt_range(m) for k, m in widths.items()}
    rng.update({f"{k}__n": r for k, r in list(rng.items())})     # (a followed output's value)
    for k, m in attr.items():
        if "width" in m:
            rng.setdefault(k, _fmt_range(m))
            rng.setdefault(f"{k}__n", _fmt_range(m))
    try:
        return _interval(_constants(e, ns), rng, {})
    except NotShareable as x:
        raise ProcessError(f"can't bound {ast.unparse(e)} ({x})") from None


def _constants(e: ast.expr, ns: Dict[str, Any]) -> ast.expr:
    """e with the module's integer constants (BIT) as their values."""
    class _C(ast.NodeTransformer):
        def visit_Name(self, n: ast.Name):
            v = ns.get(n.id)
            if isinstance(v, int) and not isinstance(v, bool) and isinstance(n.ctx, ast.Load):
                return ast.Constant(v)
            return n
    return _C().visit(copy.deepcopy(e))


def _local_widths(code, names: List[str], ranges, inputs, outputs, attr, ns,
                  fname: str, bodies=()) -> Dict[str, Dict[str, Any]]:
    """Each local's format: attr's, or the hull of what's assigned to it (a for
    loop's variable: its range's ends; in the loop's body, short of its stop),
    followed until it settles; an output's without a width in attr so too (from
    its 'init'), as out[o]."""
    from .share import NotShareable, _fmt_of, _fmt_range, _hull, _interval
    rng: Dict[str, Tuple[int, int]] = {}
    followed = [o for o in outputs if "width" not in attr.get(o, {})]
    for p in inputs + outputs:
        if p in followed:
            init = int(attr.get(p, {}).get("init", 0))
            rng[p] = rng[f"{p}__n"] = (init, init)
        else:
            rng[p] = rng[f"{p}__n"] = _fmt_range(attr[p])
    given = {v for v in names if "width" in attr.get(v, {})}
    for v in names:
        rng[v] = _fmt_range(attr[v]) if v in given else (0, 0)
    assigns = [(k, n) for k, x in enumerate(code) if x[0] == "stmt" for n in ast.walk(x[1])
               if isinstance(n, ast.Assign) and len(n.targets) == 1
               and isinstance(n.targets[0], ast.Name)]
    for v, (start, stop) in ranges.items():           # (a range: from its start to its stop)
        if v in given:
            continue
        try:
            a = _interval(_constants(start, ns), rng, {})
            b = _interval(_constants(stop, ns), rng, {})
            rng[v] = _hull(rng[v], a, b)
        except NotShareable:
            pass
    # (in a loop's body its variable is short of its stop: range(8)'s i 0..7)
    scopes = []
    for v, lo_k, hi_k, start, stop, step in bodies:
        if v in given:
            continue
        try:
            a = _interval(_constants(start, ns), rng, {})
            b = _interval(_constants(stop, ns), rng, {})
        except NotShareable:
            continue
        inner = (a[0], b[1] - 1) if step > 0 else (b[0] + 1, a[1])
        if inner[0] <= inner[1]:
            scopes.append((lo_k, hi_k, v, inner))
    first: Dict[str, ast.Assign] = {}
    for _, n in assigns:
        first.setdefault(n.targets[0].id, n)

    def where(v: str) -> str:
        """Where v is first assigned, and its name in attr (an output's: the pin's)."""
        shown = v[:-3] if v.endswith("__n") and v[:-3] in outputs else v
        return f"{_at(first[v]) if v in first else fname}: '{shown}'"
    growing: Set[str] = set()
    for _ in range(64):
        growing = set()
        for k, n in assigns:
            v = n.targets[0].id
            if v not in rng or v in given or v in ranges or v in inputs:
                continue
            seen = rng
            inside = [(u, r) for lo_k, hi_k, u, r in scopes if lo_k <= k < hi_k]
            if inside:
                seen = dict(rng)
                for u, r in inside:
                    seen[u] = r
            try:
                r = _interval(_constants(n.value, ns), seen, {})
            except NotShareable as e:
                raise ProcessError(f"{_at(n)}: can't follow the values of {where(v).split(': ')[-1]} "
                                   f"({e}): give its width in attr") from None
            new = _hull(rng[v], r)
            if new != rng[v]:
                rng[v] = new
                growing.add(v)
        for o in followed:                          # (the pin: its register's values)
            rng[o] = rng[f"{o}__n"]
        if not growing:
            break
    if growing:                                     # (a counter of its own, say: unbounded)
        raise ProcessError(f"{where(sorted(growing)[0])}: its values grow without end as "
                           f"followed: give its width in attr")
    out = {}
    for v in names + followed:
        if v in given:
            out[v] = {"signed": int(attr[v].get("signed", 0)), "width": int(attr[v]["width"])}
            continue
        lo, hi = rng[v]
        if max(abs(lo), abs(hi)) >> MAX_BITS:
            raise ProcessError(f"{where(v)}: its values grow without end as followed: give "
                               f"its width in attr")
        out[v] = dict(_fmt_of((lo, hi)))
    return out


# ==================================================================== processes together

def connect(name: str = "system", wires: Optional[Dict[str, str]] = None,
            **processes: Any) -> System:
    """
    Processes as one design (its module: name), each a generator (or a partial
    of one) by its keyword, or (it, its attr):

        echo = connect(rx=(uart_rx, {'rx': 1}), tx=(uart_tx, {'data': 8, 'start': 1}),
                       name="echo", wires={"tx.data": "rx.data", "tx.start": "rx.valid"})

    A wire gives an input of one process an output of another: that output's
    register, as on its pin (what was written before this clock), and its
    format. The design's pins: the inputs not wired and the outputs, each by
    its own name (two of a name: the process's first, rx_data). The design's
    attr: by those names, and by 'process.name' for a process's own (a local's
    width, 'tx.k'), over the process's own attr.
    """
    from .hwtypes import normalize_attr
    if not name.isidentifier():
        raise ProcessError(f"connect: the design's name {name!r} is a name (the module's)")
    if not processes:
        raise ProcessError("connect: processes, by name (connect(rx=uart_rx, tx=uart_tx, ...))")
    attrs = {}
    for p, f in list(processes.items()):
        if isinstance(f, tuple) and len(f) == 2:
            processes[p], attrs[p] = f[0], normalize_attr(f[1])
        f = processes[p]
        if isinstance(f, System) or not inspect.isgeneratorfunction(f):
            raise ProcessError(f"connect: {p} isn't a process (a generator function, or (it, "
                               f"its attr))")
    pins = {p: _pins(f) for p, f in processes.items()}

    def pin(spec: str, kind: str) -> Tuple[str, str]:
        p, _, n = spec.partition(".")
        if p not in processes:
            raise ProcessError(f"connect: {spec}: no process '{p}' (processes: "
                               f"{', '.join(processes)})")
        have = pins[p][0 if kind == "input" else 1]
        if n not in have:
            raise ProcessError(f"connect: {spec} isn't an {kind} of {p} (its {kind}s: "
                               f"{', '.join(have)})")
        return p, n
    return System(name, dict(processes),
                  {pin(dst, "input"): pin(src, "output") for dst, src in (wires or {}).items()},
                  attrs)


def _layout(s: System):
    """(each process's (inputs, outputs), each pin's name in the design, the
    design's inputs, its outputs)."""
    pins = {p: _pins(f) for p, f in s.processes.items()}
    ext_in = [(p, n) for p, (ins, _) in pins.items() for n in ins if (p, n) not in s.wires]
    ext_out = [(p, n) for p, (_, outs) in pins.items() for n in outs]
    count: Dict[str, int] = {}
    for _, n in ext_in + ext_out:
        count[n] = count.get(n, 0) + 1
    ext = {(p, n): n if count[n] == 1 else f"{p}_{n}" for p, n in ext_in + ext_out}
    if len(set(ext.values())) != len(ext):
        raise ProcessError(f"{s.name}: two pins of one name ({sorted(ext.values())}): rename one")
    return pins, ext, ext_in, ext_out


def _sub_attrs(s: System, attr: Dict[str, Any], layout,
               built: Optional[Dict[str, Process]] = None) -> Dict[str, Dict[str, Any]]:
    """Each process's attr, by its own names: its own, the design's over it; a
    wired input without a width its output's format (as built, if it is)."""
    pins, ext, _, _ = layout
    built = built or {}

    def look(p: str, n: str) -> Dict[str, Any]:
        a = dict(s.attrs.get(p, {}).get(n, {}))
        a.update(attr.get(ext[(p, n)], {}) if (p, n) in ext else {})
        a.update(attr.get(f"{p}.{n}", {}))
        return a
    out = {}
    for p, (ins, outs) in pins.items():
        a = {k: dict(v) for k, v in s.attrs.get(p, {}).items()}       # (its locals' too)
        for k, v in attr.items():
            if k.startswith(f"{p}."):
                a[k[len(p) + 1:]] = dict(a.get(k[len(p) + 1:], {}), **v)
        for n in ins + outs:
            a[n] = look(p, n)
        for n in ins:
            if (p, n) in s.wires and "width" not in a[n]:
                q, m = s.wires[(p, n)]
                src = built[q].attr[m] if q in built else look(q, m)
                a[n].update({k: v for k, v in src.items() if k in ("width", "signed", "frac")})
        out[p] = a
    return out


def _simulate_system(s: System, inputs: Sequence[Any], attr: Dict[str, Any],
                     outputs: Optional[Sequence[str]]) -> List[Dict[str, int]]:
    layout = _layout(s)
    _, ext, ext_in, ext_out = layout
    sub = _sub_attrs(s, attr, layout)
    runs = {p: _Runner(f, sub[p]) for p, f in s.processes.items()}
    names = [ext[k] for k in ext_in]
    outputs = list(outputs or [ext[k] for k in ext_out])
    rows = []
    for k, vec in enumerate(inputs):
        values = vec if isinstance(vec, dict) else dict(zip(names, vec))
        now = {p: {o: getattr(r.port, o) for o in r.outs} for p, r in runs.items()}
        row = {ext[(p, o)]: v for p, d in now.items() for o, v in d.items()}
        rows.append({o: row[o] for o in outputs})
        for p, r in runs.items():
            for n in r.ins:
                q = s.wires.get((p, n))
                setattr(r.port, n, now[q[0]][q[1]] if q else values.get(ext[(p, n)], 0))
        for r in runs.values():
            r.clock(k)
    return rows


def _compile_system(s: System, attr: Dict[str, Any]) -> Process:
    """The processes' clock functions as one: each one's state its own (named
    <process>_<name>), the outputs' registers first read (the pins' values this
    clock, what a wire carries), then each process's clock."""
    layout = _layout(s)
    pins, ext, ext_in, ext_out = layout
    # (a process once its wired inputs' formats are known: its outputs' as built)
    built: Dict[str, Process] = {}
    while len(built) < len(s.processes):
        sub = _sub_attrs(s, attr, layout, built)
        ready = [p for p in s.processes if p not in built
                 and all("width" in sub[p][n] for n in pins[p][0])]
        if not ready:
            p = next(p for p in s.processes if p not in built)
            n = next(n for n in pins[p][0] if "width" not in sub[p][n])
            raise ProcessError(f"{s.name}.{p}: give the input '{n}' its width in attr"
                               + (" (its wire's output's isn't known before it)"
                                  if (p, n) in s.wires else ""))
        for p in ready:
            try:
                built[p] = compile_process(s.processes[p], sub[p])
            except ProcessError as e:
                raise ProcessError(f"{s.name}.{p}: {e}") from None
    params = [ext[k] for k in ext_in]
    returned = [ext[k] for k in ext_out]
    copies: List[ast.stmt] = []
    clocks: List[ast.stmt] = []
    out_attr: Dict[str, Dict[str, Any]] = {}
    state: List[str] = []
    ns: Dict[str, Any] = {}
    states = 0
    for p, f in s.processes.items():
        pr = built[p]

        def name_of(n: str, p=p, pr=pr) -> str:
            if n in pr.inputs:
                return ext[s.wires[(p, n)]] if (p, n) in s.wires else ext[(p, n)]
            if n in pr.outputs:
                return ext[(p, n)]
            return f"{p}_{n}"
        fd = copy.deepcopy(pr.tree)
        for n in ast.walk(fd):
            if isinstance(n, ast.Name):
                n.id = name_of(n.id)
        k = len(pr.outputs)
        copies += fd.body[:k]                     # (o = o__q: the pins this clock)
        clocks += fd.body[k:-1]
        returned += [e.id for e in fd.body[-1].value.elts[k:]]
        params += [name_of(a.arg) for a in fd.args.args[len(pr.inputs):]]
        for key, m in pr.attr.items():
            if key in pr.inputs and (p, key) in s.wires:
                continue
            m = dict(m)
            if "next" in m:
                m["next"] = name_of(m["next"])
            out_attr[name_of(key)] = m
            if key not in pr.inputs and key not in pr.outputs:
                state.append(name_of(key))
        for g, v in _unwrap(f)[0].__globals__.items():
            ns.setdefault(g, v)
        states += pr.states
    clash = set(state) & set(ext.values())
    if clash or len(set(params)) != len(params):
        raise ProcessError(f"{s.name}: a pin named as a process's state ({sorted(clash)}): "
                           f"rename it")
    step, tree, source = _function(s.name, params, copies + clocks, returned, ns, s.name)
    return Process(func=s, step=step, attr=out_attr, inputs=[ext[k] for k in ext_in],
                   outputs=[ext[k] for k in ext_out], locals=state, states=states,
                   source=source, tree=tree, parts={p: built[p] for p in s.processes})
