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


# ==================================================================== the reference

class _Runner:
    """A generator run a clock at a time on its port."""

    def __init__(self, func: Callable, attr: Dict[str, Any]) -> None:
        self.ins, self.outs = _pins(func)
        self.port = SimpleNamespace(**{n: 0 for n in self.ins},
                                    **{o: int(attr.get(o, {}).get("init", 0)) for o in self.outs})
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
    attr = attr or {}
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

    def _clocks(self, run_stream: Callable) -> Callable[[Sequence[Dict[str, int]]], List[Dict[str, int]]]:
        pr = self.process
        if pr is None:
            raise ValueError("Not a process (a generator function): use PY_stream / TB_stream")

        def run(clocks: Sequence[Dict[str, int]]) -> List[Dict[str, int]]:
            rows = run_stream([tuple(int(c.get(p, 0)) for p in pr.inputs) for c in clocks])
            return [dict(zip(pr.outputs, r[:len(pr.outputs)])) for r in rows]
        return run

    def PY_process(self) -> Callable[[Sequence[Dict[str, int]]], List[Dict[str, int]]]:
        """The machine's bit-exact model, a clock at a time (PY_stream's)."""
        return self._clocks(self.PY_stream())

    def TB_process(self) -> Callable[[Sequence[Dict[str, int]]], List[Dict[str, int]]]:
        """The machine simulated (Verilog), a clock at a time (TB_stream's)."""
        return self._clocks(self.TB_stream())

    def GEN_process(self) -> Callable[[Sequence[Dict[str, int]]], List[Dict[str, int]]]:
        """The generator itself, run a clock at a time (simulate)."""
        pr = self.process
        return lambda clocks: simulate(pr.func, clocks, self._process_attr, pr.outputs)


# ==================================================================== the pins

def _source(func: Callable) -> ast.FunctionDef:
    fd = ast.parse(textwrap.dedent(inspect.getsource(func))).body[0]
    if not isinstance(fd, ast.FunctionDef):
        raise ProcessError(f"{func.__name__} isn't a function")
    return fd


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
                    return ex.inline(v, None, env, n.lineno)
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
                        raise ProcessError(f"line {n.lineno}: x = yield from ..., x a name")
                    return ex.inline(n.value, n.targets[0], env, n.lineno)
                return n
        out = [_T().visit(s) for s in copy.deepcopy(fd.body)]
        return pre + out

    def inline(self, y: ast.YieldFrom, target: Optional[ast.Name], env: Dict[str, Any],
               lineno: int) -> _Inline:
        call = y.value
        fn = env.get(call.func.id) if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) \
            else None
        if fn is None or not is_process(fn) or isinstance(fn, System):
            raise ProcessError(f"line {lineno}: yield from a generator function's call "
                               f"(yield from wait_until(port, 'start'))")
        base, sig = _unwrap(fn)
        if base in self.stack:
            raise ProcessError(f"line {lineno}: {base.__name__} uses itself (yield from): a step "
                               f"can't")
        if any(isinstance(a, ast.Starred) for a in call.args) or any(k.arg is None
                                                                     for k in call.keywords):
            raise ProcessError(f"line {lineno}: {base.__name__}'s arguments one by one")
        try:
            bound = sig.bind(*call.args, **{k.arg: k.value for k in call.keywords})
        except TypeError as e:
            raise ProcessError(f"line {lineno}: {base.__name__}: {e}") from None
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
        node = _Inline(body=pre + body, target=target)
        node.lineno = lineno
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

    def body(self, stmts: List[ast.stmt]) -> None:
        for s in stmts:
            self.stmt(s)

    def stmt(self, s: ast.stmt) -> None:
        c = self.code
        if isinstance(s, ast.Expr) and isinstance(s.value, ast.Yield):
            c.emit("yield", s.value.value)
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
                raise ProcessError(f"line {s.lineno}: a process returns nothing")
            c.emit("end")
            return
        if isinstance(s, (ast.Break, ast.Continue)) and self.loops:
            at = c.emit("jump", -1)
            self.loops[-1][0 if isinstance(s, ast.Break) else 1].append(at)
            return
        if not _yields([s]) and not _escapes(s):
            if any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in ast.walk(s)):
                raise ProcessError(f"line {s.lineno}: yield is a statement of its own (and yield "
                                   f"from: x = yield from ...)")
            if any(isinstance(n, ast.While) for n in ast.walk(s)):
                raise ProcessError(f"line {s.lineno}: a while loop that doesn't yield (it would "
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
                raise ProcessError(f"line {s.lineno}: a range's step is a constant, not 0")
            v = s.target.id
            self.ranges[v] = (start, stop)
            c.emit("stmt", _assign(v, start))
            top = c.here()
            skip = c.emit("jump_unless", ast.Compare(
                left=ast.Name(id=v, ctx=ast.Load()), ops=[ast.Lt() if step > 0 else ast.Gt()],
                comparators=[copy.deepcopy(stop)]), -1)
            self.loops.append(([], []))
            self.body(s.body)
            breaks, continues = self.loops.pop()
            nxt = c.here()
            for at in continues:
                c.patch(at, nxt)
            c.emit("stmt", _assign(v, ast.BinOp(left=ast.Name(id=v, ctx=ast.Load()), op=ast.Add(),
                                                right=ast.Constant(step))))
            c.emit("jump", top)
            c.patch(skip, c.here())
            for at in breaks:
                c.patch(at, c.here())
            return
        raise ProcessError(f"line {s.lineno}: {type(s).__name__} with a yield in it isn't "
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
            return ast.Assign(targets=[n.target], value=ast.BinOp(left=load, op=n.op,
                                                                    right=n.value))
    _T().visit(fd)
    ast.fix_missing_locations(fd)


def compile_process(func: Callable, attr: Optional[Dict[str, Any]] = None) -> Process:
    """The generator as a function of one clock: its inputs, then its state (the
    state's number __pc, the clocks left to wait __wait, the locals, the
    outputs' registers) as 'next' registers; it returns the outputs (their
    registers: what was written before this clock), then the state's next
    values. With the attr for it. A System: its processes' functions as one."""
    attr = {k: dict(v) for k, v in (attr or {}).items() if isinstance(v, dict)}
    if isinstance(func, System):
        return _compile_system(func, attr)
    base, _ = _unwrap(func)
    fd, port = _expand(func)
    inputs, outputs = _pins_of(fd, port)
    for pin in inputs + outputs:
        if "width" not in attr.get(pin, {}):
            raise ProcessError(f"{fd.name}: give the pin '{pin}' its width in attr")
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

    def segment(i: int, seen: Set[int]) -> List[ast.stmt]:
        out: List[ast.stmt] = []
        while True:
            if i in seen:
                raise ProcessError(f"{fd.name}: a loop that doesn't yield (a clock without end)")
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
                out += [_assign(WAIT, wait), _assign(PC, ast.Constant(state_of(i + 1)))]
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
    # same format), the locals' widths followed from what's assigned to them
    out_attr: Dict[str, Dict[str, Any]] = {}
    for p in inputs:
        out_attr[p] = dict(attr[p])
    # (the state's next values: returned for their registers, 'internal': no ports)
    for o in outputs:
        m = {k: v for k, v in attr[o].items() if k != "init"}
        out_attr[f"{o}__q"] = dict(m, next=f"{o}__n", init=int(attr[o].get("init", 0)))
        out_attr[o] = dict(m)
        out_attr[f"{o}__n"] = dict(m, internal=True)
    widths = _local_widths(code, local_names, low.ranges, inputs, outputs, attr, ns, fd.name)
    for v in local_names:
        out_attr[v] = dict(widths[v], next=v, internal=True)
    n_states = done + 1
    out_attr[PC] = {"signed": 0, "width": max(1, (n_states - 1).bit_length()), "next": PC,
                    "internal": True}
    longest = max([_bound(n, widths, attr, ns)[1] for n in waits] + [1])
    out_attr[WAIT] = {"signed": 0, "width": max(1, (longest - 1).bit_length()), "next": WAIT,
                      "internal": True}
    return Process(func=func, step=step, attr=out_attr, inputs=inputs, outputs=outputs,
                   locals=local_names, states=n_states, source=source, widths=widths,
                   tree=step_fd)


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
                  fname: str) -> Dict[str, Dict[str, Any]]:
    """Each local's format: attr's, or the hull of what's assigned to it (a for
    loop's variable: its range's ends), followed until it settles."""
    from .share import NotShareable, _fmt_of, _fmt_range, _hull, _interval
    rng: Dict[str, Tuple[int, int]] = {}
    for p in inputs + outputs:
        rng[p] = rng[f"{p}__n"] = _fmt_range(attr[p])
    given = {v for v in names if "width" in attr.get(v, {})}
    for v in names:
        rng[v] = _fmt_range(attr[v]) if v in given else (0, 0)
    assigns = [n for x in code if x[0] == "stmt" for n in ast.walk(x[1])
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
    growing: Set[str] = set()
    for _ in range(64):
        growing = set()
        for n in assigns:
            v = n.targets[0].id
            if v not in rng or v in given or v in ranges or v in inputs:
                continue
            try:
                r = _interval(_constants(n.value, ns), rng, {})
            except NotShareable as e:
                raise ProcessError(f"{fname}: can't follow the values of '{v}' ({e}): give its "
                                   f"width in attr") from None
            new = _hull(rng[v], r)
            if new != rng[v]:
                rng[v] = new
                growing.add(v)
        if not growing:
            break
    if growing:                                     # (a counter of its own, say: unbounded)
        raise ProcessError(f"{fname}: the values of {', '.join(sorted(growing))} grow without "
                           f"end as followed: give its width in attr")
    out = {}
    for v in names:
        if v in given:
            out[v] = {"signed": int(attr[v].get("signed", 0)), "width": int(attr[v]["width"])}
            continue
        lo, hi = rng[v]
        if max(abs(lo), abs(hi)) >> MAX_BITS:
            raise ProcessError(f"{fname}: the values of '{v}' grow without end as followed: give "
                               f"its width in attr")
        out[v] = dict(_fmt_of((lo, hi)))
    return out


# ==================================================================== processes together

def connect(name: str = "system", wires: Optional[Dict[str, str]] = None,
            **processes: Callable) -> System:
    """
    Processes as one design (its module: name), each a generator (or a partial
    of one) by its keyword:

        echo = connect(rx=uart_rx, tx=uart_tx, name="echo",
                       wires={"tx.data": "rx.data", "tx.start": "rx.valid"})

    A wire gives an input of one process an output of another: that output's
    register, as on its pin (what was written before this clock). The design's
    pins: the inputs not wired and the outputs, each by its own name (two of a
    name: the process's first, rx_data). attr: by those names, and by
    'process.name' for a process's own (a local's width, 'tx.k'); a wired
    input takes its output's.
    """
    if not name.isidentifier():
        raise ProcessError(f"connect: the design's name {name!r} is a name (the module's)")
    if not processes:
        raise ProcessError("connect: processes, by name (connect(rx=uart_rx, tx=uart_tx, ...))")
    for p, f in processes.items():
        if isinstance(f, System) or not inspect.isgeneratorfunction(f):
            raise ProcessError(f"connect: {p} isn't a process (a generator function)")
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
                  {pin(dst, "input"): pin(src, "output") for dst, src in (wires or {}).items()})


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


def _sub_attrs(s: System, attr: Dict[str, Any], layout) -> Dict[str, Dict[str, Any]]:
    """Each process's attr, by its own names, from the design's."""
    pins, ext, _, _ = layout

    def look(p: str, n: str) -> Dict[str, Any]:
        if f"{p}.{n}" in attr:
            return dict(attr[f"{p}.{n}"])
        return dict(attr.get(ext.get((p, n)), {}))
    out = {}
    for p, (ins, outs) in pins.items():
        a = {}
        for n in ins:
            if (p, n) in s.wires and f"{p}.{n}" not in attr:      # (its output's format)
                a[n] = {k: v for k, v in look(*s.wires[(p, n)]).items() if k not in ("init", "pin")}
            else:
                a[n] = look(p, n)
        for n in outs:
            a[n] = look(p, n)
        for k, v in attr.items():
            if k.startswith(f"{p}.") and k[len(p) + 1:] not in a:
                a[k[len(p) + 1:]] = dict(v)
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
    _, ext, ext_in, ext_out = layout
    sub = _sub_attrs(s, attr, layout)
    params = [ext[k] for k in ext_in]
    returned = [ext[k] for k in ext_out]
    copies: List[ast.stmt] = []
    clocks: List[ast.stmt] = []
    out_attr: Dict[str, Dict[str, Any]] = {}
    state: List[str] = []
    ns: Dict[str, Any] = {}
    states = 0
    for p, f in s.processes.items():
        try:
            pr = compile_process(f, sub[p])
        except ProcessError as e:
            raise ProcessError(f"{s.name}.{p}: {e}") from None

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
                   source=source, tree=tree)
