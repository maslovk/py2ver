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

The generator itself is the reference: simulate() runs it a clock at a time,
the inputs of each clock given, its outputs read; the machine built from it
(compile_process: a function of one clock with its state, see "State
registers" in the README) computes the same, clock for clock.
"""
from __future__ import annotations

import ast
import copy
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


def is_process(func: Any) -> bool:
    return inspect.isgeneratorfunction(func)


@dataclass
class Process:
    """A process built: its clock's function and attr, its pins, its registers."""
    func: Callable                       # the generator
    step: Callable                       # one clock of it
    attr: Dict[str, Dict[str, Any]]
    inputs: List[str]
    outputs: List[str]
    locals: List[str]
    states: int
    source: str = ""
    widths: Dict[str, Dict[str, Any]] = field(default_factory=dict)


# ==================================================================== the reference

def simulate(func: Callable, inputs: Sequence[Any], attr: Optional[Dict[str, Any]] = None,
             outputs: Optional[Sequence[str]] = None) -> List[Dict[str, int]]:
    """Run the generator a clock at a time: inputs, a clock's input pins each (a
    dict, or a tuple in the port's order of inputs); returns each clock's
    outputs (a dict), the values its pins have during that clock (written
    before it)."""
    ins, outs = _pins(func)
    outputs = list(outputs or outs)
    attr = attr or {}
    port = SimpleNamespace(**{o: int(attr.get(o, {}).get("init", 0)) for o in outs})
    gen = func(port)
    wait, done = 0, False
    rows = []
    for k, vec in enumerate(inputs):
        row = {o: getattr(port, o) for o in outputs}
        values = vec if isinstance(vec, dict) else dict(zip(ins, vec))
        for name, v in values.items():
            setattr(port, name, v)
        if not done:
            if wait == 0:
                try:
                    n = next(gen)
                    wait = (1 if n is None else int(n)) - 1
                    if wait < 0:
                        raise ProcessError(f"yield {n} at clock {k}: a yield is a clock at least")
                except StopIteration:
                    done = True
            else:
                wait -= 1
        rows.append(row)
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


def _port_name(fd: ast.FunctionDef) -> str:
    if len(fd.args.args) != 1 or fd.args.vararg or fd.args.kwonlyargs:
        raise ProcessError(f"{fd.name}: a process takes one argument, its port")
    return fd.args.args[0].arg


def _pins(func: Callable) -> Tuple[List[str], List[str]]:
    """(inputs, outputs) of the generator's port: its attributes read and never
    written, and written, in the order they first appear."""
    fd = _source(func)
    port = _port_name(fd)
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
            if any(isinstance(n, ast.Yield) for n in ast.walk(s)):
                raise ProcessError(f"line {s.lineno}: yield is a statement of its own")
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
    """Has s a break or continue of a loop around it (not one of its own)?"""
    def walk(n: ast.AST, inside: bool) -> bool:
        if isinstance(n, (ast.Break, ast.Continue)) and not inside:
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
    values. With the attr for it."""
    attr = {k: dict(v) for k, v in (attr or {}).items() if isinstance(v, dict)}
    fd = _source(func)
    port = _port_name(fd)
    inputs, outputs = _pins(func)
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
    body.append(ast.Return(value=ast.Tuple(elts=[ast.Name(id=n, ctx=ast.Load()) for n in returned],
                                           ctx=ast.Load())))
    params = inputs + [PC, WAIT] + local_names + [f"{o}__q" for o in outputs]
    step_name = fd.name                       # (the module's name: the generator's)
    step_fd = ast.FunctionDef(name=step_name, args=ast.arguments(
        posonlyargs=[], args=[ast.arg(arg=p) for p in params], vararg=None, kwonlyargs=[],
        kw_defaults=[], kwarg=None, defaults=[]), body=body, decorator_list=[], returns=None,
        type_comment=None, type_params=[])
    module = ast.fix_missing_locations(ast.Module(body=[step_fd], type_ignores=[]))
    source = ast.unparse(module) + "\n"
    filename = f"<py2ver process {func.__qualname__}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    ns = dict(func.__globals__)
    exec(compile(source, filename, "exec"), ns)
    step = ns[step_name]

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
                   locals=local_names, states=n_states, source=source, widths=widths)


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
