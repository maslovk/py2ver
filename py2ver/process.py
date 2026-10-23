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
(functools.partial(uart_tx, bit=217): the same transmitter, twice the baud);
a record of them too (a NamedTuple, a dataclass: a bus's settings), its fields
read where it's used (bus.q). A step, a constant or a record's class can be
named through its module (adxl345.adxl345_start, adxl345.POWER_CTL).
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
registers" in docs/reference.md) computes the same, clock for clock.
"""
from __future__ import annotations

import ast
import copy
import dataclasses
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
    notes: List[str] = field(default_factory=list)      # (the locals renamed)
    opens: Dict[str, str] = field(default_factory=dict)  # (an open-drain pin: its line's input)
    probes: List[str] = field(default_factory=list)      # (its state: ports, with probes=True)

    def explain(self) -> str:
        """A readable description: each state with the line of the generator
        where it waits, then every register and its width."""
        if self.parts:
            return "\n\n".join(f"{p}: " + pr.explain() for p, pr in self.parts.items())
        name = getattr(_unwrap(self.func)[0], "__name__", "?")
        lines = [f"{name}: {self.states} states. Each state is a place where the generator "
                 f"waits: a yield (one clock cycle) or a yield n (n cycles)."]
        width = len(str(len(self.where)))
        lines += [f"  {k:>{width}}  {w}" for k, w in enumerate(self.where)]

        def bits(names):
            return ", ".join(f"{n} {self.attr[n]['width']}" for n in names)
        lines += ["Widths in bits (the outputs, the local variables and the control are registers):",
                  f"  inputs   {bits(self.inputs)}" if self.inputs else "  inputs   -",
                  f"  outputs  {bits(self.outputs)}",
                  f"  locals   {bits(self.locals)}" if self.locals else "  locals   -",
                  f"  control  {bits([PC, WAIT])}  ({PC} is the current state; {WAIT} counts "
                  f"down a yield n)"]
        lines += [f"  Note: {n}." for n in self.notes]
        return "\n".join(lines)


# ==================================================================== the reference

def _opens(attr: Optional[Dict[str, Any]]) -> Set[str]:
    """The open-drain pins attr names ({'sda': {'open_drain': True}})."""
    return {k for k, v in (attr or {}).items() if isinstance(v, dict) and v.get("open_drain")}


class _Port:
    """A process's port as the reference runs it: an open-drain pin written as
    what it drives (0: the line low; 1: let go) and read as the line (its
    <pin>_in input, what every driver on it leaves)."""

    def __init__(self, values: Dict[str, int], opens: Set[str], drives: Dict[str, int]) -> None:
        object.__setattr__(self, "_opens", opens)
        object.__setattr__(self, "_drive", dict(drives))
        self.__dict__.update(values)

    def __setattr__(self, name: str, value: int) -> None:
        if name in self._opens:
            self._drive[name] = value
        else:
            object.__setattr__(self, name, value)

    def __getattr__(self, name: str) -> int:
        if name in self.__dict__.get("_opens", ()):
            return self.__dict__.get(f"{name}_in", 1)
        raise AttributeError(name)


class _Runner:
    """A generator run a clock at a time on its port."""

    def __init__(self, func: Callable, attr: Dict[str, Any]) -> None:
        self.opens = _opens(attr)
        fd, port = _expand(func)
        self.ins, self.outs = _pins_of(fd, port, self.opens)
        self.opens &= set(self.outs)
        init = {o: attr.get(o) if isinstance(attr.get(o), dict) else {} for o in self.outs}
        # (from reset: attr's 'init', else what the generator sets before its first yield)
        resets = _reset_values(fd, port)
        start = {o: int(init[o].get("init", resets.get(o, 1 if o in self.opens else 0)))
                 for o in self.outs}
        self.port = _Port({**{n: 0 for n in self.ins}, **{f"{o}_in": 1 for o in self.opens},
                           **{o: v for o, v in start.items() if o not in self.opens}},
                          self.opens, {o: start[o] for o in self.opens})
        self.gen = func(self.port)
        self.wait, self.done = 0, False

    def out(self, o: str) -> int:
        """An output's value this clock (an open-drain pin's: what it drives)."""
        return self.port._drive[o] if o in self.opens else getattr(self.port, o)

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


class _Sim:
    """A process, or a System's processes together, a clock at a time: its pins
    by the design's names."""

    def __init__(self, func: Any, attr: Optional[Dict[str, Any]]) -> None:
        from .hwtypes import normalize_attr
        attr = normalize_attr(attr)
        if isinstance(func, System):
            layout = _layout(func)
            _, ext, ext_in, ext_out = layout
            sub = _sub_attrs(func, attr, layout)
            self.runs = {p: _Runner(f, sub[p]) for p, f in func.processes.items()}
            self.ins, self.outs = [ext[k] for k in ext_in], [ext[k] for k in ext_out]
            self.ext, self.wires = ext, func.wires
        else:
            r = _Runner(func, attr)
            self.runs = {"": r}
            self.ins, self.outs = r.ins, r.outs
            self.ext, self.wires = {("", n): n for n in r.ins + r.outs}, {}
        # (each open-drain pin: its line's input, by the design's names)
        self.opens = {self.ext[(p, o)]: self.ext[(p, f"{o}_in")]
                      for p, r in self.runs.items() for o in r.opens}
        # (each input's source, once: a wire's output, or the design's input and the
        # value it has if not given: 0, a line 1)
        lines = set(self.opens.values())
        self.feeds = [(r, [(n, self.wires.get((p, n)),
                            None if (p, n) in self.wires else self.ext[(p, n)],
                            1 if self.ext.get((p, n)) in lines else 0) for n in r.ins])
                      for p, r in self.runs.items()]

    def row(self) -> Dict[str, int]:
        """The outputs during this clock."""
        return {self.ext[(p, o)]: r.out(o) for p, r in self.runs.items() for o in r.outs}

    def clock(self, values: Dict[str, int], k: int) -> None:
        """A clock with these inputs (one not in values: 0; a line: 1, let go)."""
        # (the outputs a wire carries: as they are before this clock)
        now = {p: {o: r.out(o) for o in r.outs} for p, r in self.runs.items()} if self.wires \
            else None
        for r, feeds in self.feeds:
            port = r.port
            for n, q, name, default in feeds:
                setattr(port, n, now[q[0]][q[1]] if q else values.get(name, default))
        for r in self.runs.values():
            r.clock(k)


def simulate(func: Any, inputs: Sequence[Any], attr: Optional[Dict[str, Any]] = None,
             outputs: Optional[Sequence[str]] = None) -> List[Dict[str, int]]:
    """Run the generator a clock at a time: inputs, a clock's input pins each (a
    dict, an input not in it 0; or a tuple in the port's order of inputs);
    returns each clock's outputs (a dict), the values its pins have during that
    clock (written before it). A System: its processes so, together."""
    sim = _Sim(func, attr)
    outputs = list(outputs or sim.outs)
    rows = []
    for k, vec in enumerate(inputs):
        row = sim.row()
        rows.append({o: row[o] for o in outputs})
        sim.clock(vec if isinstance(vec, dict) else dict(zip(sim.ins, vec)), k)
    return rows


# ==================================================================== testbenches

@dataclass
class Run:
    """A testbench's run (drive()): each clock's inputs and outputs, and what the
    testbench returned."""
    inputs: List[Dict[str, int]]
    outputs: List[Dict[str, int]]
    value: Any = None
    simulator: Optional[str] = None             # (check(): the Verilog's, icarus or verilator)


class TestbenchFailed(AssertionError):
    """A testbench's assert (drive(), check()): at which clock, where."""


def _short(path: str) -> str:
    """A source file's name to show: py2ver's as py2ver/..., one below here relative."""
    package = os.path.dirname(os.path.abspath(__file__))
    if os.path.abspath(path).startswith(package + os.sep):
        return os.path.join("py2ver", os.path.relpath(path, package))
    try:
        rel = os.path.relpath(path)
        return rel if not rel.startswith("..") else path
    except ValueError:
        return path


def _waiting(gen) -> str:
    """Where a testbench is waiting: each generator of its yield from chain."""
    out = []
    while gen is not None and getattr(gen, "gi_frame", None) is not None:
        f = gen.gi_frame
        out.append(f"{_short(f.f_code.co_filename)}:{f.f_lineno} ({f.f_code.co_name})")
        if f.f_code.co_name == "together" and f.f_code.co_filename.endswith("protocol.py"):
            # (the steps run at once: each one not done yet, where it waits)
            steps, done = f.f_locals.get("steps", ()), f.f_locals.get("done", ())
            out[-1] += " [" + "; ".join(_waiting(s) for s, d in zip(steps, done) if not d) + "]"
            break
        gen = gen.gi_yieldfrom
    return " -> ".join(out) or "?"


def _raised_at(e: BaseException) -> str:
    tb = e.__traceback__
    while tb is not None and tb.tb_next is not None:
        tb = tb.tb_next
    return f"{_short(tb.tb_frame.f_code.co_filename)}:{tb.tb_lineno}" if tb else "?"


class _Pins:
    """The design's pins as a testbench sees them: its inputs to write (each kept
    until written again), its outputs to read (this clock's). An open-drain pin
    the testbench drives too (0: low; 1, at first: let go) and reads as its
    line: low if either side pulls it low (a pull-up's bus)."""

    def __init__(self, sim: _Sim) -> None:
        lines = set(sim.opens.values())
        object.__setattr__(self, "_opens", sim.opens)
        object.__setattr__(self, "_ins", {**{n: 0 for n in sim.ins if n not in lines},
                                          **{o: 1 for o in sim.opens}})
        object.__setattr__(self, "_outs", sim.outs)
        object.__setattr__(self, "_row", sim.row())

    # (its methods' names begin with _: a pin may be called anything)
    def _line(self, o: str) -> int:
        """An open-drain pin's line: the design's drive and the testbench's."""
        return self._row[o] & self._ins[o]

    def _inputs(self) -> Dict[str, int]:
        """The design's inputs this clock (an open-drain pin's: its line)."""
        values = {n: v for n, v in self._ins.items() if n not in self._opens}
        values.update({i: self._line(o) for o, i in self._opens.items()})
        return values

    def __getattr__(self, name: str) -> int:
        if name in self._opens:
            return self._line(name)
        if name in self._row:
            return self._row[name]
        if name in self._ins:
            return self._ins[name]
        raise AttributeError(f"the design has no pin '{name}' (its inputs: "
                             f"{', '.join(self._ins)}; its outputs: {', '.join(self._outs)})")

    def __setattr__(self, name: str, value: int) -> None:
        if name not in self._ins:
            what = "an output" if name in self._outs else "no pin"
            raise AttributeError(f"'{name}' is {what} of the design: a testbench writes its "
                                 f"inputs ({', '.join(self._ins)})")
        self._ins[name] = int(value)


def drive(func: Any, tb: Callable, attr: Optional[Dict[str, Any]] = None,
          limit: int = 10_000_000) -> Run:
    """
    Run a testbench against the generator (or a System's): tb is a generator of
    one argument, the design's pins. It writes the inputs (kept until written
    again), reads the outputs (this clock's: what was written before it), and
    `yield` lets a clock pass, `yield n` n clocks; when it returns, the run
    ends (its value: the Run's). The steps of a protocol (yield from
    uart_send(dut, 'rx', b, BIT)) drive and read pins as well.
    """
    sim = _Sim(func, attr)
    pins = _Pins(sim)
    gen = tb(pins)
    inputs: List[Dict[str, int]] = []
    rows: List[Dict[str, int]] = []
    wait, value = 0, None
    while True:
        row = sim.row()
        object.__setattr__(pins, "_row", row)
        if wait == 0:
            try:
                n = next(gen)
            except StopIteration as e:
                value = e.value
                break
            except AssertionError as e:
                raise TestbenchFailed(f"the testbench at clock {len(rows):,} ({_raised_at(e)}): "
                                      f"{e}") from e
            wait = 1 if n is None else int(n)
            if wait < 1:
                raise ProcessError(f"the testbench's yield {n} at clock {len(rows)}: a clock at "
                                   f"least")
        values = pins._inputs()
        inputs.append(values)
        rows.append(row)
        sim.clock(values, len(rows) - 1)
        wait -= 1
        if len(rows) > limit:
            raise ProcessError(f"the testbench ran {limit:,} clocks without returning, waiting at "
                               f"{_waiting(gen)} (limit= for more)")
    return Run(inputs, rows, value)


class ProcessMixin:
    """Py2ver for a process: its references and simulation a clock at a time,
    each clock's inputs a dict, each clock's outputs (its pins) a dict."""

    process: Optional[Process] = None

    def _clocks(self, run_stream: Callable, vcd: Optional[str]) -> Callable[[Sequence[Dict[str, int]]], List[Dict[str, int]]]:
        pr = self.process
        if pr is None:
            raise ValueError("Not a process (a generator function): use PY_stream / TB_stream")

        def run(clocks: Sequence[Dict[str, int]]) -> List[Dict[str, int]]:
            lines = set(pr.opens.values())             # (a line not given: let go, 1)
            rows = run_stream([tuple(int(c.get(p, 1 if p in lines else 0)) for p in pr.inputs)
                               for c in clocks])
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

    def check(self, tb: Any, verilog: bool = True, vcd: Optional[str] = None,
              limit: int = 10_000_000) -> Any:
        """
        Check the design: Python, py2ver's model of the hardware and the
        simulated Verilog must all agree.

        For a function, give it input vectors: p.check([(3, 1), (7, 2)]). Each
        output is compared with what the function returns as written; one too
        narrow for its value is named, with the width it needs. Returns the
        outputs. Without a simulator installed, the Verilog is skipped, with a
        warning.

        For a process, give it a testbench, tb (see drive()), and it runs in three steps:
        1. tb runs against the generator, a clock at a time (its asserts: a
           TestbenchFailed at their clock);
        2. the inputs of each clock of that run go through the machine's model;
        3. and (verilog=True) through its Verilog: Icarus, or Verilator for a
           long run (PY2VER_SIM=icarus|verilator chooses; Run.simulator says).
        Each must give the generator's outputs, clock for clock, else a
        CheckFailed saying at which clock and pin, both runs' waveforms written
        beside (<name>_generator.vcd, <name>_model.vcd or _verilog.vcd).
        vcd: the generator's run's waveforms; limit: the clocks tb may run.
        Returns the Run: each clock's inputs and outputs, what tb returned
        (value, as the generator's run gave it).
        """
        pr = self.process
        if pr is None:
            if callable(tb):
                raise ValueError("check() takes a list of input vectors for a function (a "
                                 "testbench is for a process: a generator)")
            return self._check_function(tb, verilog)
        run = drive(pr.func, tb, self._process_attr, limit)
        if vcd:
            write_vcd(vcd, pr, run.inputs, run.outputs)
        others = [("model", self.PY_process())]
        if verilog:
            from .native import choose_simulator
            run.simulator = choose_simulator(self, len(run.inputs))    # (as TB_stream chooses)
            others.append((f"verilog ({run.simulator})", self.TB_process()))
        for what, runner in others:
            rows = runner(run.inputs)
            d = difference(run.outputs, rows)
            if d:
                base = self.outdir / self.t_name
                file = what.split()[0]
                write_vcd(f"{base}_generator.vcd", pr, run.inputs, run.outputs)
                write_vcd(f"{base}_{file}.vcd", pr, run.inputs, rows)
                raise CheckFailed(f"{self.t_name}: the {what} differs from the generator at "
                                  f"{d} (generator / {file}; the waveforms: "
                                  f"{base}_generator.vcd, {base}_{file}.vcd)")
        return run


class CheckFailed(AssertionError):
    """A design that doesn't do what its generator does (check())."""


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
    return f"{_short(f)}:{n.lineno}"


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
    returns leave it, the value to target (a Name; a Tuple of them, the values of
    a return a, b; or None)."""
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
            if isinstance(v, (bytes, bytearray)):
                v = tuple(v)                          # (a message: a table of its bytes)
            if not isinstance(v, (int, str, tuple)) and not _record(v):
                raise ProcessError(f"{fd.name}: the parameter '{name}' is {v!r}: a constant (an "
                                   f"int, bytes, a record of them: a NamedTuple, a dataclass) "
                                   f"or the port")
            if _record(v) and name in stored:
                raise ProcessError(f"{fd.name}: the parameter '{name}' is a record (its fields "
                                   f"constants): it can't be assigned")
            if name in stored:                       # (a parameter changed: a local)
                pre.append(_assign(rename.get(name, name), _value_node(v)))
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
                        return ast.copy_location(_value_node(consts[n.id]), n)
                    v = env.get(n.id)
                    # (the module's constants: an int, bytes, a tuple of ints, a table, a record)
                    if n.id not in stored and (isinstance(v, (int, bytes)) or _record(v) or (
                            isinstance(v, tuple) and v and all(type(e) is int for e in v))):
                        return ast.copy_location(_value_node(v), n)
                return n

            # (a record's field: its value; a module's constant too: adxl345.ADDRESS)
            def visit_Attribute(self, n: ast.Attribute):
                self.generic_visit(n)
                if isinstance(n.ctx, ast.Load) and isinstance(n.value, (ast.Name, ast.Attribute)):
                    v = _resolved(n, env, set(rename) | stored)
                    if isinstance(v, (int, bytes)) or _record(v) or (
                            isinstance(v, tuple) and v and all(type(e) is int for e in v)):
                        return ast.copy_location(_value_node(v), n)
                if isinstance(n.value, ast.Constant) and _record(n.value.value):
                    rec = n.value.value
                    if not isinstance(n.ctx, ast.Load):
                        raise ProcessError(f"{_at(n)}: {type(rec).__name__}'s fields are "
                                           f"constants: '{n.attr}' can't be assigned")
                    if not hasattr(rec, n.attr):
                        raise ProcessError(f"{_at(n)}: {type(rec).__name__} has no field "
                                           f"'{n.attr}'")
                    v = getattr(rec, n.attr)
                    if not isinstance(v, (int, str, bytes)) and not _record(v) and not (
                            isinstance(v, tuple) and all(type(e) is int for e in v)):
                        raise ProcessError(f"{_at(n)}: {type(rec).__name__}.{n.attr} is {v!r}: "
                                           f"a record's fields are constants (ints, strings, "
                                           f"bytes, records)")
                    return ast.copy_location(_value_node(v), n)
                return n

            def visit_Call(self, n: ast.Call):
                self.generic_visit(n)
                # (a record of constants: SPI(half, 3), half a constant; made here)
                cls = _resolved(n.func, env, set(rename))
                if _record_type(cls):
                    args = n.args + [k.value for k in n.keywords]
                    if not all(isinstance(a, ast.Constant) for a in args) \
                            or any(k.arg is None for k in n.keywords):
                        raise ProcessError(f"{_at(n)}: {cls.__name__}(...) of constants (a "
                                           f"record's fields are)")
                    try:
                        rec = cls(*[a.value for a in n.args],
                                  **{k.arg: k.value.value for k in n.keywords})
                    except TypeError as e:
                        raise ProcessError(f"{_at(n)}: {e}") from None
                    return ast.copy_location(ast.Constant(rec), n)
                # (len of a constant: its length)
                if isinstance(n.func, ast.Name) and n.func.id == "len" and len(n.args) == 1:
                    a = n.args[0]
                    if isinstance(a, ast.Tuple) and all(isinstance(e, ast.Constant) for e in a.elts):
                        return ast.copy_location(ast.Constant(len(a.elts)), n)
                    if isinstance(a, ast.Constant) and isinstance(a.value, (str, bytes)):
                        return ast.copy_location(ast.Constant(len(a.value)), n)
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
                    t = n.targets[0] if len(n.targets) == 1 else None
                    # (a, b = yield from ...: a step returning a, b)
                    if not (isinstance(t, ast.Name) or isinstance(t, ast.Tuple) and t.elts
                            and all(isinstance(e, ast.Name) for e in t.elts)):
                        raise ProcessError(f"{_at(n)}: x = yield from ..., x a name (or names: "
                                           f"x, y = yield from ...)")
                    return ex.inline(n.value, t, env, n)
                return n

            # (constants' arithmetic: its value, mode & 1 for a step's mode)
            def visit_BinOp(self, n: ast.BinOp):
                self.generic_visit(n)
                if isinstance(n.left, ast.Constant) and isinstance(n.right, ast.Constant) \
                        and type(n.left.value) is int and type(n.right.value) is int:
                    v = _value(n, {})
                    if v is not None:
                        return ast.copy_location(ast.Constant(v), n)
                return n

            # (a constant's if, msb_first's: the branch it takes)
            def visit_If(self, n: ast.If):
                self.generic_visit(n)
                if isinstance(n.test, ast.Constant):
                    return (n.body if n.test.value else n.orelse) or [ast.Pass()]
                return n

            def visit_IfExp(self, n: ast.IfExp):
                self.generic_visit(n)
                if isinstance(n.test, ast.Constant):
                    return n.body if n.test.value else n.orelse
                return n
        out: List[ast.stmt] = []
        for s in copy.deepcopy(fd.body):
            r = _T().visit(s)
            out += r if isinstance(r, list) else [r]
        for s in out:                   # (a record: its fields read, or given to a step; no more)
            for n in ast.walk(s):
                if isinstance(n, ast.Constant) and _record(n.value):
                    raise ProcessError(f"{_at(n)}: a {type(n.value).__name__} is a record of "
                                       f"constants: read its fields (bus.q), or give it to a "
                                       f"step; it isn't a value")
        return pre + out

    def inline(self, y: ast.YieldFrom, target: Optional[ast.expr], env: Dict[str, Any],
               at: ast.stmt) -> _Inline:
        call = y.value
        # (a step by its name, or a module's: adxl345.adxl345_start)
        fn = _resolved(call.func, env, set()) if isinstance(call, ast.Call) else None
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
            elif isinstance(v, ast.Constant) and (isinstance(v.value, (int, str, bytes))
                                                  or _record(v.value)):
                values[name] = v.value                   # (a record too: given on as it is)
            elif isinstance(v, ast.Tuple) and all(isinstance(e, ast.Constant) for e in v.elts):
                values[name] = tuple(e.value for e in v.elts)   # (a constant table, passed on)
            elif isinstance(v, ast.Name) and name not in stored:
                rename[name] = v.id                      # (a local's: read as it is)
            else:                                        # (an expression: a local)
                rename[name] = names[name]
                pre.append(_assign(names[name], v, at))
        for name in values:
            if name in stored:
                rename[name] = names[name]
        rets = [n.value for n in ast.walk(fd) if isinstance(n, ast.Return)]
        if target is not None:
            # (values to names: a return of as many, each)
            k = len(target.elts) if isinstance(target, ast.Tuple) else 1
            for r in [n for n in ast.walk(fd) if isinstance(n, ast.Return)]:
                got = len(r.value.elts) if isinstance(r.value, ast.Tuple) else (
                    0 if r.value is None else 1)
                if got != k and not (got == 0 and k == 1):
                    def say(n):
                        return {0: "nothing", 1: "one value"}.get(n, f"{n} values")
                    raise ProcessError(
                        f"{_at(at)}: {base.__name__} returns {say(got)} ({_at(r)}), "
                        f"{say(k)} assigned here (several: return a, b, and "
                        f"a, b = yield from ...)")
        # (a step that returns one local: that local the target itself, no copy of it)
        if isinstance(target, ast.Name) and rets and all(isinstance(r, ast.Name) and r.id == rets[0].id
                                               for r in rets) \
                and rets[0].id in stored and rets[0].id not in sig.parameters \
                and target.id not in rename.values():    # (x = yield from f(port, x): a copy)
            rename[rets[0].id] = target.id
        self.stack.append(base)
        body = self.body(base, fd, rename, values)
        self.stack.pop()
        for s in body:            # (where each yield is: in which steps, used from where)
            for n in ast.walk(s):
                n._in = getattr(n, "_in", ()) + ((base.__name__, _at(at)),)
        node = ast.copy_location(_Inline(body=pre + body, target=target), at)
        node._file = getattr(at, "_file", None)
        return node


def _record(v: Any) -> bool:
    """A record of constants (a NamedTuple's, a dataclass's: a bus's settings): its
    fields read where it's used (bus.q), it given on to steps as it is."""
    return (isinstance(v, tuple) and hasattr(type(v), "_fields")) or (
        dataclasses.is_dataclass(v) and not isinstance(v, type))


def _record_type(v: Any) -> bool:
    """A record's class (SPI(half, 3) builds one where it's used, of constants)."""
    return isinstance(v, type) and (dataclasses.is_dataclass(v) or (
        issubclass(v, tuple) and hasattr(v, "_fields")))


def _resolved(e: ast.expr, env: Dict[str, Any], local: Set[str]) -> Any:
    """What a name, or a module's attribute (adxl345.ADDRESS, protocol.SPI), is in
    the module env (None: a local's, or nothing there)."""
    if isinstance(e, ast.Name):
        return None if e.id in local else env.get(e.id)
    if isinstance(e, ast.Attribute):
        mod = _resolved(e.value, env, local)
        if inspect.ismodule(mod):
            return getattr(mod, e.attr, None)
    return None


def _value_node(v: Any) -> ast.expr:
    """A constant as code: an int or a string as itself; bytes, a tuple: a table of
    ints (text[i] a table read); a record: itself, its fields read as they're used."""
    if _record(v):
        return ast.Constant(v)
    if isinstance(v, (bytes, bytearray, tuple)):
        return ast.Tuple(elts=[ast.Constant(int(b)) for b in v], ctx=ast.Load())
    return ast.Constant(v)


def _is_port(e: ast.expr, port: str) -> bool:
    return isinstance(e, ast.Name) and e.id == port


def _is_str(e: ast.expr) -> bool:
    return isinstance(e, ast.Constant) and isinstance(e.value, str)


def _reset_values(fd: ast.FunctionDef, port: str) -> Dict[str, int]:
    """The outputs the generator sets to a constant before its first yield, and
    those constants: the values the outputs have from reset.

    Only the straight run of statements at the start counts, steps included;
    it stops at the first yield, loop or if, since what comes after may happen
    later. An output set to something not constant there gets no value here.
    """
    found: Dict[str, int] = {}

    def walk(stmts: Sequence[ast.stmt]) -> bool:
        """True once the walk reached something that may take time."""
        for s in stmts:
            if isinstance(s, _Inline):
                if walk(s.body):
                    return True
                continue
            if isinstance(s, (ast.Pass, ast.Expr)) and not _yields([s]):
                continue                                # a docstring, a call
            if isinstance(s, ast.Assign) and not _yields([s]):
                for t in s.targets:
                    if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) \
                            and t.value.id == port:
                        if isinstance(s.value, ast.Constant) and isinstance(s.value.value, int):
                            found[t.attr] = int(s.value.value)
                        else:
                            found.pop(t.attr, None)
                continue
            return True
        return False
    walk(fd.body)
    return found


def _pins(func: Any, opens: Set[str] = frozenset()) -> Tuple[List[str], List[str]]:
    """(inputs, outputs) of the generator's port: its attributes read and never
    written, and written, in the order they first appear; an open-drain pin
    (opens) written: an output (its drive) and an input <pin>_in (its line)."""
    return _pins_of(*_expand(func), opens)


def _pins_of(fd: ast.FunctionDef, port: str,
             opens: Set[str] = frozenset()) -> Tuple[List[str], List[str]]:
    read: Dict[str, None] = {}
    written: Dict[str, None] = {}
    for n in ast.walk(fd):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == port:
            (written if isinstance(n.ctx, ast.Store) else read)[n.attr] = None
    return ([a for a in read if a not in written] + [f"{a}_in" for a in written if a in opens],
            list(written))


# ==================================================================== the machine

class _Code:
    """The generator's body as instructions: statements, jumps, yields."""

    def __init__(self) -> None:
        self.ins: List[Tuple] = []
        self.where: List[Optional[ast.AST]] = []      # (each one's statement in the source)
        self.at: Optional[ast.AST] = None

    def emit(self, *x) -> int:
        self.ins.append(x)
        self.where.append(self.at)
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
        self.steps: List[Tuple[Optional[ast.expr], List[int]]] = []  # (an inlined step's target, its returns)
        self.fresh = fresh

    def body(self, stmts: List[ast.stmt]) -> None:
        for s in stmts:
            self.stmt(s)

    def stmt(self, s: ast.stmt) -> None:
        c = self.code
        outer, c.at = c.at, (s if getattr(s, "_file", None) else c.at)
        try:
            self._stmt(s)
        finally:
            c.at = outer

    def _stmt(self, s: ast.stmt) -> None:
        c = self.code
        if isinstance(s, ast.Expr) and isinstance(s.value, ast.Yield):
            c.emit("yield", s.value.value, s)
            return
        if isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant):
            return                                              # (a docstring)
        if isinstance(s, _Inline):                              # (a step: yield from)
            self.steps.append((s.target, []))
            self.body(s.body)
            _, returns = self.steps.pop()
            for at in returns:
                c.patch(at, c.here())
            return
        if isinstance(s, ast.Return) and self.steps:            # (out of the step)
            target, returns = self.steps[-1]
            if isinstance(target, ast.Tuple):                   # (a, b = ...: return x, y)
                c.emit("stmt", ast.copy_location(ast.Assign(
                    targets=[copy.deepcopy(target)], value=copy.deepcopy(s.value)), s))
            elif target is not None and s.value is not None \
                    and not (isinstance(s.value, ast.Name) and s.value.id == target.id):
                c.emit("stmt", _assign(target.id, s.value, s))
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
            c.emit("stmt", _assign(v, start, s))
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


def _assign(name: str, value: ast.expr, like: Optional[ast.AST] = None) -> ast.Assign:
    """name = value (where like is in the source, if given)."""
    out = _assign_bare(name, value)
    if like is not None and hasattr(like, "lineno"):
        ast.copy_location(out, like)
        out._file = getattr(like, "_file", None)
    return out


def _assign_bare(name: str, value: ast.expr) -> ast.Assign:
    return ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=copy.deepcopy(value))


def _desugar(fd: ast.FunctionDef, port: str, inputs: Set[str], outputs: Set[str],
             opens: Set[str] = frozenset()) -> None:
    """In place: port.x as the pin's name (an output's working value: x__n; an
    open-drain pin's read, its line x_in), augmented assignments written out."""
    class _T(ast.NodeTransformer):
        def visit_Attribute(self, n: ast.Attribute):
            self.generic_visit(n)
            if isinstance(n.value, ast.Name) and n.value.id == port:
                if n.attr in opens and isinstance(n.ctx, ast.Load):
                    return ast.Name(id=f"{n.attr}_in", ctx=n.ctx)
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
    opens = _opens(attr)
    inputs, outputs = _pins_of(fd, port, opens)
    opens &= set(outputs)
    # An output set to a constant before the first yield has that value from
    # reset (unless attr gives its 'init').
    for o, v in _reset_values(fd, port).items():
        if o in outputs:
            attr.setdefault(o, {}).setdefault("init", v)
    # (an open-drain pin: a bit; its drive let go after reset; its line an input)
    for o in opens:
        if int(attr[o].get("width", 1)) != 1:
            raise ProcessError(f"{_at(fd)}: {fd.name}: the open-drain pin '{o}' is a bit")
        attr[o].setdefault("width", 1)
        attr[o].setdefault("init", 1)
        attr[f"{o}_in"] = {"width": 1}
    from .hwtypes import pin_widths
    pin_widths(attr, inputs)               # (an input on the board's pins: theirs)
    for pin in inputs:                     # (an output's: followed from what's written)
        if "width" not in attr.get(pin, {}):
            raise ProcessError(f"{_at(fd)}: {fd.name}: give the input '{pin}' its width in attr")
    used = {n.id for n in ast.walk(fd) if isinstance(n, ast.Name)} | set(inputs) | set(outputs)
    # (a local named as a pin, data = port.data: its own name, the pins theirs)
    for n in ast.walk(fd):
        if isinstance(n, ast.Name) and n.id in set(inputs) | set(outputs):
            n.id = f"{n.id}_v"
    used |= {n.id for n in ast.walk(fd) if isinstance(n, ast.Name)}
    _desugar(fd, port, set(inputs), set(outputs), opens)
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

    pins_in = set(inputs)
    one_bit = {n for n in inputs if int(attr[n]["width"]) == 1 and not attr[n].get("signed")}
    where_of = low.code.where

    def no_clock(path: Tuple, here) -> ProcessError:
        """The loop come round without a yield: where, and the way it goes."""
        loop = [k for k, _ in path[path.index(here):]] + [here[0]]
        lines: List[str] = []
        for k in loop:
            w = where_of[k]
            if w is not None and (not lines or lines[-1] != _at(w)):
                lines.append(_at(w))
        start = where_of[here[0]]
        return ProcessError(
            f"{_at(start) if start is not None else _at(fd)}: {fd.name}: a loop that doesn't "
            f"yield: in one clock it can go {' -> '.join(lines)} (each test on the way passed "
            f"without a clock passing). A yield on the way, or tests the inputs' values rule "
            f"out (an input keeps its value through a clock: wait_until(port, 'clk', 1) then "
            f"wait_until(port, 'clk', 0) waits for an edge)")

    def segment(i: int, seen: Tuple, known: Dict[str, int]) -> List[ast.stmt]:
        """The clock from instruction i on; known: the inputs' values the tests
        passed on the way say (an input keeps its value through the clock: a
        branch it rules out isn't taken, wait_until(sclk, 1) then wait_until(sclk,
        0) waits), and the locals set to constants. An instruction come to again
        knowing the same: a loop without a clock passing."""
        out: List[ast.stmt] = []
        while True:
            here = (i, tuple(sorted(known.items())))
            if here in seen:
                raise no_clock(seen, here)
            seen = seen + (here,)
            kind, *rest = code[i]
            if kind == "stmt":
                s = rest[0]
                out.append(copy.deepcopy(s))
                # (a local set to a constant: known for the rest of the clock)
                known = {k: v for k, v in known.items() if k not in _stored([s])}
                if isinstance(s, ast.Assign) and len(s.targets) == 1 \
                        and isinstance(s.targets[0], ast.Name) and isinstance(s.value, ast.Constant) \
                        and isinstance(s.value.value, int) and "width" not in attr.get(s.targets[0].id, {}):
                    known[s.targets[0].id] = s.value.value
                i += 1
            elif kind == "jump":
                i = rest[0]
            elif kind == "jump_unless":
                test, target = rest
                decided = _decide(test, known, pins_in)
                if decided is not None:
                    i = i + 1 if decided else target
                    continue
                when_true, when_false = _facts(test, pins_in, one_bit)
                out.append(ast.If(test=copy.deepcopy(test),
                                  body=segment(i + 1, seen, {**known, **when_true}) or [ast.Pass()],
                                  orelse=segment(target, seen, {**known, **when_false})
                                  or [ast.Pass()]))
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
        segments.append(segment(order[k], (), {}))
        k += 1
    done = len(order)
    ends = False                                      # (a generator that can return)
    for seg in segments:                              # (the end: the state after the last)
        for n in ast.walk(ast.Module(body=seg, type_ignores=[])):
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) \
                    and n.targets[0].id == PC and isinstance(n.value, ast.Constant) \
                    and n.value.value == DONE:
                n.value = ast.Constant(done)
                ends = True

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
    widths = _flow_widths(segments, done, local_names, inputs, outputs, attr, ns, fd.name)
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
    n_states = done + 1 if ends else done             # (the end a state if it's reached)
    out_attr[PC] = {"signed": 0, "width": max(1, (n_states - 1).bit_length()), "next": PC,
                    "internal": True}
    longest = max([_bound(n, widths, attr, ns)[1] for n in waits] + [1])
    out_attr[WAIT] = {"signed": 0, "width": max(1, (longest - 1).bit_length()), "next": WAIT,
                      "internal": True}
    where = [f"{_at(fd)}  {_line(fd)}  (where it starts)"]
    for st in range(1, done):
        y = origin[st]
        # A yield inside a step: say which step, and where it was called from
        # ("in wait_until, called from ps2.py:21; in falling_edge, called from ...").
        chain = "; ".join(f"in {name}, called from {site}"
                          for name, site in getattr(y, "_in", ()))
        where.append(f"{_at(y)}  {_line(y)}" + (f"  ({chain})" if chain else ""))
    if ends:
        where.append("(where it ends: the generator has returned)")
    notes = [f"{v} is the local variable {v[:-2]}, renamed so that it doesn't clash "
             f"with the pin port.{v[:-2]}"
             for v in local_names if v.endswith("_v") and v[:-2] in inputs + outputs]
    return Process(func=func, step=step, attr=out_attr, inputs=inputs, outputs=outputs,
                   locals=local_names, states=n_states, source=source, widths=widths,
                   tree=step_fd, where=where, notes=notes, opens={o: f"{o}_in" for o in opens},
                   probes=[PC, WAIT] + local_names)


def _pin_test(test: ast.expr, pins: Set[str]) -> Optional[Tuple[str, Optional[bool], int]]:
    """A test of an input against a constant: (the input, True for ==, False for !=,
    the constant); a test of an input alone, (it, None, 0): true unless 0."""
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        t = _pin_test(test.operand, pins)
        if t and t[1] is None:
            return t[0], True, 0                     # (not x: x == 0)
        if t:
            return t[0], not t[1], t[2]
        return None
    if isinstance(test, ast.Name) and test.id in pins:
        return test.id, None, 0
    if isinstance(test, ast.Compare) and len(test.ops) == 1 \
            and isinstance(test.ops[0], (ast.Eq, ast.NotEq)):
        a, b = test.left, test.comparators[0]
        if isinstance(b, ast.Name) and isinstance(a, ast.Constant):
            a, b = b, a
        if isinstance(a, ast.Name) and a.id in pins and isinstance(b, ast.Constant) \
                and isinstance(b.value, int):
            return a.id, isinstance(test.ops[0], ast.Eq), b.value
    return None


def _decide(test: ast.expr, known: Dict[str, int], pins: Set[str]) -> Optional[bool]:
    """The test's value if what's known (the inputs', the locals set to constants)
    decides it."""
    v = _value(test, known)
    return None if v is None else bool(v)


_OPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
        ast.BitAnd: lambda a, b: a & b, ast.BitOr: lambda a, b: a | b,
        ast.BitXor: lambda a, b: a ^ b, ast.RShift: lambda a, b: a >> b,
        ast.LShift: lambda a, b: a << b}
_CMP = {ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b, ast.Lt: lambda a, b: a < b,
        ast.LtE: lambda a, b: a <= b, ast.Gt: lambda a, b: a > b, ast.GtE: lambda a, b: a >= b}


def _value(e: ast.expr, known: Dict[str, int]) -> Optional[int]:
    """e's value, if it's of constants and known names only."""
    if isinstance(e, ast.Constant) and isinstance(e.value, int):
        return int(e.value)
    if isinstance(e, ast.Name):
        return known.get(e.id)
    if isinstance(e, ast.UnaryOp):
        v = _value(e.operand, known)
        if v is None:
            return None
        return {ast.Not: lambda: int(not v), ast.USub: lambda: -v, ast.Invert: lambda: ~v,
                ast.UAdd: lambda: v}.get(type(e.op), lambda: None)()
    if isinstance(e, ast.BinOp) and type(e.op) in _OPS:
        a, b = _value(e.left, known), _value(e.right, known)
        if a is None or b is None or (isinstance(e.op, (ast.LShift, ast.RShift)) and b < 0):
            return None
        return _OPS[type(e.op)](a, b)
    if isinstance(e, ast.Compare) and len(e.ops) == 1 and type(e.ops[0]) in _CMP:
        a, b = _value(e.left, known), _value(e.comparators[0], known)
        return None if a is None or b is None else int(_CMP[type(e.ops[0])](a, b))
    if isinstance(e, ast.BoolOp):
        vals = [_value(x, known) for x in e.values]
        if isinstance(e.op, ast.And):
            if any(v == 0 for v in vals if v is not None):
                return 0
            return None if None in vals else int(all(vals))
        if any(v for v in vals if v is not None):
            return 1
        return None if None in vals else 0
    return None


def _facts(test: ast.expr, pins: Set[str],
           one_bit: Set[str] = frozenset()) -> Tuple[Dict[str, int], Dict[str, int]]:
    """What the test says of an input's value: (if it's true, if it's false). A
    1-bit input not 0 is 1 (while port.clk == 0: left, clk is 1)."""
    if isinstance(test, ast.BoolOp):              # (an and true, an or false: each part so)
        parts = [_facts(v, pins, one_bit) for v in test.values]
        k = 0 if isinstance(test.op, ast.And) else 1
        merged: Dict[str, int] = {}
        for p in parts:
            merged.update(p[k])
        return (merged, {}) if k == 0 else ({}, merged)
    t = _pin_test(test, pins)
    if t is None:
        return {}, {}
    name, eq, c = t
    bit = name in one_bit and c in (0, 1)
    other = {name: 1 - c} if bit else {}
    if eq is None:
        return ({name: 1} if name in one_bit else {}), {name: 0}
    return ({name: c}, other) if eq else (other, {name: c})


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


Range = Tuple[int, int]
_NEGATE = {ast.Lt: ast.GtE, ast.LtE: ast.Gt, ast.Gt: ast.LtE, ast.GtE: ast.Lt,
           ast.Eq: ast.NotEq, ast.NotEq: ast.Eq}
_FLIP = {ast.Lt: ast.Gt, ast.LtE: ast.GtE, ast.Gt: ast.Lt, ast.GtE: ast.LtE,
         ast.Eq: ast.Eq, ast.NotEq: ast.NotEq}
WIDEN_AFTER = 12              # a state's ranges changed this often: widened to its tests' constants
GIVE_UP_AFTER = 64            # and this often: a range still growing grows without end


def _flow_widths(segments: List[List[ast.stmt]], done: int, names: List[str], inputs, outputs,
                 attr, ns, fname: str) -> Dict[str, Dict[str, Any]]:
    """
    Each local's format: attr's, or every value assigned to it, as followed
    through each state's clock: a state's registers' ranges at its start, the
    clock's statements in order, an if's branches each with what its test says
    (`if pos < 255: pos = pos + 1`: pos at most 254 there), each `__pc = k`
    passing its ranges on to state k, until they settle. A range that keeps
    growing is widened to the constants it's tested against (ca reset at
    DEBOUNCE: 0..DEBOUNCE); one that grows past them, an error naming every such
    local. An output without a width in attr so too (from its 'init'), as out[o].
    """
    from .share import NotShareable, _fmt_of, _fmt_range, _hull, _interval
    followed = [o for o in outputs if "width" not in attr.get(o, {})]
    given = {v: _fmt_range(attr[v]) for v in names if "width" in attr.get(v, {})}
    for o in outputs:
        if o not in followed:
            given[f"{o}__n"] = _fmt_range(attr[o])
    fixed: Dict[str, Range] = {p: _fmt_range(attr[p]) for p in inputs}
    fixed[WAIT] = (0, 1 << 62)
    fixed[PC] = (0, done)

    def user_name(v: str) -> str:
        """A register's name as written (o__q: the output o)."""
        return v[:-3] if v.endswith("__q") else v

    def where_of(v: str) -> str:
        """Where a register is first assigned (o__q: o__n)."""
        a = f"{v[:-3]}__n" if v.endswith("__q") else v
        return _at(first[a]) if a in first else fname

    def tests_of(v: str) -> List[int]:
        """The constants a register is tested against, and one either side."""
        b = user_name(v)
        return sorted(thresholds.get(b, set()) | thresholds.get(f"{b}__n", set()))

    def start_of(env: Dict[str, Range]) -> Dict[str, Range]:
        """A clock's registers from the env at its end (o__q: o__n)."""
        return {**{v: env[v] for v in names}, **{f"{o}__q": env[f"{o}__n"] for o in outputs}}
    init = {v: given.get(v, (0, 0)) for v in names}
    for o in outputs:
        v = int(attr.get(o, {}).get("init", 0))
        init[f"{o}__q"] = _fmt_range(attr[o]) if o not in followed else (v, v)
    assigned: Dict[str, Range] = {}                  # (every value each name took)
    first: Dict[str, ast.AST] = {}
    thresholds: Dict[str, Set[int]] = {}
    for seg in segments:
        for n in ast.walk(ast.Module(body=seg, type_ignores=[])):
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name):
                first.setdefault(n.targets[0].id, n)
                # (the constants in what's assigned, too: min(pos + 1, 255))
                for c in ast.walk(_constants(n.value, ns)):
                    if isinstance(c, ast.Constant) and type(c.value) is int:
                        thresholds.setdefault(n.targets[0].id, set()).update(
                            {c.value - 1, c.value, c.value + 1})
            if isinstance(n, ast.Compare):
                for a, b in ((n.left, n.comparators[0]), (n.comparators[0], n.left)):
                    b = _constants(b, ns)
                    if isinstance(a, ast.Name) and isinstance(b, ast.Constant) \
                            and isinstance(b.value, int):
                        thresholds.setdefault(a.id, set()).update(
                            {b.value - 1, b.value, b.value + 1})

    def shown(v: str) -> str:
        return v[:-3] if v.endswith("__n") and v[:-3] in outputs else v

    def value(e: ast.expr, env, s) -> Range:
        try:
            return _interval(_constants(e, ns), env, {})
        except NotShareable as x:
            t = s.targets[0].id if isinstance(s, ast.Assign) and isinstance(s.targets[0], ast.Name) \
                else "it"
            raise ProcessError(f"{_at(s)}: can't follow the values of '{shown(t)}' ({x}): give "
                               f"its width in attr") from None

    def join(a, b):
        if a is None:
            return b
        if b is None:
            return a
        return {k: _hull(a[k], b[k]) if k in a and k in b else a.get(k, b.get(k)) for k in {*a, *b}}

    def narrow(env, name: str, op, c: Range):
        if name not in env or name in given:
            return env
        lo, hi = env[name]
        if op is ast.Lt:
            hi = min(hi, c[1] - 1)
        elif op is ast.LtE:
            hi = min(hi, c[1])
        elif op is ast.Gt:
            lo = max(lo, c[0] + 1)
        elif op is ast.GtE:
            lo = max(lo, c[0])
        elif op is ast.Eq:
            lo, hi = max(lo, c[0]), min(hi, c[1])
        elif op is ast.NotEq and c[0] == c[1]:
            lo += lo == c[0]
            hi -= hi == c[0]
        if lo > hi:
            return None
        return {**env, name: (lo, hi)}

    def refine(env, test: ast.expr, truth: bool):
        """env where test is truth (None: it can't be)."""
        if env is None:
            return None
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            return refine(env, test.operand, not truth)
        if isinstance(test, ast.BoolOp):
            if isinstance(test.op, ast.And) == truth:     # (and true, or false: each so)
                for v in test.values:
                    env = refine(env, v, truth)
                return env
            out = None                                     # (one of them so)
            for v in test.values:
                out = join(out, refine(env, v, truth))
            return out
        if isinstance(test, ast.Compare) and len(test.ops) == 1 and type(test.ops[0]) in _NEGATE:
            op = type(test.ops[0]) if truth else _NEGATE[type(test.ops[0])]
            left, right = test.left, test.comparators[0]
            try:
                lv = _interval(_constants(left, ns), env, {})
                rv = _interval(_constants(right, ns), env, {})
            except NotShareable:
                return env
            if isinstance(left, ast.Name):
                env = narrow(env, left.id, op, rv)
            if env is not None and isinstance(right, ast.Name):
                env = narrow(env, right.id, _FLIP[op], lv)
            return env
        if isinstance(test, ast.Name):
            return narrow(env, test.id, ast.NotEq if truth else ast.Eq, (0, 0))
        return env

    def assign(env, name: str, r: Range):
        if name in (PC, WAIT):
            return env
        r = given.get(name, r)
        assigned[name] = _hull(assigned.get(name, r), r)
        return {**env, name: r}

    arrivals: Dict[int, Dict[str, Range]] = {}

    def walk(stmts, env):
        for s in stmts:
            if env is None:
                return None
            if isinstance(s, ast.Assign):
                t = s.targets[0]
                if isinstance(t, ast.Name) and t.id == PC and isinstance(s.value, ast.Constant):
                    k = int(s.value.value)               # (the state after this clock)
                    arrivals[k] = join(arrivals.get(k), start_of(env))
                    continue
                if isinstance(t, ast.Name):
                    env = assign(env, t.id, value(s.value, env, s))
                elif isinstance(t, ast.Tuple) and isinstance(s.value, ast.Tuple):
                    vals = [value(v, env, s) for v in s.value.elts]
                    for x, r in zip(t.elts, vals):
                        env = assign(env, x.id, r)
                else:
                    raise ProcessError(f"{_at(s)}: can't follow what's assigned: one name at a time")
            elif isinstance(s, ast.If):
                env = join(walk(s.body, refine(env, s.test, True)),
                           walk(s.orelse, refine(env, s.test, False)))
            elif isinstance(s, ast.For) and isinstance(s.target, ast.Name):
                # (a loop within a clock, of constant bounds: its variable over them)
                args = [value(a, env, s) for a in s.iter.args]
                lo, hi = (0, args[0][1] - 1) if len(args) == 1 else (args[0][0], args[1][1] - 1)
                inner = env
                for _ in range(16):
                    out = walk(s.body, assign(inner, s.target.id, (min(lo, hi), max(lo, hi))))
                    nxt = join(inner, out)
                    if nxt == inner:
                        break
                    inner = nxt
                env = inner
            elif isinstance(s, (ast.Pass, ast.Expr)):
                pass
            else:
                raise ProcessError(f"{_at(s)}: can't follow the widths through "
                                   f"{type(s).__name__}")
        return env

    # (each state's registers at its start; the first's: their initial values)
    start: Dict[int, Optional[Dict[str, Range]]] = {k: None for k in range(len(segments))}
    start[0] = dict(init)
    changes = {k: 0 for k in start}
    todo = [0]
    while todo:
        k = todo.pop(0)
        arrivals.clear()
        o = {f"{x}__q": start[k][f"{x}__q"] for x in outputs}
        env = {**fixed, **start[k], **{x: o[f"{x}__q"] for x in outputs},
               **{f"{x}__n": o[f"{x}__q"] for x in outputs}}
        walk(segments[k], env)
        for j, arrived in arrivals.items():
            if j >= len(segments):
                continue                                  # (the end)
            new = join(start[j], arrived)
            if new == start[j]:
                continue
            changes[j] += 1
            if changes[j] > WIDEN_AFTER and start[j] is not None:
                # (still growing: to the constants it's tested against; one growing past
                # them, or without any, long after: without end)
                widened, unbounded = {}, []
                for v, r in new.items():
                    old = start[j][v]
                    ts = tests_of(v)
                    hi = r[1] if r[1] == old[1] else next((t for t in ts if t >= r[1]), r[1])
                    lo = r[0] if r[0] == old[0] else next((t for t in reversed(ts) if t <= r[0]),
                                                          r[0])
                    widened[v] = (lo, hi)
                    if r != old and changes[j] > GIVE_UP_AFTER:
                        unbounded.append(v)
                if len(unbounded) == 1:
                    v = unbounded[0]
                    raise ProcessError(f"{where_of(v)}: {_unbounded(user_name(v))}")
                if unbounded:
                    each = ", ".join(f"'{user_name(v)}' ({where_of(v)})" for v in sorted(unbounded))
                    raise ProcessError(
                        f"{fname}: py2ver can't tell how many bits {each} need: nothing in the "
                        f"code limits their values. Give each a width in attr, or limit it "
                        f"with a test such as `while x > 0` (rather than `while x != 0`).")
                new = widened
            start[j] = new
            if j not in todo:
                todo.append(j)
    out = {}
    for v in names:
        if v in given:
            out[v] = {"signed": int(attr[v].get("signed", 0)), "width": int(attr[v]["width"])}
            continue
        r = assigned.get(v, (0, 0))
        for s in start.values():
            if s:
                r = _hull(r, s[v])
        if max(abs(r[0]), abs(r[1])) >> MAX_BITS:
            raise ProcessError(f"{_at(first[v]) if v in first else fname}: {_unbounded(v)}")
        out[v] = dict(_fmt_of(r))
    for o in followed:
        v = int(attr.get(o, {}).get("init", 0))
        r = _hull(assigned.get(f"{o}__n", (v, v)), (v, v))
        if max(abs(r[0]), abs(r[1])) >> MAX_BITS:
            raise ProcessError(f"{fname}: the output {_unbounded(o)}")
        out[o] = dict(_fmt_of(r))
    return out


def _unbounded(name: str) -> str:
    """The message for a value whose width can't be worked out."""
    return (f"py2ver can't tell how many bits '{name}' needs: nothing in the code limits its "
            f"value. Give it a width in attr ({{'{name}': 16}}, say), or limit it with a test "
            f"such as `while {name.rsplit('_', 1)[-1]} > 0` (rather than `!= 0`).")


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
    opens = {p: _opens(attrs.get(p)) for p in processes}
    pins = {p: _pins(f, opens[p]) for p, f in processes.items()}

    def pin(spec: str, kind: str) -> Tuple[str, str]:
        p, _, n = spec.partition(".")
        if p not in processes:
            raise ProcessError(f"connect: {spec}: no process '{p}' (processes: "
                               f"{', '.join(processes)})")
        have = pins[p][0 if kind == "input" else 1]
        if n not in have:
            raise ProcessError(f"connect: {spec} isn't an {kind} of {p} (its {kind}s: "
                               f"{', '.join(have)})")
        if n in opens[p] or n.removesuffix("_in") in opens[p]:
            raise ProcessError(f"connect: {spec} is an open-drain pin: a bus of them isn't "
                               f"wired yet (play the other side in a testbench)")
        return p, n
    return System(name, dict(processes),
                  {pin(dst, "input"): pin(src, "output") for dst, src in (wires or {}).items()},
                  attrs)


def _layout(s: System):
    """(each process's (inputs, outputs), each pin's name in the design, the
    design's inputs, its outputs)."""
    pins = {p: _pins(f, _opens(s.attrs.get(p))) for p, f in s.processes.items()}
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


def _compile_system(s: System, attr: Dict[str, Any]) -> Process:
    """The processes' clock functions as one: each one's state its own (named
    <process>_<name>), the outputs' registers first read (the pins' values this
    clock, what a wire carries), then each process's clock."""
    layout = _layout(s)
    pins, ext, ext_in, ext_out = layout
    # (a wired input: its output's format as built; wires both ways, a bit to
    # begin with, until the formats settle)
    built: Dict[str, Process] = {}

    def formats(b: Dict[str, Process]):
        return {(p, o): (pr.attr[o]["width"], pr.attr[o].get("signed", 0))
                for p, pr in b.items() for o in pr.outputs}
    for _ in range(8):
        sub = _sub_attrs(s, attr, layout, built)
        for p, (ins, _) in pins.items():
            for n in ins:
                if (p, n) in s.wires and "width" not in sub[p][n]:
                    sub[p][n].update(width=1, signed=0)
        new = {}
        for p, f in s.processes.items():
            try:
                new[p] = compile_process(f, sub[p])
            except ProcessError as e:
                raise ProcessError(f"{s.name}.{p}: {e}") from None
        settled = bool(built) and formats(new) == formats(built)
        built = new
        if settled:
            break
    else:
        raise ProcessError(f"{s.name}: the wires' widths don't settle: give the outputs on them "
                           f"their widths in attr")
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
        # (its own names: its parameters, what it assigns, its attr's; min, max, ...: as
        # they are)
        own = {a.arg for a in pr.tree.args.args} | set(pr.attr) | {
            n.id for n in ast.walk(pr.tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}

        def name_of(n: str, p=p, pr=pr, own=own) -> str:
            if n in pr.inputs:
                return ext[s.wires[(p, n)]] if (p, n) in s.wires else ext[(p, n)]
            if n in pr.outputs:
                return ext[(p, n)]
            return f"{p}_{n}" if n in own else n
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
                   source=source, tree=tree, parts={p: built[p] for p in s.processes},
                   opens={ext[(p, o)]: ext[(p, i)] for p, pr in built.items()
                          for o, i in pr.opens.items()},
                   probes=[f"{p}_{n}" for p in s.processes for n in built[p].probes])
