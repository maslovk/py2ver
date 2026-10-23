# core.py
"""
py2ver: Python → AST → IR → Verilog.

Py2ver(func, attr) translates the function (front_end: parsed, helpers inlined,
fixed point and complex values lowered; then folded, shared or sequenced, see
schedule.py), its state as registers, and writes hdl/main.v. The rest of the
class is in its mixins:

  - memories.py   tables, register files, RAMs and the SDRAM as Verilog
  - reference.py  the Python side: the function run on the inputs, raw values
  - simulate.py   testbenches (Icarus / Verilator / cocotb), the stream wrapper
  - board.py      the DE0-Nano: its project, UART, virtual board, programming
  - syn.py        standalone synthesis in Quartus

A chain split into stages (stages.py) is built here too.
"""

from __future__ import annotations

import ast
import builtins
import copy
import re
import inspect
import logging
import os
import pickle
import pprint
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Tuple

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from . import tb_runner
from .renderer import Renderer
from .pipeline import pipeline_ir
from .semantics import exact_ir
from .inline import inline_helpers
from .names import rename_keywords
from .auto import auto_fixed
from .hwtypes import _namespace, normalize_attr, read_annotations
from .cplx import split_complex
from .fold import fold_loop
from .schedule import Schedule
from .share import ROM_MIN, share
from .comprehend import expand_comprehensions
from .control import structure, while_bound
from .choose import choose, choose_lanes
from .containers import scalarize, shapes_of
from .fixed import DEFAULT_CONST_BITS, build_model, copy_tree, is_table, lower_fixed, uses_fixed
from .visitor import FunctionVisitor
from .ast_ir import Assignment, ModuleIR, Port
from .defaults import (DEFAULT_BAUD_RATE, DEFAULT_CLK_FREQ, DEFAULT_SYN_DEVICE,  # noqa: F401
                       DEFAULT_SYN_FAMILY, DEFAULT_SYN_PERIOD_NS, DEFAULT_TB_PERIOD,
                       DEFAULT_TEMPLATE_DIR, RESULTS_PATH)
from .memories import MemoriesMixin
from .reference import ReferenceMixin
from .simulate import SimulationMixin, _drop_bytecode
from .board import BoardMixin
from .syn import SynMixin
from .process import ProcessMixin
from .pins import PinsMixin

log = logging.getLogger(__name__)


def front_end(func: Callable, attr: Dict[str, Any] | None,
              const_bits: int = DEFAULT_CONST_BITS) -> SimpleNamespace:
    """
    The function as the translator takes it: parsed, annotations read,
    helpers inlined, keywords renamed, complex values split and fixed point
    lowered to integer code, with its bit-exact model (fixed point only).
    """
    try:
        source_foo = inspect.getsource(func)
    except Exception as e:
        raise RuntimeError(f"Unable to load source for {func}: {e}") from e
    fe = SimpleNamespace()
    # Types from annotations (U[8], State(...), ...); attr entries merge over them
    tree, ann = _cached(("parse", func, source_foo),
                        lambda: read_annotations(ast.parse(source_foo), func))
    if attr is None and not ann:
        raise ValueError("Give attr (widths and formats), annotations, or samples=<example "
                         "inputs> to choose them automatically")
    attr = normalize_attr(attr)
    bound = while_bound(attr)                  # attr['while']: iterations of a while loop
    attr = {k: dict(ann.get(k, {}), **attr.get(k, {})) for k in dict.fromkeys([*ann, *attr])
            if k != "while"}
    # The function's own parameters and attr, for running it as written (PY_float)
    fe.py_attr = attr
    fe.py_params = list(inspect.signature(func).parameters)
    # The source, for comments in the Verilog: its text, first line, file
    fe.source = (source_foo, getattr(getattr(func, "__code__", None), "co_firstlineno", 1),
                 inspect.getsourcefile(func) or "")
    def prepare():
        t = inline_helpers(tree, getattr(func, "__globals__", {}))
        # List comprehensions as list literals (their lengths are known now)
        t = expand_comprehensions(t, attr, _namespace(func))
        # while / break / continue / early return as flags and guarded statements
        t = structure(t, {"while": bound})
        # Python names that are Verilog keywords (buf, time, ...) get a '_' suffix
        return rename_keywords(t)
    # Inline calls to your own helper functions and module-level int constants,
    # and the rest that doesn't depend on formats (kept for the next formats tried)
    lengths = tuple(sorted((k, m["length"]) for k, m in attr.items()
                           if isinstance(m, dict) and "length" in m))
    tree, fe.hw_name = _cached(("prepare", func, source_foo, lengths, bound), prepare)
    fe.py_name = {v: k for k, v in fe.hw_name.items()}
    attr = {
        fe.hw_name.get(k, k): (
            dict(v, next=fe.hw_name.get(v["next"], v["next"]))
            if isinstance(v, dict) and "next" in v else v
        )
        for k, v in attr.items()
    }
    ns = _namespace(func)
    # Dicts with constant keys and lists of lists (see containers.py)
    fe.shapes = shapes_of(attr)
    tree, attr = scalarize(tree, attr, ns)
    # Inputs as the caller gives them (a complex one is split in two below)
    fe.call_inputs = [fe.hw_name.get(p, p) for p in fe.py_params
                      if not {"next", "memory"} & set(attr.get(fe.hw_name.get(p, p), {}))]
    # Complex values become real / imaginary pairs (see cplx.py)
    fixed = uses_fixed(attr)
    fe.cplx_params, fe.complex_names = set(), set()
    if fixed or any(isinstance(m, dict) and m.get("complex") for m in attr.values()):
        tree, attr, fe.cplx_params, fe.complex_names = split_complex(tree, attr, ns)
    # Fixed point ('frac' in attr) becomes integer code on raw values, and
    # the same code run in Python is the bit-exact reference (see fixed.py);
    # function tables ('math.sin': {'range': ...}) become ROMs in the code
    attr = {k: Py2ver._raw_init(v) for k, v in attr.items()}
    tree, fx_attr, fe.fx_types = lower_fixed(tree, attr, ns, const_bits, fe.py_name)
    attr = {k: v for k, v in attr.items() if not is_table(v)}
    attr.update(fx_attr)
    # The reference runs the rewritten code for fixed point, and for complex
    # parameters (which the ports take as real / imaginary parts)
    fe.model = build_model(tree, attr, ns) if fixed or fe.cplx_params else None
    fe.tree, fe.attr = tree, attr
    return fe


_CACHE: Dict[Any, Any] = {}
_CACHE_SIZE = 64


def _cached(key: Any, make: Callable[[], Any]) -> Any:
    """make(), remembered by key (a few recent ones); each call gets its own copy,
    sharing only the constant tables (see fixed.copy_tree)."""
    if key not in _CACHE:
        if len(_CACHE) >= _CACHE_SIZE:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = make()
    value = _CACHE[key]
    if isinstance(value, tuple):
        return tuple(copy_tree(v) if isinstance(v, ast.AST) else copy.deepcopy(v) for v in value)
    return copy_tree(value)


# Options that can't be given together, checked in this order: (option, the
# options it rules out, the error). (Those that depend on the design - block RAM,
# 'reg' signals - are checked once the translation finds them.)
_EXCLUSIVE = (
    ("rate", ("multipliers", "fold"), ValueError,
     "rate= chooses fold / multipliers itself; give rate= or those"),
    ("sequential", ("pipeline", "fold"), NotImplementedError,
     "sequential=True can't be combined with fold= or pipeline="),
    ("multipliers", ("pipeline", "fold"), NotImplementedError,
     "multipliers= can't be combined with fold= or pipeline="),
    ("fold", ("pipeline",), NotImplementedError, "fold can't be combined with pipeline yet"),
)


def _check_options(given: Dict[str, bool]) -> None:
    """given: option -> whether it was given."""
    for option, others, error, message in _EXCLUSIVE:
        if given[option] and any(given[o] for o in others):
            raise error(message)


def _with_pins(attr: Dict[str, Any] | None, pins: Dict[str, str]) -> Dict[str, Any]:
    """attr with each port's board pin from `pins` added, as {'pin': ...}."""
    out = {k: dict(v) if isinstance(v, dict) else v for k, v in (attr or {}).items()}
    for port, pin in pins.items():
        if not isinstance(pin, str):
            raise ValueError(f"pins: the port '{port}' needs a pin's name, such as 'LED[0]' "
                             f"or 'GPIO_1[3]', not {pin!r}")
        entry = out.setdefault(port, {})
        if not isinstance(entry, dict):
            raise ValueError(f"pins: the port '{port}' is {entry!r} in attr; give it its "
                             f"width and the rest there as a dict")
        if entry.get("pin", pin) != pin:
            raise ValueError(f"pins: the port '{port}' is on {pin} here, but on "
                             f"{entry['pin']} in attr")
        entry["pin"] = pin
    return out


class Py2ver(MemoriesMixin, ReferenceMixin, SimulationMixin, BoardMixin, SynMixin, ProcessMixin,
             PinsMixin):
    """Main pipeline for Python → Verilog + optional testbench + Quartus + UART HW I/O."""

    t_name: str = ""

    def __init__(self, func: Callable, attr: Dict[str, Any] | None = None,
                 template_dir: Path = DEFAULT_TEMPLATE_DIR, pipeline: int = 0,
                 const_bits: int = DEFAULT_CONST_BITS, samples: List[Any] | None = None,
                 width: int | None = None, max_error: Any = None,
                 round: bool = False, saturate: bool = False, margin: int = 1,
                 fold: Any = None, unroll: int = 1,
                 stages: bool = False, stage_options: Dict[str, Any] | None = None,
                 multipliers: int | None = None, rate: float | None = None,
                 clock: float = DEFAULT_CLK_FREQ, sequential: bool = False,
                 sdram: Any = None, write: bool = True, probes: bool = False,
                 pins: Dict[str, str] | None = None) -> None:
        """
        func: the function, as written (a method, numpy code, a torch module
        for its forward(), or a function looping over batches: one step of
        its loop, see numpy_loops.py). attr: widths, formats and state (see
        the README). With `samples` (example inputs), formats missing from
        attr (or all of them, without attr) are chosen by auto_fixed() from
        the ranges seen: `width` bits per value (default 16), or the fewest
        bits that keep the outputs within `max_error` of the float function,
        with `round` / `saturate` added to every chosen format, and `margin`
        integer bits above the ranges seen. The chosen attr is printed.
        fold: a loop to run over several clocks instead of unrolling it (its
        variable's name, or True for the longest top-level range loop), with
        `unroll` iterations per clock (see fold.py).
        stages: build a chain as separate stages connected by streams, in
        hdl/<name>_stages.v (see stages.py); stage_options: {stage function:
        {'fold', 'unroll', 'pipeline'}} to override the automatic folding.
        multipliers: run the design over several clocks on this many
        shared multipliers, with each function table in block RAM (see
        share.py); for a sequenced design, run its multiply-accumulate loops
        that many iterations at a time.
        sequential: build the loops as they are written, as a state machine
        running a step at a time with large lists in block RAM (see
        sequencer.py), for designs too big to write out.
        sdram: a sequenced design's largest lists and tables in the DE0-Nano's
        SDRAM (True: until the rest fits its block RAM; or that many bits of
        block RAM): the module gets the SDRAM's pins and its controller
        (templates/hw/sdram), and stops while the SDRAM is busy.
        rate: samples per second the design must keep up with at `clock` Hz
        (default 50 MHz): fold / unroll / multipliers (for a sequenced design,
        its lanes) are chosen to fit, as small as possible, and the choice is
        printed (see choose.py).
        probes: a process's state (its state's number __pc, its wait __wait, its
        locals' next values) as ports too, for SignalTap on the board
        (PINS(signaltap=True) records them).
        pins: which board pin each port goes to, for PINS():
        {'led': 'LED[0]', 'tx': 'GPIO_0_TX', 'data': 'LED'} (see pins.py for the
        names). The same as giving each port {'pin': ...} in attr.
        write: write the Verilog to hdl/ in the current directory now; with
        write=False nothing is written until a simulation, a project or
        write_hdl() needs it, and the Verilog is in self.verilog (every file
        for hdl/ in self.hdl_files).
        """
        log.debug("Initializing Py2ver (func=%s)", getattr(func, "__name__", func))
        # The files for hdl/ (name -> text), written by write_hdl()
        self.outdir = Path(os.getcwd())
        self.hdl_dir = self.outdir / "hdl"
        self.hdl_files: Dict[str, str] = {}
        # numpy / methods: translated into a scalar function first (see numpy_fe.py)
        self._adapt: Callable[[Any], tuple] | None = None
        self._numpy = None
        from . import numpy_fe
        from .numpy_loops import _callable
        func = _callable(func)                       # (a torch module: its forward)
        if attr is not None:                         # (8: a width; 'LED[7:0]': a pin)
            attr = normalize_attr(attr)
        if pins:
            attr = _with_pins(attr, pins)
        # (the board's pins of the ports, attr 'pin': PINS(), see pins.py)
        self._pins = {k: v["pin"] for k, v in (attr or {}).items()
                      if isinstance(v, dict) and "pin" in v}
        # A process (a generator, a yield a clock): its clock's function and its
        # state (see process.py)
        from .process import compile_process, is_process
        if is_process(func):
            self._process_attr = normalize_attr(attr)
            self.process = compile_process(func, self._process_attr)
            func, attr = self.process.step, self.process.attr
            # (an open-drain pin: its line read from the same board pin)
            self._opens = dict(self.process.opens)
            for o, line in self._opens.items():
                if o in self._pins:
                    self._pins[line] = self._pins[o]
        elif self._pins:                             # (an input on the board's pins: theirs)
            import inspect as _inspect
            from .hwtypes import pin_widths
            try:
                pin_widths(attr, list(_inspect.signature(func).parameters))
            except (TypeError, ValueError):
                pass
        # (returned values that are 'internal': a register's next value, no port)
        self._internal = {k for k, v in (attr or {}).items()
                          if isinstance(v, dict) and v.get("internal")}
        # (probes: a process's state as ports, for SignalTap)
        self._probes = set(self.process.probes) if probes and self.process else set()
        if probes and not self.process:
            raise ValueError("probes=True: a process's state as ports (a generator's)")
        self._internal -= self._probes
        if numpy_fe.needs_frontend(func):
            from . import numpy_loops
            given = list(samples) if samples is not None else []
            if sequential or numpy_loops.loops_only(func, given):
                tr = numpy_loops.translate(func, given)
                sequential = True
            else:
                try:
                    tr = numpy_fe.translate(func, given)
                except numpy_fe.NotTranslatable as e:
                    # Too big (or too varied) to write out: its loops, sequenced
                    try:
                        tr = numpy_loops.translate(func, given)
                    except numpy_fe.NotTranslatable:
                        raise e from None
                    sequential = True
            self._numpy, self._adapt = tr, tr.adapt
            user = {k.replace("self.", "self_", 1): v for k, v in normalize_attr(attr).items()}
            attr = {k: dict(tr.attr.get(k, {}), **user.get(k, {}))
                    for k in dict.fromkeys([*tr.attr, *user])}
            samples = [tr.adapt(x) for x in (getattr(tr, "steps", None) or samples)]
            func = tr.func
        # (after numpy code's translation, which may sequence it)
        _check_options({"rate": rate is not None, "fold": fold is not None and fold is not False,
                        "multipliers": bool(multipliers), "pipeline": bool(pipeline),
                        "sequential": bool(sequential)})
        if samples is not None:
            attr = auto_fixed(func, samples, attr, width=width or 16, max_error=max_error,
                              round=round, saturate=saturate, margin=margin,
                              const_bits=const_bits)
            def shown(v):                    # (a memory's long 'init': its length)
                if isinstance(v, dict) and isinstance(v.get("init"), list) and len(v["init"]) > 8:
                    return {k: (_Abbrev(f"<{len(x)} values>") if k == "init" else x)
                            for k, x in v.items()}
                return v
            lines = "".join(f"    {k!r}: {pprint.pformat(shown(v), sort_dicts=False, width=200)},\n"
                            for k, v in attr.items())
            print(f"py2ver: formats chosen for {getattr(func, '__name__', 'the function')}"
                  f" (pass this as attr to keep them):\nattr = {{\n{lines}}}")

        fe = front_end(func, attr, const_bits)
        tree, attr = fe.tree, fe.attr
        self._py_attr, self._py_params = fe.py_attr, fe.py_params
        self._hw_name, self._py_name = fe.hw_name, fe.py_name
        # (the board's pins by the ports' names in the Verilog: byte -> byte_)
        self._pins = {self._hw_name.get(k, k): v for k, v in self._pins.items()}
        self._opens = {self._hw_name.get(k, k): self._hw_name.get(v, v)
                       for k, v in getattr(self, "_opens", {}).items()}
        self._call_inputs, self._cplx_params = fe.call_inputs, fe.cplx_params
        self._fx_types, self._model = fe.fx_types, fe.model
        self._shapes = fe.shapes             # lists of lists: (rows, ...) per argument
        self._source = fe.source
        self._complex_names = fe.complex_names
        # Several clocks a sample (a folded loop, shared multipliers or a state
        # machine): how the rewritten function runs, clock by clock (see schedule.py)
        self.schedule: Schedule | None = None
        # Memories a sequenced design keeps between samples: name -> {length, width,
        # signed, init (raw)}; and their contents after the last PY_stream()
        self._seq_mems: Dict[str, Dict[str, Any]] = {}
        self._last_memories: Dict[str, List[int]] = {}
        # rate=: the smallest structure that keeps up (fold, multipliers= or unrolled)
        self.choice = None
        if rate is not None:
            budget = int(clock // rate)
            if budget < 1:
                raise ValueError(f"{rate:g} samples/s is more than one sample per clock at "
                                 f"{clock / 1e6:g} MHz")
            self.choice = (choose_lanes(tree, attr, _namespace(func), budget) if sequential
                           else choose(tree, attr, budget))
            if pipeline and self.choice.kind != "unrolled":
                raise ValueError(f"pipeline= keeps one sample per clock; {rate:g} samples/s "
                                 f"allows {budget} clocks per sample (drop pipeline=)")
            opts = self.choice.options()
            fold, unroll = opts.get("fold", fold), opts.get("unroll", unroll)
            multipliers = opts.get("multipliers")
            print(f"py2ver: {rate:g} samples/s at {clock / 1e6:g} MHz leaves {budget} "
                  f"clock{'s' if budget != 1 else ''} per sample: {self.choice}")
        if sequential:
            from .sequencer import sequence
            # (multipliers=: lanes for the multiply-accumulate loops)
            scheduled = sequence(tree, attr, _namespace(func), lanes=int(multipliers or 1),
                                 sdram=sdram)
        elif multipliers:
            scheduled = share(tree, attr, multipliers)
        elif fold is not None and fold is not False:
            unfolded = FunctionVisitor(attr)
            unfolded.visit(copy.deepcopy(tree))
            loop = fe.hw_name.get(fold, fold) if isinstance(fold, str) else fold
            try:
                scheduled = fold_loop(tree, loop, unroll, unfolded.attr, dict(unfolded._lists))
            except ValueError as e:
                if loop is not True or "No top-level" not in str(e):
                    raise
                log.info("fold=True: no loop to fold")   # (fold=True: if there is one)
                scheduled = None
        else:
            scheduled = None
        # (one clock a sample after all, e.g. every iteration in one clock: as it was)
        if scheduled is not None and scheduled[1].phases > 1:
            tree, self.schedule = scheduled
            attr = dict(attr, **self.schedule.attr)
            self._seq_mems = self.schedule.memories
            log.info("%s", self.schedule)
        f_visitor = FunctionVisitor(attr)
        # Large constant tables are ROMs: in block RAM in a pipelined design without
        # state, else arrays read asynchronously (both readable by any simulator)
        f_visitor.rom_min = ROM_MIN
        stateless = not any(isinstance(m, dict) and "next" in m for m in attr.values())
        ir = f_visitor.visit(tree)

        # Rewrite expressions so the Verilog computes what Python computes
        ir, exact_attr = exact_ir(ir, f_visitor.attr)
        f_visitor.attr.clear()
        f_visitor.attr.update(exact_attr)

        # Explicit state: an input with {'next': '<output>'} becomes a register
        # loaded from that output every clock and reset to 'init' (default 0).
        self.func = func
        # Parameters as ports: (param, [ports] for a list argument, or None)
        fold_params = self.schedule.params if self.schedule else set()
        self.arg_groups = [g for g in f_visitor.arg_groups if g[0] not in fold_params]
        self.state_map, self.state_init, self.state_params = self._parse_state(
            attr, dict(self.arg_groups)
        )
        # State updated in place and returned (next = itself) leaves on <name>_out
        self.state_map = {s: f_visitor.output_renames.get(n, n) for s, n in self.state_map.items()}
        self._internal_ports = {f_visitor.output_renames.get(n, n) for n in self._internal}
        self._probe_ports = {f_visitor.output_renames.get(n, n) for n in self._probes}
        # Block-RAM memories: name -> {"length", "elem", ...}; final port names
        self.rams = {m: dict(r) for m, r in f_visitor._rams.items()}
        self._ram_ports = dict(f_visitor.ram_ports)
        self._ram_init = {m: attr[m].get("init", 0) for m in self.rams}
        # Automatic pipelining: `pipeline` register stages, balanced so every
        # output appears exactly `pipeline` clocks after its inputs. With state,
        # the state loop stays in one stage (see pipeline.py).
        self.pipeline_stages = int(pipeline)
        self._pipe = None
        if self.pipeline_stages:
            if self.rams:
                raise NotImplementedError("pipeline cannot be combined with block-RAM memories yet.")
            if ir.has_clk:
                raise NotImplementedError(
                    "Automatic pipelining needs a design without 'reg' signals."
                )
            self._pipe = pipeline_ir(ir, f_visitor.attr, self.pipeline_stages,
                                     state=self.state_map or None,
                                     roms={k: r for k, r in f_visitor.rom_reads.items() if "entries" in r}
                                     if stateless else None)
            ir = self._pipe.ir
            f_visitor.attr.clear()
            f_visitor.attr.update(self._pipe.attr)
            log.info(
                "Pipelined into %d stages: +%d register bits, %d table(s) in block RAM, "
                "est. stage delays %s%s",
                self.pipeline_stages + 1, self._pipe.register_bits, self._pipe.roms,
                [builtins.round(d, 2) for d in self._pipe.stage_delay],   # ('round' is a parameter)
                f", state loop in stage {self._pipe.loop_stage}" if self.state_map else "",
            )
        if self.schedule and (self.rams or ir.has_clk):
            raise NotImplementedError("fold can't be combined with block RAM or 'reg' signals yet")
        pre_inputs = list(ir.inputs)
        # (every returned value, in order: an 'internal' one too, a register's next
        # value without a port)
        self._returned_names = [p.name for p in ir.outputs if p.kind != "vout"]
        self._ext_inputs: Dict[str, int] = {}     # (sdram=: an input's port -> its SDRAM address)
        if self.state_map or self.rams or self.schedule:
            ir = self._lower_state(ir, f_visitor.attr)
        ir = self._rom_arrays(ir, f_visitor.rom_reads, f_visitor.attr)
        self.ir = ir

        self.has_clk = bool(ir.has_clk)
        self._has_ce = any(p.kind == "ce" for p in ir.inputs)

        self.t_name = ir.name
        control = ("clk", "rst", "ce", "vin")
        # (an input in the SDRAM is no port, but still an input of each sample)
        module_in = {p.name for p in ir.inputs}
        args_in = [p for p in (pre_inputs if self._ext_inputs else ir.inputs)
                   if p.kind not in control and (p.name in module_in or p.name in self._ext_inputs)]
        self.input_args_list = [p.name for p in args_in]
        self.output_args_list = [p.name for p in ir.outputs if p.kind != "vout"]
        self.input_args_width_list = [p.width for p in args_in]
        self.output_args_width_list = [p.width for p in ir.outputs if p.kind != "vout"]
        self._ext_outputs = self._sdram_outputs(f_visitor.attr)
        self._has_vin = any(p.kind == "vin" for p in ir.inputs)

        # Fixed-point parameters (meta) and outputs (fraction bits): the API
        # takes and gives numbers, the ports and the model carry raw values
        self._fx_params = {p: attr[p] for p, _ in self.arg_groups
                           if attr.get(p, {}).get("frac") is not None}
        self._fx_out = {n: f for n in self.output_args_list
                        if (f := self._fx_frac(n)) is not None}
        self._serial = self._serial_ports(attr)

        # Raw IR bit widths (core widths)
        self.input_args_bits = sum(self.input_args_width_list)
        self.output_bits = sum(self.output_args_width_list)

        # IMPORTANT: use visitor's attr (includes acc_0, acc_1, ...)
        self.attr = f_visitor.attr
        self._ssa_final = dict(f_visitor._ssa)     # variable -> its last version

        self.out_meta = {n: dict(self.attr.get(n, {})) for n in self.output_args_list}

        self.tb_latency_cycles, self.out_latency = self._compute_tb_latency()
        log.info("TB latency: %d cycles; per-output: %s", self.tb_latency_cycles, self.out_latency)

        # ---------------------- Segment widths (UART) ----------------------
        # Match DE0_Nano.v:
        #   localparam FOO_SEG_WIDTH = ((FOO_WIDTH + 7)/8)*8;
        #   RX_WIDTH = sum of *_SEG_WIDTH for inputs
        #   TX_WIDTH = sum of *_SEG_WIDTH for outputs
        self.input_seg_widths = [((w + 7) // 8) * 8 for w in self.input_args_width_list]
        self.output_seg_widths = [((w + 7) // 8) * 8 for w in self.output_args_width_list]

        self.rx_width = sum(self.input_seg_widths)
        self.tx_width = sum(self.output_seg_widths)

        log.info(
            "IR built: module=%s, inputs=%s (%d bits core, %d bits UART), "
            "outputs=%s (%d bits core, %d bits UART)",
            self.t_name,
            self.input_args_list, self.input_args_bits, self.rx_width,
            self.output_args_list, self.output_bits, self.tx_width
        )

        # ------------------------ Generate core HDL ------------------------
        renderer = Renderer(template_dir)
        sdram_pins = self._sdram
        # pass full attr (with SSA temps) into renderer; with sdram=, the SDRAM's pins
        verilog_text = renderer.render_module(ir, self.attr,
                                              self.SDRAM_PORTS if sdram_pins else (),
                                              source=(func.__name__, *self._source))
        if sdram_pins:
            # (the SDRAM's controller in the file)
            verilog_text += "\n" + (template_dir / "hw" / "sdram" / "py2ver_sdram.v").read_text(
                encoding="utf-8")
        self.verilog = verilog_text
        self.hdl_files["main.v"] = verilog_text

        # ---------------------- Jinja env & syn_data -----------------------
        self.env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            undefined=StrictUndefined,
            autoescape=False,
        )
        self._tpl_cache: Dict[str, Any] = {}
        self.template_dir = template_dir

        self.syn_data = {
            "top_name": self.t_name,
            "inputs": [
                {"name": n, "width": w}
                for n, w in zip(self.input_args_list, self.input_args_width_list)
            ],
            "inputs_size": self.input_args_bits,
            "outputs": [
                {"name": n, "width": w}
                for n, w in zip(self.output_args_list, self.output_args_width_list)
            ],
            "latency": self.tb_latency_cycles,
            "pre_step": getattr(self, "ram_phases", 1) - 1,
            "has_clk": self.has_clk,
            "has_state": bool(self.state_map or self.rams or self.schedule),
            "has_ce": self._has_ce,
            "has_vin": self._has_vin,
            "outputs_size": self.output_bits,
            # UART / loopback params for DE0_Nano template
            "CLK_HZ": DEFAULT_CLK_FREQ,
            "BAUD": DEFAULT_BAUD_RATE,
            "include_loopback": True,                 # render wrapper + wires
            "loopback_enable_expr": "SW[1] & SW[2]",  # condition for LB mux
            "force_core_loopback_off": True,          # transceiver in normal mode (LOOPBACK=0)
        }

        # HW / UART state
        self._hw_ok: bool = False
        self.syn_attr: Dict[str, Any] = {}
        self._uart = None  # type: ignore

        # A chain as separate stages (each its own design), connected by streams
        self._stages: Dict[str, Any] | None = None
        if stages:
            self._build_stages(stage_options or {}, const_bits, template_dir)
        if self._pins:                               # (the board's pins checked now, not at PINS())
            self._pin_map()
        if write:
            self.write_hdl()

    # --------------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------------

    def write_hdl(self) -> Path:
        """This design's Verilog (hdl_files) in hdl/, written again before each
        simulation or project (another design may have written there since);
        the directory."""
        self.hdl_dir.mkdir(exist_ok=True)
        for name, text in self.hdl_files.items():
            path = self.hdl_dir / name
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                path.write_text(text, encoding="utf-8")
        log.info("Generated HDL written to %s", self.hdl_dir / "main.v")
        return self.hdl_dir

    def _copy_hdl(self, project_dir: Path, core: str) -> None:
        """main.v into a project as core, with its tables' files ($readmemh's)."""
        hdl = self.write_hdl()
        shutil.copy(hdl / "main.v", project_dir / core)
        for name in self.hdl_files:
            if name.endswith(".hex"):
                shutil.copy(hdl / name, project_dir / name)


    def get_template(self, filename: str):
        if filename not in self._tpl_cache:
            self._tpl_cache[filename] = self.env.get_template(filename)
        return self._tpl_cache[filename]

    @staticmethod
    def _parse_state(attr: Dict[str, Any], groups: Dict[str, Any]):
        """
        Port-level state map (state port -> next-state output) and reset values.
        A list argument with 'next' maps element by element (xs_k -> nxt_k);
        its 'init' may be one value or a list.
        """
        state_map: Dict[str, str] = {}
        state_init: Dict[str, int] = {}
        params: List[str] = []
        for name, meta in attr.items():
            if not (isinstance(meta, dict) and "next" in meta):
                continue
            params.append(name)
            nxt, init = str(meta["next"]), meta.get("init", 0)
            ports = groups.get(name)
            if ports is None:
                state_map[name] = nxt
                state_init[name] = int(init)
                continue
            inits = list(init) if isinstance(init, (list, tuple)) else [init] * len(ports)
            if len(inits) != len(ports):
                raise ValueError(f"'init' of '{name}' needs {len(ports)} values")
            for k, port in enumerate(ports):
                state_map[port] = f"{nxt}_{k}"
                state_init[port] = int(inits[k])
        return state_map, state_init, params

    def _lower_state(self, ir: ModuleIR, attr: Dict[str, Dict[str, Any]]) -> ModuleIR:
        """
        Turn state inputs into registers:
          - state inputs are removed from the ports
          - clk, a synchronous active-high rst and a clock enable ce are added
          - each state s with next n gets:  s <= rst ? init : (ce ? n : s)
        Outputs (including next-state values) stay as ports.
        """
        in_names = [p.name for p in ir.inputs]
        out_names = [p.name for p in ir.outputs]

        if ir.has_clk and self._pipe is None:
            raise NotImplementedError(
                "State registers ('next') cannot be combined with 'reg' signals yet."
            )
        user_ports = ({q.name for q in ir.inputs if q.kind == "input"}
                      | {q.name for q in ir.outputs if q.kind != "vout"})
        for reserved in ("clk", "rst", "ce", "vin", "vout"):
            if reserved in user_ports:
                raise ValueError(f"'{reserved}' is a reserved port name in designs with state.")
        for s, n in self.state_map.items():
            if s not in in_names:
                raise ValueError(f"State '{s}' must be a function argument; arguments: {in_names}")
            if n not in out_names:
                raise ValueError(
                    f"Next state '{n}' of '{s}' must be a returned value; returned: {out_names}"
                )

        assigns = list(ir.assigns)
        # With a block RAM, a sample takes two clocks (read, then compute and
        # write); state commits in the second one.
        self.ram_phases, self._ram_schedule = self._schedule_rams(ir)
        fold = self.schedule or Schedule("none", 1)
        if self.schedule:
            self.ram_phases = fold.phases
        commit = f"(ce && {self._phase_is(self.ram_phases - 1)})" if self.ram_phases > 1 else "ce"
        if self._pipe is not None:
            # Only a real sample reaching the loop stage updates the state
            commit = f"(ce && {self._pipe.valid_at_loop})"
        for p in ir.inputs:
            if p.name not in self.state_map:
                continue
            n = self.state_map[p.name]
            init = self.state_init[p.name] & ((1 << p.width) - 1)
            n_value = self._pipe.next_at_loop.get(n, n) if self._pipe else n
            # Literal takes the next value's signedness so n is extended by its
            # own sign, matching how PY_stream() wraps it.
            n_signed = int(attr.get(n, {}).get("signed", 0)) == 1
            lit = f"{p.width}'{'s' if n_signed else ''}d{init}"
            # The hold value takes the same signedness too, or the whole
            # selection would turn unsigned and zero-extend a signed n
            hold = f"$signed({p.name})" if n_signed else f"$unsigned({p.name})"
            assigns.append(
                Assignment(left=p.name, right=f"rst ? {lit} : ({commit} ? {n_value} : {hold})",
                           is_reg=True)
            )

        # A folded loop: its phase is the phase counter, its carried values are
        # registers loaded every clock
        if self.schedule:
            # (a state machine only asks whether it is the sample's first clock: a
            # register says so, not a compare of the counter on its paths)
            assigns.append(Assignment(left=fold.phase, right="__phase_nz" if fold.sequenced
                                      else "__phase"))
            for reg, nxt in fold.regs.items():
                init = fold.reg_init.get(reg, 0)
                if init:                          # (a memory kept in registers: its 'init')
                    m = attr.get(reg, {})
                    w, sgn = int(m.get("width", 1)), int(m.get("signed", 0)) == 1
                    init = f"{w}'{'s' if sgn else ''}d{init & ((1 << w) - 1)}"
                assigns.append(Assignment(left=reg, right=f"rst ? {init} : (ce ? {nxt} : {reg})",
                                          is_reg=True))
        roms, rfs, sdram = fold.roms, fold.rfs, fold.sdram
        fold_ports = {fold.phase, *fold.regs, *roms, *rfs}
        if sdram:
            fold_ports.add("ext_q")
            # Input lists in the SDRAM: no ports (their values are written there)
            names = {p.name for p in ir.inputs}
            for a, info in sdram["lists"].items():
                if info.get("input"):
                    for i in range(int(info["length"])):
                        port = f"{a}_{i}" if f"{a}_{i}" in names else (a if i == 0 and a in names
                                                                       else None)
                        if port is not None:
                            self._ext_inputs[port] = int(info["base"]) + i * info.get("words", 1)
            fold_ports |= set(self._ext_inputs)
        inputs = [p for p in ir.inputs if p.name not in self.state_map and p.name not in fold_ports]
        have = {p.name for p in inputs}
        for name, kind in (("clk", "clk"), ("rst", "rst"), ("ce", "ce")):
            if name not in have:
                inputs.append(Port(name=name, width=1, signed=False, kind=kind))
        ext_items: List[str] = []
        ext_out: set = set()
        if sdram:
            # (the clock enable stops while the SDRAM is busy: the input is ce_in)
            inputs = [Port(name="ce_in", width=1, signed=False, kind="ce") if p.name == "ce" else p
                      for p in inputs]
            ext_items = self._sdram_items(sdram)
            ext_out = {"ext_req", "ext_we", "ext_addr", "ext_wdata"}

        return ModuleIR(
            name=ir.name,
            inputs=inputs,
            outputs=[p for p in ir.outputs if p.name not in
                     set(fold.regs.values()) | {r["index"] for r in roms.values()}
                     | {rf[k] for rf in rfs.values() for k in ("raddr", "we", "waddr", "wdata")}
                     | ext_out | self._internal_ports],
            assigns=assigns,
            has_clk=True,
            items=ext_items + list(ir.items) + self._phase_items() + self._ram_items()
                  + self._rom_items(roms, attr, self.hdl_dir, self.hdl_files)
                  + self._rf_items(rfs, attr),
        )


    def _phase_is(self, clock: int) -> str:
        if self.ram_phases <= 1:
            return "1'b1"
        return f"(__phase == {self._phase_bits}'d{clock})"

    @property
    def _phase_bits(self) -> int:
        return max(1, (self.ram_phases - 1).bit_length())

    def _phase_items(self) -> List[str]:
        """The phase counter of a design that takes several clocks per sample."""
        P, pb = self.ram_phases, self._phase_bits
        if P <= 1:
            return []
        kind = self.schedule.kind if self.schedule else "none"
        what = {"folded": "the folded loop's iterations",
                "shared": "shared multipliers and tables",
                "sequenced": "shared multipliers and tables"}.get(
                    kind, "RAM reads, then compute, then writes")
        return [
            f"// {P} clocks per sample: {what}",
            f"reg [{pb - 1}:0] __phase;",
            "always @(posedge clk) begin",
            f"  if (rst) __phase <= {pb}'d0;",
            f"  else if (ce) __phase <= (__phase == {pb}'d{P - 1}) ? {pb}'d0 : __phase + {pb}'d1;",
            "end",
        ] + ([
            "// (not the sample's first clock: __phase != 0, a register)",
            "reg __phase_nz;",
            "always @(posedge clk) begin",
            "  if (rst) __phase_nz <= 1'b0;",
            f"  else if (ce) __phase_nz <= (__phase != {pb}'d{P - 1});",
            "end",
        ] if self._sequenced else [])

    @property
    def _sequenced(self) -> bool:
        return bool(self.schedule) and self.schedule.sequenced

    @property
    def _sdram(self) -> bool:
        return bool(self.schedule and self.schedule.sdram)

    def _port_inputs(self) -> List[Tuple[str, int]]:
        """(name, width) of the inputs that are ports (not in the SDRAM)."""
        ext = getattr(self, "_ext_inputs", {})
        return [(n, w) for n, w in zip(self.input_args_list, self.input_args_width_list)
                if n not in ext]

    def _sdram_outputs(self, attr: Dict[str, Any]) -> Dict[str, int]:
        """sdram=: a returned list in the SDRAM has no ports; its values take their
        place among the outputs (output_args_list) all the same, in the return's
        order. {port: its SDRAM address}."""
        sd = (self.schedule.sdram if self.schedule else None) or {}
        returns = sd.get("returns", ())
        if not any(name for name, _ in returns):
            return {}
        ext: Dict[str, int] = {}
        ports = list(zip(self.output_args_list, self.output_args_width_list))
        merged, at = [], 0
        for name, count in returns:
            if name is None:
                merged += ports[at:at + count]
                at += count
                continue
            info = sd["lists"][name]
            w, sgn = int(info["meta"]["width"]), int(info["meta"]["signed"])
            for i in range(count):
                port = f"{name}_{i}"
                merged.append((port, w))
                ext[port] = int(info["base"]) + i * info.get("words", 1)
                attr.setdefault(port, {"signed": sgn, "width": w})
        if at != len(ports):
            raise RuntimeError(f"The outputs don't match the return: {at} of {len(ports)} ports")
        self.output_args_list = [n for n, _ in merged]
        self.output_args_width_list = [w for _, w in merged]
        return ext

    # ------------------------------------------------------------------------
    # A chain as separate stages (see stages.py)
    # ------------------------------------------------------------------------

    def _build_stages(self, options: Dict[str, Any], const_bits: int, template_dir: Path) -> None:
        from . import stages as stg
        plan, info = stg.plan(self.func, self._py_attr)
        stg.make_functions(self.func, plan)
        files: List[Path] = []
        for st in plan:
            sattr = stg.stage_attr(st, plan, info, self._py_attr, self._complex_names,
                                   self._inferred_format)
            opts = dict(options.get(st.callee, {}))
            # A block stage has as many clocks per call as samples per block
            fold, unroll = stg.auto_fold(st, sattr, info["length"], opts, const_bits)
            st.design = Py2ver(st.func, sattr, template_dir=template_dir, const_bits=const_bits,
                               fold=fold, unroll=unroll, pipeline=int(opts.get("pipeline", 0)),
                               write=False)
            # (the stage's core as <stage>.v, its wrapper and tables, beside the fused main.v)
            core = self.hdl_dir / f"{st.name}.v"
            wrapper = st.design.stream_wrapper(write=False)
            self.hdl_files.update((n, t) for n, t in st.design.hdl_files.items() if n != "main.v")
            self.hdl_files[core.name] = st.design.verilog
            files += [core, wrapper]
            log.info("Stage %d (%s, %s): %d clock(s) per call", st.index, st.callee, st.kind,
                     getattr(st.design, "ram_phases", 1))
        text, ports = stg.top_module(self.t_name, plan, info)
        top = self.hdl_dir / f"{self.t_name}_stages.v"
        self.hdl_files[top.name] = text
        files.append(top)
        # The chain's results as PY_stream() gives them: (fused port, top port, transfer)
        appended = {lst: var for st in plan for lst, var in st.appends.items()}
        results = []
        for port in self.output_args_list:
            for r in info["returns"]:
                m = re.fullmatch(rf"{re.escape(r)}(_re|_im)?(?:_(\d+))?", port)
                if m:
                    results.append((port, appended.get(r, r) + (m.group(1) or ""),
                                    int(m.group(2) or 0)))
        last = plan[-1]
        per_block = (info["per_block"][last.loop_list] if last.kind == "sample"
                     else last.design._serial["l"])
        self._stages = {"plan": plan, "info": info, "ports": ports, "files": files,
                        "results": results, "per_block": per_block}

    def _inferred_format(self, var: str) -> Dict[str, Any]:
        """The format this design gives a variable without one in attr: its last
        version's width and sign (both parts of a complex value, every element
        of a list), and its fraction bits."""
        hw = self._hw_name.get(var, var)
        metas = []
        for cand in (hw, f"{hw}_re", f"{hw}_im", f"{hw}_0", f"{hw}_re_0", f"{hw}_im_0"):
            final = self._ssa_final.get(cand, cand)
            if final in self.attr:
                metas.append(self.attr[final])
        if not metas:
            return {}
        signed = any(int(m.get("signed", 0)) for m in metas)
        out = {"signed": int(signed),
               "width": max(int(m["width"]) + (1 if signed and not int(m.get("signed", 0)) else 0)
                            for m in metas)}
        t = self._fx_types.get(hw)
        t = t[1] if t is not None and t[0] == "l" else t
        if t is not None and t[0] == "f":
            out["frac"] = t[1]
        return out

    @property
    def stages(self) -> List[Any]:
        """The stages of a chain built with stages=True (each has .design, its Py2ver)."""
        if not self._stages:
            raise ValueError("Build the chain with stages=True")
        return self._stages["plan"]

    def PY_stages(self) -> Callable[[List[Any]], List[Tuple[Any, ...]]]:
        """PY_stream() of the chain, only the results the stages give (not state)."""
        idx = [self.output_args_list.index(port) for port, _, _ in self._stages["results"]]
        ref = self.PY_stream() if (self.state_map or self.rams or self.schedule) else (
            lambda blocks: self.PY_batch()([b if isinstance(b, tuple) else (b,) for b in blocks]))
        return lambda blocks: [tuple(row[i] for i in idx) for row in ref(blocks)]

    def TB_stages(self) -> Callable[..., List[Tuple[Any, ...]]]:
        """
        Simulate <name>_stages: blocks go in one sample per transfer (s_valid
        with probability `valid`), results are taken with probability `ready`;
        returns each block's results like PY_stages(). Clocks used: stage_cycles.
        """
        return self.tb_stages_fun

    def tb_stages_fun(self, blocks: List[Any], valid: float = 1.0, ready: float = 1.0,
                      seed: int = 0) -> List[Tuple[Any, ...]]:
        if not self._stages:
            raise ValueError("Build the chain with stages=True")
        plan, info, ports = self._stages["plan"], self._stages["info"], self._stages["ports"]
        first = plan[0].design
        blocks = [b if isinstance(b, tuple) else (b,) for b in blocks]
        calls = []
        for b in blocks:                   # the first stage's calls for each block
            if plan[0].kind == "sample":
                calls += [(e,) for e in (b[0] if info["length"] > 1 else [b[0]])]
            else:
                calls.append(b)
        masks = [(1 << w) - 1 for w in first.input_args_width_list]
        vecs = [tuple(v & m for v, m in zip(first._input_vector(c), masks)) for c in calls]
        vecs, in_names = first._transfers(vecs)
        assert in_names == [n for n, _ in ports["in"]], (in_names, ports["in"])
        out_names = [n for n, _ in ports["out"]]
        n_results = len(blocks) * self._stages["per_block"]
        results_path = self.outdir / RESULTS_PATH
        if results_path.exists():
            results_path.unlink()
        self.write_hdl()
        _drop_bytecode(self.outdir, "tb_handshake")
        (self.outdir / "tb_handshake.py").write_text(self.get_template("tb_handshake.txt").render({
            "name": f"{self.t_name}_stages",
            "vectors": repr(vecs),
            "valid_p": float(valid), "ready_p": float(ready), "seed": int(seed),
            "max_cycles": 40 * (len(vecs) + n_results) + 200,
            "in_names": in_names, "out_args": out_names, "n_results": n_results,
            "out_meta": {n: {"signed": int(r.startswith("signed"))} for n, r in ports["out"]},
        }), encoding="utf-8")
        tb_runner.run_cocotb(f"{self.t_name}_stages", self._stages["files"], "tb_handshake",
                             self.outdir, build_name="sim_build_stages", timescale=("1ns", "1ps"))
        rows: List[Dict[str, int]] = []
        if results_path.exists():
            with results_path.open("rb") as handle:
                payload = pickle.load(handle)
            rows = payload["outs"]
            self.stage_cycles = payload["cycles"]
        if len(rows) != n_results:
            raise RuntimeError(f"The stages gave {len(rows)} results for {n_results} expected")
        # Raw values -> numbers with the last stage's formats
        last = plan[-1].design
        fracs = {}
        for n, _ in ports["out"]:
            src = last._serial["out"].get(n, [n])[0]
            fracs[n] = last._fx_out.get(src)
        per = self._stages["per_block"]
        out = []
        for b in range(len(blocks)):
            group = rows[b * per:(b + 1) * per]
            row = []
            for _, top, k in self._stages["results"]:
                v = group[k][top]
                row.append(v if fracs.get(top) is None else v / (1 << fracs[top]))
            out.append(tuple(row))
        return out


class _Abbrev(str):
    """Shown as it is by pprint (no quotes)."""
    def __repr__(self) -> str:
        return str(self)


# --------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    log.info("py2ver executed as script")
