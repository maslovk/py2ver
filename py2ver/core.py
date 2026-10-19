# py2ver.py
"""
py2ver: Python → AST → IR → Verilog generator + TB and Quartus project scaffolding.

This version emits the UART split modules and appends QSF assignments:
  - uart/uart_baud_gen.sv
  - uart/uart_rx_8n1.sv
  - uart/uart_tx_8n1.sv
  - uart/uart_fifo4x8.sv
  - uart/uart_prbs7_tx_test.sv
  - uart/uart_fixed_pattern_tx.sv
  - uart/uart_loopback.sv
  - uart/uart_transceiver.sv

Top-level:
  - DE0_Nano.v (template you provided) instantiates:
      * delayed_registers
      * uart_transceiver (LOOPBACK=0, TX_WIDTH/RX_WIDTH from SEG_WIDTH sums)
      * uart_loopback + PRBS + pattern generators
      * your generated core: <top_name>
"""

from __future__ import annotations

import ast
import builtins
import copy
import functools
import re
import inspect
import logging
import os
import pickle
import pprint
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from . import tb_runner
from .renderer import Renderer
from .syn_report import SynReport, read_report
from .pipeline import pipeline_ir
from .semantics import exact_ir
from .inline import inline_helpers
from .names import rename_keywords
from .auto import auto_fixed
from .hwtypes import _namespace, normalize_attr, read_annotations
from .cplx import split_complex
from .fold import fold_loop
from .share import ROM_MIN, share

ROM_FILE_MIN = 4096         # a table longer than this is read from a file ($readmemh)

# The virtual board with the SDRAM: the DE0-Nano's top and a model of the chip on its pins
_BOARD_SIM = """`timescale 1ns/1ps
module py2ver_board(input CLOCK_50, input [1:0] KEY, input [3:0] SW, input GPIO_0_RX,
                    output GPIO_0_TX);
    wire [12:0] a; wire [1:0] ba, dqm; wire cas_n, cke, dclk, cs_n, ras_n, we_n;
    wire [15:0] dq;
    DE0_Nano board(.CLOCK_50(CLOCK_50), .KEY(KEY), .SW(SW), .LED(), .DRAM_ADDR(a), .DRAM_BA(ba),
        .DRAM_CAS_N(cas_n), .DRAM_CKE(cke), .DRAM_CLK(dclk), .DRAM_CS_N(cs_n), .DRAM_DQ(dq),
        .DRAM_DQM(dqm), .DRAM_RAS_N(ras_n), .DRAM_WE_N(we_n), .GPIO_0_RX(GPIO_0_RX),
        .GPIO_0_TX(GPIO_0_TX));
    sdram_model chip(.clk(dclk), .cke(cke), .cs_n(cs_n), .ras_n(ras_n), .cas_n(cas_n),
        .we_n(we_n), .ba(ba), .sa(a), .dqm(dqm), .dq(dq));
endmodule
"""

_SDRAM_OUTS = ("[get_ports {DRAM_ADDR[*] DRAM_BA[*] DRAM_CAS_N DRAM_RAS_N DRAM_WE_N DRAM_CS_N "
               "DRAM_CKE DRAM_DQM[*] DRAM_DQ[*]}]")
# The SDRAM's timing (IS42S16160B-7 at the inverse of the design's clock): data valid
# 5.4 ns after its edge (tAC), held 2.7 ns (tOH); commands set up 1.5 ns before it
# (tIS), held 0.8 ns (tIH)
_SDRAM_SDC = (
    "\n# The SDRAM (IS42S16160B-7), its clock the inverse of the design's\n"
    "create_generated_clock -name dram_clk -source [get_ports clk] -invert [get_ports DRAM_CLK]\n"
    "set_input_delay -clock dram_clk -max 5.4 [get_ports {DRAM_DQ[*]}]\n"
    "set_input_delay -clock dram_clk -min 2.7 [get_ports {DRAM_DQ[*]}]\n"
    f"set_output_delay -clock dram_clk -max 1.5 {_SDRAM_OUTS}\n"
    f"set_output_delay -clock dram_clk -min -0.8 {_SDRAM_OUTS}\n")
from .comprehend import expand_comprehensions
from .control import structure, while_bound
from .choose import choose, choose_lanes
from .containers import flatten, nest, scalarize, shapes_of
from .fixed import (DEFAULT_CONST_BITS, build_model, copy_tree, is_table, lower_fixed, quantize,
                    uses_fixed)
from .visitor import FunctionVisitor
from .ast_ir import Assignment, ModuleIR, Port

try:
    import serial  # pyserial
except Exception:
    serial = None  # handled gracefully later

# --------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------
RESULTS_PATH = Path("results/results.pickle")
DEFAULT_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
DEFAULT_CLK_FREQ = 50_000_000
DEFAULT_BAUD_RATE = 115_200
DEFAULT_TB_PERIOD = 2000

# Standalone synthesis defaults (DE0-Nano device, 50 MHz)
DEFAULT_SYN_FAMILY = "Cyclone IV E"
DEFAULT_SYN_DEVICE = "EP4CE22F17C6"
DEFAULT_SYN_PERIOD_NS = 20.0

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


def _drop_bytecode(directory: Path, module: str) -> None:
    """
    Remove the cached bytecode of a testbench module before rewriting it:
    Python checks a .pyc only by the source's size and mtime (in seconds), so
    a rewrite of the same size within a second would run the old testbench.
    """
    for p in (directory / "__pycache__").glob(f"{module}.*.pyc"):
        try:
            log.debug("Removing stale pycache file: %s", p)
            p.unlink()
        except OSError as e:
            log.warning("Failed to remove %s: %s", p, e)


def expand_args(call_inputs: List[str], cplx_params: set, args: Tuple[Any, ...]) -> Tuple[Any, ...]:
    """Arguments as the caller gives them -> as the ports take them (a complex
    argument as its real and imaginary parts)."""
    if not cplx_params:
        return tuple(args)
    out: List[Any] = []
    for name, v in zip(call_inputs, args):
        if name in cplx_params and isinstance(v, (list, tuple)):     # a complex list
            vs = [complex(x) for x in v]
            out += [[x.real for x in vs], [x.imag for x in vs]]
        elif name in cplx_params:
            v = complex(v)
            out += [v.real, v.imag]
        else:
            out.append(v)
    return tuple(out) + tuple(args[len(call_inputs):])


class Py2ver:
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
                 sdram: Any = None, write: bool = True) -> None:
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
        self._call_inputs, self._cplx_params = fe.call_inputs, fe.cplx_params
        self._fx_types, self._model = fe.fx_types, fe.model
        self._shapes = fe.shapes             # lists of lists: (rows, ...) per argument
        self._source = fe.source
        self._complex_names = fe.complex_names
        # Folding: the loop's body runs once per clock (phase) instead of unrolled
        self._fold: Dict[str, Any] | None = None
        # Memories a sequenced design keeps between samples: name -> {length, width,
        # signed, init (raw)}; and their contents after the last PY_stream()
        self._seq_mems: Dict[str, Dict[str, Any]] = {}
        self._last_memories: Dict[str, List[int]] = {}
        # rate=: the smallest structure that keeps up (fold, multipliers= or unrolled)
        self.choice = None
        if rate is not None:
            if multipliers or (fold is not None and fold is not False):
                raise ValueError("rate= chooses fold / multipliers itself; give rate= or those")
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
            if pipeline or (fold is not None and fold is not False):
                raise NotImplementedError("sequential=True can't be combined with fold= or "
                                          "pipeline=")
            from .sequencer import sequence
            # (multipliers=: lanes for the multiply-accumulate loops)
            seq_tree, info = sequence(tree, attr, _namespace(func), lanes=int(multipliers or 1),
                                      sdram=sdram)
            if info["phases"] > 1:
                tree, self._fold = seq_tree, info
                attr = dict(attr, **info["attr"])
                self._seq_mems = info["memories"]
                log.info("Sequenced: %d blocks, %d clocks per sample, %d register file(s), "
                         "%d ROM(s), %d register bits", info["blocks"], info["phases"],
                         len(info["rfs"]), len(info["roms"]), info["register_bits"])
        elif multipliers:
            if pipeline or (fold is not None and fold is not False):
                raise NotImplementedError("multipliers= can't be combined with fold= or pipeline=")
            shared, info = share(tree, attr, multipliers)
            if info["phases"] > 1:
                tree, self._fold = shared, info
                attr = dict(attr, **info["attr"])
                log.info("Shared: %d clocks per sample, multipliers %s, %d table(s) in block "
                         "RAM, %d register bits", info["phases"], info["mul_widths"],
                         len(info["roms"]), info["register_bits"])
        elif fold is not None and fold is not False:
            if pipeline:
                raise NotImplementedError("fold can't be combined with pipeline yet")
            unfolded = FunctionVisitor(attr)
            unfolded.visit(copy.deepcopy(tree))
            loop = fe.hw_name.get(fold, fold) if isinstance(fold, str) else fold
            try:
                folded, info = fold_loop(tree, loop, unroll, unfolded.attr, dict(unfolded._lists))
            except ValueError as e:
                if loop is not True or "No top-level" not in str(e):
                    raise
                log.info("fold=True: no loop to fold")   # (fold=True: if there is one)
                info = {"phases": 1}
            if info["phases"] > 1:                 # (all iterations in one clock: nothing to fold)
                tree, self._fold = folded, info
                attr = dict(attr, **info["attr"])
                log.info("Folded loop '%s': %d clocks per sample, %d register(s)",
                         info["loop"], info["phases"], len(info["regs"]))
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
        fold_params = self._fold["params"] if self._fold else set()
        self.arg_groups = [g for g in f_visitor.arg_groups if g[0] not in fold_params]
        self.state_map, self.state_init, self.state_params = self._parse_state(
            attr, dict(self.arg_groups)
        )
        # State updated in place and returned (next = itself) leaves on <name>_out
        self.state_map = {s: f_visitor.output_renames.get(n, n) for s, n in self.state_map.items()}
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
        if self._fold and (self.rams or ir.has_clk):
            raise NotImplementedError("fold can't be combined with block RAM or 'reg' signals yet")
        pre_inputs = list(ir.inputs)
        self._ext_inputs: Dict[str, int] = {}     # (sdram=: an input's port -> its SDRAM address)
        if self.state_map or self.rams or self._fold:
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
        sdram_pins = bool(self._fold and self._fold.get("sdram"))
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
            "has_state": bool(self.state_map or self.rams or self._fold),
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

    @property
    def _hw_root(self) -> Path:
        return self.outdir / "hw"

    @property
    def _project_dir(self) -> Path:
        return self._hw_root / self.t_name

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
        fold = self._fold or {"phases": 1, "phase": None, "regs": {}}
        if self._fold:
            self.ram_phases = fold["phases"]
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
        if self._fold:
            # (a state machine only asks whether it is the sample's first clock: a
            # register says so, not a compare of the counter on its paths)
            assigns.append(Assignment(left=fold["phase"], right="__phase_nz" if self._sequenced
                                      else "__phase"))
            for reg, nxt in fold["regs"].items():
                init = fold.get("reg_init", {}).get(reg, 0)
                if init:                          # (a memory kept in registers: its 'init')
                    m = attr.get(reg, {})
                    w, sgn = int(m.get("width", 1)), int(m.get("signed", 0)) == 1
                    init = f"{w}'{'s' if sgn else ''}d{init & ((1 << w) - 1)}"
                assigns.append(Assignment(left=reg, right=f"rst ? {init} : (ce ? {nxt} : {reg})",
                                          is_reg=True))
        roms = fold.get("roms", {})
        rfs = fold.get("rfs", {})
        fold_ports = {fold["phase"], *fold["regs"], *roms, *rfs}
        sdram = fold.get("sdram")
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
                            self._ext_inputs[port] = int(info["base"]) + i
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
                     set(fold["regs"].values()) | {r["index"] for r in roms.values()}
                     | {rf[k] for rf in rfs.values() for k in ("raddr", "we", "waddr", "wdata")}
                     | ext_out],
            assigns=assigns,
            has_clk=True,
            items=ext_items + list(ir.items) + self._phase_items() + self._ram_items()
                  + self._rom_items(roms, attr, self.hdl_dir, self.hdl_files)
                  + self._rf_items(rfs, attr),
        )

    # The SDRAM's pins (the DE0-Nano's names, lower case), as a module with sdram= has them
    SDRAM_PORTS = [("output", 13, "dram_addr"), ("output", 2, "dram_ba"), ("output", 1, "dram_cas_n"),
                   ("output", 1, "dram_cke"), ("output", 1, "dram_clk"), ("output", 1, "dram_cs_n"),
                   ("inout", 16, "dram_dq"), ("output", 2, "dram_dqm"), ("output", 1, "dram_ras_n"),
                   ("output", 1, "dram_we_n"), ("output", 1, "adv"),
                   # (the host's port: reads and writes while the design is paused)
                   ("input", 1, "host_req"), ("input", 1, "host_we"), ("input", 24, "host_addr"),
                   ("input", 16, "host_wdata"), ("output", 1, "host_ready"),
                   ("output", 1, "host_rvalid"), ("output", 16, "host_rdata")]

    @staticmethod
    def _sdram_items(sdram: Dict[str, Any]) -> List[str]:
        """The SDRAM's controller and the return path of its reads: the design's
        reads (ext_req and not ext_we) get their data on ext_q `latency`
        advancing clocks later, in order, from a FIFO; the design's clock enable
        ce is ce_in, held low while an access waits (a row opened, a refresh) and
        until the SDRAM is set up. adv: the design advanced this clock."""
        L = int(sdram["latency"])
        return [
            "// The SDRAM: the lists and tables that don't fit the block RAM (sdram=)",
            "wire __ext_ready, __ext_rvalid, __ext_init;",
            "wire [15:0] __ext_rdata;",
            "wire __ext_stall = !__ext_init || (ext_req && !__ext_ready);",
            "wire ce = ce_in && !__ext_stall;",
            "assign adv = ce;",
            "assign dram_clk = ~clk;",
            "// (the host reads and writes through it while the design is paused: ce_in low)",
            "wire __host = !ce_in;",
            "assign host_ready = __host && __ext_init && __ext_ready;",
            "wire __req = (__host ? host_req : ext_req) && __ext_init;",
            "wire __we = __host ? host_we : ext_we;",
            "py2ver_sdram __sdram(.clk(clk), .rst(rst), .req(__req),",
            "    .we(__we), .addr(__host ? host_addr : ext_addr),",
            "    .wdata(__host ? host_wdata : ext_wdata), .ready(__ext_ready),",
            "    .rvalid(__ext_rvalid), .rdata(__ext_rdata), .init_done(__ext_init),",
            "    .sa(dram_addr), .ba(dram_ba), .cs_n(dram_cs_n), .ras_n(dram_ras_n),",
            "    .cas_n(dram_cas_n), .we_n(dram_we_n), .cke(dram_cke), .dqm(dram_dqm), .dq(dram_dq));",
            f"// A read's data, in order, on ext_q {L} advancing clocks after its address",
            "// (a register: taken from the FIFO the clock before)",
            "// (a read's data for whoever issued it: the design's into the FIFO)",
            "reg [15:0] __ext_fifo [0:7];",
            "reg [2:0] __ext_wp, __ext_rp;",
            f"reg [{L - 1}:0] __ext_due;",
            "reg [15:0] ext_q;",
            "reg __tag [0:7];",
            "reg [2:0] __tag_wp, __tag_rp;",
            "wire __for_host = __tag[__tag_rp];",
            "assign host_rvalid = __ext_rvalid && __for_host;",
            "assign host_rdata = __ext_rdata;",
            "always @(posedge clk) begin",
            "  if (rst) begin",
            "    __ext_wp <= 3'd0; __ext_rp <= 3'd0; __ext_due <= 0;",
            "    __tag_wp <= 3'd0; __tag_rp <= 3'd0;",
            "  end else begin",
            "    if (__req && __ext_ready && !__we) begin",
            "      __tag[__tag_wp] <= __host;",
            "      __tag_wp <= __tag_wp + 3'd1;",
            "    end",
            "    if (__ext_rvalid) __tag_rp <= __tag_rp + 3'd1;",
            "    if (__ext_rvalid && !__for_host) begin",
            "      __ext_fifo[__ext_wp] <= __ext_rdata;",
            "      __ext_wp <= __ext_wp + 3'd1;",
            "    end",
            "    if (ce) begin",
            f"      __ext_due <= {{__ext_due[{L - 2}:0], ext_req && !ext_we}};",
            f"      if (__ext_due[{L - 2}]) begin",
            "        ext_q <= __ext_fifo[__ext_rp];",
            "        __ext_rp <= __ext_rp + 3'd1;",
            "      end",
            "    end",
            "  end",
            "end",
        ]

    def _schedule_rams(self, ir: ModuleIR) -> Tuple[int, Dict[str, Any]]:
        """
        Clocks per sample with block RAMs, and the clock of each read/write.
        A read is issued once the reads its address depends on have their data;
        the outputs are computed when all read data is in; then the writes
        follow one per clock, in program order (so the last write to an
        address wins, as in Python).
        """
        if not self.rams:
            return 1, {}
        deps = {a.left: set(self._extract_idents(a.right)) for a in ir.assigns}
        rdata_of = {rd: (mem, j) for mem, ports in self._ram_ports.items()
                    for j, (_ra, rd) in enumerate(ports["reads"])}

        def reach(name: str, seen: set) -> set:
            for d in deps.get(name, ()):
                if d not in seen:
                    seen.add(d)
                    reach(d, seen)
            return seen

        issue: Dict[str, int] = {}

        def issue_clock(rdata: str) -> int:
            if rdata not in issue:
                mem, j = rdata_of[rdata]
                raddr = self._ram_ports[mem]["reads"][j][0]
                before = [r for r in reach(raddr, {raddr}) if r in rdata_of]
                issue[rdata] = max((issue_clock(r) + 1 for r in before), default=0)
            return issue[rdata]

        for rd in rdata_of:
            issue_clock(rd)
        compute = max((c + 1 for c in issue.values()), default=0)
        writes = max((len(p["writes"]) for p in self._ram_ports.values()), default=0)
        return compute + max(writes, 1), {"issue": issue, "compute": compute}

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
        what = ("shared multipliers and tables" if self._fold and self._fold.get("roms") is not None
                else "the folded loop's iterations" if self._fold
                else "RAM reads, then compute, then writes")
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
        return bool(self._fold) and self._fold.get("loop") == "(sequenced)"

    @staticmethod
    def _rom_arrays(ir: ModuleIR, rom_reads: Dict[str, Dict[str, Any]],
                    attr: Dict[str, Dict[str, Any]]) -> ModuleIR:
        """Table reads still in the assignments (not block RAM) as arrays read by
        their index: compact Verilog, and readable by Verilator (an expression of
        1024 cases is not). The array is padded to a power of two; reads of the
        same table share one array."""
        reads = {a.left: a for a in ir.assigns if a.left in rom_reads}
        if not reads:
            return ir
        items: List[str] = []
        arrays: Dict[Any, Tuple[str, int]] = {}  # table -> (array, its comment's place in items)
        for name, a in reads.items():
            rom = rom_reads[name]
            # (the index as the assignment reads it: renamed by pipelining, or a
            # sign-adjusted copy, but the one signal on the right)
            names = set(re.findall(r"(?<![$\w'])[A-Za-z_]\w*", a.right))
            if len(names) != 1:
                raise RuntimeError(f"Can't find the index of table read {name}: {a.right}")
            idx = names.pop()
            w, ab = int(rom["width"]), int(rom["address_bits"])
            sig = "signed " if rom.get("signed") else ""
            mask = (1 << w) - 1
            arr = f"{name}__table"
            iw = int(attr.get(idx, attr.get(rom["index"], {})).get("width", ab + 1))
            addr = f"{idx}[{ab - 1}:0]" if iw > ab else idx
            key = (tuple(rom.get("signals", rom.get("entries", ()))), "signals" in rom, w, sig, ab)
            if key in arrays:
                arr, at = arrays[key]
                items[at] += f", {name}"
                items.append(f"wire {sig}[{w - 1}:0] {name} = {arr}[{addr}];")
                continue
            arrays[key] = (arr, len(items))
            if "signals" in rom:                  # (a long list argument's ports)
                sigs = list(rom["signals"])
                items += [f"// A list argument read by index: {len(sigs)} x {w} bits, "
                          f"read by {name}",
                          f"wire {sig}[{w - 1}:0] {arr} [0:{(1 << ab) - 1}];"]
                items += [f"assign {arr}[{k}] = {v};" for k, v in enumerate(sigs)]
                items += [f"assign {arr}[{k}] = {w}'d0;" for k in range(len(sigs), 1 << ab)]
                items.append(f"wire {sig}[{w - 1}:0] {name} = {arr}[{addr}];")
                continue
            entries = list(rom["entries"])
            entries += [entries[-1]] * ((1 << ab) - len(entries))
            items += [f"// Table: {len(rom['entries'])} x {w} bits, read by {name}",
                      f"reg {sig}[{w - 1}:0] {arr} [0:{len(entries) - 1}];",
                      "initial begin"]
            items += [f"  {arr}[{k}] = {w}'d{int(v) & mask};" for k, v in enumerate(entries)]
            items += ["end", f"wire {sig}[{w - 1}:0] {name} = {arr}[{addr}];"]
        return ModuleIR(name=ir.name, inputs=ir.inputs, outputs=ir.outputs,
                        assigns=[a for a in ir.assigns if a.left not in reads],
                        has_clk=ir.has_clk, items=list(ir.items) + items)

    @staticmethod
    def _rom_items(roms: Dict[str, Dict[str, Any]], attr: Dict[str, Dict[str, Any]],
                   hdl_dir: Path, files: Dict[str, str]) -> List[str]:
        """Function tables in block RAM: q is the entry the index addressed a clock
        before. Two ports reading the same table share one array (a dual-port ROM:
        the same block RAM). A large table's contents go in files (name -> text),
        read from hdl_dir."""
        items: List[str] = []
        groups: Dict[Any, List[str]] = {}
        for q, rom in roms.items():
            key = (tuple(rom["entries"]), int(rom["width"]), bool(rom["signed"]))
            groups.setdefault(key, []).append(q)
        for qs in groups.values():
            for pair in (qs[i:i + 2] for i in range(0, len(qs), 2)):
                rom = roms[pair[0]]
                entries, w, aw = rom["entries"], int(rom["width"]), int(rom["address_bits"])
                sig = "signed " if rom["signed"] else ""
                mask = (1 << w) - 1
                arr = f"{pair[0]}__rom"
                ports = " and ".join(pair)
                items.append(f"// Table in block RAM: {len(entries)} x {w} bits"
                             + (f", read by {ports}" if len(pair) > 1 else ""))
                items.append(f"reg {sig}[{w - 1}:0] {arr} [0:{len(entries) - 1}];")
                items += [f"reg {sig}[{w - 1}:0] {q};" for q in pair]
                if len(entries) > ROM_FILE_MIN:
                    # (a large table from a file next to main.v: simulators and Quartus
                    # read it, instead of an assignment an entry to compile)
                    path = hdl_dir / f"{arr}.hex"
                    digits = (w + 3) // 4
                    files[path.name] = "".join(f"{int(v) & mask:0{digits}x}\n" for v in entries)
                    items.append(f'initial $readmemh("{path.as_posix()}", {arr});')
                else:
                    items.append("initial begin")
                    items += [f"  {arr}[{k}] = {w}'d{int(v) & mask};" for k, v in enumerate(entries)]
                    items.append("end")
                idxs = []
                for q in pair:
                    idx = roms[q]["index"]
                    if int(attr.get(idx, {}).get("width", aw + 1)) > aw:   # (never out of range)
                        idx = f"{idx}[{aw - 1}:0]"
                    idxs.append(idx)
                if len(pair) == 1:
                    items.append(f"always @(posedge clk) if (ce) {pair[0]} <= {arr}[{idxs[0]}];")
                    continue
                # Two ports: read every clock (with an enable Quartus won't make it one
                # dual-port ROM), at the address of the last enabled clock while ce is
                # low, so q holds as with an enable
                for q, idx in zip(pair, idxs):
                    items += [f"reg [{aw - 1}:0] {q}__a;",
                              f"always @(posedge clk) if (ce) {q}__a <= {idx};"]
                items.append("always @(posedge clk) begin "
                             + " ".join(f"{q} <= {arr}[ce ? {idx} : {q}__a];"
                                        for q, idx in zip(pair, idxs)) + " end")
        return items

    @staticmethod
    def _rf_items(rfs: Dict[str, Dict[str, Any]], attr: Dict[str, Dict[str, Any]]) -> List[str]:
        """A shared multiplier input's register file: a simple dual-port block RAM,
        written by we/waddr/wdata, q the entry raddr addressed a clock before."""
        items: List[str] = []
        for q, rf in rfs.items():
            w, depth, ab = int(rf["width"]), int(rf["depth"]), int(rf["address_bits"])
            sig = "signed " if rf["signed"] else ""
            arr = f"{q}__mem"

            def addr(name: str) -> str:
                return f"{name}[{ab - 1}:0]" if int(attr.get(name, {}).get("width", ab)) > ab \
                    else name
            items += [f"// Register file in block RAM: {depth} x {w} bits",
                      f"(* ramstyle = \"M9K\" *) reg {sig}[{w - 1}:0] {arr} [0:{depth - 1}];",
                      f"reg {sig}[{w - 1}:0] {q};"]
            if any(rf.get("init", ())):          # (a memory: its starting values)
                mask = (1 << w) - 1
                items += ["initial begin"] + [f"  {arr}[{k}] = {w}'d{int(v) & mask};"
                                              for k, v in enumerate(rf["init"])] + ["end"]
            else:
                items += [f"integer __i_{arr};",
                          f"initial for (__i_{arr} = 0; __i_{arr} < {depth}; __i_{arr} = __i_{arr} + 1)"
                          f" {arr}[__i_{arr}] = {w}'d0;"]
            items += ["always @(posedge clk) if (ce) begin",
                      f"  if ({rf['we']}) {arr}[{addr(rf['waddr'])}] <= {rf['wdata']};",
                      f"  {q} <= {arr}[{addr(rf['raddr'])}];",
                      "end"]
        return items

    def _ram_items(self) -> List[str]:
        """
        Verilog for the block-RAM memories. Each read port gets its own copy of
        the memory (all copies take every write), and the writes of a sample
        share one write port, one per clock, so every copy stays a simple
        dual-port block RAM.
        """
        if not self.rams:
            return []
        items: List[str] = []
        compute = self._ram_schedule["compute"]
        for mem, ram in self.rams.items():
            n, elem = ram["length"], ram["elem"]
            w = int(elem["width"])
            aw = max(1, (n - 1).bit_length())
            sig = "signed " if int(elem["signed"]) == 1 else ""
            reads = self._ram_ports[mem]["reads"]
            writes = self._ram_ports[mem]["writes"]
            init = self._ram_init[mem]
            values = list(init) if isinstance(init, (list, tuple)) else [init] * n
            if len(values) != n:
                raise ValueError(f"'init' of RAM '{mem}' needs {n} values")
            mask = (1 << w) - 1
            items.append(f"// RAM {mem}: {n} x {w} bits, {max(len(reads), 1)} copy(ies)")

            if writes:
                def by_clock(k: int) -> str:
                    expr = "0"
                    for i in reversed(range(len(writes))):
                        expr = f"({self._phase_is(compute + i)} ? {writes[i][k]} : {expr})"
                    return expr
                items.append(f"wire [{aw - 1}:0] {mem}__wa = {by_clock(1)};")
                items.append(f"wire [{w - 1}:0] {mem}__wd = {by_clock(2)};")
                enables = " || ".join(f"({self._phase_is(compute + i)} && {we})"
                                      for i, (we, _a, _d) in enumerate(writes))
                items.append(f"wire {mem}__wen = {enables};")

            for c in range(max(len(reads), 1)):
                arr = f"{mem}__m{c}"
                items.append(f"reg {sig}[{w - 1}:0] {arr} [0:{n - 1}];")
                if any(values):
                    items.append("initial begin")
                    items += [f"  {arr}[{k}] = {w}'d{int(v) & mask};" for k, v in enumerate(values)]
                    items.append("end")
                else:
                    items.append(f"integer __i_{arr};")
                    items.append(f"initial for (__i_{arr} = 0; __i_{arr} < {n}; "
                                 f"__i_{arr} = __i_{arr} + 1) {arr}[__i_{arr}] = {w}'d0;")
                items.append("always @(posedge clk) begin")
                if c < len(reads):
                    raddr, rdata = reads[c]
                    items.insert(-1, f"reg {sig}[{w - 1}:0] {rdata};")
                    clock = self._ram_schedule["issue"][rdata]
                    items.append(f"  if (ce && {self._phase_is(clock)}) {rdata} <= {arr}[{raddr}];")
                if writes:
                    items.append(f"  if (ce && {mem}__wen) {arr}[{mem}__wa] <= {mem}__wd;")
                items.append("end")
        return items

    @staticmethod
    def _wrap(n: int, nbits: int, signed: bool) -> int:
        n &= (1 << nbits) - 1
        if signed:
            sign_bit = 1 << (nbits - 1)
            n = (n ^ sign_bit) - sign_bit
        return n

    @staticmethod
    def tosigned(n: int, nbits: int) -> int:
        mask = (1 << nbits) - 1
        n &= mask
        sign_bit = 1 << (nbits - 1)
        return (n ^ sign_bit) - sign_bit

    @staticmethod
    def _extract_idents(expr: str) -> List[str]:
        # verilog-ish identifiers
        return re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", expr)

    def _compute_latencies(self) -> Dict[str, int]:
        """
        Conservative pipeline latency in *cycles* for each signal.

        wire assign: lat(lhs) = max(lat(deps))
        reg  assign: lat(lhs) = 1 + max(lat(deps))

        This correctly yields:
          d <= expr(inputs)  => lat(d)=1
          e <= d             => lat(e)=2
        """
        ir = getattr(self, "ir", None)
        if ir is None:
            return {n: 0 for n in self.output_args_list}

        # Known names: ports + anything ever assigned
        input_names = set(self.input_args_list)
        output_names = set(self.output_args_list)
        lhs_names = {a.left for a in ir.assigns}
        known = input_names | output_names | lhs_names

        # Seed: inputs are 0 cycles
        lat: Dict[str, int] = {n: 0 for n in known}
        for n in input_names:
            lat[n] = 0

        # Precompute deps per assignment
        assigns: List[tuple[str, bool, List[str]]] = []
        for a in ir.assigns:
            deps = []
            for ident in self._extract_idents(a.right):
                if ident.startswith("$"):
                    continue
                if ident == "clk":
                    continue
                if ident in known:
                    deps.append(ident)
            assigns.append((a.left, bool(a.is_reg), deps))

        # Fixed-point relaxation.
        # Works even if assigns are out-of-order.
        # Safe even with self-references from guarded updates (we ignore d==lhs).
        max_iters = len(known) + 10
        for _ in range(max_iters):
            changed = False
            for left, is_reg, deps in assigns:
                dep_lat = 0
                for d in deps:
                    if d == left:
                        continue  # guarded update pattern: lhs = cond ? rhs : lhs
                    dep_lat = max(dep_lat, lat.get(d, 0))
                new_lat = dep_lat + (1 if is_reg else 0)
                if new_lat > lat.get(left, 0):
                    lat[left] = new_lat
                    changed = True
            if not changed:
                break

        return lat

    def _compute_tb_latency(self) -> Tuple[int, Dict[str, int]]:
        """
        Returns:
          (cycles_to_wait, per_output_latency)
        """
        if self.state_map:
            # Stream mode: outputs are combinational in (inputs, state) and are
            # sampled every clock before the state registers update.
            return 0, {n: 0 for n in self.output_args_list}
        if self._fold:
            # Several clocks per sample (pre_step), its registers in loops: the
            # results are there a clock after the last
            return 1, {n: 1 for n in self.output_args_list}
        per_sig = self._compute_latencies()
        per_out = {n: int(per_sig.get(n, 0)) for n in self.output_args_list}
        cycles = max(per_out.values(), default=0)
        return cycles, per_out

    # ------------------------------------------------------------------------
    # Bit packing helpers for UART protocol (matches DE0_Nano + uart_transceiver)
    # ------------------------------------------------------------------------

    def _pack_inputs(self, *in_arg: Any) -> bytes:
        """
        Pack input arguments into a little-endian bit-vector with per-signal
        byte-aligned segments.

        Matches DE0_Nano.v RX side:

          localparam FOO_SEG_WIDTH = ((FOO_WIDTH+7)/8)*8;

          // RX_WIDTH is sum of FOO_SEG_WIDTH
          assign { ..., input1_seg, input0_seg } = rx_data_int;
          assign input0_int = input0_seg[FOO0_WIDTH-1:0];
          assign input1_int = input1_seg[FOO1_WIDTH-1:0];

        So we do:
          - For each input i:
              * segment width SEG_WIDTH[i]
              * write value into lower 'width[i]' bits of that segment
              * upper bits remain 0
          - Concatenate segments in order inputs[0] (LSB) .. inputs[N-1] (MSB).
        """
        acc = 0
        bit_offset = 0

        for idx, (name, width, seg_width) in enumerate(
            zip(self.input_args_list,
                self.input_args_width_list,
                self.input_seg_widths)
        ):
            v = in_arg[idx] if idx < len(in_arg) else 0
            mask = (1 << width) - 1 if width > 0 else 0
            v = int(v) & mask
            # Place v in the LSBs of this segment
            acc |= (v << bit_offset)
            bit_offset += seg_width  # move by SEG_WIDTH, not raw width

        num_bytes = (self.rx_width + 7) // 8  # should already be multiple of 8
        return acc.to_bytes(num_bytes, byteorder="little")

    def _unpack_outputs(self, data: bytes) -> Tuple[int, ...]:
        """
        Unpack output bytes (little-endian) into a tuple of ints in the
        order of self.output_args_list, applying signedness.

        Matches DE0_Nano.v TX side:

          localparam FOO_SEG_WIDTH = ((FOO_WIDTH+7)/8)*8;

          assign tx_data_int = {
              { (FOO_N_SEG_WIDTH-FOO_N_WIDTH){1'b0} }, foo_n_out_int,
              ...
              { (FOO_0_SEG_WIDTH-FOO_0_WIDTH){1'b0} }, foo_0_out_int
          };

        i.e., segments concatenated outputs[0] at LSB, outputs[1] above, etc.
        """
        if not data:
            return tuple(0 for _ in self.output_args_list)

        acc = int.from_bytes(data, byteorder="little")
        outs: List[int] = []
        bit_offset = 0

        for name, width, seg_width in zip(
            self.output_args_list,
            self.output_args_width_list,
            self.output_seg_widths
        ):
            seg_mask = (1 << seg_width) - 1 if seg_width > 0 else 0
            seg = (acc >> bit_offset) & seg_mask
            bit_offset += seg_width

            # Extract the actual value from the LSB bits of the segment
            mask_val = (1 << width) - 1 if width > 0 else 0
            v = seg & mask_val

            if self.attr.get(name, {}).get("signed") == 1:
                v = self.tosigned(v, width)
            outs.append(v)

        return tuple(outs)

    # ------------------------------------------------------------------------
    # Hardware flow
    # ------------------------------------------------------------------------

    def HW(self, syn_attr: Dict[str, Any]) -> Callable[..., Tuple[int, ...]]:
        """
        Synthesis / hardware flow.

        syn_attr must contain:
            'QUARTUS_DIR': path to quartus/bin
        and may contain:
            'UART_PORT':    serial port (default '/dev/ttyUSB0')
            'UART_BAUD':    baud rate (default DEFAULT_BAUD_RATE)
            'UART_TIMEOUT': read timeout in seconds (default 1.0)
            'FALLBACK_TO_SIM': return simulation results instead of raising
                               when the board cannot be used (default False)
            'ALLOW_TIMING_FAILURE': run even if the board design misses
                               50 MHz timing (results may be wrong)

        Returns:
            hw_fun(*args) -> tuple of outputs (same shape as TB()). For designs
            with state, each call is one sample and advances the state.
        """
        self.syn_attr = syn_attr
        self._sdram_loaded = False                   # (the SDRAM's tables: loaded on the first call)
        self._hw_ok = self.createHWproject()
        if not self._hw_ok:
            self._hw_unavailable(getattr(self, "_hw_error", "board project could not be built (see log)"))
        return self._fx_rows(self.hw_fun, single=True)

    def _hw_unavailable(self, reason: str, *in_arg: Any) -> Tuple[int, ...]:
        """Raise, or fall back to simulation if FALLBACK_TO_SIM is set."""
        if self.syn_attr.get("FALLBACK_TO_SIM") and not self.state_map:
            log.warning("HW unavailable (%s); returning simulation result.", reason)
            return self.tb_fun(*in_arg) if in_arg else ()
        raise RuntimeError(f"HW unavailable: {reason}")

    def hw_fun(self, *in_arg: Any) -> Tuple[int, ...]:
        """
        Hardware runner.

        Flow:
          - If Quartus or UART is unavailable, fall back to tb_fun().
          - Otherwise:
              * pack inputs into RX_WIDTH bits (byte-aligned segments)
              * write RX_BYTES to UART
              * read back TX_BYTES
              * unpack to tuple of outputs
        """
        if not getattr(self, "_hw_ok", False):
            return self._hw_unavailable(
                getattr(self, "_hw_error", "board project could not be built"), *in_arg
            )

        if serial is None:
            return self._hw_unavailable("pyserial is not installed", *in_arg)

        port = self.syn_attr.get("UART_PORT", "/dev/ttyUSB0")
        baud = int(self.syn_attr.get("UART_BAUD", DEFAULT_BAUD_RATE))
        timeout = float(self.syn_attr.get("UART_TIMEOUT", 1.0))

        if self._sdram:
            return self._hw_sdram_call(port, baud, timeout, in_arg)

        input_bytes = self._pack_inputs(*self._flatten_args(in_arg))
        output_bytes_expected = (self.tx_width + 7) // 8

        try:
            with serial.Serial(port, baudrate=baud, timeout=timeout) as ser:
                log.info("Opened UART port %s @ %d baud", port, baud)

                try:
                    ser.reset_input_buffer()
                    ser.reset_output_buffer()
                except Exception:
                    pass

                # --- Send input payload ---
                log.debug("Sending %d input bytes: %s",
                          len(input_bytes), input_bytes.hex(" "))
                ser.write(input_bytes)
                ser.flush()

                # --- Read response ---
                resp = b""
                while len(resp) < output_bytes_expected:
                    chunk = ser.read(output_bytes_expected - len(resp))
                    if not chunk:
                        break
                    resp += chunk

                if len(resp) == 0:
                    log.error("UART: No data received.")
                else:
                    log.info("Received %d bytes: %s",
                             len(resp), resp.hex(" "))

                if len(resp) != output_bytes_expected:
                    return self._hw_unavailable(
                        f"UART: expected {output_bytes_expected} bytes, got {len(resp)}",
                        *in_arg,
                    )

                # --- Decode ---
                outs = self._unpack_outputs(resp)
                log.info("Decoded HW outputs: %s", outs)
                return outs

        except (OSError, serial.SerialException) as e:
            return self._hw_unavailable(f"UART communication failed ({e})", *in_arg)

    def _hw_sdram_call(self, port: str, baud: int, timeout: float, in_arg) -> Tuple[int, ...]:
        """A sample on a board with the SDRAM: its tables (once), its input lists,
        then 'R' (see de0_nano_sdram_top.txt)."""
        first = not getattr(self, "_sdram_loaded", False)
        steps = self._sdram_steps(self._input_vector(tuple(in_arg)), first)
        try:
            with serial.Serial(port, baudrate=baud, timeout=max(timeout, 30.0)) as ser:
                replies = []
                for data, n in steps:
                    ser.write(data)
                    ser.flush()
                    replies.append(ser.read(n))
                    if len(replies[-1]) != n:
                        return self._hw_unavailable(
                            f"UART: expected {n} byte(s), got {len(replies[-1])}", *in_arg)
                self._sdram_loaded = True
                return self._sdram_results(replies)
        except (OSError, serial.SerialException) as e:
            return self._hw_unavailable(f"UART communication failed ({e})", *in_arg)

    def hw_run(self) -> int:
        """
        Legacy HW runner placeholder (kept for compatibility).
        Not used by the new HW()/hw_fun path.
        """
        return 1

    def HW_sim(self) -> Callable[[List[Tuple[Any, ...]]], List[Tuple[int, ...]]]:
        """
        Virtual board: simulate the full DE0-Nano project (UART, latency
        handling, core) and talk to it over its UART pins with the same bytes
        hw_fun() sends. Returns a batch callable like TB_batch().
        """
        return self._fx_rows(self.hw_sim_fun)

    def hw_sim_fun(self, vectors: List[Tuple[Any, ...]]) -> List[Tuple[int, ...]]:
        vectors = [tuple(v) for v in vectors]
        if not vectors:
            return []
        if self._sdram:
            return self._hw_sim_sdram(vectors)
        project_dir = self.gen_hw_project()
        tx_bytes = (self.tx_width + 7) // 8
        requests = [list(self._pack_inputs(*self._flatten_args(v))) for v in vectors]

        tb_src = self.get_template("tb_hw.txt").render({
            "baud": DEFAULT_BAUD_RATE,
            "tx_bytes": tx_bytes,
            "requests": repr(requests),
        })
        _drop_bytecode(project_dir, "tb_hw")
        (project_dir / "tb_hw.py").write_text(tb_src, encoding="utf-8")

        sources = (
            [project_dir / "DE0_Nano.v", project_dir / "delayed_registers.sv",
             project_dir / f"{self.t_name}.v"]
            + sorted((project_dir / "uart").glob("*.sv"))
        )
        tb_runner.run_cocotb(
            "DE0_Nano", sources, "tb_hw", project_dir, timescale=("1ns", "1ps")
        )

        results_path = project_dir / "results" / "results.pickle"
        if not results_path.exists():
            raise RuntimeError(f"Virtual board produced no results ({results_path})")
        with results_path.open("rb") as handle:
            replies = pickle.load(handle)

        outs: List[Tuple[int, ...]] = []
        for vec, reply in zip(vectors, replies):
            if len(reply) != tx_bytes or not all(isinstance(b, int) for b in reply):
                raise RuntimeError(
                    f"Virtual board reply for {vec}: expected {tx_bytes} bytes, got {reply}"
                )
            outs.append(self._unpack_outputs(bytes(reply)))
        return outs

    SIM_BAUD = 3_125_000        # the virtual SDRAM board's UART (16 clocks a bit at 50 MHz)

    def _hw_sim_sdram(self, vectors: List[Tuple[Any, ...]]) -> List[Tuple[int, ...]]:
        """The virtual board with the SDRAM: the board's project and a model of the
        chip on its pins, the same UART commands as hw_fun() (at SIM_BAUD)."""
        project_dir = self.gen_hw_project(baud=self.SIM_BAUD)
        samples = [[(list(d), n) for d, n in self._sdram_steps(self._input_vector(v), k == 0)]
                   for k, v in enumerate(vectors)]
        P = int(self.ram_phases)
        tb_src = self.get_template("tb_hw_sdram.txt").render({
            "baud": self.SIM_BAUD, "samples": repr(samples), "wait_bits": (4 * P + 50000) // 128})
        _drop_bytecode(project_dir, "tb_hw_sdram")
        (project_dir / "tb_hw_sdram.py").write_text(tb_src, encoding="utf-8")
        model = DEFAULT_TEMPLATE_DIR / "hw" / "sdram" / "sdram_model.v"
        board = project_dir / "py2ver_board.v"
        board.write_text(_BOARD_SIM, encoding="utf-8")
        sources = ([board, project_dir / "DE0_Nano.v", project_dir / f"{self.t_name}.v", model]
                   + sorted((project_dir / "uart").glob("*.sv")))
        tb_runner.run_cocotb("py2ver_board", sources, "tb_hw_sdram", project_dir,
                             timescale=("1ns", "1ps"))
        with (project_dir / "results" / "results.pickle").open("rb") as handle:
            replies = pickle.load(handle)
        outs = []
        for vec, steps, reply in zip(vectors, samples, replies):
            want = [n for _, n in steps]
            acks_ok = all(r == [0x57] for (d, _), r in zip(steps, reply) if d[0] == 0x57)
            if [len(r) for r in reply] != want or not acks_ok:
                raise RuntimeError(f"Virtual board replies for {vec}: expected {want} bytes "
                                   f"('W' acks, the outputs, the returned lists), got {reply}")
            outs.append(self._sdram_results([bytes(r) for r in reply]))
        return outs

    def gen_hw_project(self, baud: int = DEFAULT_BAUD_RATE) -> Path:
        """Write the DE0-Nano Quartus project (board top, UART, core). No compile."""
        project_dir = self._project_dir
        if project_dir.is_dir():
            shutil.rmtree(project_dir)
        project_dir.mkdir(parents=True, exist_ok=True)

        project_name = self.t_name

        (project_dir / f"{project_name}.qpf").write_text(
            f"PROJECT_REVISION = {project_name}\n", encoding="utf-8"
        )

        qsf_path = project_dir / f"{project_name}.qsf"
        self.get_template("hw/de0_nano_qsf.txt").stream({}).dump(str(qsf_path))

        # ---------------------- Emit UART split modules ----------------------
        uart_dir = project_dir / "uart"
        uart_dir.mkdir(exist_ok=True)

        self.get_template("hw/uart/uart_baud_gen.txt") \
            .stream({}).dump(str(uart_dir / "uart_baud_gen.sv"))
        self.get_template("hw/uart/uart_rx_8n1.txt") \
            .stream({}).dump(str(uart_dir / "uart_rx_8n1.sv"))
        self.get_template("hw/uart/uart_tx_8n1.txt") \
            .stream({}).dump(str(uart_dir / "uart_tx_8n1.sv"))
        self.get_template("hw/uart/uart_fifo4x8.txt") \
            .stream({}).dump(str(uart_dir / "uart_fifo4x8.sv"))
        # PRBS7 TX test leaf
        self.get_template("hw/uart/uart_prbs7_tx_test.txt") \
            .stream({}).dump(str(uart_dir / "uart_prbs7_tx_test.sv"))
        # Fixed-pattern TX generator
        self.get_template("hw/uart/uart_fixed_pattern_tx.txt") \
            .stream({}).dump(str(uart_dir / "uart_fixed_pattern_tx.sv"))
        # Loopback wrapper
        self.get_template("hw/uart/uart_loopback.txt") \
            .stream({
                "CLK_HZ": DEFAULT_CLK_FREQ,
                "BAUD": DEFAULT_BAUD_RATE
            }).dump(str(uart_dir / "uart_loopback.sv"))

        # uart_transceiver module
        self.get_template("hw/uart/uart_transceiver_top.txt") \
            .stream({}).dump(str(uart_dir / "uart_transceiver.sv"))

        # Append UART leafs to QSF
        with qsf_path.open("a", encoding="utf-8") as qsf:
            qsf.write("\n# UART split modules\n")
            qsf.write('set_global_assignment -name SYSTEMVERILOG_FILE uart/uart_baud_gen.sv\n')
            qsf.write('set_global_assignment -name SYSTEMVERILOG_FILE uart/uart_rx_8n1.sv\n')
            qsf.write('set_global_assignment -name SYSTEMVERILOG_FILE uart/uart_tx_8n1.sv\n')
            qsf.write('set_global_assignment -name SYSTEMVERILOG_FILE uart/uart_fifo4x8.sv\n')
            qsf.write('set_global_assignment -name SYSTEMVERILOG_FILE uart/uart_prbs7_tx_test.sv\n')
            qsf.write('set_global_assignment -name SYSTEMVERILOG_FILE uart/uart_fixed_pattern_tx.sv\n')
            qsf.write('set_global_assignment -name SYSTEMVERILOG_FILE uart/uart_loopback.sv\n')
            qsf.write('set_global_assignment -name SYSTEMVERILOG_FILE uart/uart_transceiver.sv\n')

        # SDC / wrappers / board top
        self.get_template("hw/de0_nano_sdc.txt") \
            .stream({}).dump(str(project_dir / "DE0_Nano.SDC"))

        if self._sdram:
            # A design with the SDRAM: the board's top takes commands (see the template)
            self.get_template("hw/de0_nano_sdram_top.txt").stream(
                self._sdram_board(baud)).dump(str(project_dir / "DE0_Nano.v"))
            with open(project_dir / "DE0_Nano.SDC", "a", encoding="utf-8") as f:
                f.write(_SDRAM_SDC.replace("[get_ports clk]", "[get_ports CLOCK_50]"))
        else:
            self.get_template("hw/delayed_registers.txt") \
                .stream(self.syn_data).dump(str(project_dir / "delayed_registers.sv"))

            # DE0_Nano top you provided
            self.get_template("hw/de0_nano_top.txt") \
                .stream(self.syn_data).dump(str(project_dir / "DE0_Nano.v"))

        # Copy generated core HDL
        self._copy_hdl(project_dir, f"{project_name}.v")

        log.info("Quartus project created at %s", project_dir)

        return project_dir

    def createHWproject(self) -> bool:
        """
        Create Quartus project, emit all HDL, run compile, and attempt
        to program the FPGA.

        Returns:
            True on successful compile (programming errors are logged but
            do not flip this to False), False on any early/compile error.
        """
        syn_tool_dir = self.syn_attr.get("QUARTUS_DIR")
        if not syn_tool_dir:
            self._hw_error = "missing 'QUARTUS_DIR'"
            log.error(self._hw_error)
            return False

        quartus_sh = os.path.join(syn_tool_dir, "quartus_sh")
        if not os.path.exists(quartus_sh):
            self._hw_error = f"quartus_sh not found: {quartus_sh}"
            log.error(self._hw_error)
            return False

        project_dir = self.gen_hw_project(int(self.syn_attr.get("UART_BAUD", DEFAULT_BAUD_RATE)))
        project_name = self.t_name

        # ---- Compile ----
        log.info("Running Quartus compile...")
        result = subprocess.run(
            [quartus_sh, "--flow", "compile", project_name],
            cwd=project_dir,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            self._hw_error = f"Quartus compile failed (rc={result.returncode}); see log"
            log.error("Quartus compile failed: rc=%d", result.returncode)
            log.error("quartus_sh stdout:\n%s", result.stdout)
            log.error("quartus_sh stderr:\n%s", result.stderr)
            return False

        # Board reports are written to the project directory
        self.hw_report = read_report(
            project_dir, project_name, self.t_name,
            1e9 / DEFAULT_CLK_FREQ, out_dir=project_dir,
        )
        log.info("Board build:\n%s", self.hw_report)
        if self.hw_report.meets_timing is False and not self.syn_attr.get("ALLOW_TIMING_FAILURE"):
            self._hw_error = (
                f"board design misses {DEFAULT_CLK_FREQ // 1_000_000} MHz timing "
                f"(Fmax {self.hw_report.fmax_mhz} MHz), so results would be unreliable; "
                "use pipeline=N, or set ALLOW_TIMING_FAILURE"
            )
            log.error(self._hw_error)
            return False

        self._program_fpga(syn_tool_dir, project_dir, project_name)
        return True

    # ------------------------------------------------------------------------
    # Valid/ready streaming wrapper
    # ------------------------------------------------------------------------

    def _stream_mode(self) -> str:
        if getattr(self, "ram_phases", 1) > 1:
            return "ram"
        if self.pipeline_stages:
            return "pipe"
        if self.has_clk and not (self.state_map or self.rams):
            raise NotImplementedError(
                "The stream wrapper needs a combinational, stateful or pipelined design; "
                "'reg' signals have no balanced latency (use pipeline=N instead)."
            )
        return "reg_out"

    def _serial_ports(self, attr: Dict[str, Any]) -> Dict[str, Any]:
        """
        Lists the stream wrapper moves one element per transfer ('serial' in
        attr): {'in': {param: [ports]}, 'out': {name: [ports]}, 'm', 'l'}
        (m inputs per call, l results per call).
        """
        sin = {p: ports for p, ports in self.arg_groups
               if ports and attr.get(p, {}).get("serial") and ports[0] in self.input_args_list}
        for p, _ in self.arg_groups:
            if attr.get(p, {}).get("serial") and p not in sin:
                raise ValueError(f"'serial' input '{p}' must be a list input (not state)")
        sout: Dict[str, List[str]] = {}
        for name, meta in attr.items():
            if not (isinstance(meta, dict) and meta.get("serial")) or name in dict(self.arg_groups):
                continue
            ports = [n for n in self.output_args_list if re.fullmatch(rf"{re.escape(name)}_\d+", n)]
            if not ports:
                raise ValueError(f"'serial' output '{name}' must be a returned list")
            sout[name] = sorted(ports, key=lambda n: int(n.rsplit("_", 1)[1]))
        lengths_in = {len(v) for v in sin.values()}
        lengths_out = {len(v) for v in sout.values()}
        if len(lengths_in) > 1 or len(lengths_out) > 1:
            raise ValueError("Serial inputs (and serial outputs) must all have the same length")
        return {"in": sin, "out": sout, "m": lengths_in.pop() if sin else 1,
                "l": lengths_out.pop() if sout else 1}

    def stream_wrapper(self, write: bool = True) -> Path:
        """
        Write hdl/<name>_stream.v (with write=False: only into hdl_files): the
        module behind a valid/ready interface (clk, rst, s_valid/s_ready/s_<input>,
        m_valid/m_ready/m_<output>). Lists marked 'serial' take or give one
        element per transfer.
        """
        mode = self._stream_mode()

        def meta(name: str) -> Tuple[int, bool]:
            m = self.attr.get(name, {})
            return int(m.get("width", 1)), int(m.get("signed", 0)) == 1

        def rng(name: str) -> str:
            w, sgn = meta(name)
            return f"{'signed ' if sgn else ''}[{w - 1}:0]"

        def joint(ports: List[str]) -> str:
            """A range holding every element: signed if any is (unsigned ones then need a bit more)."""
            signed = any(meta(n)[1] for n in ports)
            w = max(meta(n)[0] + (1 if signed and not meta(n)[1] else 0) for n in ports)
            return f"{'signed ' if signed else ''}[{w - 1}:0]"

        serial = self._serial["in"] or self._serial["out"]
        ctx = {
            "module_name": f"{self.t_name}_blocks" if serial else f"{self.t_name}_stream",
            "name": self.t_name,
            "mode": mode,
            "stages": self.pipeline_stages,
            "in_decls": [f"{rng(n)} s_{n}" for n in self.input_args_list],
            "out_decls": [f"{rng(n)} m_{n}" for n in self.output_args_list],
            "in_conns": [(n, rng(n)) for n in self.input_args_list],
            "out_conns": [(n, rng(n)) for n in self.output_args_list],
            "has_clk": self.has_clk,
            "has_rst": any(p.kind == "rst" for p in self.ir.inputs),
            "has_ce": self._has_ce,
            "has_vin": self._has_vin,
            "phases": getattr(self, "ram_phases", 1),
        }
        text = self.get_template("stream_wrapper.txt").render(ctx)
        if serial:
            s_in = [p for ports in self._serial["in"].values() for p in ports]
            s_out = [p for ports in self._serial["out"].values() for p in ports]
            plain_in = [n for n in self.input_args_list if n not in s_in]
            plain_out = [n for n in self.output_args_list if n not in s_out]
            sin = [{"name": p, "range": rng(ports[0])} for p, ports in self._serial["in"].items()]
            sout = [{"name": n, "range": joint(ports)} for n, ports in self._serial["out"].items()]
            text += self.get_template("serial_wrapper.txt").render({
                "name": self.t_name,
                "m_in": self._serial["m"], "l_out": self._serial["l"],
                "serial_in": list(self._serial["in"]), "serial_out": list(self._serial["out"]),
                "in_decls": [f"{s['range']} s_{s['name']}" for s in sin]
                            + [f"{rng(n)} s_{n}" for n in plain_in],
                "out_decls": [f"{s['range']} m_{s['name']}" for s in sout]
                             + [f"{rng(n)} m_{n}" for n in plain_out],
                "block_in": [{"port": n, "range": rng(n)} for n in self.input_args_list],
                "block_out": [{"port": n, "range": rng(n)} for n in self.output_args_list],
                "serial_in_ports": sin, "serial_out_ports": sout,
                "plain_in": [{"port": n, "range": rng(n)} for n in plain_in],
                "plain_out": [{"port": n, "range": rng(n)} for n in plain_out],
            })
        path = self.hdl_dir / f"{self.t_name}_stream.v"
        self.hdl_files[path.name] = text
        if write:
            self.write_hdl()
            log.info("Stream wrapper (%s) written to %s", mode, path)
        return path

    def _transfers(self, vecs: List[Tuple[int, ...]]) -> Tuple[List[Tuple[int, ...]], List[str]]:
        """Input vectors of calls -> the stream wrapper's transfers (a serial list
        one element per transfer), and the wrapper's input port names."""
        ser = self._serial
        if not ser["in"]:
            return vecs, list(self.input_args_list)
        s_in = [p for ports in ser["in"].values() for p in ports]
        plain = [n for n in self.input_args_list if n not in s_in]
        idx = {n: i for i, n in enumerate(self.input_args_list)}
        out = [tuple(vec[idx[ports[k]]] for ports in ser["in"].values())
               + tuple(vec[idx[n]] for n in plain)
               for vec in vecs for k in range(ser["m"])]
        return out, list(ser["in"]) + plain

    def TB_handshake(self) -> Callable[..., List[Tuple[int, ...]]]:
        return self._fx_rows(self.tb_handshake_fun)

    def tb_handshake_fun(self, vectors: List[Any], valid: float = 0.7, ready: float = 0.7,
                         seed: int = 0) -> List[Tuple[int, ...]]:
        """
        Simulate the stream wrapper with inputs offered with probability `valid`
        and results accepted with probability `ready` each clock. Returns the
        results in the order they were taken (one per input). Vectors are
        samples for a design with state (as for TB_stream), else as for TB_batch.
        """
        vectors = list(vectors)
        if not vectors:
            return []
        if self.state_map or self.rams or self._fold:
            vecs = self._stream_vectors(vectors)
        else:
            vecs = [self._input_vector(tuple(v)) for v in vectors]
        masks = [(1 << w) - 1 for w in self.input_args_width_list]
        vecs = [tuple(v & m for v, m in zip(vec, masks)) for vec in vecs]
        calls = len(vecs)
        out_names = self.output_args_list
        ser = self._serial
        s_out = [p for ports in ser["out"].values() for p in ports]
        vecs, in_names = self._transfers(vecs)
        if ser["out"]:
            out_names = list(ser["out"]) + [n for n in out_names if n not in s_out]

        wrapper = self.stream_wrapper()
        results_path = self.outdir / RESULTS_PATH
        if results_path.exists():
            results_path.unlink()
        tb_src = self.get_template("tb_handshake.txt").render({
            "name": self.t_name,
            "vectors": repr(vecs),
            "valid_p": float(valid),
            "ready_p": float(ready),
            "seed": int(seed),
            "max_cycles": (20 + getattr(self, "ram_phases", 1)) * len(vecs) * ser["l"]
                          + 100 + 4 * self.pipeline_stages,
            "in_names": in_names,
            "out_args": out_names,
            "n_results": calls * ser["l"],
            "out_meta": {**self.out_meta, **{n: {"signed": int(any(
                self.attr.get(p, {}).get("signed", 0) for p in ports))}
                for n, ports in ser["out"].items()}},
        })
        _drop_bytecode(self.outdir, "tb_handshake")
        (self.outdir / "tb_handshake.py").write_text(tb_src, encoding="utf-8")
        tb_runner.run_cocotb(
            f"{self.t_name}_stream", [self.hdl_dir / "main.v", wrapper], "tb_handshake",
            self.outdir, build_name="sim_build_handshake", timescale=("1ns", "1ps"),
        )
        rows: List[Dict[str, int]] = []
        if results_path.exists():
            with results_path.open("rb") as handle:
                payload = pickle.load(handle)
            rows = payload["outs"]
            # Clocks from reset release to the last result taken
            self.handshake_cycles = payload["cycles"]
        if len(rows) != calls * ser["l"]:
            raise RuntimeError(
                f"Stream wrapper returned {len(rows)} results for {calls * ser['l']} expected"
            )
        if ser["out"]:
            # l element transfers -> one call's outputs
            blocks = []
            for c in range(calls):
                group = rows[c * ser["l"]:(c + 1) * ser["l"]]
                row = dict(group[0])
                for name, ports in ser["out"].items():
                    for k, port in enumerate(ports):
                        row[port] = group[k][name]
                blocks.append(row)
            rows = blocks
        return self._rows_to_outputs(rows, calls)

    # ------------------------------------------------------------------------
    # Standalone synthesis (area / timing of the core only, no board)
    # ------------------------------------------------------------------------

    @property
    def _syn_dir(self) -> Path:
        return self.outdir / "syn" / self.t_name

    def SYN(self, syn_attr: Dict[str, Any] | None = None) -> SynReport:
        """
        Compile the core alone with Quartus (no board, no programming) and
        return its area and timing.

        syn_attr may contain:
            'QUARTUS_DIR':   path to quartus/bin (default: quartus_sh on PATH)
            'CLK_PERIOD_NS': target clock period (default 20.0, i.e. 50 MHz)
            'DEVICE':        FPGA part (default EP4CE22F17C6, DE0-Nano)
            'FAMILY':        device family (default "Cyclone IV E")
        """
        syn_attr = dict(syn_attr or {})
        quartus_sh = self._quartus_sh(syn_attr)
        project_dir = self.gen_syn_project(syn_attr)
        return self._run_quartus(quartus_sh, project_dir, f"{self.t_name}_syn_top", self.t_name,
                                 syn_attr)

    @staticmethod
    def _quartus_sh(syn_attr: Dict[str, Any]) -> str:
        tool_dir = syn_attr.get("QUARTUS_DIR")
        quartus_sh = (
            os.path.join(tool_dir, "quartus_sh") if tool_dir else shutil.which("quartus_sh")
        )
        if not quartus_sh or not os.path.exists(quartus_sh):
            raise RuntimeError(
                "quartus_sh not found; set 'QUARTUS_DIR' to Quartus's bin directory "
                f"(looked for: {quartus_sh or 'quartus_sh on PATH'})"
            )
        return quartus_sh

    @staticmethod
    def _run_quartus(quartus_sh: str, project_dir: Path, revision: str, top: str,
                     syn_attr: Dict[str, Any]) -> SynReport:
        log.info("Running Quartus synthesis/fit/timing for %s...", top)
        result = subprocess.run(
            [quartus_sh, "--flow", "compile", revision],
            cwd=project_dir,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            errors = [l for l in result.stdout.splitlines() if l.lstrip().startswith("Error")]
            raise RuntimeError(
                f"Quartus compile failed (rc={result.returncode}) in {project_dir}:\n"
                + "\n".join(errors[:20] or result.stdout.splitlines()[-20:])
            )

        report = read_report(
            project_dir, revision, top,
            float(syn_attr.get("CLK_PERIOD_NS", DEFAULT_SYN_PERIOD_NS)),
        )
        log.info("Synthesis report for %s:\n%s", top, report)
        return report

    def gen_syn_project(self, syn_attr: Dict[str, Any]) -> Path:
        """
        Write a Quartus project that compiles only the generated core inside a
        wrapper that registers all of its inputs and outputs (virtual pins, so
        wide cores fit). Returns the project directory.
        """
        wrapper = f"{self.t_name}_syn_top"
        core_in = [p for p in self.ir.inputs if p.kind != "clk"]
        core_out = list(self.ir.outputs)

        def _rng(w: int) -> str:
            return f"[{w - 1}:0] " if w > 1 else ""

        port_decls = ["  input clk"]
        port_decls += [f"  input {_rng(p.width)}{p.name}_in" for p in core_in]
        port_decls += [f"  output reg {_rng(p.width)}{p.name}_out" for p in core_out]

        conns = []
        if self.has_clk:
            conns.append("    .clk(clk)")
        conns += [f"    .{p.name}({p.name}_q)" for p in core_in]
        sdram = bool(self._fold and self._fold.get("sdram"))
        if sdram:
            # The SDRAM's pins: real pins (the DE0-Nano's), not registered; adv as an output
            core_out = core_out + [Port(name="adv", width=1, signed=False)]
            port_decls.append("  output reg adv_out")
            for d, w, n in self.SDRAM_PORTS:
                if n.startswith("dram_"):
                    port_decls.append(f"  {d} {_rng(w)}{n.upper()}")
                    conns.append(f"    .{n}({n.upper()})")
                elif n.startswith("host_"):                     # (registered, as the others)
                    if d == "input":
                        core_in = core_in + [Port(name=n, width=w, signed=False)]
                        port_decls.append(f"  input {_rng(w)}{n}_in")
                        conns.append(f"    .{n}({n}_q)")
                    else:
                        core_out = core_out + [Port(name=n, width=w, signed=False)]
                        port_decls.append(f"  output reg {_rng(w)}{n}_out")
        conns += [f"    .{p.name}({p.name}_w)" for p in core_out]

        ctx = {
            "top_name": self.t_name,
            "wrapper": wrapper,
            "inputs": core_in,
            "outputs": core_out,
            "port_decls": port_decls,
            "conns": conns,
            "family": syn_attr.get("FAMILY", DEFAULT_SYN_FAMILY),
            "device": syn_attr.get("DEVICE", DEFAULT_SYN_DEVICE),
            "period_ns": float(syn_attr.get("CLK_PERIOD_NS", DEFAULT_SYN_PERIOD_NS)),
            "extra_files": [],
        }

        project_dir = self._syn_dir
        if project_dir.is_dir():
            shutil.rmtree(project_dir)
        project_dir.mkdir(parents=True)

        (project_dir / f"{wrapper}.qpf").write_text(
            f"PROJECT_REVISION = {wrapper}\n", encoding="utf-8"
        )
        for tpl, name in (
            ("hw/syn/qsf.txt", f"{wrapper}.qsf"),
            ("hw/syn/sdc.txt", f"{wrapper}.sdc"),
            ("hw/syn/wrapper.txt", f"{wrapper}.v"),
        ):
            self.get_template(tpl).stream(ctx).dump(str(project_dir / name))
        self._copy_hdl(project_dir, f"{self.t_name}.v")
        if sdram:
            self._syn_sdram(project_dir, wrapper)

        log.info("Synthesis project created at %s", project_dir)
        return project_dir

    @property
    def _sdram(self) -> bool:
        return bool(self._fold and self._fold.get("sdram"))

    def _port_inputs(self) -> List[Tuple[str, int]]:
        """(name, width) of the inputs that are ports (not in the SDRAM)."""
        ext = getattr(self, "_ext_inputs", {})
        return [(n, w) for n, w in zip(self.input_args_list, self.input_args_width_list)
                if n not in ext]

    def _sdram_outputs(self, attr: Dict[str, Any]) -> Dict[str, int]:
        """sdram=: a returned list in the SDRAM has no ports; its values take their
        place among the outputs (output_args_list) all the same, in the return's
        order. {port: its SDRAM address}."""
        sd = (self._fold or {}).get("sdram") or {}
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
                ext[port] = int(info["base"]) + i
                attr.setdefault(port, {"signed": sgn, "width": w})
        if at != len(ports):
            raise RuntimeError(f"The outputs don't match the return: {at} of {len(ports)} ports")
        self.output_args_list = [n for n, _ in merged]
        self.output_args_width_list = [w for _, w in merged]
        return ext

    def _sdram_board(self, baud: int) -> Dict[str, Any]:
        """The SDRAM board top's template context."""
        ins, lo = [], 0
        for n, w in self._port_inputs():
            ins.append({"name": n, "width": w, "lo": lo})
            lo += (w + 7) // 8 * 8
        outs = [{"name": n, "width": w, "seg": (w + 7) // 8 * 8}
                for n, w in zip(self.output_args_list, self.output_args_width_list)
                if n not in self._ext_outputs]
        # (a reply to 'R' of a byte at least: the sample done, its lists ready to read)
        return {"core": self.t_name, "inputs": ins, "outputs": outs, "rx_bytes": lo // 8,
                "tx_bytes": max(1, sum(o["seg"] for o in outs) // 8), "phases": int(self.ram_phases),
                "baud": int(baud)}

    def _sdram_steps(self, vec: Tuple[int, ...], first: bool) -> List[Tuple[bytes, int]]:
        """The UART commands of a sample (vec: its raw inputs, as input_args_list):
        the tables first (first=True), its input lists into the SDRAM, then 'R'.
        [(bytes, bytes of the reply)]."""
        from .native import sdram_images, _runs
        ext = getattr(self, "_ext_inputs", {})
        steps: List[Tuple[bytes, int]] = []

        def write(base: int, words: List[int]) -> None:
            for at in range(0, len(words), 4096):
                part = words[at:at + 4096]
                head = b"W" + (base + at).to_bytes(3, "big") + len(part).to_bytes(3, "big")
                steps.append((head + b"".join((w & 0xFFFF).to_bytes(2, "big") for w in part), 1))
        if first:
            for base, values in sdram_images(self):
                write(base, values)
        names = list(self.input_args_list)
        addrs = [ext[n] for n in names if n in ext]
        values = [int(vec[i]) for i, n in enumerate(names) if n in ext]
        for start, base, count in _runs(addrs):
            write(base, values[start:start + count])
        acc, off = 0, 0
        port = {n for n, _ in self._port_inputs()}
        for i, (n, w) in enumerate(zip(names, self.input_args_width_list)):
            if n in port:
                acc |= (int(vec[i]) & ((1 << w) - 1)) << off
                off += (w + 7) // 8 * 8
        steps.append((b"R" + acc.to_bytes(off // 8, "little"), self._sdram_board(0)["tx_bytes"]))
        for base, count in self._sdram_reads():                 # (the returned lists)
            steps.append((b"X" + base.to_bytes(3, "big") + count.to_bytes(3, "big"), 2 * count))
        return steps

    def _sdram_reads(self) -> List[Tuple[int, int]]:
        """(address, words) of the 'X' reads of a sample's returned lists in the SDRAM."""
        from .native import _runs
        addrs = [self._ext_outputs[n] for n in self.output_args_list if n in self._ext_outputs]
        return [(base + at, min(4096, count - at)) for _, base, count in _runs(addrs)
                for at in range(0, count, 4096)]

    def _sdram_results(self, replies: List[bytes]) -> Tuple[int, ...]:
        """A sample's outputs from the board's replies to its steps: the ports' from
        'R' (the UART's segments), the returned lists' from the 'X' reads after it."""
        nx = len(self._sdram_reads())
        acc = int.from_bytes(replies[len(replies) - 1 - nx], "little")
        words = b"".join(replies[len(replies) - nx:])
        values: Dict[str, int] = {}
        off = 0
        ext = [n for n in self.output_args_list if n in self._ext_outputs]
        for n, w in zip(self.output_args_list, self.output_args_width_list):
            if n not in self._ext_outputs:
                values[n] = (acc >> off) & ((1 << w) - 1)
                off += (w + 7) // 8 * 8
        widths = dict(zip(self.output_args_list, self.output_args_width_list))
        for i, n in enumerate(ext):
            values[n] = int.from_bytes(words[2 * i:2 * i + 2], "big") & ((1 << widths[n]) - 1)
        return tuple(self.tosigned(values[n], widths[n]) if self.attr.get(n, {}).get("signed") == 1
                     else values[n] for n in self.output_args_list)

    def _syn_sdram(self, project_dir: Path, wrapper: str) -> None:
        """A design with the SDRAM: its pins where the DE0-Nano has them, and the
        chip's timing (IS42S16160B-7 at the inverted clock) for the timing analysis."""
        qsf = (DEFAULT_TEMPLATE_DIR / "hw" / "de0_nano_qsf.txt").read_text(encoding="utf-8")
        pins = [line for line in qsf.splitlines() if "DRAM_" in line and not line.startswith("#")]
        with open(project_dir / f"{wrapper}.qsf", "a", encoding="utf-8") as f:
            f.write("\n# The SDRAM's pins (DE0-Nano)\n" + "\n".join(pins) + "\n")
        with open(project_dir / f"{wrapper}.sdc", "a", encoding="utf-8") as f:
            f.write(_SDRAM_SDC)

    # ------------------------------------------------------------------------
    # Testbench
    # ------------------------------------------------------------------------

    def TB(self) -> Callable[..., Tuple[int, ...]]:
        return self._fx_rows(self.tb_fun, single=True)

    def TB_batch(self) -> Callable[[List[Tuple[Any, ...]]], List[Tuple[int, ...]]]:
        return self._fx_rows(self.tb_batch_fun)

    def PY_batch(self) -> Callable[[List[Tuple[Any, ...]]], List[Tuple[int, ...]]]:
        return self._fx_rows(self.py_batch_fun)

    def py_batch_fun(self, vectors: List[Tuple[Any, ...]]) -> List[Tuple[int, ...]]:
        """
        Python reference for tb_batch_fun(): the function's results (for a
        fixed-point design, its bit-exact model's), wrapped to each output's
        declared width and sign (int() truncates /).
        """
        if self.state_map or self.rams or self._seq_mems:
            raise ValueError("This design has state ('next' or 'memory' in attr); use PY_stream().")
        func = self._model or self._original()
        return [self._py_row(func(*self._py_args(vec))) for vec in vectors]

    def _original(self) -> Callable:
        """The function as written, taking the flat lists the ports take for its
        lists of lists (it gets them nested)."""
        if not self._shapes:
            return self.func
        names = [self._py_name.get(p, p) for p in self._shapes]
        shapes = {self._py_name.get(p, p): s for p, s in self._shapes.items()}
        sig = inspect.signature(self.func)

        @functools.wraps(self.func)
        def call(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            for name in names:
                if name in bound.arguments:
                    bound.arguments[name] = nest(flatten(bound.arguments[name]), shapes[name])
            return self.func(*bound.args, **bound.kwargs)
        return call

    def _py_row(self, ret: Any) -> Tuple[int, ...]:
        return tuple(
            self._wrap(int(v), w, int(self.attr.get(n, {}).get("signed", 0)) == 1)
            for v, n, w in zip(_flatten_outputs(ret), self.output_args_list,
                               self.output_args_width_list)
        )

    def hardware_random(self):
        """A context manager: inside it, the original code's random numbers (numpy
        Generators' .random, torch's dropout) are the ones the hardware draws, from
        their start, so the original can be compared with the hardware step by step."""
        from .numpy_loops import _drawing
        return _drawing(getattr(self._numpy, "generators", None) or {})

    def state_values(self, row: Tuple[Any, ...] | None = None) -> Dict[str, Any]:
        """
        numpy designs with state: the object's attributes in one result row
        of PY_stream() / TB_stream(), as arrays of their shapes (e.g. the
        weights after a hardware training step): {'W1': array, ...}. A
        sequenced design keeps its state in memories, not in its results:
        without a row, the state after the last PY_stream(), as
        {(object, attribute): array or number} (the object the original's).
        """
        import numpy as np
        if row is None and self._numpy is not None and getattr(self._numpy, "memories", None):
            if not self._last_memories:
                raise ValueError("Run PY_stream() first")
            out = {}
            for mem, (label, init, owner) in self._numpy.memories.items():
                frac = self._fx_frac(mem)
                raw = self._last_memories[mem]
                vals = [v / (1 << frac) if frac is not None else v for v in raw]
                out[(owner, label.split(".", 1)[1])] = (np.array(vals, dtype=float).reshape(np.shape(init))
                                                     if isinstance(init, np.ndarray) else vals[0])
            return out
        if self._numpy is None or not self._numpy.state:
            raise ValueError("Only for a numpy design with state")
        out, i = {}, len(self.output_args_list) - sum(v.size for v in self._numpy.state.values())
        for a, init in self._numpy.state.items():
            out[a] = np.array(row[i:i + init.size], dtype=float).reshape(init.shape)
            i += init.size
        return out

    def PY_float(self) -> Callable[[List[Any]], List[Tuple[Any, ...]]]:
        """
        The original function as written (floats), on the same inputs as the
        hardware after they are quantized: the ideal a fixed-point design
        approximates. Takes vectors like PY_batch(), or samples like
        PY_stream() for a design with state.
        """
        return self.py_float_fun

    def py_float_fun(self, vectors: List[Any]) -> List[Tuple[Any, ...]]:
        from .auto import float_reference
        if self._adapt is not None:
            vectors = [self._adapt(v) for v in vectors]
        return float_reference(self.func, self._py_attr, vectors)

    def quantization_error(self, vectors: List[Any]) -> Dict[str, Tuple[float, float]]:
        """
        Per output: (max abs error, RMS error) of the bit-exact results
        (PY_batch / PY_stream, which the hardware matches) against PY_float().
        """
        stream = bool(self.state_map or self.rams or self._seq_mems)
        exact = (self.PY_stream() if stream else self.PY_batch())(vectors)
        ideal = self.py_float_fun(vectors)
        out: Dict[str, Tuple[float, float]] = {}
        for i, name in enumerate(self.output_args_list):
            errs = [abs(float(e[i]) - float(r[i])) for e, r in zip(exact, ideal)]
            if errs:
                out[self._py_name.get(name, name)] = (
                    max(errs), (sum(x * x for x in errs) / len(errs)) ** 0.5)
        return out

    def TB_stream(self) -> Callable[[List[Any]], List[Tuple[int, ...]]]:
        return self._fx_rows(self.tb_stream_fun)

    def PY_stream(self) -> Callable[[List[Any]], List[Tuple[int, ...]]]:
        return self._fx_rows(self.py_stream_fun)

    # ------------------------------ fixed point ------------------------------

    @staticmethod
    def _raw_init(meta: Any) -> Any:
        """A fixed-point state/memory's 'init' (numbers) as raw values."""
        if not (isinstance(meta, dict) and meta.get("frac") is not None and "init" in meta):
            return meta
        q = lambda v: quantize(v, meta["frac"], meta["width"], bool(meta.get("signed")),  # noqa: E731
                               bool(meta.get("round")), bool(meta.get("saturate")))
        init = meta["init"]
        return dict(meta, init=[q(v) for v in init] if isinstance(init, (list, tuple)) else q(init))

    def _fx_frac(self, port: str) -> Optional[int]:
        """Fraction bits of an output port (xs_3 is an element of xs, a_out
        returns a, xs_3_out returns an element of xs)."""
        names = [port]
        for name in list(names):
            base = name[:-4] if name.endswith("_out") else name
            m = re.fullmatch(r"(.+)_(\d+)", base)
            names += [base] + ([m.group(1)] if m else [])
        for name in names:
            t = self._fx_types.get(name)
            if t is not None:
                t = t[1] if t[0] == "l" else t
                return t[1] if t is not None and t[0] == "f" else None
        return None

    def _py_value(self, param: str, value: Any) -> Any:
        """An argument for the ports / the bit-exact model: a fixed-point
        parameter's numbers as raw values, quantized like assignments to it."""
        if param in self._shapes:            # a list of lists: flat, row-major
            value = flatten(value)
        m = self._fx_params.get(param)
        if m is None:
            return value
        conv = lambda v: quantize(v, m["frac"], m["width"], bool(m.get("signed")),  # noqa: E731
                                  bool(m.get("round")), bool(m.get("saturate")))
        return [conv(v) for v in value] if isinstance(value, (list, tuple)) else conv(value)

    def _expand(self, args: Tuple[Any, ...]) -> Tuple[Any, ...]:
        return expand_args(self._call_inputs, self._cplx_params, args)

    def _py_args(self, vec: Tuple[Any, ...]) -> Tuple[Any, ...]:
        vec = self._expand(tuple(vec))
        params = [p for p, _ in self.arg_groups]
        return tuple(self._py_value(p, v) for p, v in zip(params, vec)) + tuple(vec[len(params):])

    def _fx_rows(self, fn: Callable, single: bool = False) -> Callable:
        """fn with its results' fixed-point outputs as numbers (raw / 2**frac),
        and (numpy designs) taking arguments as the original does: one call's
        if `single`, else a list of them first."""
        if not self._fx_out and self._adapt is None:
            return fn
        fracs = [self._fx_out.get(n) for n in self.output_args_list]

        def row(r):
            return tuple(v if f is None else int(v) / (1 << f) for v, f in zip(r, fracs))

        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            if self._adapt is not None:
                args = (self._adapt(args) if single
                        else ([self._adapt(v) for v in args[0]],) + tuple(args[1:]))
            out = fn(*args, **kwargs)
            return [row(r) for r in out] if isinstance(out, list) else row(out)

        return wrapped

    def _flatten_args(self, in_arg: Tuple[Any, ...]) -> Tuple[Any, ...]:
        """Function arguments (a list argument as a list) -> one value per input port."""
        groups = [(p, ports) for p, ports in self.arg_groups
                  if (ports or [p])[0] in self.input_args_list]   # state lives in registers
        in_arg = self._expand(tuple(in_arg))
        out: List[Any] = []
        for value, (param, ports) in zip(in_arg, groups):
            value = self._py_value(param, value)
            if ports is None:
                out.append(value)
                continue
            value = list(value)
            if len(value) != len(ports):
                raise ValueError(f"'{param}' needs {len(ports)} values, got {len(value)}")
            out.extend(value)
        return tuple(out)

    def _input_vector(self, in_arg: Tuple[Any, ...]) -> Tuple[int, ...]:
        # Missing trailing arguments default to 0
        flat = self._flatten_args(tuple(in_arg))
        return tuple(
            int(flat[i]) if i < len(flat) else 0
            for i in range(len(self.input_args_list))
        )

    def _stream_vectors(self, samples: List[Any]) -> List[Tuple[int, ...]]:
        if not (self.state_map or self.rams or self._fold):
            raise ValueError("Streams need state ('next' or 'memory' in attr); use TB_batch().")
        # A sample is a tuple of arguments; anything else is the only argument
        return [self._input_vector(s if isinstance(s, tuple) else (s,)) for s in samples]

    def _render_tb(self, template: str, vecs: List[Tuple[int, ...]]) -> None:
        # Write raw bit patterns of each port's width (cocotb rejects e.g. -1
        # on a 1-bit port)
        masks = [(1 << w) - 1 for w in self.input_args_width_list]
        vecs = [tuple(v & m for v, m in zip(vec, masks)) for vec in vecs]
        tb_out = self.get_template(template).render({
            "period": DEFAULT_TB_PERIOD,
            "in_names": self.input_args_list,
            "vectors": repr(vecs),
            "out_args": self.output_args_list,
            "has_clk": self.has_clk,
            "out_meta": self.out_meta,
            "latency_cycles": int(getattr(self, "tb_latency_cycles", 0)),
            "out_latency": dict(getattr(self, "out_latency", {})),
            "pipeline_stages": self.pipeline_stages,
            "has_ce": self._has_ce,
            "ram_phases": getattr(self, "ram_phases", 1),
            "has_vin": self._has_vin,
            # With state and a pipeline, results come out this many clocks later
            "stream_latency": self.pipeline_stages if self._has_vin else 0,
        })
        (self.outdir / "tb.py").write_text(tb_out, encoding="utf-8")

    def gen_tb(self, vectors: List[Tuple[Any, ...]]) -> None:
        vecs = [self._input_vector(tuple(v)) for v in vectors]
        log.info("gen_tb: generating tb.py with %d vector(s)", len(vecs))
        self._render_tb("tb.txt", vecs)

    def gen_tb_stream(self, samples: List[Any]) -> None:
        vecs = self._stream_vectors(samples)
        log.info("gen_tb_stream: generating tb.py with %d sample(s)", len(vecs))
        self._render_tb("tb_stream.txt", vecs)

    def tb_fun(self, *in_arg: Any) -> Tuple[int, ...]:
        return self.tb_batch_fun([in_arg])[0]

    def tb_batch_fun(self, vectors: List[Tuple[Any, ...]]) -> List[Tuple[int, ...]]:
        """Simulate all input vectors in a single cocotb run, one result per vector."""
        if self.state_map or self.rams or self._seq_mems:
            raise ValueError(
                "This design has state ('next' or 'memory' in attr); use TB_stream()."
            )
        vectors = [tuple(v) for v in vectors]
        log.info("TB called with %d vector(s)", len(vectors))
        if not vectors:
            return []
        if self._fold:              # several clocks per vector: the stream testbench
            return self.tb_stream_fun(vectors)
        native_rows = self._simulate_native([self._input_vector(v) for v in vectors])
        if native_rows is not None:
            return native_rows
        return self._simulate(len(vectors), lambda: self.gen_tb(vectors))

    def tb_stream_fun(self, samples: List[Any]) -> List[Tuple[int, ...]]:
        """
        Reset the state registers, then apply one sample per clock in a single
        cocotb run. Returns all outputs (in return order) for each sample,
        sampled before the clock edge that loads the next state.
        """
        samples = list(samples)
        self._stream_vectors(samples)  # validate before touching the filesystem
        log.info("TB stream called with %d sample(s)", len(samples))
        if not samples:
            return []
        native_rows = self._simulate_native(self._stream_vectors(samples))
        if native_rows is not None:
            return native_rows
        return self._simulate(len(samples), lambda: self.gen_tb_stream(samples))

    def _simulate_native(self, vecs: List[Tuple[int, ...]]) -> List[Tuple[int, ...]] | None:
        """The vectors through a plain Verilog testbench on Icarus or Verilator (see
        native.py), or None where the cocotb testbench is used."""
        from . import native
        clocks = len(vecs) * int(getattr(self, "ram_phases", 1)) + self.pipeline_stages
        sim = native.choose_simulator(self, clocks)
        if sim == "cocotb" or native.mode_of(self) is None:
            return None
        log.info("Simulating %d vector(s) with %s (native testbench)", len(vecs), sim)
        return self._rows_to_outputs(native.run(self, vecs, sim), len(vecs))

    def py_stream_fun(self, samples: List[Any]) -> List[Tuple[int, ...]]:
        """
        Python reference for tb_stream_fun(): calls the function (for a
        fixed-point design, its bit-exact model) once per sample, feeding each
        next-state output back into its state argument with the same
        width/sign wrapping as the hardware registers.
        """
        samples = list(samples)
        self._stream_vectors(samples)  # validates
        groups = dict(self.arg_groups)

        model = self._model is not None
        func = self._model if model else self._original()
        py = (lambda name: name) if model else (lambda name: self._py_name.get(name, name))  # noqa: E731
        params = [p for p in inspect.signature(func).parameters
                  if p not in {py(m) for m in self._seq_mems}]
        if params != [py(p) for p, _ in self.arg_groups]:
            raise ValueError(f"PY_stream() needs arguments {list(groups)}; function has {params}")
        inputs = [p for p, _ in self.arg_groups
                  if p not in self.state_params and p not in self.rams]

        def _meta(name: str) -> Tuple[int, bool]:
            m = self.attr.get(name, {})
            return int(m.get("width", 1)), int(m.get("signed", 0)) == 1

        def keep(name: str, v: Any) -> Any:
            return self._wrap(int(v), *_meta(name))

        def _state_value(param: str, value_of) -> Any:
            ports = groups[param]
            if ports is None:
                return keep(param, value_of(param))
            return [keep(port, value_of(port)) for port in ports]

        state = {p: _state_value(p, self.state_init.__getitem__) for p in self.state_params}
        # Memories: the function updates the list in place; values wrap like the RAM
        rams = {}
        for mem, ram in self.rams.items():
            init = self._ram_init[mem]
            vals = list(init) if isinstance(init, (list, tuple)) else [init] * ram["length"]
            w, sgn = int(ram["elem"]["width"]), int(ram["elem"]["signed"]) == 1
            rams[mem] = [self._wrap(int(v), w, sgn) for v in vals]
        # (a sequenced design's memories: their raw 'init', wrapped as kept)
        mems = {mem: list(m["init"]) for mem, m in self._seq_mems.items()}
        outs: List[Tuple[Any, ...]] = []
        for sample in samples:
            args = self._expand(sample if isinstance(sample, tuple) else (sample,))
            kwargs = dict(zip(inputs, args))
            for p in inputs[len(args):]:                     # missing trailing args: 0
                kwargs[p] = 0 if groups[p] is None else [0] * len(groups[p])
            kwargs = {p: self._py_value(p, v) for p, v in kwargs.items()}
            call = {py(k): v for k, v in {**kwargs, **state, **rams, **mems}.items()}
            ret = _flatten_outputs(func(**call))
            for mem, ram in self.rams.items():
                w, sgn = int(ram["elem"]["width"]), int(ram["elem"]["signed"]) == 1
                rams[mem][:] = [self._wrap(int(v), w, sgn) for v in rams[mem]]
            for mem, m in self._seq_mems.items():
                mems[mem][:] = [self._wrap(int(v), m["width"], bool(m["signed"]))
                                for v in mems[mem]]
            row = {name: keep(name, v) for name, v in zip(self.output_args_list, ret)}
            outs.append(tuple(row[n] for n in self.output_args_list))
            state = {p: _state_value(p, lambda port: row[self.state_map[port]])
                     for p in self.state_params}
        self._last_memories = {mem: list(v) for mem, v in mems.items()}
        return outs

    def _simulate(self, n_vectors: int, write_tb: Callable[[], None]) -> List[Tuple[int, ...]]:
        """Run one cocotb simulation of the tb.py written by write_tb()."""
        results_path = self.outdir / RESULTS_PATH

        # Clean previous results
        if results_path.exists():
            try:
                results_path.unlink()
            except OSError as e:
                log.warning(e)

        # Ensure a clean sim build directory so tb.py isn't stale
        sim_build = self.outdir / "sim_build"
        if sim_build.exists():
            try:
                shutil.rmtree(sim_build)
            except OSError as e:
                log.warning("Failed to remove sim_build: %s", e)

        self.write_hdl()
        _drop_bytecode(self.outdir, "tb")
        write_tb()

        # Run cocotb-based simulation
        tb_runner.test_runner(self.t_name, self.outdir)

        # Load results written by tb.py (a list of dicts; a single dict is
        # accepted as a one-vector batch)
        rows: List[Dict[str, int]] = []
        if results_path.exists():
            with results_path.open("rb") as handle:
                payload = pickle.load(handle)
            if isinstance(payload, str):
                payload = ast.literal_eval(payload)
            rows = [payload] if isinstance(payload, dict) else list(payload)
        else:
            log.warning("No TB results found at %s", results_path)

        return self._rows_to_outputs(rows, n_vectors)

    def _rows_to_outputs(self, rows: List[Dict[str, int]], n_vectors: int) -> List[Tuple[int, ...]]:
        outs: List[Tuple[int, ...]] = []
        for i in range(n_vectors):
            d = rows[i] if i < len(rows) else {}
            row: List[int] = []
            for n, w in zip(self.output_args_list, self.output_args_width_list):
                v = d.get(n, 0)
                if self.attr.get(n, {}).get("signed") == 1:
                    v = self.tosigned(v, w)
                row.append(v)
            outs.append(tuple(row))
        return outs

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
        ref = self.PY_stream() if (self.state_map or self.rams or self._fold) else (
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

    def SYN_stages(self, syn_attr: Dict[str, Any] | None = None) -> SynReport:
        """SYN() of <name>_stages: every stage and the streams between them."""
        if not self._stages:
            raise ValueError("Build the chain with stages=True")
        syn_attr = dict(syn_attr or {})
        quartus_sh = self._quartus_sh(syn_attr)
        top = f"{self.t_name}_stages"
        wrapper = f"{top}_syn_top"
        ports = self._stages["ports"]

        def width(rng: str) -> int:
            return int(re.search(r"\[(\d+):0\]", rng).group(1)) + 1

        inputs = [Port("rst", 1), Port("s_valid", 1), Port("m_ready", 1)]
        inputs += [Port(f"s_{n}", width(r)) for n, r in ports["in"]]
        outputs = [Port("s_ready", 1), Port("m_valid", 1)]
        outputs += [Port(f"m_{n}", width(r)) for n, r in ports["out"]]
        rng = lambda w: f"[{w - 1}:0] " if w > 1 else ""  # noqa: E731
        ctx = {
            "top_name": top, "wrapper": wrapper, "inputs": inputs, "outputs": outputs,
            "port_decls": ["  input clk"] + [f"  input {rng(p.width)}{p.name}_in" for p in inputs]
                          + [f"  output reg {rng(p.width)}{p.name}_out" for p in outputs],
            "conns": ["    .clk(clk)"] + [f"    .{p.name}({p.name}_q)" for p in inputs]
                     + [f"    .{p.name}({p.name}_w)" for p in outputs],
            "family": syn_attr.get("FAMILY", DEFAULT_SYN_FAMILY),
            "device": syn_attr.get("DEVICE", DEFAULT_SYN_DEVICE),
            "period_ns": float(syn_attr.get("CLK_PERIOD_NS", DEFAULT_SYN_PERIOD_NS)),
            "extra_files": [f.name for f in self._stages["files"][:-1]],
        }
        project_dir = self.outdir / "syn" / top
        if project_dir.is_dir():
            shutil.rmtree(project_dir)
        project_dir.mkdir(parents=True)
        (project_dir / f"{wrapper}.qpf").write_text(f"PROJECT_REVISION = {wrapper}\n", encoding="utf-8")
        for tpl, name in (("hw/syn/qsf.txt", f"{wrapper}.qsf"), ("hw/syn/sdc.txt", f"{wrapper}.sdc"),
                          ("hw/syn/wrapper.txt", f"{wrapper}.v")):
            self.get_template(tpl).stream(ctx).dump(str(project_dir / name))
        self.write_hdl()
        for f in self._stages["files"]:
            shutil.copy(f, project_dir / f.name)
        return self._run_quartus(quartus_sh, project_dir, wrapper, top, syn_attr)

    # ------------------------------------------------------------------------
    # FPGA Programmer
    # ------------------------------------------------------------------------

    @staticmethod
    def _program_fpga(syn_tool_dir: str, project_dir: Path, project_name: str) -> None:
        out_dir = project_dir / "output_files"
        candidates = [
            out_dir / f"{project_name}.sof",
            next(iter(out_dir.glob("*.sof")), None) if out_dir.exists() else None,
            project_dir / f"{project_name}.sof",
        ]
        sof_file = next((p for p in candidates if isinstance(p, Path) and p and p.exists()), None)

        if not sof_file:
            log.warning("No .sof found, skipping programming.")
            return

        quartus_pgm = os.path.join(syn_tool_dir, "quartus_pgm")
        if not os.path.exists(quartus_pgm):
            log.error("quartus_pgm not found at %s", quartus_pgm)
            return

        log.info("Programming FPGA with %s...", sof_file)
        result = subprocess.run(
            [quartus_pgm, "-m", "jtag", "-o", f"p;{sof_file}"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            log.error("quartus_pgm failed: %s", result.stderr)
        else:
            log.info("FPGA programmed successfully ✅")


class _Abbrev(str):
    """Shown as it is by pprint (no quotes)."""
    def __repr__(self) -> str:
        return str(self)


def _flatten_outputs(ret: Any) -> Tuple[Any, ...]:
    """
    Python results as output values, in port order: a complex value becomes
    (real, imag), a list its elements, and a list of complex values its real
    parts, then its imaginary parts (<xs>_re_*, <xs>_im_*).
    """
    out: List[Any] = []

    def add(v: Any) -> None:
        if isinstance(v, complex):
            out.extend([v.real, v.imag])
        elif isinstance(v, (list, tuple)) and any(isinstance(e, complex) for e in v):
            out.extend([complex(e).real for e in v] + [complex(e).imag for e in v])
        elif isinstance(v, (list, tuple)):
            for e in v:
                add(e)
        else:
            out.append(v)

    for v in ret if isinstance(ret, tuple) else (ret,):
        add(v)
    return tuple(out)


# --------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    log.info("py2ver executed as script")
