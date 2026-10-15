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
from .fixed import DEFAULT_CONST_BITS, build_model, lower_fixed, quantize, uses_fixed
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


class Py2ver:
    """Main pipeline for Python → Verilog + optional testbench + Quartus + UART HW I/O."""

    t_name: str = ""

    def __init__(self, func: Callable, attr: Dict[str, Any] | None = None,
                 template_dir: Path = DEFAULT_TEMPLATE_DIR, pipeline: int = 0,
                 const_bits: int = DEFAULT_CONST_BITS, samples: List[Any] | None = None,
                 width: int | None = None, max_error: Any = None) -> None:
        """
        func: the function, as written. attr: widths, formats and state (see
        the README). With `samples` (example inputs), formats missing from
        attr (or all of them, without attr) are chosen by auto_fixed() from
        the ranges seen: `width` bits per value (default 16), or the fewest
        bits that keep the outputs within `max_error` of the float function.
        The chosen attr is printed.
        """
        log.debug("Initializing Py2ver (func=%s)", getattr(func, "__name__", func))
        if samples is not None:
            attr = auto_fixed(func, samples, attr, width=width or 16, max_error=max_error,
                              const_bits=const_bits)
            lines = "".join(f"    {k!r}: {pprint.pformat(v, sort_dicts=False, width=200)},\n"
                            for k, v in attr.items())
            print(f"py2ver: formats chosen for {getattr(func, '__name__', 'the function')}"
                  f" (pass this as attr to keep them):\nattr = {{\n{lines}}}")

        try:
            source_foo = inspect.getsource(func)
        except Exception as e:
            raise RuntimeError(f"Unable to load source for {func}: {e}") from e

        tree = ast.parse(source_foo)
        # Types from annotations (U[8], State(...), ...); attr entries merge over them
        tree, ann = read_annotations(tree, func)
        if attr is None and not ann:
            raise ValueError("Give attr (widths and formats), annotations, or samples=<example "
                             "inputs> to choose them automatically")
        attr = normalize_attr(attr)
        attr = {k: dict(ann.get(k, {}), **attr.get(k, {})) for k in dict.fromkeys([*ann, *attr])}
        # Inline calls to your own helper functions and module-level int constants
        tree = inline_helpers(tree, getattr(func, "__globals__", {}))
        # Python names that are Verilog keywords (buf, time, ...) get a '_' suffix
        tree, self._hw_name = rename_keywords(tree)
        self._py_name = {v: k for k, v in self._hw_name.items()}
        attr = {
            self._hw_name.get(k, k): (
                dict(v, next=self._hw_name.get(v["next"], v["next"]))
                if isinstance(v, dict) and "next" in v else v
            )
            for k, v in attr.items()
        }
        # Fixed point ('frac' in attr) becomes integer code on raw values, and
        # the same code run in Python is the bit-exact reference (see fixed.py)
        attr = {k: self._raw_init(v) for k, v in attr.items()}
        ns = _namespace(func)
        tree, fx_attr, self._fx_types = lower_fixed(tree, attr, ns, const_bits, self._py_name)
        attr.update(fx_attr)
        self._model = build_model(tree, attr, ns) if uses_fixed(attr) else None
        f_visitor = FunctionVisitor(attr)
        ir = f_visitor.visit(tree)

        # Rewrite expressions so the Verilog computes what Python computes
        ir, exact_attr = exact_ir(ir, f_visitor.attr)
        f_visitor.attr.clear()
        f_visitor.attr.update(exact_attr)

        # Explicit state: an input with {'next': '<output>'} becomes a register
        # loaded from that output every clock and reset to 'init' (default 0).
        self.func = func
        # Parameters as ports: (param, [ports] for a list argument, or None)
        self.arg_groups = list(f_visitor.arg_groups)
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
                                     state=self.state_map or None)
            ir = self._pipe.ir
            f_visitor.attr.clear()
            f_visitor.attr.update(self._pipe.attr)
            log.info(
                "Pipelined into %d stages: +%d register bits, est. stage delays %s%s",
                self.pipeline_stages + 1, self._pipe.register_bits,
                [round(d, 2) for d in self._pipe.stage_delay],
                f", state loop in stage {self._pipe.loop_stage}" if self.state_map else "",
            )
        if self.state_map or self.rams:
            ir = self._lower_state(ir, f_visitor.attr)
        self.ir = ir

        self.has_clk = bool(ir.has_clk)
        self._has_ce = any(p.kind == "ce" for p in ir.inputs)

        self.t_name = ir.name
        control = ("clk", "rst", "ce", "vin")
        self.input_args_list = [p.name for p in ir.inputs if p.kind not in control]
        self.output_args_list = [p.name for p in ir.outputs if p.kind != "vout"]
        self.input_args_width_list = [p.width for p in ir.inputs if p.kind not in control]
        self.output_args_width_list = [p.width for p in ir.outputs if p.kind != "vout"]
        self._has_vin = any(p.kind == "vin" for p in ir.inputs)

        # Fixed-point parameters (meta) and outputs (fraction bits): the API
        # takes and gives numbers, the ports and the model carry raw values
        self._fx_params = {p: attr[p] for p, _ in self.arg_groups
                           if attr.get(p, {}).get("frac") is not None}
        self._fx_out = {n: f for n in self.output_args_list
                        if (f := self._fx_frac(n)) is not None}

        # Raw IR bit widths (core widths)
        self.input_args_bits = sum(self.input_args_width_list)
        self.output_bits = sum(self.output_args_width_list)

        # IMPORTANT: use visitor's attr (includes acc_0, acc_1, ...)
        self.attr = f_visitor.attr

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
        # pass full attr (with SSA temps) into renderer
        verilog_text = renderer.render_module(ir, self.attr)

        self.outdir = Path(os.getcwd())
        self.hdl_dir = self.outdir / "hdl"
        self.hdl_dir.mkdir(exist_ok=True)

        hdl_path = self.hdl_dir / "main.v"
        hdl_path.write_text(verilog_text, encoding="utf-8")
        log.info("Generated HDL written to %s", hdl_path)

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
            "has_state": bool(self.state_map or self.rams),
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

    # --------------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------------

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

        inputs = [p for p in ir.inputs if p.name not in self.state_map]
        have = {p.name for p in inputs}
        for name, kind in (("clk", "clk"), ("rst", "rst"), ("ce", "ce")):
            if name not in have:
                inputs.append(Port(name=name, width=1, signed=False, kind=kind))

        return ModuleIR(
            name=ir.name,
            inputs=inputs,
            outputs=list(ir.outputs),
            assigns=assigns,
            has_clk=True,
            items=list(ir.items) + self._ram_items(),
        )

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
        P, pb = self.ram_phases, self._phase_bits
        compute = self._ram_schedule["compute"]
        if P > 1:
            items += [
                f"// {P} clocks per sample: RAM reads, then compute, then writes",
                f"reg [{pb - 1}:0] __phase;",
                "always @(posedge clk) begin",
                f"  if (rst) __phase <= {pb}'d0;",
                f"  else if (ce) __phase <= (__phase == {pb}'d{P - 1}) ? {pb}'d0 : __phase + {pb}'d1;",
                "end",
            ]
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
        self._hw_ok = self.createHWproject()
        if not self._hw_ok:
            self._hw_unavailable(getattr(self, "_hw_error", "board project could not be built (see log)"))
        return self._fx_rows(self.hw_fun)

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
        project_dir = self.gen_hw_project()
        tx_bytes = (self.tx_width + 7) // 8
        requests = [list(self._pack_inputs(*self._flatten_args(v))) for v in vectors]

        tb_src = self.get_template("tb_hw.txt").render({
            "baud": DEFAULT_BAUD_RATE,
            "tx_bytes": tx_bytes,
            "requests": repr(requests),
        })
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

    def gen_hw_project(self) -> Path:
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

        self.get_template("hw/delayed_registers.txt") \
            .stream(self.syn_data).dump(str(project_dir / "delayed_registers.sv"))

        # DE0_Nano top you provided
        self.get_template("hw/de0_nano_top.txt") \
            .stream(self.syn_data).dump(str(project_dir / "DE0_Nano.v"))

        # Copy generated core HDL
        shutil.copy(self.hdl_dir / "main.v", project_dir / f"{project_name}.v")

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

        project_dir = self.gen_hw_project()
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

    def stream_wrapper(self) -> Path:
        """
        Write hdl/<name>_stream.v: the module behind a valid/ready interface
        (clk, rst, s_valid/s_ready/s_<input>, m_valid/m_ready/m_<output>).
        """
        mode = self._stream_mode()

        def rng(name: str) -> str:
            m = self.attr.get(name, {})
            w = int(m.get("width", 1))
            sig = "signed " if int(m.get("signed", 0)) == 1 else ""
            return f"{sig}[{w - 1}:0]"

        ctx = {
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
        path = self.hdl_dir / f"{self.t_name}_stream.v"
        path.write_text(self.get_template("stream_wrapper.txt").render(ctx), encoding="utf-8")
        log.info("Stream wrapper (%s) written to %s", mode, path)
        return path

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
        if self.state_map or self.rams:
            vecs = self._stream_vectors(vectors)
        else:
            vecs = [self._input_vector(tuple(v)) for v in vectors]
        masks = [(1 << w) - 1 for w in self.input_args_width_list]
        vecs = [tuple(v & m for v, m in zip(vec, masks)) for vec in vecs]

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
            "max_cycles": 20 * len(vecs) + 100 + 4 * self.pipeline_stages,
            "in_names": self.input_args_list,
            "out_args": self.output_args_list,
            "out_meta": self.out_meta,
        })
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
        if len(rows) != len(vecs):
            raise RuntimeError(
                f"Stream wrapper returned {len(rows)} results for {len(vecs)} inputs"
            )
        return self._rows_to_outputs(rows, len(vecs))

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
        tool_dir = syn_attr.get("QUARTUS_DIR")
        quartus_sh = (
            os.path.join(tool_dir, "quartus_sh") if tool_dir else shutil.which("quartus_sh")
        )
        if not quartus_sh or not os.path.exists(quartus_sh):
            raise RuntimeError(
                "quartus_sh not found; set 'QUARTUS_DIR' to Quartus's bin directory "
                f"(looked for: {quartus_sh or 'quartus_sh on PATH'})"
            )

        project_dir = self.gen_syn_project(syn_attr)
        revision = f"{self.t_name}_syn_top"

        log.info("Running Quartus synthesis/fit/timing for %s...", self.t_name)
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
            project_dir, revision, self.t_name,
            float(syn_attr.get("CLK_PERIOD_NS", DEFAULT_SYN_PERIOD_NS)),
        )
        log.info("Synthesis report for %s:\n%s", self.t_name, report)
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
        shutil.copy(self.hdl_dir / "main.v", project_dir / f"{self.t_name}.v")

        log.info("Synthesis project created at %s", project_dir)
        return project_dir

    # ------------------------------------------------------------------------
    # Testbench
    # ------------------------------------------------------------------------

    def TB(self) -> Callable[..., Tuple[int, ...]]:
        return self._fx_rows(self.tb_fun)

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
        if self.state_map or self.rams:
            raise ValueError("This design has state ('next' or 'memory' in attr); use PY_stream().")
        func = self._model or self.func
        return [self._py_row(func(*self._py_args(vec))) for vec in vectors]

    def _py_row(self, ret: Any) -> Tuple[int, ...]:
        return tuple(
            self._wrap(int(v), w, int(self.attr.get(n, {}).get("signed", 0)) == 1)
            for v, n, w in zip(_flatten_outputs(ret), self.output_args_list,
                               self.output_args_width_list)
        )

    def PY_float(self) -> Callable[[List[Any]], List[Tuple[Any, ...]]]:
        """
        The original function as written (floats), on the same inputs as the
        hardware after they are quantized: the ideal a fixed-point design
        approximates. Takes vectors like PY_batch(), or samples like
        PY_stream() for a design with state.
        """
        return self.py_float_fun

    def py_float_fun(self, vectors: List[Any]) -> List[Tuple[Any, ...]]:
        if self.state_map or self.rams:
            return self._stream_reference(vectors, exact=False)
        params = [p for p, _ in self.arg_groups]
        return [_flatten_outputs(self.func(*[self._float_value(p, self._py_value(p, v))
                                             for p, v in zip(params, vec)]))
                for vec in vectors]

    def quantization_error(self, vectors: List[Any]) -> Dict[str, Tuple[float, float]]:
        """
        Per output: (max abs error, RMS error) of the bit-exact results
        (PY_batch / PY_stream, which the hardware matches) against PY_float().
        """
        stream = bool(self.state_map or self.rams)
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
        m = self._fx_params.get(param)
        if m is None:
            return value
        conv = lambda v: quantize(v, m["frac"], m["width"], bool(m.get("signed")),  # noqa: E731
                                  bool(m.get("round")), bool(m.get("saturate")))
        return [conv(v) for v in value] if isinstance(value, (list, tuple)) else conv(value)

    def _float_value(self, param: str, raw: Any) -> Any:
        """Raw value(s) of a fixed-point parameter as numbers."""
        m = self._fx_params.get(param)
        if m is None:
            return raw
        conv = lambda v: int(v) / (1 << m["frac"])  # noqa: E731
        return [conv(v) for v in raw] if isinstance(raw, (list, tuple)) else conv(raw)

    def _py_args(self, vec: Tuple[Any, ...]) -> Tuple[Any, ...]:
        params = [p for p, _ in self.arg_groups]
        return tuple(self._py_value(p, v) for p, v in zip(params, vec)) + tuple(vec[len(params):])

    def _fx_rows(self, fn: Callable) -> Callable:
        """fn with its results' fixed-point outputs as numbers (raw / 2**frac)."""
        if not self._fx_out:
            return fn
        fracs = [self._fx_out.get(n) for n in self.output_args_list]

        def row(r):
            return tuple(v if f is None else int(v) / (1 << f) for v, f in zip(r, fracs))

        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            out = fn(*args, **kwargs)
            return [row(r) for r in out] if isinstance(out, list) else row(out)

        return wrapped

    def _flatten_args(self, in_arg: Tuple[Any, ...]) -> Tuple[Any, ...]:
        """Function arguments (a list argument as a list) -> one value per input port."""
        groups = [(p, ports) for p, ports in self.arg_groups
                  if (ports or [p])[0] in self.input_args_list]   # state lives in registers
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
        if not (self.state_map or self.rams):
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
        if self.state_map or self.rams:
            raise ValueError(
                "This design has state ('next' or 'memory' in attr); use TB_stream()."
            )
        vectors = [tuple(v) for v in vectors]
        log.info("TB called with %d vector(s)", len(vectors))
        if not vectors:
            return []
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
        return self._simulate(len(samples), lambda: self.gen_tb_stream(samples))

    def py_stream_fun(self, samples: List[Any]) -> List[Tuple[int, ...]]:
        """
        Python reference for tb_stream_fun(): calls the function (for a
        fixed-point design, its bit-exact model) once per sample, feeding each
        next-state output back into its state argument with the same
        width/sign wrapping as the hardware registers.
        """
        return self._stream_reference(samples, exact=True)

    def _stream_reference(self, samples: List[Any], exact: bool) -> List[Tuple[Any, ...]]:
        """exact: raw values wrapped like the hardware; else the original
        function on numbers, nothing wrapped (PY_float)."""
        samples = list(samples)
        self._stream_vectors(samples)  # validates
        groups = dict(self.arg_groups)

        model = exact and self._model is not None
        func = self._model if model else self.func
        py = (lambda name: name) if model else (lambda name: self._py_name.get(name, name))  # noqa: E731
        params = list(inspect.signature(func).parameters)
        if params != [py(p) for p, _ in self.arg_groups]:
            raise ValueError(f"PY_stream() needs arguments {list(groups)}; function has {params}")
        inputs = [p for p, _ in self.arg_groups
                  if p not in self.state_params and p not in self.rams]

        def _meta(name: str) -> Tuple[int, bool]:
            m = self.attr.get(name, {})
            return int(m.get("width", 1)), int(m.get("signed", 0)) == 1

        def keep(name: str, v: Any) -> Any:
            return self._wrap(int(v), *_meta(name)) if exact else v

        def _state_value(param: str, value_of) -> Any:
            ports = groups[param]
            if ports is None:
                return keep(param, value_of(param))
            return [keep(port, value_of(port)) for port in ports]

        def _init(param: str, raw: Any) -> Any:
            return raw if exact else self._float_value(param, raw)

        state = {p: _init(p, _state_value(p, self.state_init.__getitem__))
                 for p in self.state_params}
        # Memories: the function updates the list in place; values wrap like the RAM
        rams = {}
        for mem, ram in self.rams.items():
            init = self._ram_init[mem]
            vals = list(init) if isinstance(init, (list, tuple)) else [init] * ram["length"]
            w, sgn = int(ram["elem"]["width"]), int(ram["elem"]["signed"]) == 1
            rams[mem] = _init(mem, [self._wrap(int(v), w, sgn) for v in vals])
        outs: List[Tuple[Any, ...]] = []
        for sample in samples:
            args = sample if isinstance(sample, tuple) else (sample,)
            kwargs = dict(zip(inputs, args))
            for p in inputs[len(args):]:                     # missing trailing args: 0
                kwargs[p] = 0 if groups[p] is None else [0] * len(groups[p])
            kwargs = {p: self._py_value(p, v) for p, v in kwargs.items()}
            if not exact:
                kwargs = {p: self._float_value(p, v) for p, v in kwargs.items()}
            call = {py(k): v for k, v in {**kwargs, **state, **rams}.items()}
            ret = _flatten_outputs(func(**call))
            if exact:
                for mem, ram in self.rams.items():
                    w, sgn = int(ram["elem"]["width"]), int(ram["elem"]["signed"]) == 1
                    rams[mem][:] = [self._wrap(int(v), w, sgn) for v in rams[mem]]
            row = {name: keep(name, v) for name, v in zip(self.output_args_list, ret)}
            outs.append(tuple(row[n] for n in self.output_args_list))
            state = {p: _state_value(p, lambda port: row[self.state_map[port]])
                     for p in self.state_params}
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

        # 🔥 Remove stale compiled tb.py so Python can't reuse old bytecode
        pycache_dir = self.outdir / "__pycache__"
        if pycache_dir.exists():
            for p in pycache_dir.glob("tb.*.pyc"):
                try:
                    log.debug("Removing stale pycache file: %s", p)
                    p.unlink()
                except OSError as e:
                    log.warning("Failed to remove %s: %s", p, e)

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


def _flatten_outputs(ret: Any) -> Tuple[Any, ...]:
    """
    Python results as output values, in port order: a complex value becomes
    (real, imag) and a list its elements.
    """
    out: List[Any] = []

    def add(v: Any) -> None:
        if isinstance(v, complex):
            out.extend([v.real, v.imag])
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
