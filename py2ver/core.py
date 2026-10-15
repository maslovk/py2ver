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
import re
import inspect
import logging
import os
import pickle
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from . import tb_runner
from .renderer import Renderer
from .syn_report import SynReport, read_report
from .pipeline import pipeline_ir
from .semantics import exact_ir
from .inline import inline_helpers
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

    def __init__(self, func: Callable, attr: Dict[str, Any],
                 template_dir: Path = DEFAULT_TEMPLATE_DIR, pipeline: int = 0) -> None:
        log.debug("Initializing Py2ver (func=%s)", getattr(func, "__name__", func))

        try:
            source_foo = inspect.getsource(func)
        except Exception as e:
            raise RuntimeError(f"Unable to load source for {func}: {e}") from e

        tree = ast.parse(source_foo)
        # Inline calls to your own helper functions and module-level int constants
        tree = inline_helpers(tree, getattr(func, "__globals__", {}))
        f_visitor = FunctionVisitor(attr)
        ir = f_visitor.visit(tree)

        # Rewrite expressions so the Verilog computes what Python computes
        ir, exact_attr = exact_ir(ir, f_visitor.attr)
        f_visitor.attr.clear()
        f_visitor.attr.update(exact_attr)

        # Explicit state: an input with {'next': '<output>'} becomes a register
        # loaded from that output every clock and reset to 'init' (default 0).
        self.func = func
        self.state_map, self.state_init = self._parse_state(attr)
        if self.state_map:
            ir = self._lower_state(ir, f_visitor.attr)

        # Automatic pipelining: `pipeline` register stages, balanced so every
        # output appears exactly `pipeline` clocks after its inputs.
        self.pipeline_stages = int(pipeline)
        if self.pipeline_stages:
            if self.state_map:
                raise NotImplementedError(
                    "pipeline cannot be combined with state registers ('next') yet."
                )
            res = pipeline_ir(ir, f_visitor.attr, self.pipeline_stages)
            ir = res.ir
            f_visitor.attr.clear()
            f_visitor.attr.update(res.attr)
            log.info(
                "Pipelined into %d stages: +%d register bits, est. stage delays %s",
                self.pipeline_stages + 1, res.register_bits,
                [round(d, 2) for d in res.stage_delay],
            )
        self.ir = ir

        self.has_clk = bool(ir.has_clk)

        self.t_name = ir.name
        self.input_args_list = [p.name for p in ir.inputs if p.kind not in ("clk", "rst", "ce")]
        self.output_args_list = [p.name for p in ir.outputs]
        self.input_args_width_list = [p.width for p in ir.inputs if p.kind not in ("clk", "rst", "ce")]
        self.output_args_width_list = [p.width for p in ir.outputs]

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
            "has_clk": self.has_clk,
            "has_state": bool(self.state_map),
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
    def _parse_state(attr: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, int]]:
        state_map: Dict[str, str] = {}
        state_init: Dict[str, int] = {}
        for name, meta in attr.items():
            if isinstance(meta, dict) and "next" in meta:
                state_map[name] = str(meta["next"])
                state_init[name] = int(meta.get("init", 0))
        return state_map, state_init

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

        if ir.has_clk:
            raise NotImplementedError(
                "State registers ('next') cannot be combined with 'reg' signals yet."
            )
        for reserved in ("clk", "rst", "ce"):
            if reserved in in_names or reserved in out_names:
                raise ValueError(f"'{reserved}' is a reserved port name in designs with state.")
        for s, n in self.state_map.items():
            if s not in in_names:
                raise ValueError(f"State '{s}' must be a function argument; arguments: {in_names}")
            if n not in out_names:
                raise ValueError(
                    f"Next state '{n}' of '{s}' must be a returned value; returned: {out_names}"
                )

        assigns = list(ir.assigns)
        for p in ir.inputs:
            if p.name not in self.state_map:
                continue
            n = self.state_map[p.name]
            init = self.state_init[p.name] & ((1 << p.width) - 1)
            # Literal takes the next value's signedness so n is extended by its
            # own sign, matching how PY_stream() wraps it.
            n_signed = int(attr.get(n, {}).get("signed", 0)) == 1
            lit = f"{p.width}'{'s' if n_signed else ''}d{init}"
            # The hold value takes the same signedness too, or the whole
            # selection would turn unsigned and zero-extend a signed n
            hold = f"$signed({p.name})" if n_signed else f"$unsigned({p.name})"
            assigns.append(
                Assignment(left=p.name, right=f"rst ? {lit} : (ce ? {n} : {hold})", is_reg=True)
            )

        inputs = [p for p in ir.inputs if p.name not in self.state_map]
        inputs.append(Port(name="clk", width=1, signed=False, kind="clk"))
        inputs.append(Port(name="rst", width=1, signed=False, kind="rst"))
        inputs.append(Port(name="ce", width=1, signed=False, kind="ce"))

        return ModuleIR(
            name=ir.name,
            inputs=inputs,
            outputs=list(ir.outputs),
            assigns=assigns,
            has_clk=True,
        )

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
        return self.hw_fun

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

        input_bytes = self._pack_inputs(*in_arg)
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
        return self.hw_sim_fun

    def hw_sim_fun(self, vectors: List[Tuple[Any, ...]]) -> List[Tuple[int, ...]]:
        vectors = [tuple(v) for v in vectors]
        if not vectors:
            return []
        project_dir = self.gen_hw_project()
        tx_bytes = (self.tx_width + 7) // 8
        requests = [list(self._pack_inputs(*v)) for v in vectors]

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
        return self.tb_fun

    def TB_batch(self) -> Callable[[List[Tuple[Any, ...]]], List[Tuple[int, ...]]]:
        return self.tb_batch_fun

    def PY_batch(self) -> Callable[[List[Tuple[Any, ...]]], List[Tuple[int, ...]]]:
        return self.py_batch_fun

    def py_batch_fun(self, vectors: List[Tuple[Any, ...]]) -> List[Tuple[int, ...]]:
        """
        Python reference for tb_batch_fun(): the original function's results,
        wrapped to each output's declared width and sign (int() truncates /).
        """
        if self.state_map:
            raise ValueError("This design has state registers ('next' in attr); use PY_stream().")
        outs: List[Tuple[int, ...]] = []
        for vec in vectors:
            ret = self.func(*vec)
            ret = ret if isinstance(ret, tuple) else (ret,)
            outs.append(tuple(
                self._wrap(int(v), w, int(self.attr.get(n, {}).get("signed", 0)) == 1)
                for v, n, w in zip(ret, self.output_args_list, self.output_args_width_list)
            ))
        return outs

    def TB_stream(self) -> Callable[[List[Any]], List[Tuple[int, ...]]]:
        return self.tb_stream_fun

    def PY_stream(self) -> Callable[[List[Any]], List[Tuple[int, ...]]]:
        return self.py_stream_fun

    def _input_vector(self, in_arg: Tuple[Any, ...]) -> Tuple[int, ...]:
        # Missing trailing arguments default to 0
        return tuple(
            int(in_arg[i]) if i < len(in_arg) else 0
            for i in range(len(self.input_args_list))
        )

    def _stream_vectors(self, samples: List[Any]) -> List[Tuple[int, ...]]:
        if not self.state_map:
            raise ValueError("Streams need state registers ('next' in attr); use TB_batch().")
        # A scalar sample is accepted for single-input designs
        return [
            self._input_vector(tuple(s) if isinstance(s, (tuple, list)) else (s,))
            for s in samples
        ]

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
        if self.state_map:
            raise ValueError(
                "This design has state registers ('next' in attr); use TB_stream()."
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
        Python reference for tb_stream_fun(): calls the original function once
        per sample, feeding each next-state output back into its state argument
        with the same width/sign wrapping as the hardware registers.
        """
        vecs = self._stream_vectors(list(samples))

        params = list(inspect.signature(self.func).parameters)
        expected = set(self.input_args_list) | set(self.state_map)
        if set(params) != expected:
            raise ValueError(
                f"PY_stream() needs arguments {sorted(expected)}; function has {params}"
            )

        def _meta(name: str) -> Tuple[int, bool]:
            m = self.attr.get(name, {})
            return int(m.get("width", 1)), int(m.get("signed", 0)) == 1

        state = {
            name: self._wrap(self.state_init[name], *_meta(name))
            for name in self.state_map
        }
        outs: List[Tuple[int, ...]] = []
        for vec in vecs:
            ret = self.func(**dict(zip(self.input_args_list, vec)), **state)
            ret = ret if isinstance(ret, tuple) else (ret,)
            row = {
                name: self._wrap(int(v), *_meta(name))
                for name, v in zip(self.output_args_list, ret)
            }
            outs.append(tuple(row[n] for n in self.output_args_list))
            state = {
                name: self._wrap(row[nxt], *_meta(name))
                for name, nxt in self.state_map.items()
            }
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


# --------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    log.info("py2ver executed as script")
