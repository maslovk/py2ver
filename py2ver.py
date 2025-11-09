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
import inspect
import logging
import os
import pickle
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from jinja2 import Environment, FileSystemLoader, StrictUndefined

import tb_runner
from renderer import Renderer
from visitor import FunctionVisitor

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

log = logging.getLogger(__name__)


class Py2ver:
    """Main pipeline for Python → Verilog + optional testbench + Quartus + UART HW I/O."""

    t_name: str = ""

    def __init__(self, func: Callable, attr: Dict[str, Any],
                 template_dir: Path = DEFAULT_TEMPLATE_DIR) -> None:
        log.debug("Initializing Py2ver (func=%s)", getattr(func, "__name__", func))

        try:
            source_foo = inspect.getsource(func)
        except Exception as e:
            raise RuntimeError(f"Unable to load source for {func}: {e}") from e

        tree = ast.parse(source_foo)
        f_visitor = FunctionVisitor(attr)
        ir = f_visitor.visit(tree)

        self.t_name = ir.name
        self.input_args_list = [p.name for p in ir.inputs if p.kind != "clk"]
        self.output_args_list = [p.name for p in ir.outputs]
        self.input_args_width_list = [p.width for p in ir.inputs if p.kind != "clk"]
        self.output_args_width_list = [p.width for p in ir.outputs]

        # Raw IR bit widths (core widths)
        self.input_args_bits = sum(self.input_args_width_list)
        self.output_bits = sum(self.output_args_width_list)

        # IMPORTANT: use visitor's attr (includes acc_0, acc_1, ...)
        self.attr = f_visitor.attr

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
                {"name": n, "width": w, "delay": 0}
                for n, w in zip(self.input_args_list, self.input_args_width_list)
            ],
            "inputs_size": self.input_args_bits,
            "outputs": [
                {"name": n, "width": w, "delay": 0}
                for n, w in zip(self.output_args_list, self.output_args_width_list)
            ],
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
    def tosigned(n: int, nbits: int) -> int:
        mask = (1 << nbits) - 1
        n &= mask
        sign_bit = 1 << (nbits - 1)
        return (n ^ sign_bit) - sign_bit

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
            v = int(v) & ((1 << width) - 1)
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

        Returns:
            hw_fun(*args) -> tuple of outputs (same shape as TB()).
        """
        self.syn_attr = syn_attr
        self._hw_ok = self.createHWproject()
        return self.hw_fun

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
            log.error("HW() called but Quartus project/compile failed; "
                      "falling back to simulation result.")
            return self.tb_fun(*in_arg)

        if serial is None:
            log.error("pyserial not available; falling back to simulation result.")
            return self.tb_fun(*in_arg)

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
                    log.error("UART: expected %d bytes, got %d; "
                              "falling back to simulation result.",
                              output_bytes_expected, len(resp))
                    return self.tb_fun(*in_arg)

                # --- Decode ---
                outs = self._unpack_outputs(resp)
                log.info("Decoded HW outputs: %s", outs)
                return outs

        except Exception as e:
            log.error("UART communication failed (%s); falling back to simulation.", e)
            return self.tb_fun(*in_arg)

    def hw_run(self) -> int:
        """
        Legacy HW runner placeholder (kept for compatibility).
        Not used by the new HW()/hw_fun path.
        """
        return 1

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
            log.error("Missing 'QUARTUS_DIR'")
            return False

        quartus_sh = os.path.join(syn_tool_dir, "quartus_sh")
        if not os.path.exists(quartus_sh):
            log.error("quartus_sh not found: %s", quartus_sh)
            return False

        if self._hw_root.is_dir():
            shutil.rmtree(self._hw_root)
        project_dir = self._project_dir
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

        # ---- Compile ----
        log.info("Running Quartus compile...")
        result = subprocess.run(
            [quartus_sh, "--flow", "compile", project_name],
            cwd=project_dir,
            text=True,
        )
        if result.returncode != 0:
            log.error("Quartus compile failed: rc=%d", result.returncode)
            return False

        out_dir = project_dir / "output_files"
        if out_dir.exists():
            files = sorted(p.name for p in out_dir.iterdir() if p.is_file())
            log.info("Artifacts: %s", files)
        else:
            log.warning("No output_files/ directory found.")

        self._program_fpga(syn_tool_dir, project_dir, project_name)
        return True

    # ------------------------------------------------------------------------
    # Testbench
    # ------------------------------------------------------------------------

    def TB(self) -> Callable[..., Tuple[int, ...]]:
        return self.tb_fun

    def gen_tb(self, *in_arg: Any) -> None:
        # Map input names to argument values
        key_val = [
            (name, in_arg[i] if i < len(in_arg) else 0)
            for i, name in enumerate(self.input_args_list)
        ]
        log.info("gen_tb: generating tb.py with in_args=%s", key_val)

        tb_out = self.get_template("tb.txt").render({
            "period": DEFAULT_TB_PERIOD,
            "in_args": key_val,
            "out_args": self.output_args_list
        })

        # Optional: tiny snippet to confirm what ended up in the file
        first_line = tb_out.splitlines()[0] if tb_out else ""
        log.debug("gen_tb: first line of tb.py: %s", first_line)

        Path("tb.py").write_text(tb_out, encoding="utf-8")

    def tb_fun(self, *in_arg: Any) -> Tuple[int, ...]:
        # Log TB invocation so we can see arguments clearly
        log.info("TB called with args: %s", in_arg)

        # Clean previous results
        if RESULTS_PATH.exists():
            try:
                RESULTS_PATH.unlink()
            except OSError as e:
                log.warning(e)

        # Ensure a clean sim build directory so tb.py isn't stale
        sim_build = Path("sim_build")
        if sim_build.exists():
            try:
                shutil.rmtree(sim_build)
            except OSError as e:
                log.warning("Failed to remove sim_build: %s", e)

        # 🔥 Remove stale compiled tb.py so Python can't reuse old bytecode
        pycache_dir = Path("__pycache__")
        if pycache_dir.exists():
            for p in pycache_dir.glob("tb.*.pyc"):
                try:
                    log.debug("Removing stale pycache file: %s", p)
                    p.unlink()
                except OSError as e:
                    log.warning("Failed to remove %s: %s", p, e)

        # Generate testbench for this specific input set
        self.gen_tb(*in_arg)

        # Run cocotb-based simulation
        tb_runner.test_runner(self.t_name)

        # Load results written by tb.py
        if RESULTS_PATH.exists():
            with RESULTS_PATH.open("rb") as handle:
                payload = pickle.load(handle)
                d = payload if isinstance(payload, dict) else ast.literal_eval(payload)
        else:
            d = {}

        outs: List[int] = []
        for n, w in zip(self.output_args_list, self.output_args_width_list):
            v = d.get(n, 0)
            if self.attr.get(n, {}).get("signed") == 1:
                v = self.tosigned(v, w)
            outs.append(v)
        return tuple(outs)

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
