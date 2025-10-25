# py2ver.py
"""
py2ver: Python → AST → IR → Verilog generator + TB and Quartus project scaffolding.

This version emits the UART split modules and appends QSF assignments:
  - uart/uart_baud_gen.sv
  - uart/uart_rx_8n1.sv
  - uart/uart_tx_8n1.sv
  - uart/uart_fifo4x8.sv
  - uart_transceiver.sv
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
    """Main pipeline for Python → Verilog + optional testbench + Quartus."""

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

        self.input_args_bits = sum(self.input_args_width_list)
        self.output_bits = sum(self.output_args_width_list)
        self.attr = attr

        log.info(
            "IR built: module=%s, inputs=%s (%d bits), outputs=%s (%d bits)",
            self.t_name, self.input_args_list, self.input_args_bits,
            self.output_args_list, self.output_bits
        )

        renderer = Renderer(template_dir)
        verilog_text = renderer.render_module(ir)

        self.outdir = Path(os.getcwd())
        self.hdl_dir = self.outdir / "hdl"
        self.hdl_dir.mkdir(exist_ok=True)

        hdl_path = self.hdl_dir / "main.v"
        hdl_path.write_text(verilog_text, encoding="utf-8")
        log.info("Generated HDL written to %s", hdl_path)

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
        }

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
    # Hardware flow
    # ------------------------------------------------------------------------

    def HW(self, syn_attr: Dict[str, Any]) -> Callable[..., int]:
        self.syn_attr = syn_attr
        self.createHWproject()
        return self.hw_run

    def hw_run(self) -> int:
        return 1

    def createHWproject(self) -> None:
        syn_tool_dir = self.syn_attr.get("QUARTUS_DIR")
        if not syn_tool_dir:
            log.error("Missing 'QUARTUS_DIR'")
            return

        quartus_sh = os.path.join(syn_tool_dir, "quartus_sh")
        if not os.path.exists(quartus_sh):
            log.error("quartus_sh not found: %s", quartus_sh)
            return

        if self._hw_root.is_dir():
            shutil.rmtree(self._hw_root)
        project_dir = self._project_dir
        project_dir.mkdir(parents=True, exist_ok=True)

        project_name = self.t_name

        (project_dir / f"{project_name}.qpf").write_text(
            f"PROJECT_REVISION = {project_name}\n", encoding="utf-8"
        )

        qsf_path = project_dir / f"{project_name}.qsf"
        self.get_template("hw/de0_nano_qsf.txt") \
            .stream({}).dump(str(qsf_path))

        # Emit UART split modules
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
        self.get_template("hw/uart/uart_transceiver_top.txt").stream({
            "in_clk_freq": DEFAULT_CLK_FREQ,
            "baud_rate":   DEFAULT_BAUD_RATE,
            "reg_width_tx": self.output_bits,
            "reg_width_rx": self.input_args_bits
        }).dump(str(project_dir / "uart_transceiver.sv"))

        # ✅ Append UART leafs to QSF
        with qsf_path.open("a", encoding="utf-8") as qsf:
            qsf.write("\n# UART split modules\n")
            qsf.write('set_global_assignment -name SYSTEMVERILOG_FILE uart/uart_baud_gen.sv\n')
            qsf.write('set_global_assignment -name SYSTEMVERILOG_FILE uart/uart_rx_8n1.sv\n')
            qsf.write('set_global_assignment -name SYSTEMVERILOG_FILE uart/uart_tx_8n1.sv\n')
            qsf.write('set_global_assignment -name SYSTEMVERILOG_FILE uart/uart_fifo4x8.sv\n')
            qsf.write('set_global_assignment -name SYSTEMVERILOG_FILE uart_transceiver.sv\n')

        self.get_template("hw/de0_nano_sdc.txt") \
            .stream({}).dump(str(project_dir / "DE0_Nano.SDC"))

        self.get_template("hw/delayed_registers.txt") \
            .stream(self.syn_data).dump(str(project_dir / "delayed_registers.sv"))

        self.get_template("hw/de0_nano_top.txt") \
            .stream(self.syn_data).dump(str(project_dir / "DE0_Nano.v"))

        shutil.copy(self.hdl_dir / "main.v", project_dir / f"{project_name}.v")

        log.info("Quartus project created at %s", project_dir)

        log.info("Running Quartus compile...")
        result = subprocess.run(
            [quartus_sh, "--flow", "compile", project_name],
            cwd=project_dir,
            text=True,
        )
        if result.returncode != 0:
            log.error("Quartus compile failed: rc=%d", result.returncode)
            return

        out_dir = project_dir / "output_files"
        if out_dir.exists():
            files = sorted(p.name for p in out_dir.iterdir() if p.is_file())
            log.info("Artifacts: %s", files)
        else:
            log.warning("No output_files/ directory found.")

        self._program_fpga(syn_tool_dir, project_dir, project_name)

    # ------------------------------------------------------------------------
    # Testbench
    # ------------------------------------------------------------------------

    def TB(self) -> Callable[..., Tuple[int, ...]]:
        return self.tb_fun

    def gen_tb(self, *in_arg: Any) -> None:
        key_val = [
            (name, in_arg[i] if i < len(in_arg) else 0)
            for i, name in enumerate(self.input_args_list)
        ]

        tb_out = self.get_template("tb.txt").render({
            "period": DEFAULT_TB_PERIOD,
            "in_args": key_val,
            "out_args": self.output_args_list
        })
        Path("tb.py").write_text(tb_out, encoding="utf-8")

    def tb_fun(self, *in_arg: Any) -> Tuple[int, ...]:
        if RESULTS_PATH.exists():
            try:
                RESULTS_PATH.unlink()
            except OSError as e:
                log.warning(e)

        self.gen_tb(*in_arg)
        tb_runner.test_runner(self.t_name)

        if RESULTS_PATH.exists():
            with RESULTS_PATH.open("rb") as handle:
                payload = pickle.load(handle)
                d = payload if isinstance(payload, dict) else ast.literal_eval(payload)
        else:
            d = {}

        outs = []
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
