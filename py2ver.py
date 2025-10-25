# py2ver.py
"""
py2ver: Python → AST → IR → Verilog generator + TB and Quartus project scaffolding.
Readability & clarity improvements only — no behavior changes, live Quartus output.
"""

import inspect
import ast
import os
import pickle
import subprocess
import shutil
import logging
from pathlib import Path
from typing import List, Tuple, Any, Dict, Callable

from jinja2 import Environment, FileSystemLoader, StrictUndefined

import tb_runner
from visitor import FunctionVisitor
from renderer import Renderer


# --------------------------------------------------------------------
# Constants (avoid magic numbers/strings sprinkled around)
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
                 template_dir: Path = DEFAULT_TEMPLATE_DIR):
        """Extract Python function → IR → emit main.v HDL."""
        log.debug("Initializing Py2ver (func=%s)", getattr(func, "__name__", func))

        # ---- Extract source and parse AST ----
        try:
            source_foo = inspect.getsource(func)
        except Exception as e:
            raise RuntimeError(f"Unable to load source for {func}: {e}")

        tree = ast.parse(source_foo)
        f_visitor = FunctionVisitor(attr)
        ir = f_visitor.visit(tree)

        # Store module name + IO info
        self.t_name = ir.name
        self.input_args_list: List[str] = [p.name for p in ir.inputs if p.kind != 'clk']
        self.output_args_list: List[str] = [p.name for p in ir.outputs]
        self.input_args_width_list: List[int] = [p.width for p in ir.inputs if p.kind != 'clk']
        self.output_args_width_list: List[int] = [p.width for p in ir.outputs]

        self.input_args_bits = sum(self.input_args_width_list)
        self.output_bits = sum(self.output_args_width_list)
        self.attr = attr

        log.info("IR built: module=%s, inputs=%s (%d bits), outputs=%s (%d bits)",
                 self.t_name, self.input_args_list, self.input_args_bits,
                 self.output_args_list, self.output_bits)

        # ---- Verilog rendering ----
        renderer = Renderer(template_dir)
        verilog_text = renderer.render_module(ir)

        # ---- Write HDL to hdl/main.v ----
        self.outdir = Path(os.getcwd())
        self.hdl_dir = self.outdir / "hdl"
        self.hdl_dir.mkdir(exist_ok=True)

        out_path = self.hdl_dir / "main.v"
        out_path.write_text(verilog_text, encoding="utf-8")
        log.info("Generated HDL written to %s", out_path)

        # ---- Jinja env (single unified cache) ----
        self.env = Environment(
            loader=FileSystemLoader(str(template_dir)),  # str() for Path compatibility
            undefined=StrictUndefined,                  # fail fast on missing vars
            autoescape=False,                           # code/text templates
        )
        self._tpl_cache: Dict[str, Any] = {}
        self.template_dir = template_dir

        # Synthesis metadata propagated to templates
        self.syn_data = {
            'top_name': self.t_name,
            'inputs': [{'name': n, 'width': w, 'delay': 0}
                       for n, w in zip(self.input_args_list, self.input_args_width_list)],
            'inputs_size': self.input_args_bits,
            'outputs': [{'name': n, 'width': w, 'delay': 0}
                        for n, w in zip(self.output_args_list, self.output_args_width_list)],
            'outputs_size': self.output_bits
        }

    # ----------------------------------------------------------------
    # Hardware flow — API unchanged
    # ----------------------------------------------------------------
    def HW(self, syn_attr: Dict[str, Any]) -> Callable:
        """Initialize Quartus project. Return callable self.hw_run (unchanged API)."""
        self.syn_attr = syn_attr
        self.createHWproject()
        return self.hw_run

    def hw_run(self) -> int:
        """Dummy placeholder for compatibility."""
        return 1

    def createHWproject(self) -> None:
        """Emit Quartus project files using templates and run compile."""
        syn_tool_dir = self.syn_attr.get('QUARTUS_DIR')
        if not syn_tool_dir:
            log.error("Missing 'QUARTUS_DIR'")
            return

        quartus_sh = os.path.join(syn_tool_dir, "quartus_sh")
        if not os.path.exists(quartus_sh):
            log.error("quartus_sh not found in %s", quartus_sh)
            return

        # Setup hw directory
        hw_root = self.outdir / "hw"
        if hw_root.is_dir():
            log.debug("Cleaning previous hw directory")
            shutil.rmtree(hw_root)
        project_dir = hw_root / self.t_name
        project_dir.mkdir(parents=True, exist_ok=True)

        project_name = self.t_name

        # Write QPF
        (project_dir / f"{project_name}.qpf").write_text(
            f"PROJECT_REVISION = {project_name}\n", encoding="utf-8")

        # QSF
        template_qsf = self.get_template("hw/de0_nano_qsf.txt")
        (project_dir / f"{project_name}.qsf").write_text(
            template_qsf.render({}), encoding="utf-8")

        # UART transceiver
        template_rxtx = self.get_template("hw/uart_transceiver.txt")
        (project_dir / "uart_transceiver.sv").write_text(template_rxtx.render({
            'in_clk_freq': DEFAULT_CLK_FREQ,
            'baud_rate': DEFAULT_BAUD_RATE,
            'reg_width_tx': self.output_bits,
            'reg_width_rx': self.input_args_bits
        }), encoding="utf-8")

        # SDC
        template_sdc = self.get_template("hw/de0_nano_sdc.txt")
        (project_dir / "DE0_Nano.SDC").write_text(template_sdc.render({}), encoding="utf-8")

        # Delay block
        template_delay = self.get_template("hw/delayed_registers.txt")
        (project_dir / "delayed_registers.sv").write_text(
            template_delay.render(self.syn_data), encoding="utf-8")

        # Board-level top
        template_board = self.get_template("hw/de0_nano_top.txt")
        (project_dir / "DE0_Nano.v").write_text(
            template_board.render(self.syn_data), encoding="utf-8")

        # Copy generated core HDL
        shutil.copy(self.hdl_dir / "main.v", project_dir / f"{project_name}.v")

        log.info("Quartus project created at %s", project_dir)

        # ---- Quartus compile (live console output) ----
        log.info("Running Quartus compile (live output)...")
        result = subprocess.run(
            [quartus_sh, "--flow", "compile", project_name],
            cwd=project_dir,
            text=True  # let Python decode output, but don't capture it
        )
        if result.returncode != 0:
            log.error("Quartus compile failed with return code %d", result.returncode)
            return

        # List produced artifacts (usually in ./output_files)
        out_dir = project_dir / "output_files"
        if out_dir.exists():
            produced = sorted(p.name for p in out_dir.iterdir() if p.is_file())
            if produced:
                log.info("Quartus compile OK. Artifacts in %s:\n- %s",
                         out_dir, "\n- ".join(produced))
            else:
                log.warning("Quartus compile OK, but %s is empty.", out_dir)
        else:
            log.warning("Quartus compile OK, but no output_files/ directory found.")

    # ----------------------------------------------------------------
    # Testbench flow — unchanged API
    # ----------------------------------------------------------------
    def TB(self) -> Callable:
        """Return the callable testbench driver."""
        return self.tb_fun

    def get_template(self, filename: str):
        """Get Jinja template (cached by filename)."""
        if filename not in self._tpl_cache:
            self._tpl_cache[filename] = self.env.get_template(filename)
        return self._tpl_cache[filename]

    def gen_tb(self, *in_arg: Any) -> None:
        """Write tb.py from template based on given input args."""
        key_val = [
            (name, in_arg[i] if i < len(in_arg) and in_arg[i] is not None else 0)
            for i, name in enumerate(self.input_args_list)
        ]

        template = self.get_template("tb.txt")
        tb_out = template.render({
            'period': DEFAULT_TB_PERIOD,
            'in_args': key_val,
            'out_args': self.output_args_list
        })
        Path("tb.py").write_text(tb_out, encoding="utf-8")

    @staticmethod
    def tosigned(n: int, nbits: int) -> int:
        """Convert unsigned int to signed two’s complement, retain low nbits."""
        mask = (1 << nbits) - 1
        n &= mask
        sign_bit = 1 << (nbits - 1)
        return (n ^ sign_bit) - sign_bit

    def tb_fun(self, *in_arg: Any) -> Tuple[int, ...]:
        """Run testbench and return outputs in declared order."""
        log.info("tb_fun called with arguments: %s", in_arg)

        if RESULTS_PATH.exists():
            try:
                RESULTS_PATH.unlink()
            except OSError as e:
                log.warning("Failed to remove stale results file: %s", e)

        self.gen_tb(*in_arg)
        tb_runner.test_runner(self.t_name)

        if RESULTS_PATH.exists():
            with RESULTS_PATH.open('rb') as handle:
                payload = pickle.load(handle)
                d = payload if isinstance(payload, dict) else ast.literal_eval(payload)
        else:
            log.warning("No results — returning zero outputs")
            d = {}

        out_vals: List[int] = []
        for name, width in zip(self.output_args_list, self.output_args_width_list):
            val = d.get(name, 0)
            if self.attr.get(name, {}).get('signed') == 1:
                val = self.tosigned(val, width)
            out_vals.append(val)

        return tuple(out_vals)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    log.info("py2ver executed as script")
