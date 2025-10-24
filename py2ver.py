# py2ver.py
import inspect
import ast
import os
import pickle
import subprocess
import shutil
import logging
from typing import List

from jinja2 import Environment, FileSystemLoader

import tb_runner
from visitor import FunctionVisitor
from renderer import Renderer
# from ast_ir import ModuleIR  # (import if you want type hints)

DEFAULT_TEMPLATE_DIR = os.path.dirname(os.path.abspath(__file__)) + '/templates/'

log = logging.getLogger(__name__)


class Py2ver:
    t_name = ''

    def __init__(self, func, attr, template_dir: str = DEFAULT_TEMPLATE_DIR):
        log.debug("Initializing Py2ver (func=%s, template_dir=%s)", getattr(func, "__name__", func), template_dir)

        # ---- 1) Build AST and convert to IR (no rendering here) ----
        source_foo = inspect.getsource(func)
        tree = ast.parse(source_foo)
        f_visitor = FunctionVisitor(attr)
        ir = f_visitor.visit(tree)  # ModuleIR

        # ---- 2) Keep these for downstream compatibility ----
        self.t_name = ir.name
        # Inputs/outputs excluding clk for argument lists
        self.input_args_list: List[str] = [p.name for p in ir.inputs if p.kind != 'clk']
        self.output_args_list: List[str] = [p.name for p in ir.outputs]
        self.input_args_width_list: List[int] = [p.width for p in ir.inputs if p.kind != 'clk']
        self.output_args_width_list: List[int] = [p.width for p in ir.outputs]
        self.input_args_bits = sum(self.input_args_width_list)
        self.output_bits = sum(self.output_args_width_list)

        self.attr = attr
        self.template_dir = template_dir

        log.info("IR built for module '%s' (inputs=%s, outputs=%s, in_bits=%d, out_bits=%d)",
                 self.t_name, self.input_args_list, self.output_args_list,
                 self.input_args_bits, self.output_bits)

        # ---- 3) Render HDL via Renderer (IR → Verilog) ----
        renderer = Renderer(self.template_dir)
        verilog_text = renderer.render_module(ir)

        # Ensure output dir and write HDL
        os.makedirs("hdl", exist_ok=True)
        out_path = os.path.join("hdl", "main.v")
        with open(out_path, "w+", encoding="utf-8") as f:
            f.write(verilog_text)
        log.info("Wrote generated Verilog to %s", out_path)

        # Keep existing env/cache for other template renders in this class
        self.env = Environment(loader=FileSystemLoader(self.template_dir))
        self.template_cache = {}

        # Preserve your synthesis data structure for board-level templates
        self.syn_data = {
            'top_name': self.t_name,
            'inputs': [{'name': n, 'width': w, 'delay': 0}
                       for n, w in zip(self.input_args_list, self.input_args_width_list)],
            'inputs_size': self.input_args_bits,
            'outputs': [{'name': n, 'width': w, 'delay': 0}
                        for n, w in zip(self.output_args_list, self.output_args_width_list)],
            'outputs_size': self.output_bits
        }

    # ---------------- Hardware flow (unchanged API) ----------------

    def HW(self, syn_attr):
        self.syn_attr = syn_attr
        self.createHWproject()
        return self.hw_run

    def hw_run(self):
        return 1

    def createHWproject(self):
        syn_tool_dir = self.syn_attr.get('QUARTUS_DIR')
        if syn_tool_dir is None:
            log.error("No synthesis tool specified (missing 'QUARTUS_DIR').")
            return

        quartus_sh = os.path.join(syn_tool_dir, "quartus_sh")
        if not os.path.exists(quartus_sh):
            log.error("quartus_sh not found at %s", quartus_sh)
            return

        # Delete previous results
        hw_root = os.path.join(os.getcwd(), "hw")
        if os.path.isdir(hw_root):
            log.debug("Removing previous 'hw' directory at %s", hw_root)
            shutil.rmtree(hw_root)

        # Project settings
        project_name = self.t_name
        project_dir = os.path.join(hw_root, project_name)
        top_level_entity = self.t_name

        # Create project directory
        os.makedirs(project_dir, exist_ok=True)

        # Write .qpf file
        qpf_path = os.path.join(project_dir, f"{project_name}.qpf")
        with open(qpf_path, "w", encoding="utf-8") as f:
            f.write(f"PROJECT_REVISION = {project_name}\n")
        log.debug("Wrote %s", qpf_path)

        # Write .qsf file from template
        template_qsf = self.get_template("hw/de0_nano_qsf.txt")
        rslt_qsf = template_qsf.render({})
        qsf_path = os.path.join(project_dir, f"{project_name}.qsf")
        with open(qsf_path, "w", encoding="utf-8") as f:
            f.write(rslt_qsf)
        log.debug("Wrote %s", qsf_path)

        # UART transceiver (uses total widths from IR-derived fields)
        template_rxtx = self.get_template("hw/uart_transceiver.txt")
        template_rxtx_dict = {
            'in_clk_freq': 50000000,
            'baud_rate': 115200,
            'reg_width_tx': self.output_bits,
            'reg_width_rx': self.input_args_bits
        }
        rxtx_path = os.path.join(project_dir, "uart_transceiver.sv")
        with open(rxtx_path, "w", encoding="utf-8") as f:
            f.write(template_rxtx.render(template_rxtx_dict))
        log.debug("Wrote %s", rxtx_path)

        # SDC
        template_sdc = self.get_template("hw/de0_nano_sdc.txt")
        sdc_path = os.path.join(project_dir, "DE0_Nano.SDC")
        with open(sdc_path, "w", encoding="utf-8") as f:
            f.write(template_sdc.render({}))
        log.debug("Wrote %s", sdc_path)

        # Delay block
        template_delay = self.get_template("hw/delayed_registers.txt")
        delay_path = os.path.join(project_dir, "delayed_registers.sv")
        with open(delay_path, "w", encoding="utf-8") as f:
            f.write(template_delay.render(self.syn_data))
        log.debug("Wrote %s", delay_path)

        # Board-specific top (DE0-Nano)
        template_board = self.get_template("hw/de0_nano_top.txt")
        board_top_path = os.path.join(project_dir, "DE0_Nano.v")
        with open(board_top_path, "w", encoding="utf-8") as f:
            f.write(template_board.render(self.syn_data))
        log.debug("Wrote %s", board_top_path)

        # Copy generated core HDL to the project
        src_main_v = os.path.join(os.getcwd(), "hdl", "main.v")
        dst_core_v = os.path.join(project_dir, top_level_entity + ".v")
        shutil.copy(src_main_v, dst_core_v)
        log.debug("Copied %s → %s", src_main_v, dst_core_v)

        log.info("Quartus project '%s' created at %s", project_name, project_dir)

        # === Compile using Quartus command-line ===
        log.info("Starting Quartus compilation...")
        result = subprocess.run(
            [quartus_sh, "--flow", "compile", project_name],
            cwd=project_dir, capture_output=True, text=True
        )
        if result.returncode != 0:
            log.error("Quartus compile failed (rc=%d)\nSTDOUT:\n%s\nSTDERR:\n%s",
                      result.returncode, result.stdout, result.stderr)
            return
        log.info("Quartus compile finished successfully.")
        log.debug("Quartus STDOUT:\n%s", result.stdout)

        # Optional: program FPGA (commented out)
        # sof_file = os.path.join(project_dir, f"{project_name}.sof")
        # if os.path.exists(sof_file):
        #     log.info("Programming FPGA...")
        #     result = subprocess.run([
        #         os.path.join(syn_tool_dir,"quartus_pgm"), "-m", "jtag", "-o", f"p;{sof_file}"
        #     ], capture_output=True, text=True)
        #     if result.returncode != 0:
        #         log.error("quartus_pgm failed (rc=%d): %s", result.returncode, result.stderr)
        #     else:
        #         log.info("FPGA programmed successfully.")
        # else:
        #     log.warning(".sof file not found. Compilation may have failed or produced no output.")

    # ---------------- Testbench flow (unchanged API) ----------------

    def TB(self):
        return self.tb_fun

    def get_template(self, filename):
        # Reuse a tiny cache
        if not hasattr(self, "_tpl_cache"):
            self._tpl_cache = {}
        if filename in self._tpl_cache:
            return self._tpl_cache[filename]
        template = self.env.get_template(filename)
        self._tpl_cache[filename] = template
        return template

    def gen_tb(self, *in_arg):
        # Prepare name→value pairs for all non-clk inputs
        key_val = []
        for i in range(len(self.input_args_list)):
            if i < len(in_arg) and in_arg[i] is not None:
                key_val.append((self.input_args_list[i], in_arg[i]))
            else:
                key_val.append((self.input_args_list[i], 0))

        template = self.get_template("tb.txt")
        template_dict = {
            'period': 2000,  # if you want dynamic: 2000 if 'clk' present in IR
            'in_args': key_val,
            'out_args': self.output_args_list
        }
        tb_out = template.render(template_dict)
        with open("tb.py", "w+", encoding="utf-8") as f:
            f.write(tb_out)
        log.debug("Generated testbench tb.py")

    @staticmethod
    def tosigned(n: int, nbits: int) -> int:
        mask = (1 << nbits) - 1
        n &= mask
        sign_bit = 1 << (nbits - 1)
        return (n ^ sign_bit) - sign_bit

    def tb_fun(self, *in_arg):
        log.info("tb_fun called with %d arguments: %s", len(in_arg), in_arg)
        results_path = 'results/results.pickle'
        if os.path.isfile(results_path):
            try:
                os.remove(results_path)
                log.debug("Removed stale %s", results_path)
            except OSError as e:
                log.warning("Failed to remove stale results file: %s", e)

        self.gen_tb(*in_arg)
        tb_runner.test_runner(self.t_name)

        if os.path.isfile(results_path):
            with open(results_path, 'rb') as handle:
                p = pickle.load(handle)
                d = ast.literal_eval(p)

                # Build outputs in declared order and widths
                out_vals = []
                for name, width in zip(self.output_args_list, self.output_args_width_list):
                    val = d.get(name, 0)
                    if name in self.attr and int(self.attr[name].get('signed', 0)) == 1:
                        val = self.tosigned(val, width)
                    out_vals.append(val)
                b = tuple(out_vals)
                log.debug("tb_fun outputs: %s", b)
        else:
            b = (0,) * len(self.output_args_list)
            log.warning("Results file not found; returning zeros: %s", b)
        return b


if __name__ == "__main__":
    # Default logging setup if run directly
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    log.info("py2ver module executed as a script. This file is intended to be imported.")
