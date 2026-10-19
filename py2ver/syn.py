"""Py2ver's standalone synthesis (a mixin of Py2ver, see core.py)."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict

from .ast_ir import Port
from .defaults import (DEFAULT_SYN_DEVICE, DEFAULT_SYN_FAMILY, DEFAULT_SYN_PERIOD_NS,
                       DEFAULT_TEMPLATE_DIR)
from .syn_report import SynReport, read_report

log = logging.getLogger(__name__)

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


class SynMixin:
    """Py2ver's standalone synthesis: the core (or a chain of stages) in Quartus for its area and timing."""

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
        if self._sdram:
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
        if self._sdram:
            self._syn_sdram(project_dir, wrapper)

        log.info("Synthesis project created at %s", project_dir)
        return project_dir

    def _syn_sdram(self, project_dir: Path, wrapper: str) -> None:
        """A design with the SDRAM: its pins where the DE0-Nano has them, and the
        chip's timing (IS42S16160B-7 at the inverted clock) for the timing analysis."""
        qsf = (DEFAULT_TEMPLATE_DIR / "hw" / "de0_nano_qsf.txt").read_text(encoding="utf-8")
        pins = [line for line in qsf.splitlines() if "DRAM_" in line and not line.startswith("#")]
        with open(project_dir / f"{wrapper}.qsf", "a", encoding="utf-8") as f:
            f.write("\n# The SDRAM's pins (DE0-Nano)\n" + "\n".join(pins) + "\n")
        with open(project_dir / f"{wrapper}.sdc", "a", encoding="utf-8") as f:
            f.write(_SDRAM_SDC)

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
