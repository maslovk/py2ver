"""
What py2ver can do on this computer: which of the tools it uses are installed,
and what each one is for.

    python -m py2ver doctor
"""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from . import __version__


def _version(command: List[str]) -> Optional[str]:
    """The first line a tool prints about its version, or None if it can't run."""
    try:
        out = subprocess.run(command, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = (out.stdout or out.stderr).strip()
    return text.splitlines()[0].replace(" ()", "").strip() if text else ""


def _quartus_tool(name: str, quartus_dir: Optional[str]) -> Optional[str]:
    from .jtag_uart import find_tool
    try:
        return find_tool(name, quartus_dir)
    except FileNotFoundError:
        return None


def _board(quartus_dir: Optional[str]) -> Tuple[Optional[str], str]:
    """(the USB-Blaster's name, what jtagconfig said) for a board that's plugged in."""
    tool = _quartus_tool("jtagconfig", quartus_dir)
    if tool is None:
        return None, "jtagconfig not found"
    try:
        out = subprocess.run([tool], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired) as e:
        return None, str(e)
    cable = next((line.split(")", 1)[1].strip() for line in out.splitlines()
                  if line.strip()[:2].rstrip(")").isdigit() and ")" in line), None)
    device = "EP4CE22" in out
    if cable and device:
        return cable, f"{cable}, with the DE0-Nano's EP4CE22 on it"
    if cable:
        return None, f"{cable}, but no EP4CE22 on it (is the board powered?)"
    return None, "no USB-Blaster found (is the board plugged in?)"


def report(quartus_dir: Optional[str] = None, look_for_board: bool = True) -> str:
    """The report `python -m py2ver doctor` prints."""
    lines = [f"py2ver {__version__}, Python {sys.version.split()[0]}", ""]

    def line(ok: bool, what: str, detail: str, gives: str) -> None:
        lines.append(f"  [{'x' if ok else ' '}] {what:<22} {detail}")
        lines.append(f"      {'' if ok else 'without it: no '}{gives}")

    lines.append("Simulating the Verilog")
    icarus = shutil.which("iverilog") and shutil.which("vvp")
    line(bool(icarus), "Icarus Verilog",
         _version(["iverilog", "-V"]) or "" if icarus else "not found (apt install iverilog)",
         "check() and TB_batch() run the generated Verilog")
    verilator = shutil.which("verilator")
    line(bool(verilator), "Verilator",
         _version(["verilator", "--version"]) or "" if verilator
         else "not found (optional: apt install verilator)",
         "long simulations run much faster")
    cocotb = importlib.util.find_spec("cocotb") is not None
    line(cocotb, "cocotb", "installed" if cocotb else "not found (pip install cocotb)",
         "the handshake and virtual-board testbenches (TB_handshake, HW_sim)")

    lines += ["", "Python libraries for some designs"]
    for module, use in (("numpy", "designs written with numpy"),
                        ("torch", "PyTorch models"),
                        ("serial", "HW() over a USB-UART adapter (pyserial)")):
        found = importlib.util.find_spec(module) is not None
        line(found, module, "installed" if found else "not found (optional)", use)

    lines += ["", "The DE0-Nano board"]
    quartus = _quartus_tool("quartus_sh", quartus_dir)
    line(bool(quartus), "Quartus",
         str(Path(quartus).parent) if quartus
         else "not found (give QUARTUS_DIR, or install Quartus Prime Lite)",
         "SYN(), PINS() and HW(): building for the FPGA")
    if quartus:
        from .jtag_uart import _library
        library = _library(quartus_dir) is not None
        line(library, "JTAG UART library",
             "libjtag_atlantic" if library else "not found: juart-terminal is used instead",
             "fast JTAG UART (about 25 ms a round trip)" if library
             else "fast JTAG UART (juart-terminal: about 0.2 s a round trip)")
        if look_for_board:
            cable, said = _board(quartus_dir)
            line(bool(cable), "a board", said,
                 "PROGRAM(), HW(LINK='jtag') and the JTAG console on the real board")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] != ["doctor"]:
        print("usage: python -m py2ver doctor        (what py2ver can do on this computer)")
        return 2
    print(report())
    return 0
