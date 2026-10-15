# syn_report.py
"""
Parse Quartus compile reports into a small area/timing summary.

Reads, from <project>/output_files:
  - <rev>.fit.summary : design-wide resource totals after fitting
  - <rev>.map.rpt     : "Analysis & Synthesis Resource Utilization by Entity"
                        (for registers owned by the core entity)
  - <rev>.sta.summary : setup slack per timing corner
  - <rev>.sta.rpt     : "Fmax Summary" panels per timing corner
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class SynReport:
    device: Optional[str]
    clk_period_ns: float
    # Area of the core alone (the synthesis wrapper adds only registers)
    combinational: Optional[int]
    registers: Optional[int]
    multipliers_9bit: Optional[int]
    memory_bits: Optional[int]
    # Whole wrapper project, including its I/O registers
    logic_elements: Optional[int]
    # Timing (worst corner)
    fmax_mhz: Optional[float]
    setup_slack_ns: Optional[float]
    project_dir: Optional[Path] = None
    fit_summary: Dict[str, str] = field(default_factory=dict)

    @property
    def meets_timing(self) -> Optional[bool]:
        if self.setup_slack_ns is None:
            return None
        return self.setup_slack_ns >= 0.0

    def __str__(self) -> str:
        def _v(x, unit=""):
            return "n/a" if x is None else f"{x}{unit}"

        target = 1000.0 / self.clk_period_ns
        timing = {True: "met", False: "FAILED", None: "n/a"}[self.meets_timing]
        return "\n".join([
            f"Device:          {_v(self.device)}",
            f"Combinational:   {_v(self.combinational)}",
            f"Registers:       {_v(self.registers)}",
            f"9-bit mults:     {_v(self.multipliers_9bit)}",
            f"Memory bits:     {_v(self.memory_bits)}",
            f"Fmax:            {_v(self.fmax_mhz, ' MHz')}",
            f"Timing @ {target:g} MHz: {timing} (worst setup slack {_v(self.setup_slack_ns, ' ns')})",
        ])


# --------------------------------------------------------------------
# Low-level parsers (pure: take report text, return values)
# --------------------------------------------------------------------

def parse_summary(text: str) -> Dict[str, str]:
    """'Key : value' lines of a Quartus .summary file (keys stripped)."""
    out: Dict[str, str] = {}
    for line in text.splitlines():
        if " : " in line:
            key, val = line.split(" : ", 1)
            out[key.strip()] = val.strip()
    return out


def summary_int(summary: Dict[str, str], key: str) -> Optional[int]:
    """Leading integer of a summary value, e.g. '328 / 22,320 ( 1 % )' -> 328."""
    val = summary.get(key)
    if val is None:
        return None
    m = re.match(r"\s*([\d,]+)", val)
    return int(m.group(1).replace(",", "")) if m else None


def _panel_rows(text: str, title: str) -> List[List[List[str]]]:
    """
    Return the rows of every ';'-table whose title line contains `title`.
    Each table is a list of rows; each row is a list of stripped cells,
    with the header row first.
    """
    tables: List[List[List[str]]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith(";") and title in lines[i]:
            rows: List[List[str]] = []
            j = i + 1
            while j < len(lines) and (lines[j].startswith(";") or lines[j].startswith("+")):
                if lines[j].startswith(";"):
                    rows.append([c.strip() for c in lines[j].strip().strip(";").split(";")])
                j += 1
            tables.append(rows)
            i = j
        else:
            i += 1
    return tables


def parse_fmax(sta_rpt: str) -> Optional[float]:
    """Worst (lowest) restricted Fmax over all corners' Fmax Summary panels."""
    vals: List[float] = []
    for rows in _panel_rows(sta_rpt, "Fmax Summary"):
        if not rows:
            continue
        header = rows[0]
        col = header.index("Restricted Fmax") if "Restricted Fmax" in header else 0
        for row in rows[1:]:
            m = re.match(r"([\d.]+)\s*MHz", row[col]) if col < len(row) else None
            if m:
                vals.append(float(m.group(1)))
    return min(vals) if vals else None


def parse_setup_slack(sta_summary: str) -> Optional[float]:
    """Worst setup slack over all corners in a .sta.summary."""
    slacks: List[float] = []
    current_type = ""
    for line in sta_summary.splitlines():
        if line.startswith("Type"):
            current_type = line.split(":", 1)[1]
        elif line.startswith("Slack") and "Setup" in current_type:
            slacks.append(float(line.split(":", 1)[1]))
    return min(slacks) if slacks else None


def parse_entity_registers(map_rpt: str, entity: str) -> Optional[int]:
    """Dedicated logic registers of `entity` and its sub-entities (synthesis)."""
    for rows in _panel_rows(map_rpt, "Resource Utilization by Entity"):
        if not rows:
            continue
        header = rows[0]
        if "Entity Name" not in header or "Dedicated Logic Registers" not in header:
            continue
        name_col = header.index("Entity Name")
        reg_col = header.index("Dedicated Logic Registers")
        for row in rows[1:]:
            if len(row) > max(name_col, reg_col) and row[name_col] == entity:
                m = re.match(r"(\d+)", row[reg_col])
                return int(m.group(1)) if m else None
    return None


# --------------------------------------------------------------------
# Project-level entry point
# --------------------------------------------------------------------

def read_report(project_dir: Path, revision: str, core_entity: str,
                clk_period_ns: float) -> SynReport:
    out = Path(project_dir) / "output_files"

    def _read(name: str) -> str:
        p = out / f"{revision}.{name}"
        return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""

    fit = parse_summary(_read("fit.summary"))

    return SynReport(
        device=fit.get("Device"),
        clk_period_ns=clk_period_ns,
        combinational=summary_int(fit, "Total combinational functions"),
        registers=parse_entity_registers(_read("map.rpt"), core_entity),
        multipliers_9bit=summary_int(fit, "Embedded Multiplier 9-bit elements"),
        memory_bits=summary_int(fit, "Total memory bits"),
        logic_elements=summary_int(fit, "Total logic elements"),
        fmax_mhz=parse_fmax(_read("sta.rpt")),
        setup_slack_ns=parse_setup_slack(_read("sta.summary")),
        project_dir=Path(project_dir),
        fit_summary=fit,
    )
