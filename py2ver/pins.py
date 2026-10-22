"""
A design on the DE0-Nano's own pins (a mixin of Py2ver, see core.py).

A port's attr names the board's pin it goes to, as the board's pin assignments
(templates/hw/de0_nano_qsf.txt) call it:

    attr = {'rx': {'width': 1, 'pin': 'GPIO_0[0]'},
            'data': {'width': 8, 'pin': 'LED[7:0]'},     # bit 0 on LED[0]
            ...}
    p = Py2ver(uart_rx, attr)
    p.PINS({'QUARTUS_DIR': ...})       # compiled for the board, not programmed

The board's top (DE0_Nano) clocks the design with CLOCK_50, holds it in reset
at power-up and while KEY[0] is pressed (unless a port takes KEY[0]), its clock
enable high; an input pin reaches its port through two registers (a pin
changes whenever it likes: the second register's value is settled), an output
port drives its pin; a GPIO pin no output drives floats. Ports without a pin:
inputs 0, outputs unconnected.
"""
from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .defaults import DEFAULT_SYN_PERIOD_NS, DEFAULT_TEMPLATE_DIR
from .syn_report import SynReport

log = logging.getLogger(__name__)

# (the board's pins by kind: driven by the design, read, or either)
_OUT_GROUPS = {"LED"}
_IN_GROUPS = {"KEY", "SW", "GPIO_0_IN", "GPIO_1_IN", "GPIO_2_IN"}


def board_pins() -> Dict[str, int]:
    """The DE0-Nano's pins as its assignments name them: a group's width (LED: 8),
    a single pin's 0."""
    qsf = (DEFAULT_TEMPLATE_DIR / "hw" / "de0_nano_qsf.txt").read_text(encoding="utf-8")
    pins: Dict[str, int] = {}
    for m in re.finditer(r"set_location_assignment\s+\S+\s+-to\s+(\w+)(?:\[(\d+)\])?", qsf):
        name, idx = m.group(1), m.group(2)
        pins[name] = max(pins.get(name, 0), int(idx) + 1) if idx is not None else pins.get(name, 0)
    return pins


def parse_pin(spec: str, width: int, pins: Dict[str, int]) -> List[str]:
    """A port's pin attr as one pin a bit, its bit 0 first: 'LED[0]', 'LED[3:0]' (bit 0
    on LED[0]), 'LED' (the group from its bit 0), 'GPIO_0_TX' (a single pin)."""
    m = re.fullmatch(r"(\w+)(?:\[(\d+)(?::(\d+))?\])?", spec.strip())
    if not m or m.group(1) not in pins:
        raise ValueError(f"No DE0-Nano pin {spec!r} (pins: {', '.join(sorted(pins))})")
    name, a, b = m.group(1), m.group(2), m.group(3)
    size = pins[name]
    if size == 0:                                      # (a single pin)
        if a is not None or width != 1:
            raise ValueError(f"{name} is a single pin: a port of 1 bit")
        return [name]
    if a is None:
        lo, hi = 0, width - 1
    elif b is None:
        lo = hi = int(a)
    else:
        hi, lo = int(a), int(b)
    if hi < lo:
        raise ValueError(f"{spec}: [high:low]")
    if hi - lo + 1 != width:
        raise ValueError(f"{spec}: {hi - lo + 1} pins for a port of {width} bits")
    if hi >= size:
        raise ValueError(f"{spec}: {name} has {size} pins")
    return [f"{name}[{i}]" for i in range(lo, hi + 1)]


class PinsMixin:
    """Py2ver on the DE0-Nano's own pins: PINS() and its project."""

    _pins: Dict[str, str] = {}

    @property
    def _pins_dir(self) -> Path:
        return self.outdir / "syn" / f"{self.t_name}_pins"

    def _pin_map(self) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """({input port: its pins}, {output port: its pins})."""
        pins = board_pins()
        ins = {p.name: parse_pin(self._pins[p.name], p.width, pins) for p in self.ir.inputs
               if p.kind == "input" and p.name in self._pins}
        outs = {p.name: parse_pin(self._pins[p.name], p.width, pins) for p in self.ir.outputs
                if p.name in self._pins}
        unknown = set(self._pins) - set(ins) - set(outs)
        if unknown:
            raise ValueError(f"A pin for no port: {sorted(unknown)} (ports: "
                             f"{[p.name for p in self.ir.inputs + self.ir.outputs]})")
        taken: Dict[str, str] = {}
        for port, ps in {**ins, **outs}.items():
            for pin in ps:
                if pin in taken:
                    raise ValueError(f"{pin} is both {taken[pin]}'s and {port}'s")
                taken[pin] = port
        for port, ps in outs.items():
            for pin in ps:
                if pin.split("[")[0] in _IN_GROUPS:
                    raise ValueError(f"{pin} is an input pin: not for the output {port}")
        for port, ps in ins.items():
            for pin in ps:
                if pin.split("[")[0] in _OUT_GROUPS:
                    raise ValueError(f"{pin} is an output pin: not for the input {port}")
        return ins, outs

    def board_top(self) -> str:
        """The Verilog of DE0_Nano: the board's pins, the design between them."""
        if self._sdram:
            raise ValueError("A design with the SDRAM goes on the board with HW()")
        ins, outs = self._pin_map()
        pins = board_pins()
        groups = sorted({pin.split("[")[0] for ps in [*ins.values(), *outs.values()] for pin in ps}
                        | ({"KEY"} if "KEY[0]" not in {p for ps in ins.values() for p in ps}
                           else set()))
        decl = ["    input CLOCK_50"]
        for g in groups:
            size = pins[g]
            rng = f"[{size - 1}:0] " if size else ""
            kind = ("output" if g in _OUT_GROUPS else "input" if g in _IN_GROUPS
                    else "output" if size == 0 and any(g in ps for ps in outs.values())
                    else "input" if size == 0 else "inout")
            decl.append(f"    {kind} {rng}{g}")
        own_key = "KEY[0]" in {p for ps in ins.values() for p in ps}
        lines = ["// The DE0-Nano with the design between its pins (py2ver PINS())",
                 "module DE0_Nano(", ",\n".join(decl), ");",
                 "    wire clk = CLOCK_50;",
                 "    // (held in reset at power-up" + (")" if own_key else ", and while KEY[0] is pressed)"),
                 "    reg [3:0] por = 4'd0;",
                 "    always @(posedge clk) if (por != 4'd15) por <= por + 4'd1;",
                 f"    wire rst = (por != 4'd15){'' if own_key else ' || !KEY[0]'};"]
        conns = [".clk(clk)", ".rst(rst)", ".ce(1'b1)"] if self.has_clk else []
        for p in self.ir.inputs:
            if p.kind != "input":
                continue
            rng = f"[{p.width - 1}:0] " if p.width > 1 else ""
            if p.name in ins:
                bits = ins[p.name]
                cat = bits[0] if len(bits) == 1 else "{" + ", ".join(reversed(bits)) + "}"
                lines += [f"    // {p.name} from {self._pins[p.name]}, through two registers",
                          f"    reg {rng}{p.name}_s1 = 0, {p.name}_s2 = 0;",
                          f"    always @(posedge clk) begin {p.name}_s1 <= {cat}; "
                          f"{p.name}_s2 <= {p.name}_s1; end"]
                conns.append(f".{p.name}({p.name}_s2)")
            else:
                conns.append(f".{p.name}({p.width}'d0)")
        for p in self.ir.outputs:
            rng = f"[{p.width - 1}:0] " if p.width > 1 else ""
            lines.append(f"    wire {rng}{p.name}_w;")
            conns.append(f".{p.name}({p.name}_w)")
            for i, pin in enumerate(outs.get(p.name, [])):
                lines.append(f"    assign {pin} = {p.name}_w" + (f"[{i}];" if p.width > 1 else ";"))
        # (a GPIO pin no output drives: floating)
        driven = {pin for ps in outs.values() for pin in ps}
        for g in groups:
            if g.startswith("GPIO") and g not in _IN_GROUPS and pins[g]:
                free = [i for i in range(pins[g]) if f"{g}[{i}]" not in driven]
                if len(free) == pins[g]:
                    lines.append(f"    assign {g} = {pins[g]}'bz;")
                else:
                    lines += [f"    assign {g}[{i}] = 1'bz;" for i in free]
        lines += [f"    {self.t_name} u_design(", "        " + ",\n        ".join(conns) + ");",
                  "endmodule", ""]
        return "\n".join(lines)

    def gen_pins_project(self) -> Path:
        """The Quartus project of the design on the board's pins (PINS()): the
        board's assignments, its clock's constraint, DE0_Nano and the design."""
        project_dir = self._pins_dir
        if project_dir.is_dir():
            shutil.rmtree(project_dir)
        project_dir.mkdir(parents=True)
        name = "DE0_Nano"
        (project_dir / f"{name}.qpf").write_text(f"PROJECT_REVISION = {name}\n", encoding="utf-8")
        qsf = (DEFAULT_TEMPLATE_DIR / "hw" / "de0_nano_qsf.txt").read_text(encoding="utf-8")
        qsf += ("\n# The design (py2ver PINS())\n"
                "set_global_assignment -name VERILOG_FILE DE0_Nano.v\n"
                f"set_global_assignment -name VERILOG_FILE {self.t_name}.v\n")
        (project_dir / f"{name}.qsf").write_text(qsf, encoding="utf-8")
        shutil.copy(DEFAULT_TEMPLATE_DIR / "hw" / "de0_nano_sdc.txt", project_dir / "DE0_Nano.SDC")
        (project_dir / "DE0_Nano.v").write_text(self.board_top(), encoding="utf-8")
        self._copy_hdl(project_dir, f"{self.t_name}.v")
        log.info("The design on the board's pins: %s", project_dir)
        return project_dir

    def PINS(self, syn_attr: Optional[Dict[str, Any]] = None) -> SynReport:
        """
        Compile the design on the DE0-Nano's own pins (each port's attr 'pin')
        with Quartus, its area and timing; nothing is programmed. syn_attr: as
        SYN()'s ('QUARTUS_DIR'). The project: syn/<name>_pins.
        """
        syn_attr = dict(syn_attr or {})
        quartus_sh = self._quartus_sh(syn_attr)
        project_dir = self.gen_pins_project()
        syn_attr.setdefault("CLK_PERIOD_NS", DEFAULT_SYN_PERIOD_NS)
        # (the board's assignments keep the reports beside the project)
        return self._run_quartus(quartus_sh, project_dir, "DE0_Nano", self.t_name, syn_attr,
                                 out_dir=project_dir)
