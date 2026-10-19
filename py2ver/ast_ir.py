# ast_ir.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List

@dataclass
class Port:
    name: str
    width: int = 1
    signed: bool = False
    kind: str = "input"  # "input" | "output" | "inout" | "reg"

@dataclass
class Assignment:
    left: str
    right: str
    is_reg: bool = False
    line: int | None = None     # the line of the Python function it comes from

@dataclass
class ModuleIR:
    name: str
    inputs: List[Port] = field(default_factory=list)
    outputs: List[Port] = field(default_factory=list)
    assigns: List[Assignment] = field(default_factory=list)
    has_clk: bool = False
    # Raw Verilog emitted in the module body (e.g. block-RAM arrays)
    items: List[str] = field(default_factory=list)


def resolve_meta(name: str, attr: Dict[str, Dict[str, Any]],
                 ports: Dict[str, Port]) -> Dict[str, Any]:
    """Width/sign of a signal, resolved the same way the renderer declares it."""
    if name in attr:
        return dict(attr[name])
    if name in ports:
        p = ports[name]
        return {"width": p.width, "signed": int(p.signed), "type": "wire"}
    if "_" in name:
        head, tail = name.rsplit("_", 1)
        if tail.isdigit():
            return resolve_meta(head, attr, ports)
    return {"width": 1, "signed": 0, "type": "wire"}
