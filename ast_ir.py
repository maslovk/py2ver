# ast_ir.py
from dataclasses import dataclass, field
from typing import List

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

@dataclass
class ModuleIR:
    name: str
    inputs: List[Port] = field(default_factory=list)
    outputs: List[Port] = field(default_factory=list)
    assigns: List[Assignment] = field(default_factory=list)
    has_clk: bool = False
