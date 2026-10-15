"""
py2ver: turn a Python function into Verilog, and check it against Python in
simulation (cocotb + Icarus), on a simulated or real DE0-Nano board, and in
Quartus synthesis.
"""

from .core import DEFAULT_TEMPLATE_DIR, Py2ver
from .hwtypes import Array, HwType, Ram, Reg, S, State, U
from .syn_report import SynReport

__all__ = ["Py2ver", "SynReport", "DEFAULT_TEMPLATE_DIR",
           "U", "S", "Reg", "Array", "State", "Ram", "HwType"]
__version__ = "0.1.0"
