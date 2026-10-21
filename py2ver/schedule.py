"""
A design that takes several clocks a sample, as the structural passes build
it: a folded loop (fold.py), shared multipliers (share.py) or a state machine
(sequencer.py). Each rewrites the function to compute one clock's part of the
sample from a phase (the clock's number) and registers, and says so here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set


@dataclass
class Schedule:
    kind: str                       # "folded", "shared" or "sequenced" ("none": one clock)
    phases: int                     # clocks a sample (1: nothing to schedule)
    phase: str | None = None        # the parameter given the clock's number
    # register port -> the output that loads it every clock
    regs: Dict[str, str] = field(default_factory=dict)
    # the parameters the caller doesn't give: the phase, the registers, ...
    params: Set[str] = field(default_factory=set)
    # formats of the signals the rewritten function adds
    attr: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    loop: str | None = None         # folded: the loop's variable
    # a register's value at reset (default 0)
    reg_init: Dict[str, int] = field(default_factory=dict)
    # tables in block RAM: q (the entry read a clock before) -> {entries, index, ...}
    roms: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # register files in block RAM: q -> {depth, raddr, we, waddr, wdata, ...}
    rfs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # parameters / outputs the rewritten function adds for the tables and
    # register files (their q's; their addresses and data)
    extra_params: List[str] = field(default_factory=list)
    extra_returns: List[str] = field(default_factory=list)
    register_bits: int = 0
    mul_widths: List[Any] = field(default_factory=list)       # shared: the multipliers' widths
    # sequenced: lists kept between samples (name -> {length, width, signed,
    # init}), the board's SDRAM (lists and tables in it; None: none), its
    # number of blocks, and of loops pipelined
    memories: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    sdram: Dict[str, Any] | None = None
    blocks: int = 0
    pipelined: int = 0
    steps: int = 0                          # (steps.py: the steps, their templates)
    templates: int = 0

    @property
    def sequenced(self) -> bool:
        return self.kind == "sequenced"

    def __str__(self) -> str:
        if self.kind == "folded":
            return (f"Folded loop '{self.loop}': {self.phases} clocks per sample, "
                    f"{len(self.regs)} register(s)")
        if self.kind == "shared":
            return (f"Shared: {self.phases} clocks per sample, multipliers {self.mul_widths}, "
                    f"{len(self.roms)} table(s) in block RAM, {self.register_bits} register bits")
        return (f"Sequenced: {self.blocks} blocks, {self.phases} clocks per sample, "
                f"{len(self.rfs)} register file(s), {len(self.roms)} ROM(s), "
                f"{self.register_bits} register bits")
