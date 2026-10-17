# choose.py
"""
Pick how a design is built from the rate it must keep up with.

Py2ver(func, ..., rate=R) takes R samples per second at the clock (50 MHz,
the DE0-Nano's, unless clock= says otherwise), so each sample may take
C = clock / R clocks. The smallest structure that finishes in C clocks wins:

  - C == 1: every operation its own hardware (unrolled)
  - one loop holding most of the products: that loop folded, with the
    fewest copies of its body (`unroll`) that finish in time
  - else: the fewest shared multipliers (multipliers=N) whose schedule fits
  - if no schedule fits (a chain of dependent steps longer than C clocks):
    unrolled, which takes one clock
  - a sequenced design (sequential=True, or numpy too big to write out): the
    fewest lanes for its multiply-accumulate loops (choose_lanes)

The choice is printed, like automatically chosen formats.
"""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional

FOLD_SHARE = 0.8      # a loop holding this share of the products is folded


@dataclass
class Choice:
    kind: str                       # 'unrolled' | 'fold' | 'multipliers'
    clocks: int                     # clocks per sample it takes
    budget: int                     # clocks per sample the rate allows
    fold: Any = None
    unroll: int = 1
    multipliers: Optional[int] = None
    why: str = ""

    def options(self) -> Dict[str, Any]:
        if self.kind == "fold":
            return {"fold": self.fold, "unroll": self.unroll}
        if self.kind == "multipliers":
            return {"multipliers": self.multipliers}
        return {}

    def __str__(self) -> str:
        opts = ", ".join(f"{k}={v!r}" for k, v in self.options().items()) or "unrolled"
        return f"{opts} ({self.clocks} clock{'s' if self.clocks != 1 else ''} per sample; {self.why})"


def _products(node: ast.AST) -> int:
    """Products in the code that a multiplier would do (not by 0, 1 or a power of two)."""
    n = 0
    for x in ast.walk(node):
        if isinstance(x, ast.BinOp) and isinstance(x.op, ast.Mult):
            consts = [s for s in (x.left, x.right) if isinstance(s, ast.Constant)]
            if len(consts) == 2:
                continue
            if consts and isinstance(consts[0].value, int):
                c = abs(consts[0].value)
                if c & (c - 1) == 0:
                    continue
            n += 1
    return n


def _shared_products(tree: ast.Module, attr: Dict[str, Any]) -> Optional[int]:
    """Products of the whole design written out (loops unrolled), or None."""
    from .share import NotShareable, _desugared, _ToOps, _live
    try:
        fd = _desugared(tree.body[0])
        conv = _ToOps(fd, attr)
        conv.body(fd.body)
        return sum(o.kind == "mul" for o in _live(list(conv.ops.values()), conv.outputs))
    except NotShareable:
        return None


def _try_fold(tree: ast.Module, attr: Dict[str, Any], budget: int, products: Optional[int]):
    """(unroll, clocks) of the best fold of the longest loop, or None."""
    from .fold import find_loop, fold_loop
    from .visitor import FunctionVisitor
    try:
        _, loop, values = find_loop(tree.body[0], True)
    except ValueError:
        return None
    n = len(values)
    in_loop = _products(ast.Module(body=loop.body, type_ignores=[])) * n
    if products is not None and products and in_loop < FOLD_SHARE * products:
        return None                     # most of the work is elsewhere: share it all
    for unroll in range(1, n + 1):
        if n % unroll == 0 and n // unroll <= budget:
            if n // unroll == 1:
                return None             # (all of it in one clock: nothing to fold)
            try:
                unfolded = FunctionVisitor(attr)
                unfolded.visit(copy.deepcopy(tree))
                fold_loop(tree, True, unroll, unfolded.attr, dict(unfolded._lists))
            except (NotImplementedError, ValueError):
                return None
            return unroll, n // unroll
    return None


def choose(tree: ast.Module, attr: Dict[str, Any], budget: int) -> Choice:
    """The structure for a design that may take `budget` clocks per sample."""
    if budget < 1:
        raise ValueError("The rate is more than one sample per clock")
    if budget == 1:
        return Choice("unrolled", 1, budget, why="one sample per clock")
    products = _shared_products(tree, attr)
    folded = _try_fold(tree, attr, budget, products)
    if folded is not None:
        unroll, clocks = folded
        return Choice("fold", clocks, budget, fold=True, unroll=unroll,
                      why=f"the loop folded onto {unroll} cop{'ies' if unroll > 1 else 'y'} "
                          "of its body")
    if products is None:
        return Choice("unrolled", 1, budget, why="multipliers= can't schedule this design")
    from .share import share
    best = None
    k = 1
    hi = max(1, products)
    # Fewest multipliers that fit: doubling, then halving the gap
    while True:
        _, info = share(tree, attr, k)
        if info["phases"] <= budget:
            best = (k, info["phases"])
            break
        if k >= hi:
            break
        k = min(hi, k * 2)
    if best is None:
        return Choice("unrolled", 1, budget,
                      why=f"a multiplier for each of the {products} products: shared, the "
                          f"chain of dependent steps needs at least {info['phases']} clocks")
    lo = k // 2 + 1 if k > 1 else 1
    while lo < best[0]:
        mid = (lo + best[0]) // 2
        _, info = share(tree, attr, mid)
        if info["phases"] <= budget:
            best = (mid, info["phases"])
        else:
            lo = mid + 1
    if best[1] <= 1:
        return Choice("unrolled", 1, budget, why="nothing to share")
    return Choice("multipliers", best[1], budget, multipliers=best[0],
                  why=f"{products} products on {best[0]} multiplier{'s' if best[0] > 1 else ''}")


MAX_LANES = 32


def choose_lanes(tree: ast.Module, attr: Dict[str, Any], ns: Dict[str, Any],
                 budget: int) -> Choice:
    """A sequenced design (sequential=True) for `budget` clocks per sample: the
    fewest lanes for its multiply-accumulate loops (multipliers=N) that finish in
    time (see sequencer._unrolled)."""
    from .sequencer import sequence
    best = None
    for lanes in range(1, MAX_LANES + 1):
        clocks = sequence(tree, attr, ns, lanes=lanes)[1]["phases"]
        if best is not None and clocks >= best[1]:
            continue                              # (no loop splits into this many)
        best = (lanes, clocks)
        if clocks <= budget:
            return Choice("multipliers", clocks, budget, multipliers=lanes,
                          why=f"multiply-accumulate loops on {lanes} lane"
                              f"{'s' if lanes != 1 else ''}")
    raise ValueError(f"A sequenced design takes at least {best[1]} clocks per sample "
                     f"({best[0]} lanes); the rate allows {budget}")
