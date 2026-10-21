# sequencer.py
"""
Loops over arrays as a small state machine: a design too big to write out.

A neural network's forward pass is thousands of multiply-accumulates over
arrays of hundreds of values; unrolled, that is more logic than any FPGA
has. The sequencer builds the program as it is written instead, one step at
a time:

  - the program is a tree of for-range loops over straight-line blocks; each
    block is a state, run once per iteration of the loops around it, and
    after it the loop counters decide the next one (a counter steps, or the
    loop ends and the next statement follows)
  - loop counters and scalars are registers; small lists are registers too,
    and large ones (RAM_MIN elements or more) are block RAMs; constant tables
    of ROM_MIN entries or more are ROMs in block RAM; a memory argument
    ('memory' in attr) keeps its values from one sample to the next (state:
    a network's weights while it trains), starting from its 'init'
  - a division by a variable and an integer square root take a bit a clock
  - an innermost loop of one block is pipelined: an iteration a clock, its
    values carried to later stages by delay registers (pipeline_block), a
    division or a square root in it written out a bit a stage when that saves
    enough clocks (UNIT_SAVING)
  - with lanes (multipliers=N), a multiply-accumulate loop runs N iterations
    at a time, its body once per lane, and a table the lanes read at their own
    entries is banked, a ROM per lane (_unrolled, _banked)
  - a block takes a few clocks: a read from a RAM / ROM gives its value a
    clock after its address, so reads are issued level by level (a read whose
    address needs another read's value waits for it), then everything is
    computed and the registers and the RAMs written (one write per RAM per
    clock); a read after a write to the same list in a block gets the written
    value
  - every block's logic is built once, and only its own state uses it

The number of clocks is fixed by the loop bounds, so a sample takes the same
`phases` every time: the result has fold.py's / share.py's form (a phase
input, registers loaded every clock, ROMs and register files), and the core
builds, simulates and runs it like a folded design. Register widths come
from following the program with intervals (loops for as many iterations as
they run), so they hold every value Python's integers would.
"""

from __future__ import annotations

import ast
import copy
import itertools
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .schedule import Schedule
from .share import _fmt_of, _fmt_range, _hull, _interval, NotShareable

log = logging.getLogger(__name__)

RAM_MIN = 32              # a list this long or longer is a block RAM
SDRAM_WIDTH = 16          # the DE0-Nano's SDRAM: a 16-bit word an address
SDRAM_BANK_WORDS = 1 << 22  # (4 banks of 8192 rows of 512 words)
DE0_NANO_BITS = 400_000   # sdram=True: what may stay in block RAM (of the EP4CE22's 594 kbit)
ROM_MIN = 64              # a constant table this long or longer is a ROM
# A loop with dividers / square roots is pipelined (a register stage per step of
# each) only if that saves at least this part of a sample's clocks
UNIT_SAVING = 0.05


class NotSequenceable(NotImplementedError):
    """A program the sequencer can't build."""


# ==================================================================== the program

@dataclass
class Block:
    stmts: List[ast.stmt]
    index: int = -1


@dataclass
class Loop:
    var: str
    values: List[int]
    body: List[Union["Block", "Loop"]]
    index: int = -1


def _range_values(node: ast.expr) -> Optional[List[int]]:
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "range" and not node.keywords and 1 <= len(node.args) <= 3):
        return None
    try:
        args = [ast.literal_eval(a) for a in node.args]
    except ValueError:
        return None
    return list(range(*args))


def _is_table(node: ast.expr) -> bool:
    return isinstance(node, ast.Tuple) and node.elts and all(
        isinstance(e, ast.Constant) and type(e.value) is int for e in node.elts)


class _Structure:
    """The function's body as Blocks and Loops."""

    def __init__(self, lengths: Dict[str, int]) -> None:
        self.lengths = lengths
        self.loops: List[Loop] = []
        self.blocks: List[Block] = []
        self.counter = itertools.count()

    def seq(self, stmts: List[ast.stmt]) -> List[Union[Block, Loop]]:
        out: List[Union[Block, Loop]] = []
        cur: List[ast.stmt] = []

        def flush():
            if cur:
                out.append(Block(list(cur)))
                cur.clear()
        for s in stmts:
            if isinstance(s, ast.For):
                flush()
                out.append(self.loop(s))
            elif isinstance(s, ast.If) and any(isinstance(n, ast.For) for n in ast.walk(s)):
                raise NotSequenceable(f"A loop inside an if (line {s.lineno}) is not supported")
            elif isinstance(s, (ast.While, ast.With, ast.Try)):
                raise NotSequenceable(f"'{type(s).__name__}' (line {s.lineno}) is not supported")
            elif isinstance(s, ast.Return):
                flush()
                out.append(Block([s]))
            else:
                cur.append(s)
        flush()
        return out

    def loop(self, s: ast.For) -> Loop:
        if s.orelse:
            raise NotSequenceable(f"A loop's else (line {s.lineno}) is not supported")
        values = _range_values(s.iter)
        if values is None or not isinstance(s.target, ast.Name):
            # for v in xs: an index loop over a list of known length
            if isinstance(s.iter, ast.Name) and s.iter.id in self.lengths:
                k = f"__it{next(self.counter)}"
                body = [ast.Assign(targets=[copy.deepcopy(s.target)],
                                   value=ast.Subscript(value=ast.Name(id=s.iter.id, ctx=ast.Load()),
                                                       slice=ast.Name(id=k, ctx=ast.Load()),
                                                       ctx=ast.Load()))] + s.body
                return Loop(k, list(range(self.lengths[s.iter.id])), self.seq(body))
            raise NotSequenceable(f"for {ast.unparse(s.target)} in {ast.unparse(s.iter)} "
                                  f"(line {s.lineno}): loops need constant range(...) bounds")
        return Loop(s.target.id, values, self.seq(s.body))


def _walk_items(items):
    for it in items:
        yield it
        if isinstance(it, Loop):
            yield from _walk_items(it.body)


# ==================================================================== list allocations

def _allocations(items, rams: Set[str], sq) -> List[Union[Block, Loop]]:
    out: List[Union[Block, Loop]] = []
    for k, it in enumerate(items):
        if isinstance(it, Loop):
            it.body = _allocations(it.body, rams, sq)
            out.append(it)
            continue
        cur: List[ast.stmt] = []
        stmts = it.stmts
        for j, s in enumerate(stmts):
            name = s.targets[0].id if (isinstance(s, ast.Assign) and len(s.targets) == 1
                                       and isinstance(s.targets[0], ast.Name)) else None
            elems = _list_elements(s.value) if name in rams else None
            if elems is None:
                cur.append(s)
                continue
            if cur:
                out.append(Block(cur))
                cur = []
            last_here = j == len(stmts) - 1
            nxt = items[k + 1] if k + 1 < len(items) else None
            if last_here and isinstance(nxt, Loop) and _writes_all(nxt, name, len(elems)):
                continue                              # (every element written first)
            first = elems[0]
            if all(e is first for e in elems) or len({ast.dump(e) for e in elems}) == 1:
                var = sq.fresh("__fill")
                out.append(Loop(var, list(range(len(elems))), [Block([ast.Assign(
                    targets=[ast.Subscript(value=ast.Name(id=name, ctx=ast.Load()),
                                           slice=ast.Name(id=var, ctx=ast.Load()),
                                           ctx=ast.Store())],
                    value=copy.deepcopy(elems[0]))])]))
            else:
                out.append(Block([ast.Assign(
                    targets=[ast.Subscript(value=ast.Name(id=name, ctx=ast.Load()),
                                           slice=ast.Constant(i), ctx=ast.Store())],
                    value=e) for i, e in enumerate(elems)]))
        if cur:
            out.append(Block(cur))
    return out


def _writes_all(loop: Loop, name: str, n: int, limit: int = 1 << 20) -> bool:
    """Does the loop nest write list name at every index 0..n-1, unconditionally,
    without reading it? (Each write's index a function of the loop counters.)"""
    reads, writes = False, []

    def scan(items, vars_):
        nonlocal reads
        for it in items:
            if isinstance(it, Loop):
                scan(it.body, vars_ + [(it.var, it.values)])
                continue
            for s in it.stmts:
                for x in ast.walk(s):
                    if isinstance(x, ast.Subscript) and isinstance(x.value, ast.Name) \
                            and x.value.id == name and isinstance(x.ctx, ast.Load):
                        reads = True
                    if isinstance(x, ast.If) and any(
                            isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
                            and t.value.id == name
                            for st in ast.walk(x) if isinstance(st, ast.Assign) for t in st.targets):
                        reads = True                  # (a conditional write: not sure)
                if isinstance(s, ast.Assign) and isinstance(s.targets[0], ast.Subscript) \
                        and isinstance(s.targets[0].value, ast.Name) \
                        and s.targets[0].value.id == name:
                    writes.append((s.targets[0].slice, list(vars_)))
    scan([loop], [])
    if reads or not writes:
        return False
    seen: Set[int] = set()
    for idx, vars_ in writes:
        free = {x.id for x in ast.walk(idx) if isinstance(x, ast.Name)}
        if not free <= {v for v, _ in vars_}:
            return False
        if math.prod(len(vals) for _, vals in vars_) > limit:
            return False
        code = compile(ast.Expression(ast.fix_missing_locations(copy.deepcopy(idx))), "<i>", "eval")
        names = [v for v, _ in vars_]
        for combo in itertools.product(*[vals for _, vals in vars_]):
            seen.add(eval(code, {}, dict(zip(names, combo))))
    return seen >= set(range(n))


# ==================================================================== SDRAM

def _external(program, rams: Set[str], fmt, attr, memories, budget,
              lengths: Dict[str, int]) -> Dict[str, Any]:
    """
    The lists (block RAMs) and tables (ROMs) that go to the SDRAM: the largest,
    one after another, until what stays (their lengths times their widths) is
    within budget bits. Each gets an address range in a bank (the banks taken in
    turn, so a product's two operands keep two rows open), its words
    SDRAM_WIDTH bits: {'lists': {name: info}, 'tables': {entries: info}}, info
    {'base', 'length', 'meta', 'init' (the values it starts with, or None),
    'words' (a value's words: 2 for a list of values up to twice as wide)}.
    """
    budget = DE0_NANO_BITS if budget is True else int(budget)
    items = []
    for a in rams:
        meta = fmt(a)
        n = lengths[a]
        init = attr[a].get("init") if a in memories else None
        items.append(("list", a, n, meta, init))
    seen = set()
    for it in _walk_items(program):
        if isinstance(it, Block):
            for s in it.stmts:
                for x in ast.walk(s):
                    if _is_table(x) and len(x.elts) >= ROM_MIN:
                        entries = tuple(e.value for e in x.elts)
                        if entries not in seen:
                            seen.add(entries)
                            meta = _fmt_of((min(entries), max(entries)))
                            items.append(("table", entries, len(entries), meta, list(entries)))
    items.sort(key=lambda t: -t[2] * int(t[3]["width"]))
    total = sum(n * int(m["width"]) for _, _, n, m, _ in items)
    out: Dict[str, Any] = {"lists": {}, "tables": {}}
    free = [0, 0, 0, 0]
    k = 0
    for kind, key, n, meta, init in items:
        if total <= budget:
            break
        words = -(-int(meta["width"]) // SDRAM_WIDTH)
        if words > 2 or (words > 1 and (kind == "table" or init is not None)):
            raise NotSequenceable(f"{'A table' if kind == 'table' else repr(key)} of "
                                  f"{meta['width']}-bit values can't be in the SDRAM "
                                  f"({SDRAM_WIDTH}-bit words); use width={SDRAM_WIDTH}")
        bank = k % 4
        k += 1
        if free[bank] + n * words > SDRAM_BANK_WORDS:
            raise NotSequenceable("The lists don't fit the SDRAM")
        info = {"base": (bank << 22) | free[bank], "length": n, "meta": meta, "init": init,
                "words": words}
        free[bank] += -(-n * words // 512) * 512          # (each from a row's start)
        out["lists" if kind == "list" else "tables"][key] = info
        total -= n * int(meta["width"])
    return out


def _external_ports(ext: Dict[str, Any], names: Set[str], kind: str, fmt,
                    lengths: Dict[str, int]) -> None:
    """Lists that would be ports by the hundred (INPUT_ARRAY_MIN values) go to the
    SDRAM too: an input's values written there before each sample (by the
    testbench, by the board's loader) and read like its other lists; a returned
    list's written there and read back after (kind: 'input' or 'output')."""
    top = [0, 0, 0, 0]
    for what in ("lists", "tables"):
        for info in ext[what].values():
            b = info["base"] >> 22
            top[b] = max(top[b], (info["base"] & (SDRAM_BANK_WORDS - 1))
                         + -(-info["length"] // 512) * 512)
    for a in sorted(names):
        n = lengths[a]
        meta = fmt(a)
        if int(meta["width"]) > SDRAM_WIDTH:
            raise NotSequenceable(f"The {kind} {a!r} of {meta['width']}-bit values can't be in "
                                  f"the SDRAM ({SDRAM_WIDTH}-bit words)")
        bank = min(range(4), key=lambda b: top[b])
        ext["lists"][a] = {"base": (bank << 22) | top[bank], "length": n, "meta": meta,
                           "init": None, kind: True}
        top[bank] += -(-n // 512) * 512


# ==================================================================== lanes

def _products(stmts: List[ast.stmt]) -> bool:
    """Does the code multiply two computed values (not a value by a constant)?"""
    return any(isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult)
               and not isinstance(n.left, ast.Constant) and not isinstance(n.right, ast.Constant)
               for s in stmts for n in ast.walk(s))


def _writes_lists(stmts: List[ast.stmt]) -> bool:
    return any(isinstance(t, ast.Subscript) for s in stmts for n in ast.walk(s)
               if isinstance(n, (ast.Assign, ast.AugAssign))
               for t in (n.targets if isinstance(n, ast.Assign) else [n.target]))


def _unrolled(items, lanes: int, fresh, program=None,
              outer: Optional[Dict[str, List[int]]] = None) -> List[Union[Block, Loop]]:
    """
    Innermost loops of one block that multiply computed values and write no
    list (a matrix product's multiply-accumulates) run `lanes` iterations at a
    time: the body is written once per lane, and a new counter j steps over the
    groups of lanes iterations, lane u reading the loop variable as
    first + (j * lanes + u) * step. The copies run in order, so a value carried
    through them (acc = acc + x[i] * w[i]) is computed as Python computes it;
    each lane reads through ports of its own. The lanes divide the loop's length
    (a loop of 10 asked for 4 runs on 2). A table each lane reads at its own
    entries (w[i * 128 + o]) is banked: lane u gets a table of the entries it
    reads, so the lanes together hold the table once (see _banked).
    fresh: makes new names; outer: the enclosing loops' values.
    """
    program = items if program is None else program
    outer = outer or {}
    out: List[Union[Block, Loop]] = []
    for it in items:
        if isinstance(it, Loop):
            body = it.body
            if len(body) == 1 and isinstance(body[0], Block) and _products(body[0].stmts) \
                    and not _writes_lists(body[0].stmts) and len(it.values) > 1 \
                    and not _reads_elsewhere(program, it):
                n = len(it.values)
                k = max(d for d in range(1, lanes + 1) if n % d == 0)
                step = it.values[1] - it.values[0]
                if k > 1:
                    j = fresh(f"{it.var}_j")
                    stmts = []
                    for u in range(k):
                        at = _plus(_times(_name(j), k * step), it.values[0] + u * step)
                        for st in body[0].stmts:
                            st = _banked(st, it, k, u, j, outer)
                            stmts.append(_substituted(st, it.var, at))
                    out.append(Loop(j, list(range(n // k)), [Block(stmts)]))
                    continue
            it.body = _unrolled(body, lanes, fresh, program, dict(outer, **{it.var: it.values}))
        out.append(it)
    return out


def _reads_elsewhere(items, loop: Loop) -> bool:
    """Is loop's variable read outside it (other than by loops of their own)?"""
    for it in items:
        if it is loop:
            continue
        if isinstance(it, Loop):
            if it.var != loop.var and _reads_elsewhere(it.body, loop):
                return True
        elif any(isinstance(n, ast.Name) and n.id == loop.var and isinstance(n.ctx, ast.Load)
                 for s in it.stmts for n in ast.walk(s)):
            return True
    return False


def _name(x: str) -> ast.Name:
    return ast.Name(id=x, ctx=ast.Load())


def _times(e: ast.expr, c: int) -> ast.expr:
    return e if c == 1 else ast.BinOp(left=e, op=ast.Mult(), right=ast.Constant(c))


def _plus(e: ast.expr, c: int) -> ast.expr:
    return e if c == 0 else ast.BinOp(left=e, op=ast.Add(), right=ast.Constant(c))


def _substituted(stmt: ast.stmt, var: str, value: ast.expr) -> ast.stmt:
    """stmt with the variable var read as value."""
    class _S(ast.NodeTransformer):
        def visit_Name(self, n: ast.Name):
            if n.id == var and isinstance(n.ctx, ast.Load):
                return copy.deepcopy(value)
            return n
    return _S().visit(copy.deepcopy(stmt))


def _affine(e: ast.expr, names: Set[str]) -> Optional[Dict[Optional[str], int]]:
    """e as {name: coefficient, None: constant} over the given names, if it is a sum
    of those times constants; else None."""
    if isinstance(e, ast.Name):
        return {e.id: 1} if e.id in names else None
    if isinstance(e, ast.Constant) and type(e.value) is int:
        return {None: e.value}
    if isinstance(e, ast.UnaryOp) and isinstance(e.op, ast.USub):
        a = _affine(e.operand, names)
        return None if a is None else {v: -c for v, c in a.items()}
    if isinstance(e, ast.BinOp) and isinstance(e.op, (ast.Add, ast.Sub)):
        a, b = _affine(e.left, names), _affine(e.right, names)
        if a is None or b is None:
            return None
        sign = 1 if isinstance(e.op, ast.Add) else -1
        out = dict(a)
        for v, c in b.items():
            out[v] = out.get(v, 0) + sign * c
        return out
    if isinstance(e, ast.BinOp) and isinstance(e.op, ast.Mult):
        a, b = _affine(e.left, names), _affine(e.right, names)
        if a is None or b is None:
            return None
        for x, y in ((a, b), (b, a)):
            if set(x) <= {None}:
                return {v: c * x.get(None, 0) for v, c in y.items()}
    return None


def _banked(stmt: ast.stmt, loop: Loop, k: int, u: int, j: str,
            outer: Dict[str, List[int]]) -> ast.stmt:
    """
    stmt (lane u of loop on k lanes, its counter j) with each read of a table in
    block RAM at a index a * i + r (i the loop's variable, a != 0, r a sum of the
    enclosing loops' variables times constants) made a read of lane u's own
    table: the entries the lane reads, row j (an iteration of the lanes), column
    (r - its least value) / g, g the gcd of r's coefficients. The lanes' tables
    together hold the entries of the table that are read; it is banked if they
    are no larger than the copies the lanes would read otherwise (a table read by
    two ports is one block RAM).
    """
    names = set(outer) | {loop.var}
    n, step, first = len(loop.values), loop.values[1] - loop.values[0], loop.values[0]

    class _B(ast.NodeTransformer):
        def visit_Subscript(self, s: ast.Subscript):
            self.generic_visit(s)
            if not (_is_table(s.value) and len(s.value.elts) >= ROM_MIN) \
                    or isinstance(s.slice, ast.Slice):
                return s
            f = _affine(s.slice, names)
            if f is None or not f.get(loop.var) or any(not outer[v] for v in f if v not in (None, loop.var)):
                return s
            a = f.pop(loop.var)
            c = f.pop(None, 0)
            f = {v: b for v, b in f.items() if b}
            g = math.gcd(*f.values()) if f else 1
            low = c + sum(b * (min(outer[v]) if b > 0 else max(outer[v])) for v, b in f.items())
            high = c + sum(b * (max(outer[v]) if b > 0 else min(outer[v])) for v, b in f.items())
            cols = (high - low) // g + 1
            entries = [e.value for e in s.value.elts]
            if n * cols > (k + 1) // 2 * len(entries):
                return s
            table = []
            for row in range(n // k):
                i = first + (row * k + u) * step
                for col in range(cols):
                    x = a * i + low + g * col
                    table.append(entries[x] if 0 <= x < len(entries) else 0)
            addr: ast.expr = _times(_name(j), cols)
            for v, b in f.items():
                addr = ast.BinOp(left=addr, op=ast.Add(), right=_times(_name(v), b // g))
            addr = _plus(addr, -sum(b // g * (min(outer[v]) if b > 0 else max(outer[v]))
                                    for v, b in f.items()))
            return ast.Subscript(value=ast.Tuple(elts=[ast.Constant(x) for x in table],
                                                 ctx=ast.Load()),
                                 slice=addr, ctx=ast.Load())
    return _B().visit(copy.deepcopy(stmt))


# ==================================================================== intervals

class _Ranges:
    """Every value each variable takes, following the program with intervals."""

    def __init__(self, attr, lists: Dict[str, int], params: List[str]) -> None:
        self.attr = attr
        self.rng: Dict[str, Tuple[int, int]] = {}
        self.lists = set(lists)
        for p in params:
            m = attr.get(p, {})
            if "width" in m:
                self.rng[p] = _fmt_range(m)
        self.seen: Dict[str, Tuple[int, int]] = dict(self.rng)

    def value(self, node: ast.expr) -> Tuple[int, int]:
        return _interval(node, self.rng, self.attr)

    def set(self, name: str, r: Tuple[int, int], declared: bool = True) -> None:
        m = self.attr.get(name, {})
        if declared and "width" in m:
            f = _fmt_range(m)
            if not (f[0] <= r[0] and r[1] <= f[1]):
                r = f                                 # (it wraps)
        if name in self.lists:                        # a list: the hull of its elements
            r = _hull(self.rng[name], r) if name in self.rng else r
        self.rng[name] = r
        self.seen[name] = _hull(self.seen[name], r) if name in self.seen else r

    def stmts(self, stmts: List[ast.stmt]) -> None:
        for s in stmts:
            self.stmt(s)

    def stmt(self, s: ast.stmt) -> None:
        if isinstance(s, ast.Assign):
            t = s.targets[0]
            if isinstance(t, ast.Name):
                if isinstance(s.value, (ast.List, ast.BinOp)) and t.id in self.lists:
                    elems = _list_elements(s.value)
                    rs = [self.value(e) for e in elems] if elems else [(0, 0)]
                    r = _hull(*rs)
                    self.rng[t.id] = r                 # (a new list)
                    self.seen[t.id] = _hull(self.seen[t.id], r) if t.id in self.seen else r
                    return
                self.set(t.id, self.value(s.value))
            elif isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name):
                self.set(t.value.id, self.value(s.value))
            elif isinstance(t, ast.Tuple):
                for tt, v in zip(t.elts, s.value.elts):
                    self.set(tt.id, self.value(v))
        elif isinstance(s, ast.If):
            before = dict(self.rng)
            self.stmts(s.body)
            taken = dict(self.rng)
            self.rng = dict(before)
            self.stmts(s.orelse)
            for k in set(taken) | set(self.rng):
                if k in taken and k in self.rng:
                    self.rng[k] = _hull(taken[k], self.rng[k])
                else:
                    self.rng[k] = taken.get(k, self.rng.get(k))

    def items(self, items) -> None:
        for it in items:
            if isinstance(it, Block):
                self.stmts([s for s in it.stmts if not isinstance(s, ast.Return)])
            else:
                lo, hi = min(it.values, default=0), max(it.values, default=0)
                self.set(it.var, (lo, hi), declared=False)
                for _ in it.values:                   # (as many times as it runs)
                    before = dict(self.rng)
                    self.items(it.body)
                    if self.rng == before:
                        break                         # (nothing grows any more)


def _list_elements(node: ast.expr) -> Optional[List[ast.expr]]:
    if isinstance(node, ast.List):
        return list(node.elts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        for lst, k in ((node.left, node.right), (node.right, node.left)):
            if isinstance(lst, ast.List) and isinstance(k, ast.Constant):
                return list(lst.elts) * int(k.value)
    return None


# ==================================================================== the state machine

@dataclass
class _Port:
    """A read port of a RAM / ROM: its reads, by (block, clock) -> address."""
    array: str
    q: str
    issues: List[Tuple[int, int, ast.expr]] = field(default_factory=list)


class _Seq:
    def __init__(self, fd: ast.FunctionDef, attr, ns) -> None:
        self.fd = fd
        self.attr = attr
        self.params = [a.arg for a in fd.args.args]
        self.used = {n.id for n in ast.walk(fd) if isinstance(n, ast.Name)} | set(self.params)
        self.lists: Dict[str, int] = {}
        for p in self.params:
            if "length" in attr.get(p, {}):
                self.lists[p] = int(attr[p]["length"])
        for n in ast.walk(fd):
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name):
                elems = _list_elements(n.value)
                if elems is not None:
                    self.lists[n.targets[0].id] = len(elems)
        self.counter = itertools.count()

    def fresh(self, base: str) -> str:
        name = base
        while name in self.used:
            name = f"{base}_{next(self.counter)}"
        self.used.add(name)
        return name


def sequence(tree: ast.Module, attr: Dict[str, Dict[str, Any]], ns: Dict[str, Any],
             pipelined: bool = True, lanes: int = 1,
             sdram: Any = None) -> Tuple[ast.Module, Schedule]:
    """
    Build the module's first function (lowered integer code with for-range
    loops) as a state machine; returns (tree, its schedule), phases == 1 if it
    has no loops to sequence. pipelined: innermost
    loops start an iteration every clock where they can (see pipeline_block).
    lanes: multiply-accumulate loops run that many iterations at a time (see
    _unrolled).
    sdram: True or a number of bits: the largest lists and tables go to the
    board's SDRAM until the rest (block RAM) is within that many bits (True: the
    DE0-Nano's, DE0_NANO_BITS); see _external.
    """
    from .visitor import _Desugar
    fd = _Desugar().visit(copy.deepcopy(tree.body[0]))
    sq = _Seq(fd, attr, ns)
    st = _Structure(sq.lists)
    program = st.seq(fd.body)
    if not any(isinstance(it, Loop) for it in program):
        return tree, Schedule("sequenced", 1)
    return _build(sq, program, attr, ns, pipelined, lanes, sdram)


def _build(sq: _Seq, program, attr, ns, pipelined: bool = True, lanes: int = 1, sdram=None):
    params = sq.params
    inputs = set(params)
    # Number the blocks and loops in program order
    blocks: List[Block] = []
    loops: List[Loop] = []
    for it in _walk_items(program):
        if isinstance(it, Block):
            it.index = len(blocks)
            blocks.append(it)
        else:
            it.index = len(loops)
            loops.append(it)
    ret_block = next((b for b in blocks if any(isinstance(s, ast.Return) for s in b.stmts)), None)
    if ret_block is None or not isinstance(ret_block.stmts[-1], ast.Return) \
            or ret_block is not blocks[-1] or not any(it is ret_block for it in program):
        raise NotSequenceable("The function must end with one return, outside any loop")
    ret = ret_block.stmts[-1]
    ret_block.stmts = ret_block.stmts[:-1]

    # ----- which lists are block RAMs, which registers
    # (a memory argument, 'memory' in attr, is state: a block RAM or registers
    # that keep their values from one sample to the next, starting from 'init')
    memories = [p for p in params if "memory" in attr.get(p, {})]
    inputs -= set(memories)
    returned = {n.id for n in ast.walk(ret) if isinstance(n, ast.Name)}
    if returned & set(memories):
        raise NotSequenceable(f"A memory is returned: {sorted(returned & set(memories))}")
    rams = {name for name, n in sq.lists.items()
            if n >= RAM_MIN and name not in inputs and name not in returned}
    # (sdram=: a long list returned as it is goes to the SDRAM too, no ports: read
    # back from there after each sample)
    from .visitor import INPUT_ARRAY_MIN
    ret_vals = ret.value.elts if isinstance(ret.value, ast.Tuple) else [ret.value]
    ext_outputs = ({v.id for v in ret_vals if isinstance(v, ast.Name) and v.id not in inputs
                    and v.id not in memories and sq.lists.get(v.id, 0) >= INPUT_ARRAY_MIN}
                   if sdram is not None and sdram is not False else set())
    rams |= ext_outputs
    reg_lists = {name for name in sq.lists if name not in rams and name not in inputs}
    # A list in block RAM made by [c] * n: filled by a loop, unless the loop after
    # it writes every element before anything reads it
    program = _allocations(program, rams, sq)
    if lanes > 1:
        program = _unrolled(program, lanes, sq.fresh)
    blocks, loops = [], []
    for it in _walk_items(program):
        if isinstance(it, Block):
            it.index = len(blocks)
            blocks.append(it)
        else:
            it.index = len(loops)
            loops.append(it)

    # ----- ranges (for register widths)
    rg = _Ranges(attr, sq.lists, params)
    rg.items(program)
    rng = rg.seen

    def fmt(name: str) -> Dict[str, Any]:
        r = rng.get(name)
        if r is None:
            raise NotSequenceable(f"Can't bound the values of '{name}'")
        return dict(_fmt_of(r), type="wire")

    # ----- lists and tables in the SDRAM (sdram=): the largest, until the rest fits
    ext = (_external(program, rams - ext_outputs, fmt, attr, memories, sdram, sq.lists)
           if sdram is not None and sdram is not False else None)
    if ext is not None:
        _external_ports(ext, {a for a in inputs if sq.lists.get(a, 0) >= INPUT_ARRAY_MIN},
                        "input", fmt, sq.lists)
        _external_ports(ext, ext_outputs, "output", fmt, sq.lists)

    # ----- the clocks of each block: read levels, then writes
    from .sequencer_blocks import BlockCode
    codes: Dict[int, "BlockCode"] = {}
    # Which loop a block's loop variables refer to (innermost enclosing, else last seen)
    counter_of: Dict[str, str] = {}
    counters: Dict[int, str] = {}
    for lp in loops:
        counters[lp.index] = sq.fresh(f"cnt_{lp.var}")

    scopes: Dict[int, Dict[str, str]] = {}
    times: Dict[int, int] = {}                     # loop -> how often it runs per sample

    def walk(items, scope: Dict[str, str], n: int):
        for it in items:
            if isinstance(it, Loop):
                scope[it.var] = counters[it.index]
                times[it.index] = n
                walk(it.body, scope, n * len(it.values))
            else:
                scopes[it.index] = dict(scope)
                codes[it.index] = BlockCode(sq, it, scopes[it.index], rams, reg_lists, inputs,
                                            attr, rng, ns, ext=ext)
    walk(program, {}, 1)

    def clocks_per_sample() -> int:
        """(a pipelined loop: N + its stages - 1 clocks each time it runs)"""
        total = 0

        def count(items, n: int):
            nonlocal total
            for it in items:
                if isinstance(it, Block):
                    total += n * codes[it.index].clocks
                elif len(it.body) == 1 and isinstance(it.body[0], Block) \
                        and codes[it.body[0].index].pipe is not None:
                    c = codes[it.body[0].index]
                    total += n * ((len(it.values) - 1) * c.pipe.get("ii", 1) + c.clocks)
                else:
                    count(it.body, n * len(it.values))
        count(program, 1)
        return total

    # ----- innermost loops of one block: an iteration a clock (see pipeline_block)
    from .sequencer_blocks import pipeline_block
    known: Dict[str, Tuple[int, int]] = {f"r_{v}": r for v, r in rng.items()}
    for lp in loops:
        known[f"{counters[lp.index]}_now"] = (min(lp.values, default=0), max(lp.values, default=0))
    for a in reg_lists:
        for e in range(sq.lists[a]):
            if a in rng:
                known[f"r_{a}_{e}"] = rng[a]

    def read_ranges(code) -> None:
        for site in code.sites:
            known[site.data] = ((min(site.array), max(site.array)) if site.rom
                                else rng.get(site.array, (0, 0)))
    for code in codes.values():
        read_ranges(code)

    def one_block(lp: Loop) -> bool:
        return pipelined and len(lp.body) == 1 and isinstance(lp.body[0], Block)
    for lp in loops:
        if one_block(lp):
            pipeline_block(codes[lp.body[0].index], counters[lp.index], len(lp.values), known,
                           attr)
    # A loop that divides or takes square roots: pipelined with them written out a
    # step a stage (registers for every step), where that saves enough clocks
    before = clocks_per_sample()
    for lp in loops:
        k = lp.body[0].index if one_block(lp) else None
        if k is None or codes[k].pipe is not None or not codes[k].extra_regs:
            continue
        alt = BlockCode(sq, lp.body[0], scopes[k], rams, reg_lists, inputs, attr, rng, ns,
                        unrolled=True, ext=ext)
        read_ranges(alt)
        if pipeline_block(alt, counters[lp.index], len(lp.values), known, attr):
            n = len(lp.values)
            saved = times[lp.index] * (n * codes[k].clocks
                                       - ((n - 1) * alt.pipe.get("ii", 1) + alt.clocks))
            if saved >= UNIT_SAVING * before:
                codes[k] = alt
                log.info("Loop of block %d pipelined, its dividers and square roots written "
                         "out: %d clocks saved", k, saved)
            else:
                log.info("Loop of block %d kept a step at a time: pipelining it written out "
                         "would save %d of %d clocks a sample (under %.0f %%)", k, saved, before,
                         100 * UNIT_SAVING)

    # ----- the control: successor of each block, how often each runs
    n = len(blocks)
    DONE = n
    parent: Dict[int, Optional[Loop]] = {}
    order: List[Union[Block, Loop]] = []

    def link(items, par):
        for k, it in enumerate(items):
            parent[id(it)] = par
            nxt = items[k + 1] if k + 1 < len(items) else None
            it._next = nxt
            if isinstance(it, Loop):
                link(it.body, it)
    link(program, None)

    def enter(it) -> Tuple[ast.expr, Dict[str, ast.expr]]:
        """(state, counter settings) of entering an item (a loop from its start)."""
        if it is None:
            return ast.Constant(DONE), {}
        if isinstance(it, Block):
            return ast.Constant(it.index), {}
        if not it.values:                        # (runs no times)
            return after(it)
        state, sets = enter(it.body[0]) if it.body else after(it)
        return state, dict(sets, **{counters[it.index]: ast.Constant(it.values[0])})

    def after(it) -> Tuple[ast.expr, Dict[str, ast.expr]]:
        """(state, counter settings) once an item is done."""
        if it._next is not None:
            return enter(it._next)
        par = parent[id(it)]
        if par is None:
            return ast.Constant(DONE), {}
        # The loop goes round again unless its counter was the last value
        cnt = counters[par.index]
        last = par.values[-1]
        step = par.values[1] - par.values[0] if len(par.values) > 1 else 1
        again_state, again_sets = enter(par.body[0])
        again_sets = dict(again_sets, **{cnt: ast.BinOp(left=ast.Name(id=f"{cnt}_now", ctx=ast.Load()),
                                                         op=ast.Add(), right=ast.Constant(step))})
        out_state, out_sets = after(par)
        test = ast.Compare(left=ast.Name(id=f"{cnt}_now", ctx=ast.Load()), ops=[ast.Eq()],
                           comparators=[ast.Constant(last)])
        keys = set(again_sets) | set(out_sets)
        sets = {}
        for key in keys:
            a = again_sets.get(key, ast.Name(id=f"{key}_now", ctx=ast.Load()))
            b = out_sets.get(key, ast.Name(id=f"{key}_now", ctx=ast.Load()))
            sets[key] = ast.IfExp(test=copy.deepcopy(test), body=b, orelse=a)
        state = ast.IfExp(test=test, body=out_state, orelse=again_state)
        return state, sets

    succ = {b.index: after(b) for b in blocks}
    start_state, start_sets = enter(program[0])

    # A block's products (not pipelined: its values are wires) through registers:
    # they are used a clock after their multipliers (BlockCode.muls), and a
    # register cuts the path there
    for code in codes.values():
        if code.pipe is not None or not code.muls:
            continue
        ranges = dict(known)
        # (the divider's and the square root's own registers: their formats)
        ranges.update({r: _fmt_range(meta) for r, (_, meta) in code.extra_regs.items()
                       if r not in ranges})
        unknown: Dict[str, str] = {}          # (values that can't be bounded: why)
        for s in code.out:
            t = s.targets[0].id
            try:
                ranges[t] = _interval(s.value, ranges, attr)
            except NotShareable as e:
                if t in code.formats:          # (it wraps to its declared format)
                    ranges[t] = _fmt_range(code.formats[t])
                else:
                    unknown[t] = str(e)
        held = {m: sq.fresh(f"r_{m}") for m in sorted(code.muls) if m in ranges}
        lost = sorted(set(code.muls) - set(held))
        if lost:
            # (no register to size: the product and the next one may share a clock)
            log.warning("Block %d: no register after %s (can't bound it: %s); two multipliers "
                        "may be on one path in a clock", code.k, ", ".join(lost),
                        unknown.get(lost[0], "?"))
        if not held:
            continue

        class _Held(ast.NodeTransformer):
            def visit_Name(self, x: ast.Name):
                return ast.Name(id=held[x.id], ctx=x.ctx) if x.id in held else x
        code.out = [s if s.targets[0].id in held else _Held().visit(s) for s in code.out]
        code.held = {held[m]: (m, dict(_fmt_of(ranges[m]), type="wire")) for m in held}

    # How many times each block runs, and the clocks of a sample
    phases = clocks_per_sample() + 1

    from .sequencer_blocks import emit_machine
    return emit_machine(sq, program, blocks, codes, counters, succ, start_state, start_sets,
                        DONE, phases, ret, rams, reg_lists, rng, fmt, attr, memories, ext)
