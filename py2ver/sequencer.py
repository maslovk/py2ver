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
    values carried to later stages by delay registers (pipeline_block)
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
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .share import _fmt_of, _fmt_range, _hull, _interval, NotShareable

RAM_MIN = 32              # a list this long or longer is a block RAM
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
            consts = {ast.dump(e) for e in elems}
            if len(consts) == 1:
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
             pipelined: bool = True) -> Tuple[ast.Module, Dict[str, Any]]:
    """
    Build the module's first function (lowered integer code with for-range
    loops) as a state machine; returns (tree, info) in share.py's form, or
    info['phases'] == 1 if it has no loops to sequence. pipelined: innermost
    loops start an iteration every clock where they can (see pipeline_block).
    """
    from .visitor import _Desugar
    fd = _Desugar().visit(copy.deepcopy(tree.body[0]))
    sq = _Seq(fd, attr, ns)
    st = _Structure(sq.lists)
    program = st.seq(fd.body)
    if not any(isinstance(it, Loop) for it in program):
        return tree, {"phases": 1}
    return _build(sq, program, attr, ns, pipelined)


def _build(sq: _Seq, program, attr, ns, pipelined: bool = True):
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
    reg_lists = {name for name in sq.lists if name not in rams and name not in inputs}
    # A list in block RAM made by [c] * n: filled by a loop, unless the loop after
    # it writes every element before anything reads it
    program = _allocations(program, rams, sq)
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
                                            attr, rng, ns)
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
                    total += n * (len(it.values) + codes[it.body[0].index].clocks - 1)
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
                        unrolled=True)
        read_ranges(alt)
        if pipeline_block(alt, counters[lp.index], len(lp.values), known, attr):
            n = len(lp.values)
            saved = times[lp.index] * (n * codes[k].clocks - (n + alt.clocks - 1))
            if saved >= UNIT_SAVING * before:
                codes[k] = alt

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

    # How many times each block runs, and the clocks of a sample
    phases = clocks_per_sample() + 1

    from .sequencer_blocks import emit_machine
    return emit_machine(sq, program, blocks, codes, counters, succ, start_state, start_sets,
                        DONE, phases, ret, rams, reg_lists, rng, fmt, attr, memories)
