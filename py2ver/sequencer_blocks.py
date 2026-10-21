# sequencer_blocks.py
"""
The sequencer's blocks as logic, and the per-clock function of the machine
(see sequencer.py).

A block's statements are written once, with every name made the block's own
(b<k>_...): a scalar not yet assigned in the block reads its register, an
if assigns through a selection (both branches are computed), a list in block
RAM is read through a port (its value a clock after its address) and written
through the RAM's write port (one write per clock), and a list in registers
is read through a multiplexer and written element by element.
"""

from __future__ import annotations

import ast
import copy
import logging
import os
from typing import Any, Dict, List, Optional, Set, Tuple

from .schedule import Schedule
from .share import _fmt_of, _hull

log = logging.getLogger(__name__)


def _name(n: str) -> ast.Name:
    return ast.Name(id=n, ctx=ast.Load())


def _assign(n: str, v: ast.expr) -> ast.Assign:
    return ast.Assign(targets=[ast.Name(id=n, ctx=ast.Store())], value=v)


def _and(a: Optional[ast.expr], b: Optional[ast.expr]) -> Optional[ast.expr]:
    if a is None:
        return b
    if b is None:
        return a
    return ast.BoolOp(op=ast.And(), values=[a, b])


def _not(a: ast.expr) -> ast.expr:
    return ast.Compare(left=a, ops=[ast.Eq()], comparators=[ast.Constant(0)])


def _eq(a: ast.expr, k: int) -> ast.expr:
    return ast.Compare(left=a, ops=[ast.Eq()], comparators=[ast.Constant(k)])


def _key(v: ast.expr) -> Any:
    if isinstance(v, ast.Name):
        return ("n", v.id)
    if isinstance(v, ast.Constant):
        return ("c", type(v.value), v.value)
    return ast.dump(v)


def _copy(v: ast.expr) -> ast.expr:
    if isinstance(v, ast.Name):
        return ast.Name(id=v.id, ctx=ast.Load())
    if isinstance(v, ast.Constant):
        return ast.Constant(v.value)
    return copy.deepcopy(v)


def _sel(values: List[ast.expr], idx: ast.expr) -> ast.expr:
    """values[idx] (a balanced multiplexer), or the value if they're all the same."""
    first = _key(values[0])
    if all(_key(v) == first for v in values):
        return _copy(values[0])
    return ast.Subscript(value=ast.Tuple(elts=[_copy(v) for v in values], ctx=ast.Load()),
                         slice=idx, ctx=ast.Load())


EXT_LATENCY = 7          # a read of the SDRAM: its data this many clocks after its address
EXT_HELD = EXT_LATENCY + 1   # (outside a pipelined loop: from a register a clock later)
CHAIN_BLOCKS = 32         # a port shared by more blocks: chosen by a search on the state
DSP_ELEMENTS = 132        # the EP4CE22's 9-bit multipliers: products past them share (share_products)


class _Site:
    def __init__(self, array: Any, addr: str, data: str, clock: int, rom: bool,
                 ext: Optional[int] = None, half: Optional[str] = None) -> None:
        self.array = array            # a RAM list's name, or a ROM table's entries
        self.addr = addr
        self.data = data
        self.clock = clock
        self.rom = rom
        self.ext = ext                # in the SDRAM: its base address
        self.half = half              # (a value in two words: 'lo' or 'hi')


class _Write:
    def __init__(self, array: str, addr: str, data: str, cond: Optional[str], ready: int,
                 ext: Optional[int] = None, index: Optional[str] = None,
                 value: Optional[str] = None, half: Optional[str] = None) -> None:
        self.array, self.addr, self.data, self.cond, self.ready = array, addr, data, cond, ready
        self.clock = 0
        self.ext = ext                # in the SDRAM: its base address
        # (a value in two words, half 'lo' / 'hi': addr and data a word's, index and
        # value the element's)
        self.index, self.value, self.half = index or addr, value or data, half


class BlockCode:
    """One block's statements as the machine's logic."""

    def __init__(self, sq, block, scope: Dict[str, str], rams: Set[str], reg_lists: Set[str],
                 inputs: Set[str], attr, rng, ns, unrolled: bool = False,
                 ext: Optional[Dict[str, Any]] = None) -> None:
        from .sequencer import ROM_MIN, _is_table, _list_elements, NotSequenceable
        self.NotSequenceable = NotSequenceable
        self._is_table, self._list_elements, self.ROM_MIN = _is_table, _list_elements, ROM_MIN
        self.sq = sq
        self.k = block.index
        self.scope = scope                     # loop variable -> its counter
        self.rams, self.reg_lists, self.inputs = rams, reg_lists, inputs
        self.attr = attr
        self.lengths = sq.lists
        self.out: List[ast.stmt] = []          # the block's statements, as emitted
        self.env: Dict[str, str] = {}          # scalar -> its value now (a name)
        self.elems: Dict[str, List[str]] = {}  # register list -> its elements now
        self.ready: Dict[str, int] = {}        # emitted name -> the clock its value is there
        self.read_regs: Set[str] = set()       # scalars read before the block sets them
        self.sites: List[_Site] = []
        self.writes: List[_Write] = []
        self.formats: Dict[str, Dict[str, Any]] = {}   # names that wrap to a declared format
        self.rng = rng
        self.extra_regs: Dict[str, Tuple[str, Dict[str, Any]]] = {}   # register -> (next, format)
        self.reads: Dict[Any, ast.expr] = {}   # (array, address) -> its value (read once)
        self.pipe: Optional[Dict[str, Any]] = None    # set by pipeline_block
        # unrolled: dividers and square roots written out a step per stage (for a
        # pipelined loop: a step's results are a stage after its inputs, `bumped`)
        self.unrolled = unrolled
        self.bumped: Set[str] = set()
        self.muls: Set[str] = set()          # (bumped: a register after the multiplier)
        self.held: Dict[str, Tuple[str, Dict[str, Any]]] = {}   # register -> (value, format)
        # (a product on a shared multiplier: its register -> (the unit, the clock it loads))
        self.shared_held: Dict[str, Tuple[str, int]] = {}
        # Lists and tables in the SDRAM: one access a clock (its port), data
        # EXT_LATENCY clocks after the address
        self.ext_lists = (ext or {}).get("lists", {})
        self.ext_tables = (ext or {}).get("tables", {})
        self.ext_used: Set[int] = set()
        for s in block.stmts:
            self.stmt(s, None)
        self.schedule()

    def ext_slot(self, c: int) -> int:
        """The first clock from c the SDRAM's port is free (taken)."""
        while c in self.ext_used:
            c += 1
        self.ext_used.add(c)
        return c

    # ------------------------------------------------------------ names
    def tmp(self, base: str) -> str:
        return self.sq.fresh(f"b{self.k}_{base}")

    def emit(self, base: str, value: ast.expr, bump: bool = False) -> str:
        name = self.tmp(base)
        self.out.append(_assign(name, value))
        self.ready[name] = self.when(value) + (1 if bump else 0)
        if bump:
            self.bumped.add(name)
        return name

    def steps(self, base: str, regs: Dict[str, Tuple[str, int]], step, n: int) -> Dict[str, str]:
        """A unit written out step by step (unrolled): regs maps each of its values
        to (its first value, width); step(values) gives their next values (ast);
        each step's values are a stage after the step before."""
        cur = {k: v for k, (v, _) in regs.items()}
        for _ in range(n):
            nxt = step(cur)
            new = {}
            for k, e in nxt.items():
                new[k] = self.emit(f"{base}{k}", e, bump=True)
                self.formats[new[k]] = {"signed": 0, "width": regs[k][1], "type": "wire"}
            cur = new
        return cur

    def when(self, node: ast.expr) -> int:
        return max([0] + [self.ready.get(n.id, 0) for n in ast.walk(node)
                          if isinstance(n, ast.Name)])

    def scalar(self, v: str) -> str:
        """Scalar v's value now, as a name."""
        if v in self.env:
            return self.env[v]
        if v in self.scope:
            return f"{self.scope[v]}_now"
        if v in self.inputs:
            return v
        self.read_regs.add(v)
        return f"r_{v}"

    def elements(self, a: str) -> List[str]:
        if a not in self.elems:
            self.elems[a] = [f"r_{a}_{e}" for e in range(self.lengths[a])]
        return self.elems[a]

    # ------------------------------------------------------------ expressions
    def expr(self, n: ast.expr) -> ast.expr:
        if isinstance(n, ast.Name):
            if n.id in self.lengths and n.id not in self.env:
                raise self.NotSequenceable(f"The list '{n.id}' can only be indexed here")
            return _name(self.scalar(n.id))
        if isinstance(n, ast.Subscript) and not isinstance(n.slice, ast.Slice):
            base = n.value
            if isinstance(base, ast.Name) and base.id in self.lengths:
                return self.read(base.id, self.expr(n.slice))
            if self._is_table(base) and len(base.elts) >= self.ROM_MIN:
                idx = self.expr(n.slice)
                return self.rom(tuple(e.value for e in base.elts), idx)
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div) \
                and not isinstance(n.right, ast.Constant):
            return self.divide(n)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) \
                and n.func.id == "__py2ver_isqrt" and len(n.args) == 1:      # (fixed.ISQRT)
            return self.sqrt(n.args[0])
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "len" \
                and len(n.args) == 1 and isinstance(n.args[0], ast.Name) \
                and n.args[0].id in self.lengths:
            return ast.Constant(self.lengths[n.args[0].id])
        n = copy.copy(n)
        for f, v in ast.iter_fields(n):
            if isinstance(n, ast.Call) and f == "func":
                continue                              # (max, min, abs: functions, not values)
            if isinstance(v, ast.expr):
                setattr(n, f, self.expr(v))
            elif isinstance(v, list):
                setattr(n, f, [self.expr(x) if isinstance(x, ast.expr) else x for x in v])
        return n

    def ext_base(self, a: str) -> Tuple[int, Optional[ast.expr]]:
        """(base, base register) of a list in the SDRAM: a template's (steps.py) at
        its slot's base (the slot its high bits, its low ones 0: the base's bits
        beside the address's, not a sum), another's at its own."""
        info = self.ext_lists[a]
        if info.get("param"):
            return 0, ast.BinOp(left=_name(self.scalar(info["param"])), op=ast.LShift(),
                                right=ast.Constant(int(info["shift"])))
        return info["base"], None

    @staticmethod
    def based(base_reg: Optional[ast.expr], word: ast.expr) -> ast.expr:
        """A word's address from its list's base register (ext_base), if it has one."""
        if base_reg is None:
            return word
        return ast.BinOp(left=copy.deepcopy(base_reg), op=ast.BitOr(), right=word)

    def at_base(self, a: str, addr: str, base_reg: Optional[ast.expr], what: str) -> str:
        """addr from its list's base register, if it has one."""
        if base_reg is None:
            return addr
        return self.emit(f"{a}_{what}", self.based(base_reg, _name(addr)))

    def two_words(self, array: Any, addr: str, base: int, prefix: str, rom: bool,
                  base_reg: Optional[ast.expr] = None) -> str:
        """A value in two SDRAM words (a list's or a table's): its low word,
        unsigned, and its high one, read in turn and put together."""
        halves = {}
        # (both words at the low one's address, even: the high one's is it, the
        # SDRAM's port adding its odd bit, ext_odd)
        word = ast.BinOp(left=_name(addr), op=ast.Mult(), right=ast.Constant(2))
        if base_reg is not None:
            word = self.based(base_reg, word)
        even = self.emit(f"{prefix}alo", word)
        for half in ("lo", "hi"):
            wa = even if half == "lo" else self.emit(f"{prefix}ahi", _name(even))
            halves[half] = self.tmp(f"{prefix}d{half}")
            clock = self.ext_slot(self.ready[wa])
            self.sites.append(_Site(array, wa, halves[half], clock, rom=rom, ext=base, half=half))
            self.ready[halves[half]] = clock + EXT_HELD
        return self.emit(f"{prefix}d", ast.BinOp(
            left=ast.BinOp(left=_name(halves["hi"]), op=ast.LShift(), right=ast.Constant(16)),
            op=ast.Add(), right=_name(halves["lo"])))

    def read(self, a: str, idx: ast.expr) -> ast.expr:
        if a in self.inputs and a not in self.ext_lists:
            return ast.Subscript(value=_name(a), slice=idx, ctx=ast.Load())
        if a in self.rams or a in self.ext_lists:
            key = (a, ast.dump(idx))
            if key in self.reads:
                return copy.deepcopy(self.reads[key])
            addr = self.emit(f"{a}_ra", self.products(idx, values_only=True))
            data = self.tmp(f"{a}_rd")
            if a in self.ext_lists and self.ext_lists[a].get("words", 1) == 2:
                base, base_reg = self.ext_base(a)
                data = self.two_words(a, addr, base, f"{a}_r", rom=False, base_reg=base_reg)
            elif a in self.ext_lists:
                base, base_reg = self.ext_base(a)
                at = self.at_base(a, addr, base_reg, "rb")
                clock = self.ext_slot(self.ready[at])
                self.sites.append(_Site(a, at, data, clock, rom=False, ext=base))
                self.ready[data] = clock + EXT_HELD
            else:
                clock = self.ready[addr]
                self.sites.append(_Site(a, addr, data, clock, rom=False))
                self.ready[data] = clock + 1
            value: ast.expr = _name(data)
            # A write earlier in the block to the same element: its value
            for w in self.writes:
                if w.array == a and w.half != "hi":
                    test = ast.Compare(left=_name(addr), ops=[ast.Eq()], comparators=[_name(w.index)])
                    if w.cond:
                        test = _and(_name(w.cond), test)
                    value = ast.IfExp(test=test, body=_name(w.value), orelse=value)
            value = _name(self.emit(f"{a}_v", value)) if not isinstance(value, ast.Name) else value
            self.reads[key] = value
            return value
        elems = self.elements(a)
        try:
            k = ast.literal_eval(idx)
            return _name(elems[k])
        except (ValueError, TypeError, IndexError):
            return _sel([_name(e) for e in elems], idx)

    def divide(self, n: ast.BinOp) -> ast.expr:
        """a / b (truncating; 0 for b == 0) by a variable: a restoring divider, a
        quotient bit a clock (a combinational divider would set the clock rate on
        its own)."""
        from .share import _interval
        (lo_a, hi_a), (lo_b, hi_b) = (_interval(n.left, self.rng, self.attr),
                                      _interval(n.right, self.rng, self.attr))
        bits = max(1, max(abs(lo_a), abs(hi_a)).bit_length())
        bbits = max(1, max(abs(lo_b), abs(hi_b)).bit_length())
        a = self.emit("dva", self.expr(n.left))
        b = self.emit("dvb", self.expr(n.right))
        if self.unrolled:
            return self.divide_steps(a, b, bits, bbits)
        start = max(self.ready[a], self.ready[b])
        absa = self.emit("dvA", ast.IfExp(test=ast.Compare(left=_name(a), ops=[ast.Lt()],
                                                            comparators=[ast.Constant(0)]),
                                         body=ast.UnaryOp(op=ast.USub(), operand=_name(a)),
                                         orelse=_name(a)))
        absb = self.emit("dvB", ast.IfExp(test=ast.Compare(left=_name(b), ops=[ast.Lt()],
                                                            comparators=[ast.Constant(0)]),
                                         body=ast.UnaryOp(op=ast.USub(), operand=_name(b)),
                                         orelse=_name(b)))
        neg = self.emit("dvn", ast.Compare(
            left=ast.Compare(left=_name(a), ops=[ast.Lt()], comparators=[ast.Constant(0)]),
            ops=[ast.NotEq()],
            comparators=[ast.Compare(left=_name(b), ops=[ast.Lt()], comparators=[ast.Constant(0)])]))
        q, r = self.sq.fresh(f"r_b{self.k}_dq"), self.sq.fresh(f"r_b{self.k}_dr")
        mask = (1 << bits) - 1
        here = _eq(_name("st_now"), self.k)
        load = self.emit("dvload", _and(copy.deepcopy(here), _eq(_name("cyc_now"), start)))
        step = self.emit("dvstep", _and(copy.deepcopy(here), ast.BoolOp(op=ast.And(), values=[
            ast.Compare(left=_name("cyc_now"), ops=[ast.Gt()], comparators=[ast.Constant(start)]),
            ast.Compare(left=_name("cyc_now"), ops=[ast.LtE()],
                        comparators=[ast.Constant(start + bits)])])))
        rem_in = self.emit("dvrem", ast.BinOp(
            left=ast.BinOp(left=_name(r), op=ast.LShift(), right=ast.Constant(1)), op=ast.BitOr(),
            right=ast.BinOp(left=ast.BinOp(left=_name(q), op=ast.RShift(),
                                           right=ast.Constant(bits - 1)),
                            op=ast.BitAnd(), right=ast.Constant(1))))
        ge = self.emit("dvge", ast.Compare(left=_name(rem_in), ops=[ast.GtE()],
                                          comparators=[_name(absb)]))
        qn = self.emit("dvqn", ast.IfExp(test=_name(load), body=_name(absa), orelse=ast.IfExp(
            test=_name(step), body=ast.BinOp(
                left=ast.BinOp(left=ast.BinOp(left=_name(q), op=ast.LShift(), right=ast.Constant(1)),
                               op=ast.BitAnd(), right=ast.Constant(mask)),
                op=ast.BitOr(), right=_name(ge)), orelse=_name(q))))
        rn = self.emit("dvrn", ast.IfExp(test=_name(load), body=ast.Constant(0), orelse=ast.IfExp(
            test=_name(step), body=ast.IfExp(test=_name(ge), body=ast.BinOp(
                left=_name(rem_in), op=ast.Sub(), right=_name(absb)), orelse=_name(rem_in)),
            orelse=_name(r))))
        uq = {"signed": 0, "width": bits}
        ur = {"signed": 0, "width": bbits + 1}
        self.extra_regs[q] = (qn, uq)
        self.extra_regs[r] = (rn, ur)
        self.formats[qn] = dict(uq, type="wire")
        self.formats[rn] = dict(ur, type="wire")
        self.ready[q] = start + bits + 1
        out = self.emit("dvq", ast.IfExp(          # (by 0: 0, as the model)
            test=_eq(_name(b), 0), body=ast.Constant(0), orelse=ast.IfExp(
                test=_name(neg), body=ast.UnaryOp(op=ast.USub(), operand=_name(q)),
                orelse=_name(q))))
        self.ready[out] = start + bits + 1
        return _name(out)

    def divide_steps(self, a: str, b: str, bits: int, bbits: int) -> ast.expr:
        """The divider written out, a quotient bit a stage."""
        def neg_of(x):
            return ast.Compare(left=_name(x), ops=[ast.Lt()], comparators=[ast.Constant(0)])

        def absolute(x):
            return ast.IfExp(test=neg_of(x), body=ast.UnaryOp(op=ast.USub(), operand=_name(x)),
                             orelse=_name(x))
        absa, absb = self.emit("dvA", absolute(a)), self.emit("dvB", absolute(b))
        neg = self.emit("dvn", ast.Compare(left=neg_of(a), ops=[ast.NotEq()],
                                          comparators=[neg_of(b)]))
        zero = self.emit("dvz", _eq(_name(b), 0))
        mask = (1 << bits) - 1

        def step(v):
            rem_in = ast.BinOp(left=ast.BinOp(left=_name(v["r"]), op=ast.LShift(), right=ast.Constant(1)),
                               op=ast.BitOr(), right=ast.BinOp(
                                   left=ast.BinOp(left=_name(v["q"]), op=ast.RShift(),
                                                  right=ast.Constant(bits - 1)),
                                   op=ast.BitAnd(), right=ast.Constant(1)))
            rem = self.emit("dvi", rem_in)
            ge = self.emit("dvg", ast.Compare(left=_name(rem), ops=[ast.GtE()],
                                             comparators=[_name(absb)]))
            return {"q": ast.BinOp(left=ast.BinOp(left=ast.BinOp(left=_name(v["q"]), op=ast.LShift(),
                                                                 right=ast.Constant(1)),
                                                  op=ast.BitAnd(), right=ast.Constant(mask)),
                                   op=ast.BitOr(), right=_name(ge)),
                    "r": ast.IfExp(test=_name(ge), body=ast.BinOp(left=_name(rem), op=ast.Sub(),
                                                                 right=_name(absb)),
                                   orelse=_name(rem))}
        r0 = self.emit("dvr", ast.Constant(0))
        q = self.steps("dv", {"q": (absa, bits), "r": (r0, bbits + 1)}, step, bits)["q"]
        return _name(self.emit("dvq", ast.IfExp(test=_name(zero), body=ast.Constant(0),
                                               orelse=ast.IfExp(test=_name(neg), body=ast.UnaryOp(
                                                   op=ast.USub(), operand=_name(q)),
                                                   orelse=_name(q)))))

    def sqrt_steps(self, x: str, bits: int) -> ast.expr:
        """The square root written out, a root bit (two radicand bits) a stage."""
        steps = bits // 2
        x0 = self.emit("sqX", ast.IfExp(test=ast.Compare(left=_name(x), ops=[ast.Lt()],
                                                         comparators=[ast.Constant(0)]),
                                        body=ast.Constant(0), orelse=_name(x)))

        def step(v):
            rem = self.emit("sqi", ast.BinOp(
                left=ast.BinOp(left=_name(v["r"]), op=ast.LShift(), right=ast.Constant(2)),
                op=ast.BitOr(), right=ast.BinOp(left=ast.BinOp(left=_name(v["x"]), op=ast.RShift(),
                                                               right=ast.Constant(bits - 2)),
                                                op=ast.BitAnd(), right=ast.Constant(3))))
            trial = self.emit("sqt", ast.BinOp(
                left=ast.BinOp(left=_name(v["q"]), op=ast.LShift(), right=ast.Constant(2)),
                op=ast.BitOr(), right=ast.Constant(1)))
            ge = self.emit("sqg", ast.Compare(left=_name(rem), ops=[ast.GtE()],
                                             comparators=[_name(trial)]))
            return {"x": ast.BinOp(left=ast.BinOp(left=_name(v["x"]), op=ast.LShift(),
                                                  right=ast.Constant(2)),
                                   op=ast.BitAnd(), right=ast.Constant((1 << bits) - 1)),
                    "r": ast.IfExp(test=_name(ge), body=ast.BinOp(left=_name(rem), op=ast.Sub(),
                                                                 right=_name(trial)),
                                   orelse=_name(rem)),
                    "q": ast.BinOp(left=ast.BinOp(left=ast.BinOp(left=_name(v["q"]), op=ast.LShift(),
                                                                 right=ast.Constant(1)),
                                                  op=ast.BitAnd(), right=ast.Constant((1 << steps) - 1)),
                                   op=ast.BitOr(), right=_name(ge))}
        z = self.emit("sqz", ast.Constant(0))
        q = self.steps("sq", {"x": (x0, bits), "r": (z, steps + 2), "q": (z, steps)}, step,
                       steps)["q"]
        return _name(self.emit("sqq", _name(q)))

    def sqrt(self, arg: ast.expr) -> ast.expr:
        """The integer square root (0 below 0), two radicand bits a clock: each
        step brings the next two bits into the remainder and takes the next
        bit of the root (the restoring method, like the divider)."""
        from .share import _interval
        hi = _interval(arg, self.rng, self.attr)[1]
        bits = max(2, hi.bit_length())
        bits += bits % 2
        steps = bits // 2
        x = self.emit("sqx", self.expr(arg))
        if self.unrolled:
            return self.sqrt_steps(x, bits)
        start = self.ready[x]
        xr, rr, qr = (self.sq.fresh(f"r_b{self.k}_{n}") for n in ("sqx", "sqr", "sqq"))
        here = _eq(_name("st_now"), self.k)
        load = self.emit("sqload", _and(copy.deepcopy(here), _eq(_name("cyc_now"), start)))
        step = self.emit("sqstep", _and(copy.deepcopy(here), ast.BoolOp(op=ast.And(), values=[
            ast.Compare(left=_name("cyc_now"), ops=[ast.Gt()], comparators=[ast.Constant(start)]),
            ast.Compare(left=_name("cyc_now"), ops=[ast.LtE()],
                        comparators=[ast.Constant(start + steps)])])))
        rem_in = self.emit("sqrem", ast.BinOp(
            left=ast.BinOp(left=_name(rr), op=ast.LShift(), right=ast.Constant(2)), op=ast.BitOr(),
            right=ast.BinOp(left=ast.BinOp(left=_name(xr), op=ast.RShift(),
                                           right=ast.Constant(bits - 2)),
                            op=ast.BitAnd(), right=ast.Constant(3))))
        trial = self.emit("sqtry", ast.BinOp(
            left=ast.BinOp(left=_name(qr), op=ast.LShift(), right=ast.Constant(2)), op=ast.BitOr(),
            right=ast.Constant(1)))
        ge = self.emit("sqge", ast.Compare(left=_name(rem_in), ops=[ast.GtE()],
                                          comparators=[_name(trial)]))
        xn = self.emit("sqxn", ast.IfExp(test=_name(load), body=ast.IfExp(
            test=ast.Compare(left=_name(x), ops=[ast.Lt()], comparators=[ast.Constant(0)]),
            body=ast.Constant(0), orelse=_name(x)), orelse=ast.IfExp(
            test=_name(step), body=ast.BinOp(
                left=ast.BinOp(left=_name(xr), op=ast.LShift(), right=ast.Constant(2)),
                op=ast.BitAnd(), right=ast.Constant((1 << bits) - 1)), orelse=_name(xr))))
        rn = self.emit("sqrn", ast.IfExp(test=_name(load), body=ast.Constant(0), orelse=ast.IfExp(
            test=_name(step), body=ast.IfExp(test=_name(ge), body=ast.BinOp(
                left=_name(rem_in), op=ast.Sub(), right=_name(trial)), orelse=_name(rem_in)),
            orelse=_name(rr))))
        qn = self.emit("sqqn", ast.IfExp(test=_name(load), body=ast.Constant(0), orelse=ast.IfExp(
            test=_name(step), body=ast.BinOp(
                left=ast.BinOp(left=ast.BinOp(left=_name(qr), op=ast.LShift(), right=ast.Constant(1)),
                               op=ast.BitAnd(), right=ast.Constant((1 << steps) - 1)),
                op=ast.BitOr(), right=_name(ge)), orelse=_name(qr))))
        for r, nxt, w in ((xr, xn, bits), (rr, rn, steps + 2), (qr, qn, steps)):
            meta = {"signed": 0, "width": w}
            self.extra_regs[r] = (nxt, meta)
            self.formats[nxt] = dict(meta, type="wire")
        self.ready[qr] = start + steps + 1
        out = self.emit("sqq", _name(qr))
        self.ready[out] = start + steps + 1
        return _name(out)

    def rom(self, entries: Tuple[int, ...], idx: ast.expr) -> ast.expr:
        key = (entries, ast.dump(idx))
        if key in self.reads:
            return copy.deepcopy(self.reads[key])
        addr = self.emit("rom_a", self.products(idx, values_only=True))
        data = self.tmp("rom_d")
        if entries in self.ext_tables and self.ext_tables[entries].get("words", 1) == 2:
            data = self.two_words(entries, addr, self.ext_tables[entries]["base"], "rom_",
                                  rom=True)
        elif entries in self.ext_tables:
            clock = self.ext_slot(self.ready[addr])
            self.sites.append(_Site(entries, addr, data, clock, rom=True,
                                    ext=self.ext_tables[entries]["base"]))
            self.ready[data] = clock + EXT_HELD
        else:
            clock = self.ready[addr]
            self.sites.append(_Site(entries, addr, data, clock, rom=True))
            self.ready[data] = clock + 1
        self.reads[key] = _name(data)
        return _name(data)

    # ------------------------------------------------------------ statements
    def stmt(self, s: ast.stmt, cond: Optional[ast.expr]) -> None:
        if isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant):
            return
        if isinstance(s, ast.Pass):
            return
        if isinstance(s, ast.If):
            test = _name(self.emit("if", self.expr(s.test)))
            for st in s.body:
                self.stmt(st, _and(copy.deepcopy(cond), copy.deepcopy(test)))
            for st in s.orelse:
                self.stmt(st, _and(copy.deepcopy(cond), _not(copy.deepcopy(test))))
            return
        if isinstance(s, ast.Assign) and len(s.targets) == 1:
            t = s.targets[0]
            if isinstance(t, ast.Tuple) and isinstance(s.value, ast.Tuple):
                vals = [self.expr(v) for v in s.value.elts]
                names = [self.emit("tup", v) for v in vals]
                for tt, nm in zip(t.elts, names):
                    self.set_scalar(tt.id, _name(nm), cond)
                return
            if isinstance(t, ast.Name):
                elems = self._list_elements(s.value)
                if t.id in self.lengths and elems is not None:
                    self.store_all(t.id, elems, cond)
                    return
                self.set_scalar(t.id, self.expr(s.value), cond)
                return
            if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) \
                    and t.value.id in self.lengths:
                self.store(t.value.id, self.expr(t.slice), self.expr(s.value), cond)
                return
        raise self.NotSequenceable(f"Unsupported statement in a sequenced design: {ast.unparse(s)}")

    def products(self, value: ast.expr, values_only: bool = False) -> ast.expr:
        """A product (of two computed values, or by a constant that isn't a power of
        two: a multiplier either way): a register after its multiplier (its value a
        clock later), so that a clock has one multiplier on a path at most (two in
        a row, with the adders around them, take longer than 20 ns). values_only:
        products of two values only (an address: by a constant, a few adders)."""
        code = self

        def shift(x: ast.expr) -> bool:
            return (isinstance(x, ast.Constant) and isinstance(x.value, int)
                    and x.value > 0 and x.value & (x.value - 1) == 0)

        class _Mul(ast.NodeTransformer):
            def visit_BinOp(self, n: ast.BinOp):
                self.generic_visit(n)
                consts = [x for x in (n.left, n.right) if isinstance(x, ast.Constant)]
                if values_only and consts:
                    return n
                if isinstance(n.op, ast.Mult) and len(consts) < 2 and not any(map(shift, consts)):
                    name = code.emit("mul", n, bump=True)
                    code.muls.add(name)
                    return _name(name)
                return n
        return _Mul().visit(value)

    def set_scalar(self, v: str, value: ast.expr, cond: Optional[ast.expr]) -> None:
        if v in self.scope:
            raise self.NotSequenceable(f"The loop variable '{v}' is assigned in its loop")
        value = self.products(value)
        if cond is not None:
            value = ast.IfExp(test=cond, body=value, orelse=_name(self.scalar(v)))
        self.env[v] = self.emit(v, value)
        self.declare(self.env[v], v)

    def declare(self, name: str, var: str) -> None:
        """name holds var's value: it wraps to var's format, as the model does."""
        m = self.attr.get(var, {})
        if "width" in m:
            self.formats[name] = {"signed": int(m.get("signed", 0)), "width": int(m["width"]),
                                  "type": "wire"}

    def store_all(self, a: str, elems: List[ast.expr], cond) -> None:
        for k, e in enumerate(elems):
            self.store(a, ast.Constant(k), self.expr(e), cond)

    def store(self, a: str, idx: ast.expr, value: ast.expr, cond) -> None:
        if a in self.inputs:
            raise self.NotSequenceable(f"The argument '{a}' is written")
        if a in self.rams:
            self.reads = {k: v for k, v in self.reads.items() if k[0] != a}
            addr = self.emit(f"{a}_wa", self.products(idx, values_only=True))
            data = self.emit(f"{a}_wd", self.products(value))
            self.declare(data, a)
            c = self.emit(f"{a}_we", cond) if cond is not None else None
            base, base_reg = self.ext_base(a) if a in self.ext_lists else (None, None)
            if a in self.ext_lists and self.ext_lists[a].get("words", 1) == 2:
                # (a value in two words: its low 16 bits, then the rest, both at the
                # low one's address, the port adding the high one's odd bit)
                word = ast.BinOp(left=_name(addr), op=ast.Mult(), right=ast.Constant(2))
                if base_reg is not None:
                    word = self.based(base_reg, word)
                even = self.emit(f"{a}_walo", word)
                for half in ("lo", "hi"):
                    wa = even if half == "lo" else self.emit(f"{a}_wahi", _name(even))
                    part = (ast.BinOp(left=_name(data), op=ast.BitAnd(), right=ast.Constant(0xFFFF))
                            if half == "lo" else
                            ast.BinOp(left=_name(data), op=ast.RShift(), right=ast.Constant(16)))
                    wd = self.emit(f"{a}_wd{half}", part)
                    self.writes.append(_Write(a, wa, wd, c, max(
                        self.ready[wa], self.ready[wd], self.ready.get(c, 0) if c else 0),
                        ext=base, index=addr, value=data, half=half))
                return
            at = self.at_base(a, addr, base_reg, "wb")
            self.writes.append(_Write(a, at, data, c, max(self.ready[at], self.ready[data],
                                                          self.ready.get(c, 0) if c else 0),
                                      ext=base, index=addr))
            return
        elems = self.elements(a)
        try:
            k = ast.literal_eval(idx)
            hit = {k}
        except (ValueError, TypeError):
            k = None
            hit = set(range(len(elems)))
        new = []
        idx_name = self.emit(f"{a}_i", idx) if k is None else None
        val = self.emit(f"{a}_w", value)
        for e, cur in enumerate(elems):
            if e not in hit:
                new.append(cur)
                continue
            test = cond
            if k is None:
                test = _and(copy.deepcopy(cond), _eq(_name(idx_name), e))
            v = _name(val) if test is None else ast.IfExp(test=test, body=_name(val),
                                                         orelse=_name(cur))
            new.append(self.emit(f"{a}_{e}", v))
            self.declare(new[-1], a)
        self.elems[a] = new

    # ------------------------------------------------------------ clocks
    def schedule(self) -> None:
        """Writes one per clock per RAM (in order); the block's clocks."""
        last = max([0] + [self.ready.get(v, 0) for v in self.env.values()]
                   + [self.ready.get(e, 0) for es in self.elems.values() for e in es])
        busy: Dict[str, int] = {}
        for w in self.writes:
            w.clock = max(w.ready, busy.get(w.array, -1) + 1)
            if w.ext is not None:                      # (the SDRAM's port: one access a clock)
                w.clock = self.ext_slot(w.clock)
            busy[w.array] = w.clock
            last = max(last, w.clock)
        last = max([last] + [s.clock for s in self.sites])
        self.clocks = last + 1


def share_products(codes, ranges_of: Dict[int, Dict[str, Tuple[int, int]]], attr,
                   fresh) -> List[Dict[str, Any]]:
    """
    The products of two values (not by a constant) on a few multipliers, every
    block's: the machine runs one block at a time, so a multiplier serves them
    all; when they'd take more than the device's multipliers (DSP_ELEMENTS:
    fewer, each its own, without operands to pick between, a few logic
    elements fewer). A block's products at one clock each take their own (a pipelined
    loop's: all of them, at every clock); a unit serves blocks of one kind,
    pipelined or not, and has two users at least (one alone keeps its
    multiplier). A user's operands are picked by the state (and its clock,
    outside a pipelined loop); outside one, the product is read only from its
    register, which loads at its clock (references to the product's wire are
    to the register). Returns the units: {'name', 'pipelined', 'users': [{'k',
    'clock' (None: a pipelined loop's), 't' (the product), 'a', 'b' (its
    operands), 'ra', 'rb' (their ranges), 'held' (its register)}]}.
    """
    from .share import NotShareable, _interval
    mode = os.environ.get("PY2VER_SHARE_PRODUCTS", "auto")     # (1: always, 0: never)
    if mode == "0":
        return []
    wires = {s.targets[0].id for code in codes.values() for s in code.out}
    users: Dict[bool, List[Dict[str, Any]]] = {False: [], True: []}
    for k in sorted(codes):
        code = codes[k]
        rng = ranges_of.get(k)
        if rng is None or not code.muls:
            continue
        piped = code.pipe is not None
        held_of = {m: r for r, (m, _) in code.held.items()}
        slots: Dict[Optional[int], int] = {}
        for s in code.out:
            t, v = s.targets[0].id, s.value
            if t not in code.muls or not (isinstance(v, ast.BinOp) and isinstance(v.op, ast.Mult)) \
                    or any(isinstance(x, ast.Constant) for x in (v.left, v.right)):
                continue
            if piped and any(isinstance(x, ast.Name) and x.id in wires for x in ast.walk(v)):
                continue                          # (its operands: registers, at a stage)
            if not piped and t not in held_of:
                continue                          # (no register: used the clock it's made)
            try:
                ra, rb = _interval(v.left, rng, attr), _interval(v.right, rng, attr)
            except NotShareable:
                continue
            clock = None if piped else code.ready[t] - 1
            unit = slots.get(clock, 0)
            slots[clock] = unit + 1
            users[piped].append({"k": k, "clock": clock, "t": t, "a": v.left, "b": v.right,
                                 "ra": ra, "rb": rb, "held": held_of.get(t), "unit": unit})
    # (products the device's multipliers take as they are: theirs, no operands to pick)
    def elements(u: Dict[str, Any]) -> int:
        wa, wb = (int(_fmt_of(r)["width"]) for r in (u["ra"], u["rb"]))
        return 1 if wa <= 9 and wb <= 9 else 2 * -(-wa // 18) * -(-wb // 18)
    if mode == "auto" and sum(elements(u) for us in users.values() for u in us) <= DSP_ELEMENTS:
        return []
    units: List[Dict[str, Any]] = []
    for piped in (True, False):
        for i in range(max([u["unit"] for u in users[piped]] + [-1]) + 1):
            mine = [u for u in users[piped] if u["unit"] == i]
            if len(mine) < 2:
                continue
            units.append({"name": fresh(f"pmul{len(units)}"), "pipelined": piped, "users": mine})
    # (a shared product another's operand, outside a pipelined loop: its register)
    regs_of = {(u["k"], u["t"]): u["held"] for unit in units if not unit["pipelined"]
               for u in unit["users"]}
    for unit in units:
        for u in unit["users"]:
            for side in ("a", "b"):
                e = copy.deepcopy(u[side])
                for x in ast.walk(e):
                    if isinstance(x, ast.Name) and (u["k"], x.id) in regs_of:
                        x.id = regs_of[(u["k"], x.id)]
                u[side] = e
    for unit in units:
        for u in unit["users"]:
            code = codes[u["k"]]
            if unit["pipelined"]:                 # (the product: the unit's, at its stage)
                code.out = [_assign(u["t"], _name(unit["name"])) if s.targets[0].id == u["t"]
                            else s for s in code.out]
                continue
            held = u["held"]

            class _Reg(ast.NodeTransformer):
                def visit_Name(self, x: ast.Name):
                    return _name(held) if x.id == u["t"] else x
            code.out = [_Reg().visit(s) for s in code.out if s.targets[0].id != u["t"]]
            code.shared_held[held] = (unit["name"], u["clock"])
    if units:
        log.info("%d products on %d multipliers", sum(len(u["users"]) for u in units), len(units))
    return units


def emit_units(units, codes, issue_chain, new_attr, fresh, piped: bool) -> List[ast.stmt]:
    """The shared multipliers (share_products) of one kind: each user's operands
    in the unit's formats, picked by the state (and clock), and their product."""
    from .share import _hull
    out: List[ast.stmt] = []
    for unit in units:
        if unit["pipelined"] != piped:
            continue
        operands = {}
        for side in ("a", "b"):
            meta = dict(_fmt_of(_hull((-1, 0), *[u["r" + side] for u in unit["users"]])),
                        type="wire")
            items = []
            for u in unit["users"]:
                w = fresh(f"{unit['name']}_{side}")
                out.append(_assign(w, copy.deepcopy(u[side])))
                new_attr[w] = dict(meta)
                items.append((u["k"], u["clock"] or 0, _name(w)))
            operands[side] = fresh(f"{unit['name']}_{side}")
            out.append(_assign(operands[side], issue_chain(items, ast.Constant(0))))
            new_attr[operands[side]] = dict(meta)
        out.append(_assign(unit["name"], ast.BinOp(left=_name(operands["a"]), op=ast.Mult(),
                                                   right=_name(operands["b"]))))
        lo_a, hi_a = _hull((-1, 0), *[u["ra"] for u in unit["users"]])
        lo_b, hi_b = _hull((-1, 0), *[u["rb"] for u in unit["users"]])
        corners = [x * y for x in (lo_a, hi_a) for y in (lo_b, hi_b)]
        new_attr[unit["name"]] = dict(_fmt_of((min(corners), max(corners))), type="wire")
    return out


def emit_machine(sq, program, blocks, codes, counters, succ, start_state, start_sets,
                 DONE, phases, ret, rams, reg_lists, rng, fmt, attr, memories=(), ext=None,
                 units=(), control_regs=(), steps=(0, 0), aliases=None):
    """The per-clock function (share.py's form) and its schedule. control_regs:
    scalars the control reads (steps.py's slots: loop counts, the state to go to);
    steps: (steps, templates) of steps.py; aliases: a template's name of a slot ->
    the slot (a wire from its register, in the template's format)."""
    from .sequencer import Loop, _walk_items
    fd = sq.fd
    n = len(blocks)
    stmts: List[ast.stmt] = []
    new_attr: Dict[str, Dict[str, Any]] = {}
    regs: Dict[str, str] = {}                    # register -> its next value
    phase = sq.fresh("fphase")
    new_attr[phase] = {"signed": 0, "width": max(1, (phases - 1).bit_length()), "type": "wire"}

    def reg(name: str, meta: Dict[str, Any]) -> str:
        new_attr[name] = dict(meta, type="wire")
        new_attr[f"rn_{name[2:]}" if name.startswith("r_") else f"{name}_n"] = dict(meta, type="wire")
        return name

    def bits(n_values: int) -> Dict[str, Any]:
        return {"signed": 0, "width": max(1, (n_values - 1).bit_length())}

    def at_start(value_now: str, register: str, start: Optional[ast.expr]) -> None:
        stmts.append(_assign(value_now, _name(register) if start is None else ast.IfExp(
            test=_eq(_name(phase), 0), body=start, orelse=_name(register))))

    # ----- control registers: state, clock within the block, loop counters
    st, cyc = "r_st", "r_cyc"
    reg(st, bits(DONE + 1))
    # A block's clocks each time it runs (a pipelined loop's block: the whole loop)
    run = {k: ((codes[k].pipe["n"] - 1) * codes[k].pipe.get("ii", 1) + codes[k].clocks
               if codes[k].pipe else codes[k].clocks) for k in codes}
    max_c = max(run.values())
    reg(cyc, bits(max_c + 1))

    def iterations(k: int, less: int = 0) -> ast.expr:
        """A pipelined loop's iterations (less fewer) times its ii: a slot's count
        (steps.py) or its own."""
        ii = codes[k].pipe.get("ii", 1)
        if codes[k].pipe.get("count"):
            n = ast.BinOp(left=_name(codes[k].pipe["count"]), op=ast.Sub(),
                          right=ast.Constant(less)) if less else _name(codes[k].pipe["count"])
            return n if ii == 1 else ast.BinOp(left=n, op=ast.Mult(), right=ast.Constant(ii))
        return ast.Constant((codes[k].pipe["n"] - less) * ii)

    def last_clock(k: int) -> ast.expr:
        """The clock a block ends at, each time it runs."""
        if codes[k].pipe is not None and codes[k].pipe.get("count"):
            return ast.BinOp(left=iterations(k, 1), op=ast.Add(),
                             right=ast.Constant(codes[k].clocks - 1))
        return ast.Constant(run[k] - 1)
    at_start("st_now", st, start_state)
    at_start("cyc_now", cyc, ast.Constant(0))
    new_attr["st_now"] = dict(bits(DONE + 1), type="wire")
    new_attr["cyc_now"] = dict(bits(max_c + 1), type="wire")
    # (cyc_now % ii for a pipelined loop starting an iteration every ii clocks: a
    # counter beside cyc, not a divider)
    iis = sorted({codes[k].pipe.get("ii", 1) for k in codes if codes[k].pipe} - {1})
    for ii in iis:
        reg(f"r_cm{ii}", bits(ii))
        at_start(f"cm{ii}_now", f"r_cm{ii}", ast.Constant(0))
        new_attr[f"cm{ii}_now"] = dict(bits(ii), type="wire")
    loop_of = {counters[lp.index]: lp for lp in _walk_items(program) if isinstance(lp, Loop)}
    for cnt, lp in loop_of.items():
        meta = _fmt_of((min(lp.values), max(lp.values)))
        reg(f"r_{cnt}", meta)
        new_attr[f"{cnt}_now"] = dict(meta, type="wire")
        at_start(f"{cnt}_now", f"r_{cnt}", start_sets.get(cnt))

    # ----- a template's slots (steps.py): its registers', in its own format
    for name, q in sorted((aliases or {}).items()):
        stmts.append(_assign(name, _name(f"r_{q}")))
        new_attr[name] = fmt(name)

    # ----- every block's logic
    piped_units_at = len(stmts)                   # (the shared multipliers of pipelined loops)
    for k in range(n):
        stmts += codes[k].out
        for r, (v, meta) in codes[k].held.items():      # (a product's register)
            reg(r, meta)
            regs[r] = v
        new_attr.update(codes[k].formats)
        for r, (nxt, meta) in codes[k].extra_regs.items():     # (dividers)
            regs[r] = nxt
            new_attr[r] = dict(meta, type="wire")
    units_at = len(stmts)                         # (the other shared multipliers)

    # ----- pipelined loops: which stages hold an iteration (valid), the delays
    valid: Dict[Tuple[int, int], str] = {}
    for k in range(n):
        pipe = codes[k].pipe
        if pipe is None:
            continue
        v0 = sq.fresh(f"b{k}_valid0")
        ii = pipe.get("ii", 1)
        first = _and(_eq(_name("st_now"), k), ast.Compare(
            left=_name("cyc_now"), ops=[ast.Lt()], comparators=[iterations(k)]))
        if ii > 1:                                   # (an iteration starts every ii clocks)
            first = _and(first, _eq(_name(f"cm{ii}_now"), 0))
        stmts.append(_assign(v0, first))
        new_attr[v0] = {"signed": 0, "width": 1, "type": "wire"}
        valid[(k, 0)] = v0
        # (each loaded from a wire: a register's next value can't be another register)
        for c in range(1, codes[k].clocks):
            r = sq.fresh(f"r_b{k}_valid{c}")
            regs[r] = sq.fresh(f"{r}_n")
            stmts.append(_assign(regs[r], _name(valid[(k, c - 1)])))
            new_attr[r] = new_attr[regs[r]] = {"signed": 0, "width": 1, "type": "wire"}
            valid[(k, c)] = r
        for r, prev in pipe["regs"].items():
            regs[r] = sq.fresh(f"{r}_n")
            stmts.append(_assign(regs[r], _name(prev)))
            new_attr[r] = dict(pipe["formats"][r])
            new_attr[regs[r]] = dict(pipe["formats"][r])

    # ----- registers: scalars, register lists
    # Registers for the scalars read in a block before it sets them (in another
    # block, or a loop's previous iteration) or returned; the rest are wires
    ret_names = {x.id for x in ast.walk(ret) if isinstance(x, ast.Name)}
    assigned = {v for k in codes for v in codes[k].env}
    scalars = sorted({v for k in codes for v in codes[k].read_regs} | (ret_names & assigned)
                     | set(control_regs))
    end = _name("blk_end")
    counted = [k for k in range(n) if codes[k].pipe is not None and codes[k].pipe.get("count")]
    blk_end: ast.expr = ast.Compare(
        left=_name("cyc_now"), ops=[ast.Eq()],
        comparators=[_sel([ast.Constant(0 if k in counted else run[k] - 1) for k in range(n)]
                          + [ast.Constant(0)], _name("st_now"))])
    for k in counted:                            # (a slot's count of iterations: its own test)
        blk_end = ast.IfExp(test=_eq(_name("st_now"), k), body=ast.Compare(
            left=_name("cyc_now"), ops=[ast.Eq()], comparators=[last_clock(k)]), orelse=blk_end)
    stmts.append(_assign("blk_end", blk_end))

    def next_value(per_block: List[ast.expr], keep: ast.expr) -> ast.expr:
        # (a register only a few blocks set: a chain over those, not a multiplexer over
        # every block; a state machine of hundreds of blocks made gigantic ones)
        kept = _key(keep)
        writers = [(k, v) for k, v in enumerate(per_block) if _key(v) != kept]
        # (the blocks that set it to one value: one of the chain, a test of them
        # all; the ends of steps.py's templates, say, one next step)
        values: Dict[Any, Tuple[ast.expr, List[int]]] = {}
        for k, v in writers:
            values.setdefault(_key(v), (v, []))[1].append(k)
        if len(values) <= max(8, len(per_block) // 4):
            value: ast.expr = _copy(keep)
            for v, ks in reversed(list(values.values())):
                test = _eq(_name("st_now"), ks[0]) if len(ks) == 1 else ast.BoolOp(
                    op=ast.Or(), values=[_eq(_name("st_now"), k) for k in ks])
                value = ast.IfExp(test=test, body=_copy(v), orelse=value)
        else:
            value = _sel(per_block + [keep], _name("st_now"))
        return ast.IfExp(test=copy.deepcopy(end), body=value, orelse=keep)

    for v in scalars:
        if v not in rng:
            raise _err(f"'{v}' is read before it is set")
        reg(f"r_{v}", fmt(v))
        per = [_name(codes[k].env.get(v, f"r_{v}") if codes[k].pipe is None else f"r_{v}")
               for k in range(n)]
        value = next_value(per, _name(f"r_{v}"))
        for k in range(n):                 # (a pipelined block: at its stage, each iteration)
            if codes[k].pipe is not None and v in codes[k].env:
                value = ast.IfExp(test=_name(valid[(k, codes[k].pipe["stage"][v])]),
                                  body=_name(codes[k].env[v]), orelse=value)
        stmts.append(_assign(f"rn_{v}", value))
        regs[f"r_{v}"] = f"rn_{v}"
    # Memories: their 'init' (raw), wrapped to the format they are kept in
    reg_init: Dict[str, int] = {}
    mem_info: Dict[str, Dict[str, Any]] = {}
    for a in memories:
        meta = fmt(a)
        init = attr[a].get("init", 0)
        init = list(init) if isinstance(init, (list, tuple)) else [init] * sq.lists[a]
        if len(init) != sq.lists[a]:
            raise _err(f"'init' of memory '{a}' needs {sq.lists[a]} values")
        w, sgn = int(meta["width"]), bool(meta["signed"])
        mem_info[a] = {"length": sq.lists[a], "width": w, "signed": int(sgn),
                       "init": [_wrap(int(v), w, sgn) for v in init]}
    for a in sorted(reg_lists):
        meta = fmt(a)
        for e in range(sq.lists[a]):
            r = f"r_{a}_{e}"
            reg(r, meta)
            if a in mem_info and mem_info[a]["init"][e]:
                reg_init[r] = mem_info[a]["init"][e]
            per = [_name(codes[k].elems[a][e]) if a in codes[k].elems else _name(r)
                   for k in range(n)]
            stmts.append(_assign(f"rn_{a}_{e}", next_value(per, _name(r))))
            regs[r] = f"rn_{a}_{e}"

    # ----- control next values
    stmts.append(_assign("rn_cyc", ast.IfExp(test=copy.deepcopy(end), body=ast.Constant(0),
                                             orelse=ast.BinOp(left=_name("cyc_now"), op=ast.Add(),
                                                              right=ast.Constant(1)))))
    regs[cyc] = "rn_cyc"
    for ii in iis:
        last = _eq(_name(f"cm{ii}_now"), ii - 1)
        stmts.append(_assign(f"rn_cm{ii}", ast.IfExp(
            test=ast.BoolOp(op=ast.Or(), values=[copy.deepcopy(end), last]), body=ast.Constant(0),
            orelse=ast.BinOp(left=_name(f"cm{ii}_now"), op=ast.Add(), right=ast.Constant(1)))))
        regs[f"r_cm{ii}"] = f"rn_cm{ii}"
    stmts.append(_assign("rn_st", next_value([succ[k][0] for k in range(n)],
                                             _name("st_now"))))
    regs[st] = "rn_st"
    for cnt in loop_of:
        per = [succ[k][1].get(cnt, _name(f"{cnt}_now")) for k in range(n)]
        value = next_value(per, _name(f"{cnt}_now"))
        for k in range(n):                 # (its pipelined loop: a step a clock, to the last)
            pipe = codes[k].pipe
            if pipe is not None and pipe["counter"] == cnt:
                lp = loop_of[cnt]
                step = lp.values[1] - lp.values[0]
                ii = pipe.get("ii", 1)
                test = _and(_eq(_name("st_now"), k), ast.Compare(
                    left=_name("cyc_now"), ops=[ast.Lt()], comparators=[iterations(k, 1)]))
                if ii > 1:                           # (a step every ii clocks)
                    test = _and(test, _eq(_name(f"cm{ii}_now"), ii - 1))
                value = ast.IfExp(test=test,
                    body=ast.BinOp(left=_name(f"{cnt}_now"), op=ast.Add(), right=ast.Constant(step)),
                    orelse=value)
        stmts.append(_assign(f"rn_{cnt}", value))
        regs[f"r_{cnt}"] = f"rn_{cnt}"

    # ----- RAM ports (a register file per read port, all taking the writes)
    rfs: Dict[str, Dict[str, Any]] = {}
    roms: Dict[str, Dict[str, Any]] = {}
    extra_params: List[str] = []
    extra_returns: List[str] = []

    def issue_chain(items: List[Tuple[int, int, ast.expr]], hold: ast.expr,
                    write: bool = False) -> ast.expr:
        """(block, clock, value) -> the value in that block's clock, else hold. In a
        pipelined block: a write at its stage while an iteration is there, a read
        every clock."""
        def chain(part: List[Tuple[int, int, ast.expr]]) -> ast.expr:
            # (a block's items of one value: one of the chain, its clocks together)
            merged: Dict[Any, Tuple[ast.expr, List[ast.expr]]] = {}
            for k, c, v in part:
                if codes[k].pipe is None:
                    test = _and(_eq(_name("st_now"), k), _eq(_name("cyc_now"), c))
                else:
                    test = _name(valid[(k, c)]) if write else _eq(_name("st_now"), k)
                key = (k, _key(v))
                if key not in merged:
                    merged[key] = (v, [])
                merged[key][1].append(test)
            out = hold
            for v, tests in reversed(list(merged.values())):
                test = tests[0] if len(tests) == 1 else ast.BoolOp(op=ast.Or(), values=tests)
                out = ast.IfExp(test=test, body=v, orelse=out)
            return out
        blocks: Dict[int, List[Tuple[int, int, ast.expr]]] = {}
        for item in items:
            blocks.setdefault(item[0], []).append(item)
        if len(blocks) <= CHAIN_BLOCKS:
            return chain(items)
        # (many blocks: a search on the state, as deep as the log of their number;
        # a chain as long as them nests past what a parser or a simulator takes)
        keys = sorted(blocks)

        def search(lo: int, hi: int) -> ast.expr:
            if hi - lo == 1:
                return chain(blocks[keys[lo]])
            mid = (lo + hi) // 2
            return ast.IfExp(test=ast.Compare(left=_name("st_now"), ops=[ast.Lt()],
                                              comparators=[ast.Constant(keys[mid])]),
                             body=search(lo, mid), orelse=search(mid, hi))
        return search(0, len(keys))

    # ----- shared multipliers (share_products): a pipelined loop's operands are
    # registers, its units first; the others' after every block, with the
    # registers that load their products
    if units:
        late = emit_units(units, codes, issue_chain, new_attr, sq.fresh, piped=False)
        for k in range(n):
            for r, (unit, c) in codes[k].shared_held.items():
                nxt = sq.fresh(f"{r}_n")
                late.append(_assign(nxt, ast.IfExp(
                    test=_and(_eq(_name("st_now"), k), _eq(_name("cyc_now"), c)),
                    body=_name(unit), orelse=_name(r))))
                new_attr[nxt] = dict(new_attr[r])
                regs[r] = nxt
        stmts[units_at:units_at] = late
        stmts[piped_units_at:piped_units_at] = emit_units(units, codes, issue_chain, new_attr,
                                                          sq.fresh, piped=True)

    ext_lists = (ext or {}).get("lists", {})
    for a in sorted(set(rams) - set(ext_lists)):
        meta = fmt(a)
        depth = sq.lists[a]
        ab = bits(depth)
        nports = max([sum(1 for s in codes[k].sites if s.array == a) for k in codes] + [1])
        writes = [(k, w.clock, w) for k in codes for w in codes[k].writes if w.array == a]
        wnames = {key: sq.fresh(f"{a}_{key}") for key in ("we", "waddr", "wdata")}
        stmts.append(_assign(wnames["we"], issue_chain(
            [(k, c, _name(w.cond) if w.cond else ast.Constant(1)) for k, c, w in writes],
            ast.Constant(0), write=True)))
        stmts.append(_assign(wnames["waddr"], issue_chain(
            [(k, c, _name(w.addr)) for k, c, w in writes], ast.Constant(0), write=True)))
        stmts.append(_assign(wnames["wdata"], issue_chain(
            [(k, c, _name(w.data)) for k, c, w in writes], ast.Constant(0), write=True)))
        new_attr[wnames["wdata"]] = dict(meta, type="wire")
        for p in range(nports):
            q = sq.fresh(f"{a}_q{p}")
            hold = sq.fresh(f"r_{a}_ra{p}")
            items = []
            for k in codes:
                mine = [s for s in codes[k].sites if s.array == a]
                if p < len(mine):
                    s = mine[p]
                    items.append((k, s.clock, _name(s.addr)))
                    stmts.insert(0, _assign(s.data, _name(q)))
            raddr = sq.fresh(f"{a}_raddr{p}")
            stmts.append(_assign(raddr, issue_chain(items, _name(hold))))
            new_attr[hold] = dict(ab, type="wire")
            new_attr[raddr] = dict(ab, type="wire")
            regs[hold] = raddr
            names = dict(wnames)
            if p:                                     # (each copy its own write port names)
                for key in ("we", "waddr", "wdata"):
                    names[key] = sq.fresh(f"{a}_{key}{p}")
                    stmts.append(_assign(names[key], _name(wnames[key])))
                new_attr[names["wdata"]] = dict(meta, type="wire")
            new_attr[q] = dict(meta, type="wire")
            rfs[q] = dict(meta, depth=depth, address_bits=ab["width"], raddr=raddr, **names)
            if a in mem_info:
                rfs[q]["init"] = mem_info[a]["init"]
            extra_params.append(q)
            extra_returns += [raddr, names["we"], names["waddr"], names["wdata"]]

    # ----- ROM ports (one per read, by table)
    tables: Dict[Tuple[int, ...], List[Tuple[int, _Site]]] = {}
    for k in codes:
        for s in codes[k].sites:
            if s.rom and s.ext is None:
                tables.setdefault(s.array, []).append((k, s))
    for t, (entries, sites) in enumerate(tables.items()):
        nports = max(sum(1 for kk, _ in sites if kk == k) for k in codes)
        lo, hi = min(entries), max(entries)
        meta = _fmt_of((lo, hi))
        ab = bits(len(entries))
        for p in range(nports):
            q = sq.fresh(f"tab{t}_q{p}")
            hold = sq.fresh(f"r_tab{t}_a{p}")
            items = []
            for k in codes:
                mine = [s for kk, s in sites if kk == k]
                if p < len(mine):
                    items.append((k, mine[p].clock, _name(mine[p].addr)))
                    stmts.insert(0, _assign(mine[p].data, _name(q)))
            idx = sq.fresh(f"tab{t}_i{p}")
            stmts.append(_assign(idx, issue_chain(items, _name(hold))))
            new_attr[hold] = dict(ab, type="wire")
            new_attr[idx] = dict(ab, type="wire")
            regs[hold] = idx
            new_attr[q] = dict(meta, type="wire")
            roms[q] = {"index": idx, "entries": list(entries), "width": meta["width"],
                       "signed": meta["signed"], "address_bits": ab["width"]}
            extra_params.append(q)
            extra_returns.append(idx)

    # ----- the SDRAM's port (sdram=): one access a clock, its address the list's
    # base plus the index; a read's data on ext_q EXT_LATENCY clocks after it
    sdram_info = None
    ext_reads = [(k, s) for k in codes for s in codes[k].sites if s.ext is not None]
    ext_writes = [(k, w) for k in codes for w in codes[k].writes if w.ext is not None]
    if ext_reads or ext_writes:
        # (an address that's another's copy, a high word's: that one, so a block's
        # items of it are one)
        copies = {s.targets[0].id: s.value.id for code in codes.values() for s in code.out
                  if isinstance(s.value, ast.Name)}

        def at_base(base: int, a: str) -> ast.expr:
            while a in copies:
                a = copies[a]
            return ast.BinOp(left=ast.Constant(base), op=ast.Add(), right=_name(a))
        # (in a pipelined loop a read only while an iteration is at its stage: the
        # reads of an iteration take turns on the port)
        rd = issue_chain([(k, s.clock, ast.Constant(1)) for k, s in ext_reads], ast.Constant(0),
                         write=True)
        wr = issue_chain([(k, w.clock, _name(w.cond) if w.cond else ast.Constant(1))
                          for k, w in ext_writes], ast.Constant(0), write=True)
        raddr = issue_chain([(k, s.clock, at_base(s.ext, s.addr)) for k, s in ext_reads],
                            ast.Constant(0), write=True)
        waddr = issue_chain([(k, w.clock, at_base(w.ext, w.addr)) for k, w in ext_writes],
                            ast.Constant(0), write=True)
        # (a value's high word: at its low word's address, which is even, plus one)
        odd = issue_chain([(k, s.clock, ast.Constant(1)) for k, s in ext_reads if s.half == "hi"]
                          + [(k, w.clock, ast.Constant(1)) for k, w in ext_writes
                             if w.half == "hi"], ast.Constant(0), write=True)
        stmts.append(_assign("ext_we", wr))
        stmts.append(_assign("ext_req", ast.BoolOp(op=ast.Or(), values=[_name("ext_we"), rd])))
        stmts.append(_assign("ext_odd", odd))
        stmts.append(_assign("ext_addr", ast.BinOp(
            left=ast.IfExp(test=_name("ext_we"), body=waddr, orelse=raddr), op=ast.BitOr(),
            right=_name("ext_odd"))))
        stmts.append(_assign("ext_wdata", issue_chain(
            [(k, w.clock, _name(w.data)) for k, w in ext_writes], ast.Constant(0), write=True)))
        one = {"signed": 0, "width": 1, "type": "wire"}
        new_attr.update({"ext_we": one, "ext_req": one, "ext_odd": one,
                         "ext_addr": {"signed": 0, "width": 24, "type": "wire"},
                         "ext_wdata": {"signed": 0, "width": 16, "type": "wire"},
                         "ext_q": {"signed": 0, "width": 16, "type": "wire"}})
        # (outside a pipelined loop: a block's j-th read held in the j-th of a few
        # registers, every block's, loaded the clock its data comes, read after)
        pool: List[str] = []
        arrive: List[Dict[int, int]] = []           # j -> {block: the clock its j-th comes}
        nth: Dict[int, int] = {}
        for k, s in sorted(ext_reads, key=lambda ks: (ks[0], ks[1].clock)):
            meta = (_fmt_of((min(s.array), max(s.array))) if s.rom else fmt(s.array))
            if s.half == "lo":                           # (a value in two words)
                meta = {"signed": 0, "width": 16}
            elif s.half == "hi":
                meta = dict(meta, width=int(meta["width"]) - 16)
            new_attr[s.data] = dict(meta, type="wire")
            if codes[k].pipe is not None:
                # (a pipelined loop: the data at its stage, delayed after as any value)
                stmts.insert(0, _assign(s.data, _name("ext_q")))
                continue
            j = nth.get(k, 0)
            nth[k] = j + 1
            if j == len(pool):
                pool.append(reg(sq.fresh(f"r_extq{j}"), {"signed": 0, "width": 16}))
                arrive.append({})
            arrive[j][k] = s.clock + EXT_LATENCY
            stmts.insert(0, _assign(s.data, _name(pool[j])))
        for j, r in enumerate(pool):
            ks = sorted(arrive[j])
            if len(ks) <= 8:
                load: ast.expr = ast.BoolOp(op=ast.Or(), values=[
                    _and(_eq(_name("st_now"), k), _eq(_name("cyc_now"), arrive[j][k])) for k in ks]) \
                    if len(ks) > 1 else _and(_eq(_name("st_now"), ks[0]),
                                             _eq(_name("cyc_now"), arrive[j][ks[0]]))
            else:                                     # (by the state: its clock, if it has one)
                load = _and(_sel([ast.Constant(int(k in arrive[j])) for k in range(n)]
                                 + [ast.Constant(0)], _name("st_now")),
                            ast.Compare(left=_name("cyc_now"), ops=[ast.Eq()], comparators=[
                                _sel([ast.Constant(arrive[j].get(k, 0)) for k in range(n)]
                                     + [ast.Constant(0)], _name("st_now"))]))
            nxt = sq.fresh(f"{r}_n")
            stmts.append(_assign(nxt, ast.IfExp(test=load, body=_name("ext_q"), orelse=_name(r))))
            new_attr[nxt] = dict(new_attr[r])
            regs[r] = nxt
        extra_params.append("ext_q")
        extra_returns += ["ext_req", "ext_we", "ext_addr", "ext_wdata"]
        sdram_info = dict(ext, latency=EXT_LATENCY)

    # ----- the outputs, from the registers (read in the last clock)
    class _Out(ast.NodeTransformer):
        def visit_Name(self, x: ast.Name):
            if x.id in reg_lists:
                return ast.List(elts=[_name(f"r_{x.id}_{e}") for e in range(sq.lists[x.id])],
                                ctx=ast.Load())
            if x.id in scalars:
                return _name(f"r_{x.id}")
            return x
    vals = ret.value.elts if isinstance(ret.value, ast.Tuple) else [ret.value]
    out_names = []
    # (the return's values in order: a list in the SDRAM by its name, else how many
    # ports it has)
    returns: List[Tuple[Optional[str], int]] = []
    for i, v in enumerate(vals):
        if isinstance(v, ast.Name) and ext_lists.get(v.id, {}).get("output"):
            returns.append((v.id, sq.lists[v.id]))
            continue
        if isinstance(v, ast.Name):
            returns.append((None, 1 if v.id in scalars else sq.lists.get(v.id, 1)))
        else:
            returns.append((None, len(v.elts) if isinstance(v, (ast.List, ast.Tuple)) else 1))
        name = v.id if isinstance(v, ast.Name) else (f"ret{i}" if len(vals) > 1 else "ret")
        if isinstance(v, ast.Name) and v.id in sq.params and v.id not in scalars:
            out_names.append(name)                    # (an argument returned as it is)
            continue
        stmts.append(_assign(name, _Out().visit(copy.deepcopy(v))))
        out_names.append(name)

    if sdram_info is not None:
        sdram_info["returns"] = returns
    reg_names = list(regs)
    ret_tuple = ast.Tuple(elts=[_name(x) for x in out_names] + [_name(regs[r]) for r in reg_names]
                          + [_name(x) for x in extra_returns], ctx=ast.Load())
    stmts.append(ast.Return(value=ret_tuple))
    # (reads of a port's data come first: the data is an input of the clock)
    new_fd = copy.deepcopy(fd)
    new_fd.body = stmts
    new_fd.args.args = ([x for x in fd.args.args if x.arg not in mem_info] + [ast.arg(arg=phase)]
                        + [ast.arg(arg=r) for r in reg_names] + [ast.arg(arg=q) for q in extra_params])
    tree = ast.fix_missing_locations(ast.Module(body=[new_fd], type_ignores=[]))
    for r in reg_names:                               # (next-value formats as the registers)
        new_attr.setdefault(regs[r], dict(new_attr[r]))
    return tree, Schedule(
        "sequenced", phases, phase=phase, regs=regs,
        params={phase} | set(reg_names) | set(extra_params), attr=new_attr, reg_init=reg_init,
        roms=roms, rfs=rfs, extra_params=extra_params, extra_returns=extra_returns,
        register_bits=sum(int(new_attr[r]["width"]) for r in reg_names), memories=mem_info,
        sdram=sdram_info, blocks=n, pipelined=sum(1 for k in codes if codes[k].pipe is not None),
        steps=steps[0], templates=steps[1])


def _wrap(n: int, w: int, signed: bool) -> int:
    n &= (1 << w) - 1
    return n - (1 << w) if signed and n >> (w - 1) else n


def _err(msg: str):
    from .sequencer import NotSequenceable
    return NotSequenceable(msg)


# ==================================================================== pipelined loops

def _expand(expr: ast.expr, defs: Dict[str, ast.expr], depth: int = 0) -> ast.expr:
    """expr with the block's own temporaries replaced by their definitions."""
    if depth > 20:
        return expr

    class _X(ast.NodeTransformer):
        def visit_Name(self, n: ast.Name):
            if n.id in defs:
                return _expand(copy.deepcopy(defs[n.id]), defs, depth + 1)
            return n
    return _X().visit(copy.deepcopy(expr))


def _coefficient(e: ast.expr, now: str, varying: Set[str]) -> Optional[int]:
    """e's coefficient of the loop counter `now` if e is that times a constant plus
    values that don't change in the loop; else None."""
    if isinstance(e, ast.Name):
        return 1 if e.id == now else (None if e.id in varying else 0)
    if isinstance(e, ast.Constant) and type(e.value) is int:
        return 0
    if isinstance(e, ast.BinOp) and isinstance(e.op, (ast.Add, ast.Sub)):
        a, b = _coefficient(e.left, now, varying), _coefficient(e.right, now, varying)
        if a is None or b is None:
            return None
        return a + b if isinstance(e.op, ast.Add) else a - b
    if isinstance(e, ast.BinOp) and isinstance(e.op, ast.Mult):
        for c, x in ((e.left, e.right), (e.right, e.left)):
            if isinstance(c, ast.Constant) and type(c.value) is int:
                k = _coefficient(x, now, varying)
                return None if k is None else c.value * k
    if isinstance(e, ast.UnaryOp) and isinstance(e.op, ast.USub):
        k = _coefficient(e.operand, now, varying)
        return None if k is None else -k
    names = {n.id for n in ast.walk(e) if isinstance(n, ast.Name)}
    return 0 if not names & (varying | {now}) else None


def pipeline_block(code: "BlockCode", cnt: str, n_iter: int, known: Dict[str, Tuple[int, int]],
                   attr) -> bool:
    """
    Retime an innermost loop's one block to start an iteration every clock: each
    value is computed at its stage (the clock of the iteration it belongs to at
    which its inputs are there), and one used at a later stage reaches it through
    delay registers. A value carried from one iteration to the next (acc = acc +
    ...) is computed and stored at one stage, so the next iteration, a clock
    later, finds it there. A loop of N iterations then takes N + stages - 1
    clocks instead of N * stages. Returns False (and leaves the block as it is)
    if the block can't: a divider or a square root (several clocks each), a
    list in registers written, two writes to one RAM, a RAM read elsewhere than
    where it is written (a later iteration could read before an earlier one
    writes), or a value carried through a RAM read.
    known: ranges of the registers and counters (for the delay registers' widths).
    """
    from .share import NotShareable, _fmt_of, _fmt_range, _interval

    def no(why: str) -> bool:                         # (said why, at -v)
        log.info("Loop of block %d not pipelined: %s", code.k, why)
        return False
    if n_iter < 2:
        return False
    if code.extra_regs:
        return no("a divider or a square root, a bit a clock (tried again written out, a "
                  "step a stage)")
    if code.elems:
        return no("a list in registers written")
    # An iteration every ii clocks: as many as its SDRAM accesses (one a clock)
    ii = max(1, sum(1 for s in code.sites if s.ext is not None)
             + sum(1 for w in code.writes if w.ext is not None))
    now = f"{cnt}_now"
    defs = {s.targets[0].id: s.value for s in code.out}
    carried = {v for v in code.read_regs if v in code.env}
    varying = set(defs) | {s.data for s in code.sites} | {now} | {f"r_{v}" for v in carried}
    # One write per RAM; a RAM written here is read only where it is written, an
    # element a different one each iteration
    written: Dict[str, _Write] = {}
    for w in code.writes:
        if w.array in written:
            return no(f"two writes to {w.array}")
        written[w.array] = w
    for s in code.sites:
        if not s.rom and s.array in written:
            ra, wa = _expand(defs[s.addr], defs), _expand(defs[written[s.array].addr], defs)
            if ast.dump(ra) != ast.dump(wa) or not _coefficient(ra, now, varying - {now}):
                return no(f"{s.array} read elsewhere than where it is written")

    # Stages: a statement's is the latest of its inputs'; a carried value's register
    # is there at the stage its new value is
    ready: Dict[str, int] = {}
    data_of = {s.addr: s.data for s in code.sites}
    latency = {s.addr: EXT_LATENCY if s.ext is not None else 1 for s in code.sites}
    ext_addrs = [s.addr for s in code.sites if s.ext is not None]
    # (an SDRAM read issued this much later: a stage after its address at least, the
    # port's address from a register, not from the stage's products and sums)
    bump: Dict[str, int] = {a: 1 for a in ext_addrs}

    def stage(e: ast.expr) -> int:
        return max([0] + [ready.get(n.id, 0) for n in ast.walk(e)
                          if isinstance(n, ast.Name) and n.id in varying])

    def settle() -> bool:
        for _ in range(len(code.out) + 3):
            before = dict(ready)
            for v in carried:
                ready[f"r_{v}"] = ready.get(code.env[v], 0)
            for s in code.out:
                t = s.targets[0].id
                ready[t] = stage(s.value) + (1 if t in code.bumped else 0)
                if t in data_of:
                    ready[data_of[t]] = ready[t] + bump.get(t, 0) + latency[t]
            if ready == before:
                return True
        return False                                  # (carried through a read: no fixed stage)
    if not settle():
        return no("a value carried through a read (no fixed stage)")
    if ii > 1:
        # The SDRAM's reads at different clocks of the ii (a read a stage later where
        # two would meet), the stages settled again
        for _ in range(4 * ii):
            used: Set[int] = set()
            moved = False
            for a in ext_addrs:
                at = ready[a] + bump[a]
                while at % ii in used:
                    at += 1
                    moved = True
                bump[a] = at - ready[a]
                used.add(at % ii)
            if not moved:
                break
            if not settle():
                return no("a value carried through a read (no fixed stage)")

    # Ranges of the block's values (for the delay registers' widths); one that
    # can't be bounded takes its declared format
    rng = dict(known)
    for s in code.out:
        t = s.targets[0].id
        try:
            r = _interval(s.value, rng, attr)
        except NotShareable as e:
            if t not in code.formats:
                log.warning("Loop of block %d not pipelined: can't bound %s (%s)", code.k, t, e)
                return False
            r = _fmt_range(code.formats[t])
        if t in code.formats:
            f = _fmt_range(code.formats[t])
            if not (f[0] <= r[0] and r[1] <= f[1]):
                r = f
        rng[t] = r

    delays: Dict[Tuple[str, int], str] = {}
    regs: Dict[str, str] = {}
    formats: Dict[str, Dict[str, Any]] = {}

    def delayed(x: str, d: int) -> str:
        prev = x
        for j in range(1, d + 1):
            if (x, j) not in delays:
                delays[(x, j)] = code.sq.fresh(f"{x}_p{j}")
                regs[delays[(x, j)]] = prev
                formats[delays[(x, j)]] = dict(_fmt_of(rng[x]), type="wire")
            prev = delays[(x, j)]
        return prev

    def at(e: ast.expr, s: int) -> ast.expr:
        class _T(ast.NodeTransformer):
            def visit_Name(self, n: ast.Name):
                if n.id in varying and ready.get(n.id, 0) < s:
                    return _name(delayed(n.id, s - ready.get(n.id, 0)))
                return n
        return _T().visit(copy.deepcopy(e))
    for x in varying:
        rng.setdefault(x, known.get(x, (0, 0)))
    out = []
    for s in code.out:
        t = s.targets[0].id
        out.append(_assign(t, at(s.value, ready[t])))
    last = max([0] + list(ready.values()))
    taken = {(ready[a] + bump[a]) % ii for a in ext_addrs}
    for w in code.writes:
        w.clock = max(ready.get(w.addr, 0), ready.get(w.data, 0), ready.get(w.cond, 0) if w.cond else 0)
        if w.ext is not None:                        # (the SDRAM's: from registers, a stage on)
            w.clock += 1
        if w.ext is not None and ii > 1:             # (its own clock of the ii, too)
            while w.clock % ii in taken:
                w.clock += 1
            taken.add(w.clock % ii)
        w.addr, w.data = (delayed(w.addr, w.clock - ready.get(w.addr, 0)),
                          delayed(w.data, w.clock - ready.get(w.data, 0)))
        if w.cond:
            w.cond = delayed(w.cond, w.clock - ready.get(w.cond, 0))
        last = max(last, w.clock)
    for s in code.sites:
        b = bump.get(s.addr, 0)
        s.clock = ready[s.addr] + b
        if b:
            s.addr = delayed(s.addr, b)
            last = max(last, s.clock)
    code.out = out
    code.clocks = last + 1
    code.pipe = {"counter": cnt, "n": n_iter, "ii": ii, "regs": regs, "formats": formats,
                 "stage": {v: ready.get(code.env[v], 0) for v in code.env}}
    return True
