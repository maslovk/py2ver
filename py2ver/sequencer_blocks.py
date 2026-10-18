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
from typing import Any, Dict, List, Optional, Set, Tuple

from .share import _fmt_of, _hull


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


class _Site:
    def __init__(self, array: Any, addr: str, data: str, clock: int, rom: bool) -> None:
        self.array = array            # a RAM list's name, or a ROM table's entries
        self.addr = addr
        self.data = data
        self.clock = clock
        self.rom = rom


class _Write:
    def __init__(self, array: str, addr: str, data: str, cond: Optional[str], ready: int) -> None:
        self.array, self.addr, self.data, self.cond, self.ready = array, addr, data, cond, ready
        self.clock = 0


class BlockCode:
    """One block's statements as the machine's logic."""

    def __init__(self, sq, block, scope: Dict[str, str], rams: Set[str], reg_lists: Set[str],
                 inputs: Set[str], attr, rng, ns, unrolled: bool = False) -> None:
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
        for s in block.stmts:
            self.stmt(s, None)
        self.schedule()

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

    def read(self, a: str, idx: ast.expr) -> ast.expr:
        if a in self.inputs:
            return ast.Subscript(value=_name(a), slice=idx, ctx=ast.Load())
        if a in self.rams:
            key = (a, ast.dump(idx))
            if key in self.reads:
                return copy.deepcopy(self.reads[key])
            addr = self.emit(f"{a}_ra", idx)
            data = self.tmp(f"{a}_rd")
            clock = self.ready[addr]
            self.sites.append(_Site(a, addr, data, clock, rom=False))
            self.ready[data] = clock + 1
            value: ast.expr = _name(data)
            # A write earlier in the block to the same element: its value
            for w in self.writes:
                if w.array == a:
                    test = ast.Compare(left=_name(addr), ops=[ast.Eq()], comparators=[_name(w.addr)])
                    if w.cond:
                        test = _and(_name(w.cond), test)
                    value = ast.IfExp(test=test, body=_name(w.data), orelse=value)
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
        addr = self.emit("rom_a", idx)
        data = self.tmp("rom_d")
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

    def set_scalar(self, v: str, value: ast.expr, cond: Optional[ast.expr]) -> None:
        if v in self.scope:
            raise self.NotSequenceable(f"The loop variable '{v}' is assigned in its loop")
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
            addr = self.emit(f"{a}_wa", idx)
            data = self.emit(f"{a}_wd", value)
            self.declare(data, a)
            c = self.emit(f"{a}_we", cond) if cond is not None else None
            self.writes.append(_Write(a, addr, data, c, max(self.ready[addr], self.ready[data],
                                                            self.ready.get(c, 0) if c else 0)))
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
            busy[w.array] = w.clock
            last = max(last, w.clock)
        last = max([last] + [s.clock for s in self.sites])
        self.clocks = last + 1


def emit_machine(sq, program, blocks, codes, counters, succ, start_state, start_sets,
                 DONE, phases, ret, rams, reg_lists, rng, fmt, attr, memories=()):
    """The per-clock function (share.py's form) and its info."""
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
    run = {k: (codes[k].pipe["n"] + codes[k].clocks - 1 if codes[k].pipe else codes[k].clocks)
           for k in codes}
    max_c = max(run.values())
    reg(cyc, bits(max_c + 1))
    at_start("st_now", st, start_state)
    at_start("cyc_now", cyc, ast.Constant(0))
    new_attr["st_now"] = dict(bits(DONE + 1), type="wire")
    new_attr["cyc_now"] = dict(bits(max_c + 1), type="wire")
    loop_of = {counters[lp.index]: lp for lp in _walk_items(program) if isinstance(lp, Loop)}
    for cnt, lp in loop_of.items():
        meta = _fmt_of((min(lp.values), max(lp.values)))
        reg(f"r_{cnt}", meta)
        new_attr[f"{cnt}_now"] = dict(meta, type="wire")
        at_start(f"{cnt}_now", f"r_{cnt}", start_sets.get(cnt))

    # ----- every block's logic
    for k in range(n):
        stmts += codes[k].out
        new_attr.update(codes[k].formats)
        for r, (nxt, meta) in codes[k].extra_regs.items():     # (dividers)
            regs[r] = nxt
            new_attr[r] = dict(meta, type="wire")

    # ----- pipelined loops: which stages hold an iteration (valid), the delays
    valid: Dict[Tuple[int, int], str] = {}
    for k in range(n):
        pipe = codes[k].pipe
        if pipe is None:
            continue
        v0 = sq.fresh(f"b{k}_valid0")
        stmts.append(_assign(v0, _and(_eq(_name("st_now"), k), ast.Compare(
            left=_name("cyc_now"), ops=[ast.Lt()], comparators=[ast.Constant(pipe["n"])]))))
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
    scalars = sorted({v for k in codes for v in codes[k].read_regs} | (ret_names & assigned))
    end = _name("blk_end")
    stmts.append(_assign("blk_end", ast.Compare(
        left=_name("cyc_now"), ops=[ast.Eq()],
        comparators=[_sel([ast.Constant(run[k] - 1) for k in range(n)]
                          + [ast.Constant(0)], _name("st_now"))])))

    def next_value(per_block: List[ast.expr], keep: ast.expr) -> ast.expr:
        return ast.IfExp(test=copy.deepcopy(end), body=_sel(per_block + [keep], _name("st_now")),
                         orelse=keep)

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
                value = ast.IfExp(test=_and(_eq(_name("st_now"), k), ast.Compare(
                    left=_name("cyc_now"), ops=[ast.Lt()], comparators=[ast.Constant(pipe["n"] - 1)])),
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
        out = hold
        for k, c, v in reversed(items):
            if codes[k].pipe is None:
                test = _and(_eq(_name("st_now"), k), _eq(_name("cyc_now"), c))
            else:
                test = _name(valid[(k, c)]) if write else _eq(_name("st_now"), k)
            out = ast.IfExp(test=test, body=v, orelse=out)
        return out

    for a in sorted(rams):
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
            if s.rom:
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
    for i, v in enumerate(vals):
        name = v.id if isinstance(v, ast.Name) else (f"ret{i}" if len(vals) > 1 else "ret")
        if isinstance(v, ast.Name) and v.id in sq.params and v.id not in scalars:
            out_names.append(name)                    # (an argument returned as it is)
            continue
        stmts.append(_assign(name, _Out().visit(copy.deepcopy(v))))
        out_names.append(name)

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
    info = {"phases": phases, "phase": phase, "loop": "(sequenced)", "regs": regs,
            "roms": roms, "rfs": rfs, "extra_params": extra_params,
            "extra_returns": extra_returns, "params": {phase} | set(reg_names) | set(extra_params),
            "attr": new_attr, "register_bits": sum(int(new_attr[r]["width"]) for r in reg_names),
            "blocks": n, "mul_widths": [], "memories": mem_info, "reg_init": reg_init,
            "pipelined": sum(1 for k in codes if codes[k].pipe is not None)}
    return tree, info


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
    if code.extra_regs or code.elems or n_iter < 2:
        return False
    now = f"{cnt}_now"
    defs = {s.targets[0].id: s.value for s in code.out}
    carried = {v for v in code.read_regs if v in code.env}
    varying = set(defs) | {s.data for s in code.sites} | {now} | {f"r_{v}" for v in carried}
    # One write per RAM; a RAM written here is read only where it is written, an
    # element a different one each iteration
    written: Dict[str, _Write] = {}
    for w in code.writes:
        if w.array in written:
            return False
        written[w.array] = w
    for s in code.sites:
        if not s.rom and s.array in written:
            ra, wa = _expand(defs[s.addr], defs), _expand(defs[written[s.array].addr], defs)
            if ast.dump(ra) != ast.dump(wa) or not _coefficient(ra, now, varying - {now}):
                return False

    # Stages: a statement's is the latest of its inputs'; a carried value's register
    # is there at the stage its new value is
    ready: Dict[str, int] = {}
    data_of = {s.addr: s.data for s in code.sites}

    def stage(e: ast.expr) -> int:
        return max([0] + [ready.get(n.id, 0) for n in ast.walk(e)
                          if isinstance(n, ast.Name) and n.id in varying])
    for _ in range(len(code.out) + 3):
        before = dict(ready)
        for v in carried:
            ready[f"r_{v}"] = ready.get(code.env[v], 0)
        for s in code.out:
            t = s.targets[0].id
            ready[t] = stage(s.value) + (1 if t in code.bumped else 0)
            if t in data_of:
                ready[data_of[t]] = ready[t] + 1
        if ready == before:
            break
    else:
        return False                                  # (carried through a read: no fixed stage)

    # Ranges of the block's values (for the delay registers' widths)
    rng = dict(known)
    try:
        for s in code.out:
            t = s.targets[0].id
            r = _interval(s.value, rng, attr)
            if t in code.formats:
                f = _fmt_range(code.formats[t])
                if not (f[0] <= r[0] and r[1] <= f[1]):
                    r = f
            rng[t] = r
    except NotShareable:
        return False

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
    for w in code.writes:
        w.clock = max(ready.get(w.addr, 0), ready.get(w.data, 0), ready.get(w.cond, 0) if w.cond else 0)
        w.addr, w.data = (delayed(w.addr, w.clock - ready.get(w.addr, 0)),
                          delayed(w.data, w.clock - ready.get(w.data, 0)))
        if w.cond:
            w.cond = delayed(w.cond, w.clock - ready.get(w.cond, 0))
        last = max(last, w.clock)
    for s in code.sites:
        s.clock = ready[s.addr]
    code.out = out
    code.clocks = last + 1
    code.pipe = {"counter": cnt, "n": n_iter, "regs": regs, "formats": formats,
                 "stage": {v: ready.get(code.env[v], 0) for v in code.env}}
    return True
