"""
Loop nests of the same form written once: a big model as a loop over its steps.

A model's loop program (a U-Net: hundreds of loop nests, each convolution's
pad, im2col copy, product and bias, each GroupNorm's passes, ...) has a few
dozen forms of nest: the same loops and statements but for their numbers
(sizes, strides, shifts, rounding), the lists they read and write and their
tables. Written out, every nest is its own logic; here each form is written
once, a template, and the program becomes a loop over its steps (the
top-level loop nests and blocks, in order):

    for __step in range(S):
        __q0 = __desc[__step * W]; __q1 = ...; __entry = __desc[__step * W + NP]
        -> the step's template, which goes back to the loop when it's done

A template's numbers that differ between its steps are read from the slots
__q0, __q1, ... (one register each, every template's): a constant, a loop's
count (range(__q3): Loop.count), a list's place in the SDRAM (the lists that
differ are all there: a template reads and writes its step's at the slot's
base), a table's offset in one table of all of its steps'. __desc, a list in
the SDRAM, holds every step's slots and the state its template starts at.
Scalars stay in their step (one set in a step and read in another keeps its
name in every template: steps that differ there are of different forms).
"""
from __future__ import annotations

import ast
import copy
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .share import _fmt_of, _hull

MIN_NESTS = 64          # a program of this many top-level loop nests or more (PY2VER_STEPS=1: any)
SDRAM_SPAN = 1 << 24    # (a list's base: an address of the SDRAM)
SLOT_LIMIT = 1 << 31    # (a slot's value: two SDRAM words, signed)


def _fold(node: ast.AST) -> ast.AST:
    """Products, sums, ... of constants as their values (0 * 32 + i: i)."""
    class _F(ast.NodeTransformer):
        def visit_BinOp(self, n: ast.BinOp):
            self.generic_visit(n)
            if isinstance(n.left, ast.Constant) and isinstance(n.right, ast.Constant) \
                    and type(n.left.value) is int and type(n.right.value) is int \
                    and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.LShift, ast.RShift)):
                a, b = n.left.value, n.right.value
                if isinstance(n.op, (ast.LShift, ast.RShift)) and not 0 <= b < 64:
                    return n
                return ast.Constant({ast.Add: a + b, ast.Sub: a - b, ast.Mult: a * b,
                                     ast.LShift: a << b if isinstance(n.op, ast.LShift) else 0,
                                     ast.RShift: a >> b if isinstance(n.op, ast.RShift) else 0}
                                    [type(n.op)])
            if isinstance(n.op, ast.Add):             # (x + 0, 0 + x: x)
                for x, y in ((n.left, n.right), (n.right, n.left)):
                    if isinstance(y, ast.Constant) and y.value == 0 and type(y.value) is int:
                        return x
            if isinstance(n.op, ast.Mult):            # (0 * x: 0)
                for x, y in ((n.left, n.right), (n.right, n.left)):
                    if isinstance(y, ast.Constant) and y.value == 0 and type(y.value) is int:
                        return ast.Constant(0)
            return n
    return _F().visit(node)


def _items_fold(items) -> None:
    from .sequencer import Block, Loop
    for it in items:
        if isinstance(it, Loop):
            _items_fold(it.body)
        elif isinstance(it, Block):
            it.stmts = [ast.fix_missing_locations(_fold(s)) for s in it.stmts]


@dataclass
class _Ctx:
    literal: Set[str]                    # names kept as they are in every template
    movable: Set[str]                    # lists a template may take from its step
    words: Dict[str, int]                # a list's SDRAM words a value
    attr: Dict[str, Dict[str, Any]]


class _Walker:
    """A step's form (decide None: its key and its leaves, the values a template
    may take from its step) or its template (decide: leaf -> what it is)."""

    def __init__(self, ctx: _Ctx, decide=None) -> None:
        self.ctx = ctx
        self.decide = decide
        self.leaves: List[Tuple[str, Any]] = []
        self.arrays: Dict[str, int] = {}
        self.scalars: Dict[str, int] = {}

    def _fmt(self, name: str) -> str:
        m = self.ctx.attr.get(name, {})
        return f"{m.get('signed', '-')}.{m.get('width', '-')}"

    def leaf(self, kind: str, value: Any, token: str):
        """The leaf's token (a form) or its node in the template."""
        i = len(self.leaves)
        self.leaves.append((kind, value))
        if self.decide is None:
            return token
        return self.decide(i, kind, value)

    def name(self, n: ast.Name) -> ast.AST:
        x = n.id
        if x in self.ctx.literal:
            return n
        if x in self.ctx.movable:
            j = self.arrays.setdefault(x, len(self.arrays))
            r = self.leaf("a", x, f"§a{j}:{self.ctx.words.get(x, 1)}:{self._fmt(x)}")
        else:
            j = self.scalars.setdefault(x, len(self.scalars))
            r = self.leaf("s", x, f"§s{j}:{self._fmt(x)}")
        if isinstance(r, ast.Name):
            r = r.id
        return ast.Name(id=r, ctx=n.ctx) if isinstance(r, str) else r

    def expr(self, n: ast.AST) -> ast.AST:
        from .sequencer import _is_table
        if isinstance(n, ast.Subscript) and _is_table(n.value):
            n.slice = self.expr(n.slice)
            r = self.leaf("t", tuple(e.value for e in n.value.elts), "§t")
            if isinstance(r, str):
                n.value = ast.Name(id=r, ctx=ast.Load())
                return n
            return r(n)                       # (the template's: a table and its offset)
        if isinstance(n, ast.Constant) and type(n.value) is int:
            if not -SLOT_LIMIT <= n.value < SLOT_LIMIT:
                return n                      # (wider than a slot: as it is, in the form too)
            r = self.leaf("c", n.value, "§c")
            return ast.Name(id=r, ctx=ast.Load()) if isinstance(r, str) else r
        if isinstance(n, ast.Tuple) and n.elts and all(isinstance(e, ast.Constant) for e in n.elts):
            return n                          # (a table not read by an index: as it is)
        if isinstance(n, ast.Name):
            return self.name(n)
        if isinstance(n, ast.Call):           # (a function's name as it is)
            n.args = [self.expr(a) for a in n.args]
            return n
        for f, v in ast.iter_fields(n):
            if isinstance(v, ast.AST):
                setattr(n, f, self.expr(v))
            elif isinstance(v, list):
                setattr(n, f, [self.expr(x) if isinstance(x, ast.AST) else x for x in v])
        return n

    def stmt(self, s: ast.stmt) -> ast.stmt:
        from .sequencer import _list_elements
        s = copy.deepcopy(s)
        if isinstance(s, ast.Assign) and _list_elements(s.value) is not None:
            s.targets = [self.expr(t) for t in s.targets]      # (a new list: as written)
            return s
        return self.expr(s)

    def item(self, it):
        """(key, template item)."""
        from .sequencer import Block, Loop
        if isinstance(it, Block):
            stmts = [self.stmt(s) for s in it.stmts]
            return ("B", tuple(ast.dump(s) for s in stmts)), Block(stmts)
        var = self.name(ast.Name(id=it.var, ctx=ast.Store()))
        var = var.id
        values, count = list(it.values), None
        if values and values == list(range(len(values))):     # (a loop that runs: a count)
            r = self.leaf("n", len(values), "§n")
            if not isinstance(r, str):
                values, count = r
            key_values: Any = "§n"
        else:
            key_values = tuple(values)
        body = [self.item(b) for b in it.body]
        return (("L", var, key_values, tuple(k for k, _ in body)),
                Loop(var, values, [t for _, t in body], count=count))


@dataclass
class Steps:
    """The program as a loop over its steps (see the module's doc)."""
    program: list                         # [the loop over the steps, the last block, templates]
    order: list                           # the steps as they were (the lists' lifetimes)
    forced: Set[str]                      # lists that go to the SDRAM (the steps', __desc)
    placeholders: Dict[str, Dict[str, Any]]   # a template's list -> {slot, length, words, lists}
    desc: str
    width: int                            # a step's entries in __desc
    slots: List[str]
    entry: str
    jump: Any                             # the block that goes to the step's template
    calls: List[Tuple[Any, List[Any]]]    # each step: (its template, its slots' values)
    attr: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # (a template's name of a slot -> (the slot, its steps' values): a wire from the
    # slot's register in the template's own format)
    aliases: Dict[str, Tuple[str, List[Any]]] = field(default_factory=dict)
    align: Dict[str, int] = field(default_factory=dict)   # a list: its base a multiple of 2^this

    def ranges(self, rng: Dict[str, Tuple[int, int]], n_blocks: int) -> None:
        """The ranges of the slots (every template's values), the templates' names of
        them (their own), __entry and __desc (a base's high bits: any address's)."""
        def values(vs):
            return [(0, (SDRAM_SPAN - 1) >> v[1]) if isinstance(v, tuple) else (v, v)
                    for v in vs]
        every = []
        for j, q in enumerate(self.slots):
            rs = values(vals[j] for _, vals in self.calls if j < len(vals))
            rng[q] = _hull((0, 0), *rs)
            every.append(rng[q])
        for name, (_, vs) in self.aliases.items():
            rng[name] = _hull(*values(vs))
        rng[self.entry] = (0, n_blocks)
        rng[self.desc] = _hull((0, n_blocks), *every)

    def counts(self, values: List[Any]) -> Dict[str, int]:
        """A step's numbers by the templates' names of the slots."""
        at = {q: j for j, q in enumerate(self.slots)}
        return {name: values[at[q]] for name, (q, _) in self.aliases.items()
                if at[q] < len(values) and not isinstance(values[at[q]], tuple)}

    def resolve(self, ext: Dict[str, Any]) -> None:
        """The steps' values once the lists have their places in the SDRAM: the bases
        in __desc, which starts with them; the templates' lists as their slots."""
        lists = ext["lists"]
        values: List[int] = []
        for proc, vals in self.calls:
            row = [int(lists[v[2]]["base"]) >> v[1] if isinstance(v, tuple) else int(v)
                   for v in vals]
            row += [0] * (self.width - 1 - len(row))
            values += row + [proc.body[0].index]
        lists[self.desc]["init"] = values
        for ph, info in self.placeholders.items():
            lists[ph] = {"base": 0, "param": info["slot"], "shift": info["shift"],
                         "length": info["length"], "meta": info["meta"], "init": None,
                         "words": info["words"], "virtual": True}


def as_steps(program, sq, rams: Set[str], reg_lists: Set[str], fixed: Set[str],
             rng: Dict[str, Tuple[int, int]], attr) -> Optional[Steps]:
    """The program (Blocks and Loops, its lists' allocations done) as a loop over
    its steps, or None: fewer than MIN_NESTS loop nests (a smaller program fits as
    it is, without the steps' clocks; PY2VER_STEPS=1: two at least, 0: never), or
    no two of a form.
    fixed: lists a template can't take from its step (inputs, memories,
    returned lists: ports or state). rng: the program's ranges (the templates'
    from their steps')."""
    from .sequencer import Block, Loop, Proc
    mode = os.environ.get("PY2VER_STEPS", "auto")
    if mode == "0":
        return None
    items = list(program)
    tail = [items.pop()] if items and isinstance(items[-1], Block) else []
    if sum(isinstance(it, Loop) for it in items) < (MIN_NESTS if mode == "auto" else 2):
        return None
    _items_fold(items)
    movable = (rams | reg_lists) - set(fixed)
    words = {a: -(-int(_fmt_of(rng[a])["width"]) // 16) if a in rng else 1 for a in movable}
    ctx, forms, groups = _forms(items, tail, sq, movable, words, attr)
    # (a list in registers a template takes from its step: in the SDRAM, made by a
    # loop, one write, not a write an element; the forms again)
    to_sdram = {v for members in groups.values() if len(members) > 1
                for vec in _list_leaves([forms[i][1] for i in members]) for v in vec} & reg_lists
    if to_sdram:
        items = _filled(items, to_sdram, sq)
        ctx, forms, groups = _forms(items, tail, sq, movable, words, attr)
    if all(len(g) < 2 for g in groups.values()):
        return None

    slots: List[str] = []
    calls: List[Optional[Tuple[Any, List[Any]]]] = [None] * len(items)
    procs: List[Any] = []
    placeholders: Dict[str, Dict[str, Any]] = {}
    aliases: Dict[str, Tuple[str, List[Any]]] = {}    # a template's slot -> (the slot, values)
    align: Dict[str, int] = {}                # a list -> its base a multiple of 2^this
    forced: Set[str] = set()
    extra_attr: Dict[str, Dict[str, Any]] = {}
    for g, members in enumerate(groups.values()):
        leaves = [forms[i][1] for i in members]
        slot_of: Dict[Tuple, int] = {}

        def slot(vec: Tuple) -> str:
            """The template's name for the slot of these values (a wire from the slot's
            register, its range the template's own)."""
            if vec not in slot_of:
                j = slot_of[vec] = len(slot_of)
                while len(slots) <= j:
                    slots.append(sq.fresh(f"__q{len(slots)}"))
                name = sq.fresh(f"__t{g}_q{j}")
                aliases[name] = (slots[j], list(vec))
                alias_of[vec] = name
            return alias_of[vec]
        alias_of: Dict[Tuple, str] = {}
        decided: Dict[int, Any] = {}
        arrays: Dict[Tuple, str] = {}
        for i, (kind, v0) in enumerate(leaves[0]):
            vec = tuple(ls[i][1] for ls in leaves)
            same = all(v == v0 for v in vec)
            if kind == "s":                       # (the first step's name, its range all of theirs)
                rs = [rng[v] for v in vec if v in rng]
                if rs:
                    rng[v0] = _hull(*rs)
                decided[i] = ast.Name(id=v0, ctx=ast.Load())
            elif same:
                decided[i] = None                 # (as it is)
            elif kind in ("c", "n"):
                decided[i] = slot(vec)
            elif kind == "a":
                if vec not in arrays:
                    ph = sq.fresh(f"__a{g}_{len(arrays)}")
                    arrays[vec] = ph
                    # (its lists at multiples of 2^shift words, its words fewer: the
                    # slot the base's high bits, an address that bits, not a sum)
                    length = max(sq.lists[v] for v in vec)
                    shift = max(9, (length * words.get(v0, 1) - 1).bit_length())
                    base = slot(tuple(("base", shift, v) for v in vec))
                    placeholders[ph] = {"slot": base, "lists": list(vec), "length": length,
                                        "words": words.get(v0, 1), "shift": shift}
                    for v in vec:
                        align[v] = max(align.get(v, 0), shift)
                    sq.lists[ph] = placeholders[ph]["length"]
                    rng[ph] = _hull(*[rng[v] for v in vec if v in rng])
                    placeholders[ph]["meta"] = dict(_fmt_of(rng[ph]), type="wire")
                    if v0 in attr:
                        extra_attr[ph] = dict(attr[v0])
                    forced |= set(vec)
                decided[i] = arrays[vec]
            elif kind == "t":                     # (one table of every step's, at an offset)
                merged: List[int] = []
                at: Dict[Tuple, int] = {}
                for t in vec:
                    if t not in at:
                        at[t] = len(merged)
                        merged += list(t)
                decided[i] = (tuple(merged), slot(tuple(at[t] for t in vec)))
        def decide(i, kind, value, decided=decided, leaves=leaves):
            d = decided[i]
            if kind == "n":
                return (list(range(max(ls[i][1] for ls in leaves))), d) if d \
                    else (list(range(value)), None)
            if kind == "t":
                if d is None:
                    return lambda n: n
                merged, q = d

                def table(n: ast.Subscript) -> ast.Subscript:
                    n.value = ast.Tuple(elts=[ast.Constant(e) for e in merged], ctx=ast.Load())
                    n.slice = ast.BinOp(left=ast.Name(id=q, ctx=ast.Load()), op=ast.Add(),
                                        right=n.slice)
                    return n
                return table
            if d is None:
                return ast.Constant(value) if kind == "c" else ast.Name(id=value, ctx=ast.Load())
            if isinstance(d, ast.Name):
                return d
            return ast.Name(id=d, ctx=ast.Load())
        _, template = _Walker(ctx, decide).item(items[members[0]])
        # (a shift by a slot: by its smallest value, a constant, then by the rest, a
        # narrow slot: a barrel shifter of a few places, not of every place its
        # format reaches)
        for x in _blocks(template):
            for s in x.stmts:
                for n in ast.walk(s):
                    if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.LShift, ast.RShift)) \
                            and isinstance(n.right, ast.Name) and n.right.id in aliases \
                            and n.right.id in alias_of.values():
                        vs = aliases[n.right.id][1]
                        lo = min(vs)
                        if lo > 0 and all(isinstance(v, int) for v in vs):
                            n.left = ast.BinOp(left=n.left, op=type(n.op)(), right=ast.Constant(lo))
                            n.right = ast.Name(id=slot(tuple(v - lo for v in vs)), ctx=ast.Load())
        vals: List[List[Any]] = [[] for _ in members]
        for vec, j in sorted(slot_of.items(), key=lambda x: x[1]):
            for m in range(len(members)):
                vals[m].append(vec[m])
        ast.fix_missing_locations(ast.Module(body=[s for x in _blocks(template) for s in x.stmts],
                                             type_ignores=[]))
        proc = Proc([Block([]), template])
        procs.append(proc)
        for m, i in enumerate(members):
            calls[i] = (proc, vals[m])

    S = len(items)
    np_ = len(slots)
    width = 1 << max(1, np_.bit_length())         # (a power of two: no multiplier)
    while width < np_ + 1:
        width <<= 1
    step, entry, desc = sq.fresh("__step"), sq.fresh("__entry"), sq.fresh("__desc")
    sq.lists[desc] = S * width

    def at(j: int) -> ast.Subscript:
        return ast.Subscript(value=ast.Name(id=desc, ctx=ast.Load()), slice=ast.BinOp(
            left=ast.BinOp(left=ast.Name(id=step, ctx=ast.Load()), op=ast.Mult(),
                           right=ast.Constant(width)), op=ast.Add(), right=ast.Constant(j)),
            ctx=ast.Load())
    loads = Block([ast.Assign(targets=[ast.Name(id=q, ctx=ast.Store())], value=at(j))
                   for j, q in enumerate(slots)]
                  + [ast.Assign(targets=[ast.Name(id=entry, ctx=ast.Store())], value=at(width - 1))])
    for s in loads.stmts:
        ast.fix_missing_locations(s)
    jump = Block([], jump=entry)
    loop = Loop(step, list(range(S)), [loads, jump])
    return Steps(program=[loop] + tail + procs, order=items + tail, forced=forced | {desc},
                 placeholders=placeholders, desc=desc, width=width, slots=slots, entry=entry,
                 jump=jump, calls=list(calls), attr=extra_attr, aliases=aliases, align=align)


def _forms(items, tail, sq, movable: Set[str], words: Dict[str, int], attr):
    """(the walkers' context, each step's (form, leaves), the steps of each form)."""
    from .sequencer import Block, Loop, _walk_items

    def names(it, ctx_type) -> Set[str]:
        out: Set[str] = set()
        for x in _walk_items([it]):
            if isinstance(x, Loop) and ctx_type is ast.Store:
                out.add(x.var)
            if isinstance(x, Block):
                for s in x.stmts:
                    out |= {n.id for n in ast.walk(s) if isinstance(n, ast.Name)
                            and isinstance(n.ctx, ctx_type)}
        return out
    lists = set(sq.lists)
    stores = [names(it, ast.Store) - lists for it in items + tail]
    loads = [names(it, ast.Load) - lists for it in items + tail]
    # (a scalar set in one step and read in another, or set in two: by its name)
    shared: Set[str] = set()
    every_store: Set[str] = set().union(*stores) if stores else set()
    for i, ld in enumerate(loads):
        for v in ld & every_store:
            if any(v in st for j, st in enumerate(stores) if j != i):
                shared.add(v)
    for i, st in enumerate(stores):
        for v in st:
            if sum(v in s for s in stores) > 1:
                shared.add(v)
    used = set().union(*loads, *stores) if loads else set()
    never_set = used - every_store
    ctx = _Ctx(literal=(lists - movable) | shared | never_set, movable=movable, words=words,
               attr=attr)
    forms = []
    for it in items:
        w = _Walker(ctx)
        key, _ = w.item(it)
        forms.append((key, w.leaves))
    groups: Dict[Any, List[int]] = {}
    for i, (key, _) in enumerate(forms):
        groups.setdefault(key, []).append(i)
    return ctx, forms, groups


def _list_leaves(leaves: List[List[Tuple[str, Any]]]) -> List[Tuple]:
    """The lists a form's steps take in each of its places, where they differ."""
    out = []
    for i, (kind, _) in enumerate(leaves[0]):
        if kind == "a":
            vec = tuple(ls[i][1] for ls in leaves)
            if len(set(vec)) > 1:
                out.append(vec)
    return out


def _filled(items, names: Set[str], sq):
    """The steps with the lists names made (name = [c] * n, in a block) by a loop
    writing c to each element: a step of its own."""
    from .sequencer import Block, Loop, _list_elements
    out = []
    for it in items:
        if not isinstance(it, Block):
            out.append(it)
            continue
        cur: List[ast.stmt] = []
        for s in it.stmts:
            t = s.targets[0] if isinstance(s, ast.Assign) and len(s.targets) == 1 else None
            elems = _list_elements(s.value) if isinstance(t, ast.Name) and t.id in names else None
            if not elems or len({ast.dump(e) for e in elems}) != 1:
                cur.append(s)
                continue
            if cur:
                out.append(Block(cur))
                cur = []
            var = sq.fresh("__fill")
            out.append(Loop(var, list(range(len(elems))), [Block([ast.fix_missing_locations(
                ast.Assign(targets=[ast.Subscript(value=ast.Name(id=t.id, ctx=ast.Load()),
                                                  slice=ast.Name(id=var, ctx=ast.Load()),
                                                  ctx=ast.Store())],
                           value=copy.deepcopy(elems[0])))])]))
        if cur:
            out.append(Block(cur))
    return out


def _blocks(it):
    from .sequencer import Block, _walk_items
    return [x for x in _walk_items([it]) if isinstance(x, Block)]
