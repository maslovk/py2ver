# containers.py
"""
Dicts with constant keys and nested lists, as the translator takes them.

Hardware has no dictionaries or lists of lists, but both are fixed in shape
when their keys and sizes are known, so before translation:

  - a dict is one variable per key:

        gain = {'lo': 3, 'hi': x * 2}           gain_lo = 3
        y = gain['hi'] + gain['lo']       ->    gain_hi = x * 2
        gain['lo'] += y                         y = gain_hi + gain_lo
                                                gain_lo = gain_lo + y

    d[k], d[k] = v, d[k] += v with a constant key (a literal, a module
    constant, or the variable of a loop over the dict); k in d, len(d),
    d.get(k, default), d.update({...}), loops over d, d.keys(), d.values()
    and d.items() (unrolled); d2 = d (the same dict, as in Python), dict(d) /
    d.copy() (a copy), dict comprehensions; module-level constant dicts. With
    int keys, a key computed at run time reads (or updates) the existing keys.
    attr['d'] gives every entry its format.
  - a list of lists (any depth) is one flat list in row-major order:
    m[i][j] is m[i * C + j] (i, j may be variables), m[i] a row (a list),
    for row in m a loop over the rows, len(m) / len(m[0]) constants. A list
    argument declares its shape with attr {'m': {..., 'shape': (R, C)}} and
    takes nested lists; a returned one gives its elements in row-major order.
    Writing through a row (row = m[0]; row[1] = 5) would change m in Python,
    so it is refused: write m[0][1] = 5.
"""

from __future__ import annotations

import ast
import copy
import itertools
import math
import re
from typing import Any, Dict, List, Optional, Tuple

from .names import walk_code

_MISSING = object()


class _Dict:
    """A dict's storage: key -> variable name, in insertion order."""

    def __init__(self, owner: str, number: int) -> None:
        self.owner = owner
        self.number = number
        self.vars: Dict[Any, str] = {}


def _slug(key: Any) -> str:
    if isinstance(key, bool):
        return str(key)
    if isinstance(key, int):
        return f"k{key}" if key >= 0 else f"kn{-key}"
    if isinstance(key, str):
        s = re.sub(r"\W", "_", key)
        return s if s and not s[0].isdigit() else f"k{s}"
    if isinstance(key, tuple):
        return "_".join(_slug(k) for k in key) or "empty"
    return re.sub(r"\W", "_", repr(key))


def _flat_index(idx: List[ast.expr], shape: Tuple[int, ...]) -> ast.expr:
    """(i, j, ...) -> i * (C * ...) + j * (...) + ... (constants folded)."""
    out: Optional[ast.expr] = None
    for k, i in enumerate(idx):
        stride = math.prod(shape[k + 1:])
        term = i if stride == 1 else ast.BinOp(left=i, op=ast.Mult(), right=ast.Constant(stride))
        out = term if out is None else ast.BinOp(left=out, op=ast.Add(), right=term)
    return _fold(out)


def _fold(node: ast.expr) -> ast.expr:
    """Constant subexpressions as constants."""
    if not any(isinstance(n, (ast.Name, ast.Call, ast.Subscript, ast.Attribute))
               for n in ast.walk(node)):
        try:
            return ast.Constant(eval(compile(ast.Expression(ast.fix_missing_locations(node)),
                                             "<index>", "eval"), {}))
        except Exception:
            return node
    return node


def _shape_of_literal(node: ast.expr) -> Optional[Tuple[int, ...]]:
    """The shape of a nested list literal (None if it isn't one, or is ragged)."""
    if not isinstance(node, (ast.List, ast.Tuple)) or not node.elts:
        return None
    subs = [_shape_of_literal(e) for e in node.elts]
    if all(s is None for s in subs):
        if any(isinstance(e, (ast.List, ast.Tuple)) for e in node.elts):
            return None
        return (len(node.elts),)
    if any(s is None for s in subs) or len(set(subs)) != 1:
        raise NotImplementedError(f"A list of lists must be rectangular: {ast.unparse(node)}")
    return (len(node.elts),) + subs[0]


def _flatten_literal(node: ast.expr) -> List[ast.expr]:
    if isinstance(node, (ast.List, ast.Tuple)) and any(
            isinstance(e, (ast.List, ast.Tuple)) for e in node.elts):
        return [x for e in node.elts for x in _flatten_literal(e)]
    if isinstance(node, (ast.List, ast.Tuple)):
        return list(node.elts)
    return [node]


class _Rewrite:
    def __init__(self, fd: ast.FunctionDef, attr: Dict[str, Any], ns: Dict[str, Any]) -> None:
        self.attr = attr
        self.ns = ns
        self.params = [a.arg for a in fd.args.args]
        self.locals = {n.id for n in ast.walk(fd) if isinstance(n, ast.Name)
                       and isinstance(n.ctx, ast.Store)} | set(self.params)
        self.used = {n.id for n in ast.walk(fd) if isinstance(n, ast.Name)} | set(self.params)
        self.shapes: Dict[str, Tuple[int, ...]] = {}
        for p in self.params:
            shape = (attr.get(p) or {}).get("shape") if isinstance(attr.get(p), dict) else None
            if shape is not None:
                self.shapes[p] = tuple(int(s) for s in shape)
        self.dicts: Dict[str, _Dict] = {}
        self.views: Dict[str, str] = {}        # row variable -> the list it is a view of
        self.subst: Dict[str, ast.expr] = {}  # loop variables over dict keys
        self.new_attr: Dict[str, Any] = {}
        self.counter = itertools.count()
        self.n_dicts: Dict[str, int] = {}

    # ------------------------------------------------------------ helpers
    def fresh(self, base: str) -> str:
        name = base
        while name in self.used:
            name = f"{base}_{next(self.counter)}"
        self.used.add(name)
        return name

    def const(self, node: ast.expr) -> Any:
        """A key known when translating, or _MISSING."""
        node = self.substitute(node)
        try:
            return ast.literal_eval(node)
        except (ValueError, SyntaxError, TypeError):
            pass
        if isinstance(node, ast.Name) and node.id not in self.locals and node.id in self.ns:
            v = self.ns[node.id]
            if isinstance(v, (int, str, tuple, bool)):
                return v
        try:
            if not any(isinstance(n, ast.Name) and n.id in self.locals for n in ast.walk(node)):
                v = eval(compile(ast.Expression(ast.fix_missing_locations(copy.deepcopy(node))),
                                 "<key>", "eval"), dict(self.ns))
                if isinstance(v, (int, str, tuple, bool)):
                    return v
        except Exception:
            pass
        return _MISSING

    def substitute(self, node: ast.expr) -> ast.expr:
        if not self.subst:
            return node
        subst = self.subst

        class _S(ast.NodeTransformer):
            def visit_Name(self, n: ast.Name):
                if isinstance(n.ctx, ast.Load) and n.id in subst:
                    return copy.deepcopy(subst[n.id])
                return n
        return _S().visit(copy.deepcopy(node))

    def new_dict(self, name: str) -> _Dict:
        k = self.n_dicts.get(name, 0)
        self.n_dicts[name] = k + 1
        return _Dict(name if k == 0 else f"{name}{k}", k)

    def dict_var(self, d: _Dict, key: Any) -> str:
        if key not in d.vars:
            d.vars[key] = self.fresh(f"{d.owner}_{_slug(key)}")
            meta = self.attr.get(d.owner) or self.attr.get(re.sub(r"\d+$", "", d.owner))
            if isinstance(meta, dict) and "width" in meta:
                self.new_attr[d.vars[key]] = {k: v for k, v in meta.items() if k != "dict"}
        return d.vars[key]

    def module_dict(self, node: ast.expr) -> Optional[dict]:
        if isinstance(node, ast.Name) and node.id not in self.locals \
                and isinstance(self.ns.get(node.id), dict):
            return self.ns[node.id]
        return None

    @staticmethod
    def value_node(v: Any) -> ast.expr:
        if isinstance(v, (int, float, complex, bool, str)):
            return ast.Constant(v)
        if isinstance(v, (list, tuple)):
            return ast.Tuple(elts=[_Rewrite.value_node(x) for x in v], ctx=ast.Load())
        raise NotImplementedError(f"A module-level dict value {v!r} can't be a constant")

    # ------------------------------------------------------------ expressions
    def shape(self, node: ast.expr) -> Optional[Tuple[int, ...]]:
        """The shape of a list-of-lists valued expression (rows: the rest)."""
        if isinstance(node, ast.Name) and node.id in self.shapes:
            return self.shapes[node.id]
        if isinstance(node, ast.Subscript):
            base, idx = self.chain(node)
            if isinstance(base, ast.Name) and base.id in self.shapes \
                    and not any(isinstance(i, ast.Slice) for i in idx):
                rest = self.shapes[base.id][len(idx):]
                return rest if rest else None
        return None

    @staticmethod
    def chain(node: ast.Subscript) -> Tuple[ast.expr, List[ast.expr]]:
        idx: List[ast.expr] = []
        while isinstance(node, ast.Subscript):
            idx.insert(0, node.slice)
            node = node.value
        return node, idx

    def nd_read(self, name: str, idx: List[ast.expr], ctx) -> ast.expr:
        shape = self.shapes[name]
        if len(idx) > len(shape):
            raise NotImplementedError(f"Too many indices for '{name}' (shape {shape})")
        idx = [self.expr(i) for i in idx]
        if len(idx) == len(shape):
            return ast.Subscript(value=ast.Name(id=name, ctx=ast.Load()),
                                 slice=_flat_index(idx, shape), ctx=ctx)
        # A row (or block): the list of its elements
        rest = shape[len(idx):]

        def block(prefix: List[ast.expr], dims: Tuple[int, ...]) -> ast.expr:
            if not dims:
                return ast.Subscript(value=ast.Name(id=name, ctx=ast.Load()),
                                     slice=_flat_index(prefix, shape), ctx=ast.Load())
            return ast.List(elts=[block(prefix + [ast.Constant(k)], dims[1:])
                                  for k in range(dims[0])], ctx=ast.Load())
        return block(idx, rest)

    def expr(self, node: ast.expr) -> ast.expr:
        node = self.substitute(node)
        if isinstance(node, ast.Subscript) and not isinstance(node.slice, ast.Slice):
            base, idx = self.chain(node)
            if isinstance(base, ast.Name) and base.id in self.dicts:
                if len(idx) != 1:
                    raise NotImplementedError(f"Dict values can't be indexed: {ast.unparse(node)}")
                return self.dict_read(self.dicts[base.id], idx[0], node)
            mod = self.module_dict(base)
            if mod is not None:
                key = self.const(idx[0])
                if key is _MISSING and mod and all(isinstance(k, int) and not isinstance(k, bool)
                                                   for k in mod):
                    value = self.choose(self.expr(idx[0]),
                                        [(k, self.value_node(v)) for k, v in mod.items()])
                elif key is _MISSING or key not in mod:
                    raise NotImplementedError(f"{ast.unparse(node)}: the key must be one of "
                                              f"{list(mod)}, known when translating (or the "
                                              "keys ints)")
                else:
                    value = self.value_node(mod[key])
                for i in idx[1:]:
                    value = ast.Subscript(value=value, slice=self.expr(i), ctx=ast.Load())
                return value
            if isinstance(base, ast.Name) and base.id in self.shapes:
                return self.nd_read(base.id, idx, ast.Load())
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id == "len" and len(node.args) == 1:
                a = node.args[0]
                if isinstance(a, ast.Name) and a.id in self.dicts:
                    return ast.Constant(len(self.dicts[a.id].vars))
                if self.module_dict(a) is not None:
                    return ast.Constant(len(self.module_dict(a)))
                s = self.shape(a)
                if s is not None:
                    return ast.Constant(s[0])
            if isinstance(fn, ast.Attribute) and fn.attr == "get" and isinstance(fn.value, ast.Name) \
                    and (fn.value.id in self.dicts or self.module_dict(fn.value) is not None):
                key = self.const(node.args[0])
                default = self.expr(node.args[1]) if len(node.args) > 1 else ast.Constant(None)
                if fn.value.id in self.dicts:
                    d = self.dicts[fn.value.id]
                    if key is _MISSING:
                        raise NotImplementedError(f"{ast.unparse(node)}: the key must be known "
                                                  "when translating")
                    return ast.Name(id=d.vars[key], ctx=ast.Load()) if key in d.vars else default
                mod = self.module_dict(fn.value)
                return self.value_node(mod[key]) if key in mod else default
        if isinstance(node, ast.Compare) and len(node.ops) == 1 \
                and isinstance(node.ops[0], (ast.In, ast.NotIn)):
            c = node.comparators[0]
            keys = (list(self.dicts[c.id].vars) if isinstance(c, ast.Name) and c.id in self.dicts
                    else list(self.module_dict(c)) if self.module_dict(c) is not None else None)
            if keys is not None:
                key = self.const(node.left)
                if key is _MISSING:
                    raise NotImplementedError(f"{ast.unparse(node)}: the key must be known when "
                                              "translating")
                return ast.Constant((key in keys) == isinstance(node.ops[0], ast.In))
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in self.dicts:
            raise NotImplementedError(f"The dict '{node.id}' can only be used as d[key], len(d), "
                                      "k in d, d.get(...), or in a for loop")
        if isinstance(node, ast.Dict):
            raise NotImplementedError("A dict can be assigned to a variable, not used in an "
                                      f"expression: {ast.unparse(node)}")
        # Anything else: rewrite inside
        for field, value in ast.iter_fields(node):
            if isinstance(value, ast.expr):
                setattr(node, field, self.expr(value))
            elif isinstance(value, list):
                setattr(node, field, [self.expr(v) if isinstance(v, ast.expr) else v
                                      for v in value])
        return node

    def dict_read(self, d: _Dict, key_node: ast.expr, node: ast.AST) -> ast.expr:
        key = self.const(key_node)
        if key is not _MISSING:
            if key not in d.vars:
                raise NotImplementedError(f"{ast.unparse(node)}: no key {key!r} (KeyError); "
                                          f"keys: {list(d.vars)}")
            return ast.Name(id=d.vars[key], ctx=ast.Load())
        if not all(isinstance(k, int) and not isinstance(k, bool) for k in d.vars):
            raise NotImplementedError(f"{ast.unparse(node)}: a key computed at run time needs "
                                      "int keys")
        return self.choose(self.expr(key_node),
                           [(k, ast.Name(id=v, ctx=ast.Load())) for k, v in d.vars.items()])

    @staticmethod
    def choose(idx: ast.expr, entries: List[Tuple[Any, ast.expr]]) -> ast.expr:
        """The entry whose key idx equals (a missing key: some entry)."""
        out = entries[-1][1]
        for k, value in reversed(entries[:-1]):
            out = ast.IfExp(test=ast.Compare(left=copy.deepcopy(idx), ops=[ast.Eq()],
                                             comparators=[ast.Constant(k)]),
                            body=value, orelse=out)
        return out

    # ------------------------------------------------------------ statements
    def block(self, stmts: List[ast.stmt]) -> List[ast.stmt]:
        out: List[ast.stmt] = []
        for s in stmts:
            out += self.stmt(s)
        return out

    def assign(self, name: str, value: ast.expr) -> ast.stmt:
        return ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=value)

    def forget(self, name: str) -> None:
        self.dicts.pop(name, None)
        self.shapes.pop(name, None)
        self.views.pop(name, None)

    def dict_value(self, value: ast.expr) -> Optional[List[Tuple[Any, ast.expr]]]:
        """(key, value) pairs of a dict-valued expression, or None."""
        value = self.substitute(value)
        if isinstance(value, ast.Dict):
            pairs = []
            for k, v in zip(value.keys, value.values):
                if k is None:                              # {**other}
                    src = self.dict_value(v)
                    if src is None:
                        raise NotImplementedError(f"Can't unpack {ast.unparse(v)} into a dict")
                    pairs += src
                    continue
                key = self.const(k)
                if key is _MISSING:
                    raise NotImplementedError(f"A dict key must be known when translating: "
                                              f"{ast.unparse(k)}")
                pairs.append((key, v))
            return pairs
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) \
                and value.func.id == "dict":
            pairs = []
            for a in value.args:
                src = self.dict_value(a)
                if src is None:
                    raise NotImplementedError(f"dict({ast.unparse(a)}) is not supported")
                pairs += src
            pairs += [(kw.arg, kw.value) for kw in value.keywords if kw.arg is not None]
            return pairs
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) \
                and value.func.attr == "copy" and not value.args:
            return self.dict_value(value.func.value)
        if isinstance(value, ast.Name) and value.id in self.dicts:
            d = self.dicts[value.id]
            return [(k, ast.Name(id=v, ctx=ast.Load())) for k, v in d.vars.items()]
        mod = self.module_dict(value)
        if mod is not None:
            return [(k, self.value_node(v)) for k, v in mod.items()]
        return None

    def stmt(self, s: ast.stmt) -> List[ast.stmt]:
        if isinstance(s, ast.AugAssign) and isinstance(s.target, ast.Subscript):
            load = copy.deepcopy(s.target)
            for n in ast.walk(load):
                if hasattr(n, "ctx"):
                    n.ctx = ast.Load()
            return self.stmt(ast.copy_location(ast.Assign(
                targets=[s.target], value=ast.BinOp(left=load, op=s.op, right=s.value)), s))
        if isinstance(s, ast.Assign) and len(s.targets) == 1:
            t = s.targets[0]
            if isinstance(t, ast.Name):
                return self.assign_name(t.id, s)
            if isinstance(t, ast.Subscript):
                return self.assign_item(t, s)
        if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call):
            f = s.value.func
            if isinstance(f, ast.Attribute) and f.attr == "update" \
                    and isinstance(f.value, ast.Name) and f.value.id in self.dicts:
                d = self.dicts[f.value.id]
                out = []
                pairs = []
                for a in s.value.args:
                    src = self.dict_value(a)
                    if src is None:
                        raise NotImplementedError(f"update({ast.unparse(a)}) is not supported")
                    pairs += src
                pairs += [(kw.arg, kw.value) for kw in s.value.keywords]
                values = [self.expr(v) for _, v in pairs]
                for (k, _), v in zip(pairs, values):
                    out.append(self.assign(self.dict_var(d, k), v))
                return out
        if isinstance(s, ast.For):
            return self.loop(s)
        if isinstance(s, ast.If):
            s.test = self.expr(s.test)
            shapes, dicts = dict(self.shapes), dict(self.dicts)
            s.body = self.block(s.body) or [ast.Pass()]
            after_body, dicts_body = dict(self.shapes), dict(self.dicts)
            self.shapes, self.dicts = dict(shapes), dict(dicts)
            s.orelse = self.block(s.orelse)
            for name in set(dicts_body) | set(self.dicts):
                if dicts_body.get(name) is not self.dicts.get(name) \
                        or dicts_body.get(name) is not dicts.get(name):
                    raise NotImplementedError(
                        f"'{name}' is made a new dict in a branch of the if on line {s.lineno}; "
                        "set its keys instead (d[k] = v)")
            for name, shape in after_body.items():
                if name in self.shapes and self.shapes[name] != shape:
                    raise NotImplementedError(f"'{name}' has different shapes in the branches "
                                              f"of the if on line {s.lineno}")
                self.shapes.setdefault(name, shape)
            return [s]
        if isinstance(s, ast.Return) and s.value is not None:
            vals = s.value.elts if isinstance(s.value, ast.Tuple) else [s.value]
            for v in vals:
                if isinstance(v, ast.Name) and v.id in self.dicts:
                    raise NotImplementedError(f"Returning the dict '{v.id}' is not supported; "
                                              "return its values")
            s.value = self.returned(s.value)
            return [s]
        # Anything else: its expressions
        for field, value in ast.iter_fields(s):
            if isinstance(value, ast.expr):
                setattr(s, field, self.expr(value))
            elif isinstance(value, list) and value and isinstance(value[0], ast.stmt):
                setattr(s, field, self.block(value))
        return [s]

    def returned(self, node: ast.expr) -> ast.expr:
        """A returned value, rewritten: a nested list literal flat (in a tuple too)."""
        if isinstance(node, ast.Tuple):
            node.elts = [self.returned(e) for e in node.elts]
            return node
        if isinstance(node, ast.List):
            info = self.as_elements(node)          # (rewrites the elements: use them)
            if info is not None:
                return ast.List(elts=info[1], ctx=ast.Load())
        return self.expr(node)

    def as_elements(self, node: ast.expr) -> Optional[Tuple[Tuple[int, ...], List[ast.expr]]]:
        """(shape, flat elements, rewritten) of a list-valued expression made of
        lists (literals, [v] * n, list(...), rows of a list of lists), else None."""
        node = self.substitute(node)
        if isinstance(node, (ast.List, ast.Tuple)):
            subs = [self.as_elements(e) for e in node.elts]
            if not node.elts or all(x is None for x in subs):
                return (len(node.elts),), [self.expr(e) for e in node.elts]
            if any(x is None for x in subs) or len({x[0] for x in subs}) != 1:
                raise NotImplementedError(f"A list of lists must be rectangular: "
                                          f"{ast.unparse(node)}")
            return (len(node.elts),) + subs[0][0], [e for _, es in subs for e in es]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            for lst, k in ((node.left, node.right), (node.right, node.left)):
                if isinstance(lst, (ast.List, ast.Tuple)):
                    n = self.const(k)
                    inner = self.as_elements(lst)
                    if n is _MISSING or inner is None:
                        return None
                    if len(inner[0]) > 1:
                        raise NotImplementedError(
                            f"'{ast.unparse(node)}' repeats the same inner list (a change to one "
                            "row changes all); build it with a comprehension: "
                            "[[0] * C for _ in range(R)]")
                    return (inner[0][0] * int(n),), [copy.deepcopy(e) for _ in range(int(n))
                                                     for e in inner[1]]
            return None
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in ("list", "tuple") and len(node.args) == 1:
            return self.as_elements(node.args[0])
        shape = self.shape(node)
        if shape is not None and (isinstance(node, ast.Subscript) or isinstance(node, ast.Name)):
            if isinstance(node, ast.Name):
                block = self.nd_read(node.id, [], ast.Load())
            else:
                base, idx = self.chain(node)
                block = self.nd_read(base.id, idx, ast.Load())
            return shape, _flatten_literal(block)
        return None

    def assign_name(self, name: str, s: ast.Assign) -> List[ast.stmt]:
        value = s.value
        pairs = self.dict_value(value)
        if pairs is not None:
            if isinstance(value, ast.Name) and value.id in self.dicts:
                self.forget(name)
                self.dicts[name] = self.dicts[value.id]           # the same dict, as in Python
                return []
            values = [self.expr(v) for _, v in pairs]
            self.forget(name)
            d = self.new_dict(name)
            out = [self.assign(self.dict_var(d, k), v) for (k, _), v in zip(pairs, values)]
            self.dicts[name] = d
            return out
        info = self.as_elements(value) if isinstance(value, (ast.List, ast.Tuple, ast.BinOp,
                                                              ast.Call)) else None
        if info is not None:                        # (its elements are rewritten: use them)
            shape, elts = info
            self.forget(name)
            if len(shape) > 1:
                self.shapes[name] = shape
                self.note_shape(name, shape)
            s.value = ast.List(elts=elts, ctx=ast.Load())
            return [s]
        src_shape = self.shape(value)
        view_of = None
        if isinstance(value, ast.Subscript):
            base, _ = self.chain(value)
            if isinstance(base, ast.Name) and base.id in self.shapes:
                view_of = base.id
        if isinstance(value, ast.Name) and value.id in self.shapes:
            s.value = value
            self.forget(name)
            self.shapes[name] = self.shapes[value.id]
            return [s]
        s.value = self.expr(value)
        self.forget(name)
        if view_of is not None:
            self.views[name] = view_of               # a row: a view of the list
            if src_shape is not None and len(src_shape) > 1:
                self.shapes[name] = src_shape
                s.value = ast.List(elts=_flatten_literal(s.value), ctx=ast.Load())
        return [s]

    def note_shape(self, name: str, shape: Tuple[int, ...]) -> None:
        meta = self.attr.get(name)
        if isinstance(meta, dict) and "width" in meta:
            self.new_attr[name] = dict({k: v for k, v in meta.items() if k != "shape"},
                                       length=math.prod(shape))

    def assign_item(self, t: ast.Subscript, s: ast.Assign) -> List[ast.stmt]:
        base, idx = self.chain(t)
        if isinstance(base, ast.Name) and base.id in self.views:
            raise NotImplementedError(
                f"'{base.id}' is a row of '{self.views[base.id]}' (the same list in Python): "
                f"write {self.views[base.id]}[...][...] = ... instead")
        info = (self.as_elements(s.value) if isinstance(base, ast.Name) and base.id in self.shapes
                and len(idx) < len(self.shapes[base.id]) else None)
        value = ast.List(elts=info[1], ctx=ast.Load()) if info is not None else self.expr(s.value)
        if isinstance(base, ast.Name) and base.id in self.dicts:
            d = self.dicts[base.id]
            if len(idx) != 1:
                raise NotImplementedError(f"Dict values can't be indexed: {ast.unparse(t)}")
            key = self.const(idx[0])
            if key is not _MISSING:
                return [self.assign(self.dict_var(d, key), value)]
            if not all(isinstance(k, int) and not isinstance(k, bool) for k in d.vars):
                raise NotImplementedError(f"{ast.unparse(t)}: a key computed at run time needs "
                                          "int keys")
            k_node = self.expr(idx[0])
            tmp = self.fresh("__dv")
            out = [self.assign(tmp, value)]
            for k, var in d.vars.items():              # (only the keys it already has)
                out.append(self.assign(var, ast.IfExp(
                    test=ast.Compare(left=copy.deepcopy(k_node), ops=[ast.Eq()],
                                     comparators=[ast.Constant(k)]),
                    body=ast.Name(id=tmp, ctx=ast.Load()), orelse=ast.Name(id=var, ctx=ast.Load()))))
            return out
        if isinstance(base, ast.Name) and base.id in self.shapes:
            shape = self.shapes[base.id]
            if len(idx) == len(shape):
                target = self.nd_read(base.id, idx, ast.Store())
                return [ast.Assign(targets=[target], value=value)]
            # A whole row (or block): element by element
            rows = self.nd_read(base.id, idx, ast.Load())
            targets = _flatten_literal(rows)
            tmp = self.fresh("__row")
            out = [self.assign(tmp, value)]
            for k, tgt in enumerate(targets):
                tgt.ctx = ast.Store()
                out.append(ast.Assign(targets=[tgt], value=ast.Subscript(
                    value=ast.Name(id=tmp, ctx=ast.Load()), slice=ast.Constant(k), ctx=ast.Load())))
            return out
        t.value = self.expr(t.value) if not isinstance(t.value, ast.Name) else t.value
        t.slice = self.expr(t.slice)
        s.value = value
        return [s]

    def loop(self, s: ast.For) -> List[ast.stmt]:
        it = self.substitute(s.iter)
        # Over a dict: unrolled, the loop variable(s) its keys / values
        target_d, kind = None, None
        if isinstance(it, ast.Name) and (it.id in self.dicts or self.module_dict(it) is not None):
            target_d, kind = it, "keys"
        elif isinstance(it, ast.Call) and isinstance(it.func, ast.Attribute) and not it.args \
                and it.func.attr in ("keys", "values", "items") \
                and isinstance(it.func.value, ast.Name) \
                and (it.func.value.id in self.dicts or self.module_dict(it.func.value) is not None):
            target_d, kind = it.func.value, it.func.attr
        if target_d is not None:
            if s.orelse:
                raise NotImplementedError("A loop over a dict with an 'else' is not supported")
            if target_d.id in self.dicts:
                d = self.dicts[target_d.id]
                entries = [(k, ast.Name(id=v, ctx=ast.Load())) for k, v in d.vars.items()]
            else:
                entries = [(k, self.value_node(v)) for k, v in self.module_dict(target_d).items()]
            out: List[ast.stmt] = []
            names = [n.id for n in ast.walk(s.target) if isinstance(n, ast.Name)]
            saved = {n: self.subst.get(n) for n in names}
            for key, value in entries:
                item = {"keys": ast.Constant(key), "values": value,
                        "items": ast.Tuple(elts=[ast.Constant(key), value], ctx=ast.Load())}[kind]
                if isinstance(s.target, ast.Name):
                    self.subst[s.target.id] = item
                elif isinstance(s.target, ast.Tuple) and isinstance(item, ast.Tuple) \
                        and len(s.target.elts) == len(item.elts):
                    for t, v in zip(s.target.elts, item.elts):
                        if not isinstance(t, ast.Name):
                            raise NotImplementedError("Unpack a dict's items into names")
                        self.subst[t.id] = v
                else:
                    raise NotImplementedError(f"Can't unpack {ast.unparse(item)} into "
                                              f"{ast.unparse(s.target)}")
                out += self.block(copy.deepcopy(s.body))
            for n, v in saved.items():
                if v is None:
                    self.subst.pop(n, None)
                else:
                    self.subst[n] = v
            return out
        # Over a list of lists: a loop over its rows (each row a view)
        rows_of, index_var = None, None
        if isinstance(it, ast.Name) and it.id in self.shapes and len(self.shapes[it.id]) > 1:
            rows_of = it.id
        elif isinstance(it, ast.Call) and isinstance(it.func, ast.Name) \
                and it.func.id == "enumerate" and len(it.args) == 1 \
                and isinstance(it.args[0], ast.Name) and it.args[0].id in self.shapes \
                and isinstance(s.target, ast.Tuple) and len(s.target.elts) == 2:
            rows_of = it.args[0].id
            index_var = s.target.elts[0].id
            s.target = s.target.elts[1]
        if rows_of is not None:
            if not isinstance(s.target, ast.Name):
                raise NotImplementedError("Loop over a list of lists into one name: for row in m")
            row = s.target.id
            i = index_var or self.fresh("__r")
            pre = [self.assign(row, self.nd_read(rows_of, [ast.Name(id=i, ctx=ast.Load())],
                                                 ast.Load()))]
            self.forget(row)
            self.views[row] = rows_of
            rest = self.shapes[rows_of][1:]
            if len(rest) > 1:
                self.shapes[row] = rest
                pre[0].value = ast.List(elts=_flatten_literal(pre[0].value), ctx=ast.Load())
            body = self.block(s.body)
            new = ast.For(target=ast.Name(id=i, ctx=ast.Store()),
                          iter=ast.Call(func=ast.Name(id="range", ctx=ast.Load()),
                                        args=[ast.Constant(self.shapes[rows_of][0])], keywords=[]),
                          body=pre + body, orelse=self.block(s.orelse), type_comment=None)
            return [ast.copy_location(new, s)]
        s.iter = self.expr(it)
        for n in ast.walk(s.target):
            if isinstance(n, ast.Name):
                self.forget(n.id)
        s.body = self.block(s.body)
        s.orelse = self.block(s.orelse)
        return [s]


def scalarize(tree: ast.Module, attr: Dict[str, Any], ns: Dict[str, Any]):
    """(tree, attr) with dicts and lists of lists rewritten (see the module doc);
    attr gains each dict entry's format and a list of lists' flat 'length'."""
    fd = tree.body[0]
    if not isinstance(fd, ast.FunctionDef):
        return tree, attr
    has = any(isinstance(n, (ast.Dict, ast.DictComp)) for n in walk_code(fd)) \
        or any(isinstance(m, dict) and "shape" in m for m in attr.values()) \
        or any(isinstance(n, ast.List) and any(
            isinstance(e, (ast.List, ast.Tuple, ast.BinOp, ast.Call)) for e in n.elts)
               for n in walk_code(fd)) \
        or any(isinstance(n, ast.Name) and isinstance(ns.get(n.id), dict) for n in walk_code(fd)) \
        or any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "dict"
               for n in walk_code(fd))
    if not has:
        return tree, attr
    tree = copy.deepcopy(tree)
    fd = tree.body[0]
    rw = _Rewrite(fd, attr, ns)
    fd.body = rw.block(fd.body)
    attr = dict(attr)
    for name, meta in attr.items():
        if isinstance(meta, dict) and "shape" in meta:
            attr[name] = dict({k: v for k, v in meta.items() if k != "shape"},
                              length=math.prod(int(x) for x in meta["shape"]))
    attr.update(rw.new_attr)
    return ast.fix_missing_locations(tree), attr


def shapes_of(attr: Dict[str, Any]) -> Dict[str, Tuple[int, ...]]:
    return {k: tuple(int(x) for x in m["shape"]) for k, m in (attr or {}).items()
            if isinstance(m, dict) and "shape" in m}


def flatten(value: Any) -> List[Any]:
    """A nested list (or numpy array) as a flat row-major list."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [x for v in value for x in flatten(v)]
    return [value]


def nest(flat: List[Any], shape: Tuple[int, ...]) -> Any:
    """A flat row-major list in a shape."""
    if len(shape) == 1:
        return list(flat)
    step = math.prod(shape[1:])
    return [nest(flat[k * step:(k + 1) * step], shape[1:]) for k in range(shape[0])]
