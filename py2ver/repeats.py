"""
Repeated layers folded into a loop.

A model's identical layers (a Transformer's blocks: the same code, each with its
own weights) are written out one after the other in a loop program. Here a run
of k copies of the same top-level statements becomes

    for rep in range(k):
        <the first copy, each weight table T read as T_all[rep * len(T) + i]>
        <the lists the next copy reads copied back: x = x_out>

so the hardware has one copy of the layer's loops instead of k. The copies must
match exactly but for their names: their own lists and values (renamed to the
first copy's), their tables (of the same length: one table of all of them), and
what each reads from the copy before (the layer's input: copied at the end of
each turn).
"""
import ast
import copy
import os
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

MIN_STMTS = 3          # a repeat of this many top-level statements or more
# (in the namespace) loop variable -> the names its copies' code assigns: their
# values are profiled per copy, for formats as fine as each copy's (auto.py)
REPEATS_KEY = "__py2ver_repeats__"


def _is_table(ns: Dict[str, Any], name: str) -> bool:
    return isinstance(ns.get(name), tuple)


def _key(stmt: ast.stmt, ns: Dict[str, Any]) -> str:
    """The statement's code with its names left out (tables told apart)."""
    s = copy.deepcopy(stmt)
    for n in ast.walk(s):
        if isinstance(n, ast.Name):
            n.id = "T" if _is_table(ns, n.id) else "v"
    return ast.dump(s)


def _stored(stmts: List[ast.stmt]) -> Set[str]:
    out = set()
    for s in stmts:
        for n in ast.walk(s):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                out.add(n.id)
    return out


def _pairs(a: List[ast.stmt], b: List[ast.stmt]):
    """Corresponding names of two copies (the same code but for its names)."""
    for x, y in zip(a, b):
        for m, n in zip(ast.walk(x), ast.walk(y)):
            if isinstance(m, ast.Name):
                yield m.id, n.id


def _list_length(stmts: List[ast.stmt], name: str) -> Optional[int]:
    """n of `name = [0] * n` among the statements."""
    for s in stmts:
        if (isinstance(s, ast.Assign) and len(s.targets) == 1
                and isinstance(s.targets[0], ast.Name) and s.targets[0].id == name
                and isinstance(s.value, ast.BinOp) and isinstance(s.value.op, ast.Mult)
                and isinstance(s.value.left, ast.List) and isinstance(s.value.right, ast.Constant)):
            return int(s.value.right.value)
    return None


def _candidates(keys: List[str]) -> List[Tuple[int, int, int]]:
    """(start, period, copies) of the runs of repeated statements, the most
    statements saved first."""
    n = len(keys)
    out = []
    for p in range(MIN_STMTS, n // 2 + 1):
        for s in range(0, n - 2 * p + 1):
            k = 1
            while s + (k + 1) * p <= n and keys[s + k * p:s + (k + 1) * p] == keys[s:s + p]:
                k += 1
            if k >= 2:
                out.append((s, p, k))
    out.sort(key=lambda c: (-(c[2] - 1) * c[1], c[0]))
    return out


def _fold(body: List[ast.stmt], s: int, p: int, k: int, ns: Dict[str, Any],
          fresh: Callable[[str], str]) -> Optional[Tuple[List[ast.stmt], Dict[str, tuple]]]:
    copies = [body[s + j * p:s + (j + 1) * p] for j in range(k)]
    head, tail = body[:s], body[s + k * p:]
    own = [_stored(c) for c in copies]
    # (each copy's names for the first copy's)
    maps: List[Dict[str, str]] = [{} for _ in range(k)]
    for j in range(k):
        back: Dict[str, str] = {}
        for a, b in _pairs(copies[0], copies[j]):
            if maps[j].setdefault(a, b) != b or back.setdefault(b, a) != a:
                return None
    tables: Dict[str, List[str]] = {}
    carried: Dict[str, str] = {}            # the first copy's input -> what it is in the next
    for a in maps[0]:
        ids = [maps[j][a] for j in range(k)]
        if a in own[0]:
            if not all(ids[j] in own[j] for j in range(k)):
                return None
        elif _is_table(ns, a):
            if not all(_is_table(ns, t) and len(ns[t]) == len(ns[a]) for t in ids):
                return None
            if any(ns[t] != ns[a] for t in ids):      # (else the same values: one table)
                tables[a] = ids
        elif any(t != a for t in ids):
            # (an input of the layer: the copy before's value of one of its own)
            if any(t in ns for t in ids):
                return None
            o = ids[1]
            if o not in own[0] or any(ids[j] != maps[j - 1][o] for j in range(2, k)):
                return None
            carried[a] = o
    # A weight table read only by index (one table of all the copies')
    for c in copies[0]:
        for n in ast.walk(c):
            for f, v in ast.iter_fields(n):
                vals = v if isinstance(v, list) else [v]
                for x in vals:
                    if isinstance(x, ast.Name) and x.id in tables and not (
                            isinstance(n, ast.Subscript) and f == "value"):
                        return None
    # After the run: only the last copy's values, by their first-copy names, and
    # not the run's first inputs (copied over)
    later = {maps[k - 1][a]: a for a in own[0]}
    earlier = set().union(*own[:-1]) - set(later)
    used_after = {n.id for x in tail for n in ast.walk(x) if isinstance(n, ast.Name)}
    if used_after & (earlier | set(carried)):
        return None
    for a, o in carried.items():
        # (a list made by [0] * n, as long as the one it is copied into; or a value)
        indexed = {n.value.id for x in head + copies[0] for n in ast.walk(x)
                   if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)}
        if (o in indexed or a in indexed) and (
                _list_length(copies[0], o) is None
                or _list_length(head, a) != _list_length(copies[0], o)):
            return None

    rep = fresh("rep")
    loop_body = copy.deepcopy(copies[0])
    merged: Dict[str, tuple] = {}
    names: Dict[str, str] = {}
    from .auto import WEIGHT_TABLES
    for a, ids in tables.items():
        name = fresh(f"{a}_all")
        merged[name] = tuple(v for t in ids for v in ns[t])
        names[a] = name
        if any(t in ns.get(WEIGHT_TABLES, ()) for t in ids):   # (a format from its values)
            ns[WEIGHT_TABLES].add(name)

    class _Tables(ast.NodeTransformer):
        def visit_Subscript(self, x: ast.Subscript):
            self.generic_visit(x)
            if isinstance(x.value, ast.Name) and x.value.id in names:
                n = len(ns[x.value.id])
                x.slice = ast.BinOp(left=ast.BinOp(left=ast.Name(id=rep, ctx=ast.Load()),
                                                   op=ast.Mult(), right=ast.Constant(n)),
                                    op=ast.Add(), right=x.slice)
                x.value = ast.Name(id=names[x.value.id], ctx=ast.Load())
            return x
    loop_body = [_Tables().visit(c) for c in loop_body]
    for a, o in carried.items():                  # (the next turn's inputs)
        n = _list_length(copies[0], o)
        if n is None:
            loop_body.append(ast.Assign(targets=[ast.Name(id=a, ctx=ast.Store())],
                                        value=ast.Name(id=o, ctx=ast.Load())))
            continue
        t = fresh("t")
        loop_body.append(ast.For(
            target=ast.Name(id=t, ctx=ast.Store()),
            iter=ast.Call(func=ast.Name(id="range", ctx=ast.Load()), args=[ast.Constant(n)],
                          keywords=[]),
            body=[ast.Assign(targets=[ast.Subscript(value=ast.Name(id=a, ctx=ast.Load()),
                                                    slice=ast.Name(id=t, ctx=ast.Load()),
                                                    ctx=ast.Store())],
                             value=ast.Subscript(value=ast.Name(id=o, ctx=ast.Load()),
                                                 slice=ast.Name(id=t, ctx=ast.Load()),
                                                 ctx=ast.Load()))],
            orelse=[]))
    ns.setdefault(REPEATS_KEY, {})[rep] = sorted(own[0])
    loop = ast.For(target=ast.Name(id=rep, ctx=ast.Store()),
                   iter=ast.Call(func=ast.Name(id="range", ctx=ast.Load()),
                                 args=[ast.Constant(k)], keywords=[]),
                   body=loop_body, orelse=[])

    class _Last(ast.NodeTransformer):
        def visit_Name(self, x: ast.Name):
            if x.id in later:
                x.id = later[x.id]
            return x
    tail = [_Last().visit(copy.deepcopy(x)) for x in tail]
    return head + [loop] + tail, merged


def fold_repeats(body: List[ast.stmt], ns: Dict[str, Any],
                 taken: Set[str]) -> Tuple[List[ast.stmt], Dict[str, tuple]]:
    """The top-level statements with their longest runs of repeated copies as loops
    (see the module's doc). taken: names in use. Returns (statements, the new
    tables: name -> values, also put in ns)."""
    if os.environ.get("PY2VER_FOLD", "1") == "0":
        return body, {}
    used = set(taken) | {n.id for s in body for n in ast.walk(s) if isinstance(n, ast.Name)}

    def fresh(base: str) -> str:
        k = 0
        while f"{base}{k}" in used or f"{base}{k}" in ns:
            k += 1
        used.add(f"{base}{k}")
        return f"{base}{k}"
    new_tables: Dict[str, tuple] = {}
    while True:
        keys = [_key(x, ns) for x in body]
        for s, p, k in _candidates(keys):
            done = _fold(body, s, p, k, ns, fresh)
            if done is not None:
                body, merged = done
                ns.update(merged)
                new_tables.update(merged)
                break
        else:
            return body, new_tables
