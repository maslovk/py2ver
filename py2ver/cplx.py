# cplx.py
"""
Split complex values into real and imaginary parts, for fixed point and for
complex parameters.

Every complex variable z becomes z_re / z_im (the names the rest of py2ver
uses for complex values), so the fixed-point pass and the translator only
see real arithmetic. A parameter with 'complex': True in attr becomes two
ports, <p>_re and <p>_im; its attr entry (format, 'next', 'init') applies to
both parts. A returned complex value is two outputs, <z>_re and <z>_im (or
ret_re / ret_im for an expression).

Supported: complex(a, b), complex constants (also module-level, and tables
of them indexed by a variable), + - *, / by a real value, unary - and +,
x if c else y, .real, .imag and .conjugate().

A list of complex values is two lists, <xs>_re and <xs>_im: literals,
append / extend, indexing and assigning elements, slices, +, [v] * n,
list(), len(), sum() and `for v in xs` work on it. A complex list parameter
has 'complex': True and 'length' in attr.
"""

from __future__ import annotations

import ast
import copy
from typing import Any, Dict, List, Set, Tuple


def _simple(node: ast.expr) -> bool:
    return isinstance(node, (ast.Name, ast.Constant))


def _num(x: float) -> ast.expr:
    """A real constant, as an int when it is one (so integer designs stay integer)."""
    v = int(x) if float(x).is_integer() else float(x)
    if v < 0:
        return ast.UnaryOp(op=ast.USub(), operand=ast.Constant(-v))
    return ast.Constant(v)


class _Split:
    def __init__(self, fd: ast.FunctionDef, attr: Dict[str, Dict[str, Any]],
                 ns: Dict[str, Any]) -> None:
        self.ns = ns
        self.params = [a.arg for a in fd.args.args]
        self.locals: Set[str] = set(self.params) | {
            n.id for n in ast.walk(fd) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
        }
        self.used = {n.id for n in ast.walk(fd) if isinstance(n, ast.Name)} | set(self.params)
        cparams = {p for p in self.params if attr.get(p, {}).get("complex")}
        self.clists: Set[str] = {p for p in cparams if "length" in attr.get(p, {})}
        self.cplx: Set[str] = cparams - self.clists
        self.pre: List[ast.stmt] = []
        self.n_temps = 0

    # ---------------- which values are complex ----------------
    def _global(self, name: str) -> Any:
        return self.ns.get(name) if name not in self.locals else None

    def _ctable(self, name: str) -> bool:
        table = self._global(name)
        return isinstance(table, (tuple, list)) and any(isinstance(x, complex) for x in table)

    def is_cl(self, n: ast.expr) -> bool:
        """Is n a list of complex values?"""
        if isinstance(n, ast.Name):
            return n.id in self.clists or self._ctable(n.id)
        if isinstance(n, (ast.List, ast.Tuple)):
            return any(self.is_c(e) for e in n.elts)
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Mult)):
            return self.is_cl(n.left) or self.is_cl(n.right)
        if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Slice):
            return self.is_cl(n.value)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "list" \
                and len(n.args) == 1:
            return self.is_cl(n.args[0])
        return False

    def is_c(self, n: ast.expr) -> bool:
        if isinstance(n, ast.Constant):
            return isinstance(n.value, complex)
        if isinstance(n, ast.Name):
            return n.id in self.cplx or isinstance(self._global(n.id), complex)
        if isinstance(n, ast.Subscript):
            return not isinstance(n.slice, ast.Slice) and self.is_cl(n.value)
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name) and f.id == "complex":
                return True
            if isinstance(f, ast.Name) and f.id == "sum" and len(n.args) == 1:
                return self.is_cl(n.args[0])
            return (isinstance(f, ast.Attribute) and f.attr == "conjugate" and self.is_c(f.value))
        if isinstance(n, ast.BinOp):
            return self.is_c(n.left) or self.is_c(n.right)
        if isinstance(n, ast.UnaryOp):
            return not isinstance(n.op, ast.Not) and self.is_c(n.operand)
        if isinstance(n, ast.IfExp):
            return self.is_c(n.body) or self.is_c(n.orelse)
        return False

    def infer(self, fd: ast.FunctionDef) -> None:
        """Variables given a complex value anywhere are complex (until nothing changes)."""
        changed = True
        while changed:
            changed = False
            for s in ast.walk(fd):
                pairs: List[Tuple[ast.expr, ast.expr]] = []
                if isinstance(s, ast.Assign) and len(s.targets) == 1:
                    t, v = s.targets[0], s.value
                    if isinstance(t, ast.Tuple) and isinstance(v, ast.Tuple) \
                            and len(t.elts) == len(v.elts):
                        pairs = list(zip(t.elts, v.elts))
                    else:
                        pairs = [(t, v)]
                elif isinstance(s, ast.AugAssign):
                    pairs = [(s.target, ast.BinOp(left=s.target, op=s.op, right=s.value))]
                elif (isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)
                      and isinstance(s.value.func, ast.Attribute)
                      and isinstance(s.value.func.value, ast.Name) and len(s.value.args) == 1):
                    base, arg = s.value.func.value.id, s.value.args[0]
                    if base not in self.clists and (
                            (s.value.func.attr == "append" and self.is_c(arg))
                            or (s.value.func.attr == "extend" and self.is_cl(arg))):
                        self.clists.add(base)
                        changed = True
                elif isinstance(s, ast.For) and isinstance(s.target, ast.Name):
                    if s.target.id not in self.cplx and self.is_cl(s.iter):
                        self.cplx.add(s.target.id)
                        changed = True
                for t, v in pairs:
                    if isinstance(t, ast.Name) and t.id not in self.cplx and self.is_c(v):
                        self.cplx.add(t.id)
                        changed = True
                    elif isinstance(t, ast.Name) and t.id not in self.clists and self.is_cl(v):
                        self.clists.add(t.id)
                        changed = True
                    elif (isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
                          and t.value.id not in self.clists and self.is_c(v)):
                        self.clists.add(t.value.id)
                        changed = True
        both = self.cplx & self.clists
        if both:
            raise NotImplementedError(f"'{sorted(both)[0]}' is both a complex value and a list")

    # ---------------- rewriting ----------------
    def _temp(self) -> str:
        while True:
            name = f"cx{self.n_temps}"
            self.n_temps += 1
            if name not in self.used and f"{name}_re" not in self.used:
                return name

    def _hoist(self, node: ast.expr) -> ast.expr:
        """A real value that is used more than once, as a name or constant."""
        if _simple(node):
            return node
        tmp = self._temp()
        self.pre.append(ast.Assign(targets=[ast.Name(id=tmp, ctx=ast.Store())], value=node))
        return ast.Name(id=tmp, ctx=ast.Load())

    def _hoist_c(self, parts: Tuple[ast.expr, ast.expr]) -> Tuple[ast.expr, ast.expr]:
        if all(_simple(p) for p in parts):
            return parts
        tmp = self._temp()
        self.pre.append(ast.Assign(
            targets=[ast.Tuple(elts=[ast.Name(id=f"{tmp}_{k}", ctx=ast.Store()) for k in ("re", "im")],
                               ctx=ast.Store())],
            value=ast.Tuple(elts=list(parts), ctx=ast.Load())))
        return ast.Name(id=f"{tmp}_re", ctx=ast.Load()), ast.Name(id=f"{tmp}_im", ctx=ast.Load())

    @staticmethod
    def _err(node: ast.AST, msg: str):
        line = getattr(node, "lineno", None)
        raise NotImplementedError(f"{msg} (line {line})" if line else msg)

    def comp(self, n: ast.expr) -> Tuple[ast.expr, ast.expr]:
        """(real, imaginary) parts of an expression (a real one has imaginary 0)."""
        if not self.is_c(n):
            return self.rx(n), ast.Constant(0)
        if isinstance(n, ast.Constant):
            return _num(n.value.real), _num(n.value.imag)
        if isinstance(n, ast.Name):
            if n.id in self.cplx:
                return (ast.Name(id=f"{n.id}_re", ctx=ast.Load()),
                        ast.Name(id=f"{n.id}_im", ctx=ast.Load()))
            v = self._global(n.id)
            return _num(v.real), _num(v.imag)
        if isinstance(n, ast.Subscript):                    # xs[i] of a complex list
            idx = self._hoist(self.rx(n.slice))
            re, im = self.comp_list(n.value)
            return (ast.Subscript(value=re, slice=idx, ctx=ast.Load()),
                    ast.Subscript(value=im, slice=copy.deepcopy(idx), ctx=ast.Load()))
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name) and f.id == "sum":    # sum(xs) of a complex list
                re, im = self.comp_list(n.args[0])
                return (ast.Call(func=ast.Name(id="sum", ctx=ast.Load()), args=[re], keywords=[]),
                        ast.Call(func=ast.Name(id="sum", ctx=ast.Load()), args=[im], keywords=[]))
            if isinstance(f, ast.Name):                     # complex(a, b)
                if n.keywords or not 1 <= len(n.args) <= 2 or any(self.is_c(a) for a in n.args):
                    self._err(n, "complex() takes one or two real values")
                re = self.rx(n.args[0])
                return re, (self.rx(n.args[1]) if len(n.args) == 2 else ast.Constant(0))
            re, im = self.comp(f.value)                     # z.conjugate()
            return re, ast.UnaryOp(op=ast.USub(), operand=im)
        if isinstance(n, ast.UnaryOp):
            re, im = self.comp(n.operand)
            if isinstance(n.op, ast.USub):
                return ast.UnaryOp(op=ast.USub(), operand=re), ast.UnaryOp(op=ast.USub(), operand=im)
            if isinstance(n.op, ast.UAdd):
                return re, im
            self._err(n, f"{type(n.op).__name__} is not defined for complex values")
        if isinstance(n, ast.IfExp):
            test = self._hoist(self.rx(n.test))
            (br, bi), (orr, oi) = self.comp(n.body), self.comp(n.orelse)
            return (ast.IfExp(test=test, body=br, orelse=orr),
                    ast.IfExp(test=copy.deepcopy(test), body=bi, orelse=oi))
        if isinstance(n, ast.BinOp):
            op = n.op
            if isinstance(op, (ast.Add, ast.Sub)):
                (a, b), (c, d) = self.comp(n.left), self.comp(n.right)
                return ast.BinOp(left=a, op=op, right=c), ast.BinOp(left=b, op=copy.deepcopy(op), right=d)
            if isinstance(op, ast.Mult):
                lc, rc = self.is_c(n.left), self.is_c(n.right)
                if lc and rc:
                    (a, b), (c, d) = self._hoist_c(self.comp(n.left)), self._hoist_c(self.comp(n.right))
                    cp = copy.deepcopy
                    return (ast.BinOp(left=ast.BinOp(left=a, op=ast.Mult(), right=c), op=ast.Sub(),
                                      right=ast.BinOp(left=b, op=ast.Mult(), right=d)),
                            ast.BinOp(left=ast.BinOp(left=cp(a), op=ast.Mult(), right=cp(d)), op=ast.Add(),
                                      right=ast.BinOp(left=cp(b), op=ast.Mult(), right=cp(c))))
                z, r = (n.left, n.right) if lc else (n.right, n.left)
                re, im = self.comp(z)
                r = self._hoist(self.rx(r))
                return (ast.BinOp(left=re, op=ast.Mult(), right=r),
                        ast.BinOp(left=im, op=ast.Mult(), right=copy.deepcopy(r)))
            if isinstance(op, ast.Div) and not self.is_c(n.right):
                re, im = self.comp(n.left)
                d = self._hoist(self.rx(n.right))
                return (ast.BinOp(left=re, op=ast.Div(), right=d),
                        ast.BinOp(left=im, op=ast.Div(), right=copy.deepcopy(d)))
            self._err(n, f"'{ast.unparse(n)}': complex values support + - *, and / by a real value")
        self._err(n, f"Unsupported complex expression: {ast.unparse(n)}")

    def comp_list(self, n: ast.expr) -> Tuple[ast.expr, ast.expr]:
        """(real, imaginary) lists of a list expression (a real list has imaginary zeros)."""
        if isinstance(n, ast.Name):
            if n.id in self.clists:
                return (ast.Name(id=f"{n.id}_re", ctx=ast.Load()),
                        ast.Name(id=f"{n.id}_im", ctx=ast.Load()))
            if self._ctable(n.id):
                table = [complex(x) for x in self._global(n.id)]
                return tuple(ast.Tuple(elts=[_num(getattr(x, part)) for x in table], ctx=ast.Load())
                             for part in ("real", "imag"))
        if isinstance(n, (ast.List, ast.Tuple)):
            parts = [self.comp(e) for e in n.elts]
            return (ast.List(elts=[p[0] for p in parts], ctx=ast.Load()),
                    ast.List(elts=[p[1] for p in parts], ctx=ast.Load()))
        if not self.is_cl(n):                               # a real list: imaginary zeros
            re = self.rx(n)
            zeros = ast.BinOp(left=ast.List(elts=[ast.Constant(0)], ctx=ast.Load()), op=ast.Mult(),
                              right=ast.Call(func=ast.Name(id="len", ctx=ast.Load()),
                                             args=[copy.deepcopy(re)], keywords=[]))
            return re, zeros
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add):
            (a, b), (c, d) = self.comp_list(n.left), self.comp_list(n.right)
            return ast.BinOp(left=a, op=ast.Add(), right=c), ast.BinOp(left=b, op=ast.Add(), right=d)
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult):      # [v] * n
            lst, count = (n.left, n.right) if self.is_cl(n.left) else (n.right, n.left)
            re, im = self.comp_list(lst)
            count = self.rx(count)
            return (ast.BinOp(left=re, op=ast.Mult(), right=count),
                    ast.BinOp(left=im, op=ast.Mult(), right=copy.deepcopy(count)))
        if isinstance(n, ast.Subscript):                    # a slice
            re, im = self.comp_list(n.value)
            sl = self.rx(n.slice)
            return (ast.Subscript(value=re, slice=sl, ctx=ast.Load()),
                    ast.Subscript(value=im, slice=copy.deepcopy(sl), ctx=ast.Load()))
        if isinstance(n, ast.Call):                         # list(xs)
            re, im = self.comp_list(n.args[0])
            return (ast.Call(func=ast.Name(id="list", ctx=ast.Load()), args=[re], keywords=[]),
                    ast.Call(func=ast.Name(id="list", ctx=ast.Load()), args=[im], keywords=[]))
        self._err(n, f"Unsupported list of complex values: {ast.unparse(n)}")

    def rx(self, n: ast.expr) -> ast.expr:
        """A real expression, with .real / .imag of complex values replaced by their parts."""
        if self.is_c(n):
            self._err(n, f"'{ast.unparse(n)}' is complex where a real value is needed")
        if self.is_cl(n):
            self._err(n, f"'{ast.unparse(n)}' is a list of complex values where a real value is needed")

        outer = self

        class _Real(ast.NodeTransformer):
            def visit_Attribute(self, a: ast.Attribute):
                if a.attr in ("real", "imag") and outer.is_c(a.value):
                    re, im = outer.comp(a.value)
                    return re if a.attr == "real" else im
                return self.generic_visit(a)

            def visit_Call(self, c: ast.Call):
                if (isinstance(c.func, ast.Name) and c.func.id == "len" and len(c.args) == 1
                        and outer.is_cl(c.args[0])):
                    return ast.Call(func=c.func, args=[outer.comp_list(c.args[0])[0]], keywords=[])
                if (isinstance(c.func, ast.Name) and c.func.id == "abs" and len(c.args) == 1
                        and outer.is_c(c.args[0])):
                    outer._err(c, "abs() of a complex value (a square root) is not supported "
                                  "here; use the squared magnitude z.real * z.real + z.imag * z.imag")
                return self.generic_visit(c)

        return _Real().visit(n)

    @staticmethod
    def _copy_list(node: ast.expr) -> ast.expr:
        """A list value to assign: a list name is copied (lists aren't shared)."""
        if isinstance(node, ast.Name):
            return ast.Call(func=ast.Name(id="list", ctx=ast.Load()), args=[node], keywords=[])
        return node

    def _assign(self, targets: List[ast.expr], values: List[ast.expr],
                s: ast.stmt) -> List[ast.stmt]:
        """targets = values (simultaneously), complex targets split in two."""
        if any(isinstance(t, ast.Name) and t.id in self.clists for t in targets) \
                or any(isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
                       and t.value.id in self.clists for t in targets):
            return self._assign_lists(targets, values, s)
        ts, vs = [], []
        for t, v in zip(targets, values):
            if isinstance(t, ast.Name) and t.id in self.cplx:
                re, im = self.comp(v)
                ts += [ast.Name(id=f"{t.id}_re", ctx=ast.Store()), ast.Name(id=f"{t.id}_im", ctx=ast.Store())]
                vs += [re, im]
            elif self.is_c(v):
                self._err(s, "Complex values can only be assigned to plain names")
            else:
                ts.append(self.rx(t) if isinstance(t, ast.Subscript) else t)
                vs.append(self.rx(v))
        if len(ts) == 1:
            return [ast.copy_location(ast.Assign(targets=ts, value=vs[0]), s)]
        return [ast.copy_location(ast.Assign(targets=[ast.Tuple(elts=ts, ctx=ast.Store())],
                                             value=ast.Tuple(elts=vs, ctx=ast.Load())), s)]

    def _assign_lists(self, targets: List[ast.expr], values: List[ast.expr],
                      s: ast.stmt) -> List[ast.stmt]:
        """Assignments involving complex lists: every value is computed into
        temporaries first, then assigned (so they still happen at once)."""
        out: List[ast.stmt] = []
        later: List[Tuple[ast.expr, ast.expr]] = []

        def store(name: str) -> ast.Name:
            return ast.Name(id=name, ctx=ast.Store())

        t0, v0 = targets[0], values[0]
        if (len(targets) == 1 and isinstance(t0, ast.Name)
                and t0.id not in {n.id for n in ast.walk(v0) if isinstance(n, ast.Name)}):
            re, im = self.comp_list(v0)                 # xs = <list not using xs>: directly
            return [ast.copy_location(ast.Assign(targets=[store(f"{t0.id}_{k}")],
                                                 value=self._copy_list(p)), s)
                    for k, p in (("re", re), ("im", im))]

        for t, v in zip(targets, values):
            if isinstance(t, ast.Name) and t.id in self.clists:
                re, im = self.comp_list(v)
                tmp = self._temp()
                out += [ast.Assign(targets=[store(f"{tmp}_re")], value=self._copy_list(re)),
                        ast.Assign(targets=[store(f"{tmp}_im")], value=self._copy_list(im))]
                later += [(store(f"{t.id}_re"), self._copy_list(ast.Name(id=f"{tmp}_re", ctx=ast.Load()))),
                          (store(f"{t.id}_im"), self._copy_list(ast.Name(id=f"{tmp}_im", ctx=ast.Load())))]
            elif isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) \
                    and t.value.id in self.clists:
                if isinstance(t.slice, ast.Slice):
                    self._err(s, "Assigning a slice of a complex list is not supported")
                idx = self._hoist(self.rx(t.slice))
                re, im = self._hoist_c(self.comp(v))
                for part, value in (("re", re), ("im", im)):
                    later.append((ast.Subscript(value=ast.Name(id=f"{t.value.id}_{part}", ctx=ast.Load()),
                                                slice=copy.deepcopy(idx), ctx=ast.Store()), value))
            else:
                stmts = self._assign([t], [v], s)
                tmp_targets = []
                for st in stmts:                            # value into temporaries first
                    tgts = st.targets[0].elts if isinstance(st.targets[0], ast.Tuple) else [st.targets[0]]
                    vals = st.value.elts if isinstance(st.targets[0], ast.Tuple) else [st.value]
                    for tg, vl in zip(tgts, vals):
                        tmp = self._temp()
                        out.append(ast.Assign(targets=[store(tmp)], value=vl))
                        tmp_targets.append((tg, ast.Name(id=tmp, ctx=ast.Load())))
                later += tmp_targets
        out += [ast.Assign(targets=[t], value=v) for t, v in later]
        return [ast.copy_location(st, s) for st in out]

    def block(self, stmts: List[ast.stmt]) -> List[ast.stmt]:
        saved, out = self.pre, []
        for s in stmts:
            self.pre = []
            new = self.stmt(s)
            out.extend(self.pre)
            out.extend(new)
        self.pre = saved
        return out

    def stmt(self, s: ast.stmt) -> List[ast.stmt]:
        if isinstance(s, ast.AugAssign):
            load = copy.deepcopy(s.target)
            for node in ast.walk(load):
                if hasattr(node, "ctx"):
                    node.ctx = ast.Load()
            s = ast.copy_location(ast.Assign(
                targets=[s.target], value=ast.BinOp(left=load, op=s.op, right=s.value)), s)
        if isinstance(s, ast.Assign) and len(s.targets) == 1:
            t, v = s.targets[0], s.value
            if isinstance(t, ast.Tuple) and isinstance(v, ast.Tuple) and len(t.elts) == len(v.elts):
                return self._assign(t.elts, v.elts, s)
            return self._assign([t], [v], s)
        if isinstance(s, ast.If):
            s.test = self.rx(s.test)
            s.body, s.orelse = self.block(s.body), self.block(s.orelse)
            return [s]
        if isinstance(s, ast.For) and self.is_cl(s.iter):
            # for v in xs (complex): for i in range(len(xs)): v = xs[i]
            if not isinstance(s.target, ast.Name):
                self._err(s, "Loop over a complex list with a single variable")
            re, im = self.comp_list(s.iter)
            if not isinstance(re, ast.Name):
                tmp = self._temp()
                self.pre += [ast.Assign(targets=[ast.Name(id=f"{tmp}_{k}", ctx=ast.Store())],
                                        value=self._copy_list(p)) for k, p in (("re", re), ("im", im))]
                re, im = (ast.Name(id=f"{tmp}_{k}", ctx=ast.Load()) for k in ("re", "im"))
            i = self._temp()
            v = s.target.id
            take = ast.Assign(
                targets=[ast.Tuple(elts=[ast.Name(id=f"{v}_re", ctx=ast.Store()),
                                         ast.Name(id=f"{v}_im", ctx=ast.Store())], ctx=ast.Store())],
                value=ast.Tuple(elts=[ast.Subscript(value=p, slice=ast.Name(id=i, ctx=ast.Load()),
                                                    ctx=ast.Load()) for p in (re, im)], ctx=ast.Load()))
            s.target = ast.Name(id=i, ctx=ast.Store())
            s.iter = ast.Call(func=ast.Name(id="range", ctx=ast.Load()), keywords=[], args=[
                ast.Call(func=ast.Name(id="len", ctx=ast.Load()), args=[copy.deepcopy(re)], keywords=[])])
            s.body = [take] + self.block(s.body)
            s.orelse = self.block(s.orelse)
            return [s]
        if isinstance(s, ast.For):
            if self.is_c(s.iter) or (isinstance(s.target, ast.Name) and s.target.id in self.cplx):
                self._err(s, "Loops over complex values are not supported")
            s.iter = self.rx(s.iter)
            s.body, s.orelse = self.block(s.body), self.block(s.orelse)
            return [s]
        if isinstance(s, ast.Return) and s.value is not None:
            vals = s.value.elts if isinstance(s.value, ast.Tuple) else [s.value]
            out: List[ast.expr] = []
            for i, v in enumerate(vals):
                if self.is_cl(v):                           # a complex list: <xs>_re, <xs>_im
                    if isinstance(v, ast.Name) and v.id in self.clists:
                        name = v.id
                    else:
                        name = f"ret{i}" if len(vals) > 1 else "ret"
                        re, im = self.comp_list(v)
                        self.pre += [ast.Assign(targets=[ast.Name(id=f"{name}_{k}", ctx=ast.Store())],
                                                value=self._copy_list(p))
                                     for k, p in (("re", re), ("im", im))]
                    out += [ast.Name(id=f"{name}_re", ctx=ast.Load()),
                            ast.Name(id=f"{name}_im", ctx=ast.Load())]
                    continue
                if not self.is_c(v):
                    if isinstance(v, ast.Name):
                        out.append(v)
                        continue
                    # A real expression is the output ret / ret<i>, even if it
                    # becomes a name here (z.real -> z_re)
                    name = f"ret{i}" if len(vals) > 1 else "ret"
                    if name in self.locals:
                        self._err(s, f"Output name '{name}' is already used")
                    self.pre.append(ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())],
                                               value=self.rx(v)))
                    out.append(ast.Name(id=name, ctx=ast.Load()))
                    continue
                if isinstance(v, ast.Name) and v.id in self.cplx:
                    name = v.id
                else:                                   # a complex expression: ret_re / ret_im
                    name = f"ret{i}" if len(vals) > 1 else "ret"
                    re, im = self.comp(v)
                    self.pre.append(ast.Assign(
                        targets=[ast.Tuple(elts=[ast.Name(id=f"{name}_re", ctx=ast.Store()),
                                                 ast.Name(id=f"{name}_im", ctx=ast.Store())],
                                           ctx=ast.Store())],
                        value=ast.Tuple(elts=[re, im], ctx=ast.Load())))
                out += [ast.Name(id=f"{name}_re", ctx=ast.Load()),
                        ast.Name(id=f"{name}_im", ctx=ast.Load())]
            s.value = ast.Tuple(elts=out, ctx=ast.Load()) if len(out) > 1 else out[0]
            return [s]
        if (isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)
                and isinstance(s.value.func, ast.Attribute) and isinstance(s.value.func.value, ast.Name)
                and s.value.func.value.id in self.clists and s.value.func.attr in ("append", "extend")):
            base, method, arg = s.value.func.value.id, s.value.func.attr, s.value.args[0]
            parts = self._hoist_c(self.comp(arg)) if method == "append" else self.comp_list(arg)
            return [ast.copy_location(ast.Expr(ast.Call(
                func=ast.Attribute(value=ast.Name(id=f"{base}_{k}", ctx=ast.Load()), attr=method,
                                   ctx=ast.Load()), args=[p], keywords=[])), s)
                for k, p in zip(("re", "im"), parts)]
        if isinstance(s, ast.Expr) and not isinstance(s.value, ast.Constant):
            s.value = self.rx(s.value)
        return [s]


def _returned_complex(fd: ast.FunctionDef, sp: _Split) -> Set[str]:
    ret = next((s for s in reversed(fd.body) if isinstance(s, ast.Return)), None)
    if ret is None or ret.value is None:
        return set()
    vals = ret.value.elts if isinstance(ret.value, ast.Tuple) else [ret.value]
    return {f"ret{i}" if len(vals) > 1 else "ret"
            for i, v in enumerate(vals) if not isinstance(v, ast.Name) and (sp.is_c(v) or sp.is_cl(v))}


def _may_be_complex(fd: ast.FunctionDef, attr: Dict[str, Dict[str, Any]],
                    ns: Dict[str, Any]) -> bool:
    """Could anything in the function be complex? (A quick look, so designs
    without complex values skip the type inference.)"""
    if any(isinstance(m, dict) and m.get("complex") for m in attr.values()):
        return True
    for n in ast.walk(fd):
        if isinstance(n, ast.Constant) and isinstance(n.value, complex):
            return True
        if isinstance(n, ast.Attribute) and n.attr in ("real", "imag", "conjugate"):
            return True
        if isinstance(n, ast.Name):
            v = ns.get(n.id)
            if n.id == "complex" or isinstance(v, complex) or getattr(v, "__name__", "") == "cmath":
                return True
            if isinstance(v, (list, tuple)) and any(isinstance(x, complex) for x in v):
                return True
    return False


def split_complex(tree: ast.Module, attr: Dict[str, Dict[str, Any]], ns: Dict[str, Any]):
    """
    Rewrite the module's first function so complex values are pairs of real
    ones. Returns (tree, attr with complex entries split, set of complex
    parameter names, set of all complex names: values and lists).
    """
    fd = tree.body[0]
    if not _may_be_complex(fd, attr, ns):
        return tree, attr, set(), set()
    sp = _Split(fd, attr, ns)
    sp.infer(fd)
    if not sp.cplx and not sp.clists and not any(sp.is_c(n) for n in ast.walk(fd)
                                                 if isinstance(n, ast.expr)):
        return tree, attr, set(), set()
    complex_names = sp.cplx | sp.clists | _returned_complex(fd, sp)
    fd.body = sp.block(fd.body)
    args = []
    for a in fd.args.args:
        if a.arg in sp.cplx or a.arg in sp.clists:
            args += [ast.arg(arg=f"{a.arg}_re"), ast.arg(arg=f"{a.arg}_im")]
        else:
            args.append(a)
    fd.args.args = args
    ast.fix_missing_locations(tree)

    out: Dict[str, Dict[str, Any]] = {}
    for name, meta in attr.items():
        if name not in complex_names or not isinstance(meta, dict):
            out[name] = meta
            continue
        for k, part in (("re", "real"), ("im", "imag")):
            m = {key: v for key, v in meta.items() if key != "complex"}
            if "next" in m:
                m["next"] = f"{m['next']}_{k}"
            if "init" in m:
                init = m["init"]
                m["init"] = ([getattr(complex(v), part) for v in init]
                             if isinstance(init, (list, tuple)) else getattr(complex(init), part))
            out[f"{name}_{k}"] = m
    return tree, out, {p for p in sp.params if p in sp.cplx or p in sp.clists}, complex_names
