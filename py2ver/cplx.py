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
        self.cplx: Set[str] = {p for p in self.params if attr.get(p, {}).get("complex")}
        self.pre: List[ast.stmt] = []
        self.n_temps = 0

    # ---------------- which values are complex ----------------
    def _global(self, name: str) -> Any:
        return self.ns.get(name) if name not in self.locals else None

    def is_c(self, n: ast.expr) -> bool:
        if isinstance(n, ast.Constant):
            return isinstance(n.value, complex)
        if isinstance(n, ast.Name):
            return n.id in self.cplx or isinstance(self._global(n.id), complex)
        if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name):
            table = self._global(n.value.id)
            return isinstance(table, (tuple, list)) and any(isinstance(x, complex) for x in table)
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name) and f.id == "complex":
                return True
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
                for t, v in pairs:
                    if isinstance(t, ast.Name) and t.id not in self.cplx and self.is_c(v):
                        self.cplx.add(t.id)
                        changed = True

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
        if isinstance(n, ast.Subscript):
            table = [complex(x) for x in self._global(n.value.id)]
            if isinstance(n.slice, ast.Slice):
                self._err(n, "Slices of complex tables are not supported")
            idx = self._hoist(self.rx(n.slice))
            parts = []
            for part in ("real", "imag"):
                elts = [_num(getattr(x, part)) for x in table]
                parts.append(ast.Subscript(value=ast.Tuple(elts=elts, ctx=ast.Load()),
                                           slice=copy.deepcopy(idx), ctx=ast.Load()))
            return parts[0], parts[1]
        if isinstance(n, ast.Call):
            f = n.func
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

    def rx(self, n: ast.expr) -> ast.expr:
        """A real expression, with .real / .imag of complex values replaced by their parts."""
        if self.is_c(n):
            self._err(n, f"'{ast.unparse(n)}' is complex where a real value is needed")

        outer = self

        class _Real(ast.NodeTransformer):
            def visit_Attribute(self, a: ast.Attribute):
                if a.attr in ("real", "imag") and outer.is_c(a.value):
                    re, im = outer.comp(a.value)
                    return re if a.attr == "real" else im
                return self.generic_visit(a)

            def visit_Call(self, c: ast.Call):
                if (isinstance(c.func, ast.Name) and c.func.id == "abs" and len(c.args) == 1
                        and outer.is_c(c.args[0])):
                    outer._err(c, "abs() of a complex value (a square root) is not supported "
                                  "here; use the squared magnitude z.real * z.real + z.imag * z.imag")
                return self.generic_visit(c)

        return _Real().visit(n)

    def _assign(self, targets: List[ast.expr], values: List[ast.expr], s: ast.stmt) -> ast.stmt:
        """targets = values (simultaneously), complex targets split in two."""
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
            return ast.copy_location(ast.Assign(targets=ts, value=vs[0]), s)
        return ast.copy_location(ast.Assign(targets=[ast.Tuple(elts=ts, ctx=ast.Store())],
                                            value=ast.Tuple(elts=vs, ctx=ast.Load())), s)

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
                return [self._assign(t.elts, v.elts, s)]
            return [self._assign([t], [v], s)]
        if isinstance(s, ast.If):
            s.test = self.rx(s.test)
            s.body, s.orelse = self.block(s.body), self.block(s.orelse)
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
        if isinstance(s, ast.Expr) and not isinstance(s.value, ast.Constant):
            s.value = self.rx(s.value)
        return [s]


def _returned_complex(fd: ast.FunctionDef, sp: _Split) -> Set[str]:
    ret = next((s for s in reversed(fd.body) if isinstance(s, ast.Return)), None)
    if ret is None or ret.value is None:
        return set()
    vals = ret.value.elts if isinstance(ret.value, ast.Tuple) else [ret.value]
    return {f"ret{i}" if len(vals) > 1 else "ret"
            for i, v in enumerate(vals) if not isinstance(v, ast.Name) and sp.is_c(v)}


def split_complex(tree: ast.Module, attr: Dict[str, Dict[str, Any]], ns: Dict[str, Any]):
    """
    Rewrite the module's first function so complex values are pairs of real
    ones. Returns (tree, attr with complex entries split, set of complex
    parameter names).
    """
    fd = tree.body[0]
    sp = _Split(fd, attr, ns)
    sp.infer(fd)
    if not sp.cplx and not any(sp.is_c(n) for n in ast.walk(fd) if isinstance(n, ast.expr)):
        return tree, attr, set()
    complex_names = sp.cplx | _returned_complex(fd, sp)
    fd.body = sp.block(fd.body)
    args = []
    for a in fd.args.args:
        if a.arg in sp.cplx:
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
    return tree, out, {p for p in sp.params if p in sp.cplx}
