# inline.py
"""
Inline calls to your own helper functions before the visitor runs.

    def mac(acc, x, k):            def top(a, b):
        p = x * k                      y = mac(a, b, 3) + 1
        return acc + p                 return y

becomes, inside top:

    __mac0_acc = a
    __mac0_x = b
    __mac0_k = 3
    __mac0_p = __mac0_x * __mac0_k
    __mac0_ret = __mac0_acc + __mac0_p
    y = __mac0_ret + 1

Helpers are found in the calling function's globals and must be ordinary
Python functions whose source is available. Their body may use anything the
main function may use, and must end with a single `return` (a tuple return
works with `x, y = helper(...)`). Helpers may call helpers; recursion is
rejected. Module-level integer constants used by name (e.g. MASK = 0xFFFF)
are replaced by their values, and module-level tuples/lists of integers
(tables such as an S-box) by literals.
"""

from __future__ import annotations

import ast
import copy
import inspect
import itertools
import textwrap
from typing import Any, Dict, List, Optional, Set, Tuple

from .hwtypes import strip_helper_annotations

_BUILTINS = {"range", "min", "max", "abs", "complex"}


def _assigned_names(stmts: List[ast.stmt]) -> Set[str]:
    names: Set[str] = set()
    for s in stmts:
        for node in ast.walk(s):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                names.add(node.id)
            elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                names.add(node.target.id)
    return names


class _Inliner:
    def __init__(self, globals_: Dict[str, Any]):
        self.globals = globals_
        self.counter = itertools.count()
        self.stack: List[str] = []

    # --- lookups ---
    def _helper(self, name: str) -> Optional[ast.FunctionDef]:
        if name in _BUILTINS or name in self.globals.get("__py2ver_tables__", ()):
            return None                     # (a function built as a table isn't inlined)
        obj = self.globals.get(name)
        if not inspect.isfunction(obj):
            return None
        try:
            src = textwrap.dedent(inspect.getsource(obj))
        except (OSError, TypeError) as e:
            raise NotImplementedError(f"Cannot read the source of helper '{name}': {e}") from e
        fd = ast.parse(src).body[0]
        if not isinstance(fd, ast.FunctionDef):
            raise NotImplementedError(f"Helper '{name}' is not a plain function")
        from .control import structure_function
        return structure_function(strip_helper_annotations(fd))  # (early returns, while, ...)

    def _constants(self, locals_: Set[str]):
        """Transformer replacing names of module-level int constants with values."""
        g = self.globals

        class _Consts(ast.NodeTransformer):
            def visit_Name(self, node: ast.Name):
                if not isinstance(node.ctx, ast.Load) or node.id in locals_:
                    return node
                value = g.get(node.id)
                if isinstance(value, int):
                    return ast.copy_location(ast.Constant(value=int(value)), node)
                if (isinstance(value, (tuple, list)) and value
                        and all(isinstance(v, int) for v in value)):
                    # A module-level table becomes a literal (a ROM when indexed by a variable)
                    table = ast.Tuple(elts=[ast.Constant(value=int(v)) for v in value],
                                      ctx=ast.Load())
                    return ast.copy_location(table, node)
                return node

            def visit_Call(self, node: ast.Call):
                # math / cmath functions (and complex()) of constants: their value
                self.generic_visit(node)
                root = node.func
                while isinstance(root, ast.Attribute):
                    root = root.value
                if (node.keywords or not isinstance(root, ast.Name) or root.id in locals_
                        or any(isinstance(n, ast.Name) and n.id in locals_
                               for a in node.args for n in ast.walk(a))):
                    return node
                try:
                    fn = eval(compile(ast.Expression(node.func), "<call>", "eval"), dict(g))
                    if fn is not complex and getattr(fn, "__module__", None) not in ("math", "cmath"):
                        return node
                    value = eval(compile(ast.Expression(node), "<call>", "eval"), dict(g))
                except Exception:
                    return node
                if isinstance(value, (int, float, complex)) and not isinstance(value, bool):
                    return ast.copy_location(ast.Constant(value=value), node)
                return node

        return _Consts()

    # --- statements ---
    def block(self, stmts: List[ast.stmt], locals_: Set[str]) -> List[ast.stmt]:
        out: List[ast.stmt] = []
        for s in stmts:
            pre, s2 = self.stmt(s, locals_)
            out.extend(pre)
            out.append(s2)
        return out

    def stmt(self, s: ast.stmt, locals_: Set[str]) -> Tuple[List[ast.stmt], ast.stmt]:
        s = self._constants(locals_).visit(s)
        if isinstance(s, ast.If):
            pre, s.test = self.expr(s.test, locals_)
            s.body = self.block(s.body, locals_)
            s.orelse = self.block(s.orelse, locals_)
            return pre, s
        if isinstance(s, ast.For):
            pre, s.iter = self.expr(s.iter, locals_)
            s.body = self.block(s.body, locals_)
            return pre, s
        if isinstance(s, (ast.Assign, ast.AugAssign, ast.Return, ast.Expr)) and s.value is not None:
            pre, s.value = self.expr(s.value, locals_)
            return pre, s
        return [], s

    # --- expressions ---
    def expr(self, node: ast.expr, locals_: Set[str]) -> Tuple[List[ast.stmt], ast.expr]:
        pre: List[ast.stmt] = []
        inliner = self

        class _Calls(ast.NodeTransformer):
            def visit_Call(self, call: ast.Call):
                self.generic_visit(call)  # inner calls first
                if not isinstance(call.func, ast.Name):
                    return call
                fd = inliner._helper(call.func.id)
                if fd is None:
                    return call
                stmts, value = inliner.inline(fd, call)
                pre.extend(stmts)
                return ast.copy_location(value, call)

        return pre, _Calls().visit(node)

    def inline(self, fd: ast.FunctionDef, call: ast.Call) -> Tuple[List[ast.stmt], ast.expr]:
        if fd.name in self.stack:
            raise NotImplementedError(f"Recursive call to '{fd.name}' cannot be inlined")
        if call.keywords or fd.args.vararg or fd.args.kwarg or fd.args.kwonlyargs:
            raise NotImplementedError(f"Call to '{fd.name}': only positional arguments are supported")
        params = [a.arg for a in fd.args.args]
        if len(call.args) != len(params) or fd.args.defaults:
            raise NotImplementedError(
                f"Call to '{fd.name}' needs exactly {len(params)} positional arguments"
            )

        body = [s for s in fd.body
                if not (isinstance(s, ast.Expr) and isinstance(getattr(s, "value", None), ast.Constant))]
        if not body or not isinstance(body[-1], ast.Return) or body[-1].value is None:
            raise NotImplementedError(f"Helper '{fd.name}' must end with 'return <value>'")
        if any(isinstance(n, ast.Return) for s in body[:-1] for n in ast.walk(s)):
            raise NotImplementedError(f"Helper '{fd.name}' may only return at its end")

        # Arguments are bound by copying (a list too), so a helper may not
        # change a list argument in place: the caller wouldn't see it
        for n in (n for st in body for n in ast.walk(st)):
            base = None
            if isinstance(n, (ast.Assign, ast.AugAssign)):
                for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                    if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name):
                        base = t.value.id
            elif (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                  and isinstance(n.func.value, ast.Name)
                  and n.func.attr in ("append", "extend", "insert", "pop", "remove")):
                base = n.func.value.id
            if base in params and base not in _assigned_names(body):
                raise NotImplementedError(
                    f"Helper '{fd.name}' changes its argument '{base}' in place; copy it first "
                    f"({base} = list({base})) and return it")

        k = next(self.counter)
        prefix = f"__{fd.name}{k}_"
        locals_ = set(params) | _assigned_names(body)
        rename = {n: prefix + n for n in locals_}

        class _Rename(ast.NodeTransformer):
            def visit_Name(self, node: ast.Name):
                if node.id in rename:
                    return ast.copy_location(ast.Name(id=rename[node.id], ctx=node.ctx), node)
                return node

        body = [_Rename().visit(copy.deepcopy(s)) for s in body]
        renamed_locals = set(rename.values())

        stmts: List[ast.stmt] = [
            ast.Assign(targets=[ast.Name(id=prefix + p, ctx=ast.Store())], value=arg)
            for p, arg in zip(params, call.args)
        ]
        self.stack.append(fd.name)
        try:
            stmts += self.block(body[:-1], renamed_locals)
            ret_pre, ret_val = self.expr(self._constants(renamed_locals).visit(body[-1].value),
                                         renamed_locals)
        finally:
            self.stack.pop()
        stmts += ret_pre

        if isinstance(ret_val, ast.Tuple):
            names = []
            for i, elt in enumerate(ret_val.elts):
                name = f"{prefix}ret{i}"
                stmts.append(ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=elt))
                names.append(ast.Name(id=name, ctx=ast.Load()))
            return stmts, ast.Tuple(elts=names, ctx=ast.Load())
        name = f"{prefix}ret"
        stmts.append(ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=ret_val))
        return stmts, ast.Name(id=name, ctx=ast.Load())


def inline_helpers(tree: ast.Module, globals_: Dict[str, Any]) -> ast.Module:
    """Inline helper calls (and int constants) in the module's first function."""
    tree = copy.deepcopy(tree)
    fd = tree.body[0]
    if not isinstance(fd, ast.FunctionDef):
        return tree
    params = {a.arg for a in fd.args.args}
    locals_ = params | _assigned_names(fd.body)
    inliner = _Inliner(globals_)
    inliner.stack.append(fd.name)
    fd.body = inliner.block(fd.body, locals_)
    ast.fix_missing_locations(tree)
    return tree
