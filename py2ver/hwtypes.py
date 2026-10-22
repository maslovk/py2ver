# hwtypes.py
"""
Hardware types for annotations, an alternative to the `attr` dict:

    def lpf_step(x: S[12], y_prev: State(S[16], next="y")) -> S[16]:
        err: S[16] = x - y_prev
        delta: S[16] = err >> 3
        y = y_prev + delta
        return y

    U[8], S[12]                  unsigned / signed of that width
    Reg(U[9])                    registered signal ('type': 'reg')
    Array(U[8], 8)               list of 8 elements ('length')
    State(S[16], next="y")       state register loaded from y ('next', 'init')
    Ram(U[16], 256)              block-RAM memory ('memory': 'ram')

Annotations on parameters, on assignments (`t: U[8] = ...`, or a bare
`t: U[8]`) and on the return (`-> (U[9], S[9])`, one type per returned value)
become attr entries. An explicit attr entry for the same name is merged over
the annotation.
"""

from __future__ import annotations

import ast
import builtins
import inspect
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Optional, Tuple


@dataclass(frozen=True)
class HwType:
    width: int
    signed: bool = False
    reg: bool = False
    length: Optional[int] = None
    next: Optional[str] = None
    init: Any = None
    memory: Optional[str] = None

    def attr(self) -> Dict[str, Any]:
        """The equivalent attr entry."""
        d: Dict[str, Any] = {"signed": int(self.signed), "width": self.width,
                             "type": "reg" if self.reg else "wire"}
        if self.length is not None:
            d["length"] = self.length
        if self.next is not None:
            d["next"] = self.next
        if self.init is not None:
            d["init"] = self.init
        if self.memory is not None:
            d["memory"] = self.memory
        return d

    def __repr__(self) -> str:
        r = f"{'S' if self.signed else 'U'}[{self.width}]"
        if self.length is not None:
            r = f"{'Ram' if self.memory else 'Array'}({r}, {self.length})"
        if self.reg:
            r = f"Reg({r})"
        if self.next is not None or (self.init is not None and not self.memory):
            r = f"State({r}, next={self.next!r}, init={self.init!r})"
        return r


class _Int:
    def __init__(self, signed: bool) -> None:
        self.signed = signed

    def __getitem__(self, width: int) -> HwType:
        if not isinstance(width, int) or isinstance(width, bool) or width < 1:
            raise ValueError(f"width must be a positive int, got {width!r}")
        return HwType(width=width, signed=self.signed)

    def __repr__(self) -> str:
        return "S" if self.signed else "U"


U = _Int(False)
S = _Int(True)


def _scalar(t: HwType, what: str) -> HwType:
    if not isinstance(t, HwType):
        raise TypeError(f"{what} takes a type such as U[8], got {t!r}")
    return t


def Reg(t: HwType) -> HwType:
    """A registered signal: one clock of latency ('type': 'reg')."""
    return replace(_scalar(t, "Reg"), reg=True)


def Array(t: HwType, length: int) -> HwType:
    """A list of `length` elements of type t ('length' in attr)."""
    t = _scalar(t, "Array")
    if t.length is not None:
        raise NotImplementedError("Nested lists are not supported")
    return replace(t, length=int(length))


def State(t: HwType, next: str, init: Any = None) -> HwType:
    """A state argument loaded from the returned value `next` every sample."""
    return replace(_scalar(t, "State"), next=next, init=init)


def Ram(t: HwType, length: int, init: Any = None) -> HwType:
    """A block-RAM memory argument of `length` elements, updated in place."""
    return replace(Array(t, length), memory="ram", init=init)


# ----------------------------- reading annotations -----------------------------

def _namespace(func: Callable) -> Dict[str, Any]:
    ns = dict(vars(builtins))
    ns.update(getattr(func, "__globals__", {}))
    try:
        ns.update(inspect.getclosurevars(func).nonlocals)
    except (TypeError, ValueError):
        pass
    return ns


def _eval(node: ast.expr, ns: Dict[str, Any], where: str) -> Any:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        node = ast.parse(node.value, mode="eval").body
    try:
        return eval(compile(ast.Expression(node), "<annotation>", "eval"), ns)
    except Exception as e:
        raise ValueError(f"Cannot evaluate the annotation of {where}: {e}") from e


def _return_names(fd: ast.FunctionDef):
    """Names that the function's return annotation types: a returned variable's
    own name, or ret / ret<i> for a returned expression (as the visitor names them)."""
    ret = next((s for s in reversed(fd.body) if isinstance(s, ast.Return)), None)
    if ret is None or ret.value is None:
        return []
    vals = ret.value.elts if isinstance(ret.value, ast.Tuple) else [ret.value]
    return [v.id if isinstance(v, ast.Name)
            else (f"ret{i}" if len(vals) > 1 else "ret")
            for i, v in enumerate(vals)]


def read_annotations(tree: ast.Module, func: Callable) -> Tuple[ast.Module, Dict[str, Dict[str, Any]]]:
    """
    Collect hardware types from the annotations of the module's first function
    and strip them; returns (tree, attr). Annotations that aren't hardware types
    (int, ...) on parameters and the return are ignored.
    """
    fd = tree.body[0]
    if not isinstance(fd, ast.FunctionDef):
        return tree, {}
    ns = _namespace(func)
    found: Dict[str, HwType] = {}
    params = {a.arg for a in fd.args.args}

    def add(name: str, t: Any, where: str, strict: bool) -> None:
        if not isinstance(t, HwType):
            if strict:
                raise TypeError(f"{where} is annotated with {t!r}, not a type such as U[8]")
            return
        if name in found and found[name] != t:
            raise ValueError(f"'{name}' is annotated as both {found[name]!r} and {t!r}")
        found[name] = t

    for a in fd.args.args:
        if a.annotation is not None:
            add(a.arg, _eval(a.annotation, ns, f"parameter '{a.arg}'"), f"parameter '{a.arg}'", False)
            a.annotation = None

    if fd.returns is not None:
        rt = _eval(fd.returns, ns, "the return")
        types = list(getattr(rt, "__args__", ())) if getattr(rt, "__origin__", None) is tuple \
            else list(rt) if isinstance(rt, tuple) else [rt]
        if any(isinstance(t, HwType) for t in types):
            names = _return_names(fd)
            if len(types) != len(names):
                raise ValueError(f"The return annotation has {len(types)} types for "
                                 f"{len(names)} returned values")
            for name, t in zip(names, types):
                if name in params:
                    raise ValueError(f"Returned parameter '{name}' takes its type from "
                                     "the parameter; don't annotate it in the return")
                add(name, t, f"returned value '{name}'", True)
        fd.returns = None

    class _Strip(ast.NodeTransformer):
        def visit_AnnAssign(self, node: ast.AnnAssign):
            if not isinstance(node.target, ast.Name):
                raise NotImplementedError("Only plain names can be annotated "
                                          f"(line {node.lineno})")
            name = node.target.id
            add(name, _eval(node.annotation, ns, f"'{name}'"), f"'{name}'", True)
            if node.value is None:
                return None
            return ast.copy_location(
                ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=node.value),
                node,
            )

        def generic_visit(self, node):
            super().generic_visit(node)
            # A block left empty by removing bare declarations still needs a statement
            if getattr(node, "body", None) == []:
                node.body.append(ast.Pass())
            return node

    fd.body = [_Strip().visit(s) for s in fd.body]
    fd.body = [s for s in fd.body if s is not None] or [ast.Pass()]
    ast.fix_missing_locations(tree)
    return tree, {k: t.attr() for k, t in found.items()}


def strip_helper_annotations(fd: ast.FunctionDef) -> ast.FunctionDef:
    """Helpers' parameter/return annotations are ignored; typed locals are rejected."""
    for a in fd.args.args:
        a.annotation = None
    fd.returns = None
    for node in ast.walk(fd):
        if isinstance(node, ast.AnnAssign):
            raise NotImplementedError(
                f"Helper '{fd.name}': annotate variables in the top function, "
                f"not in helpers (line {node.lineno})"
            )
    return fd


def normalize_attr(attr: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """attr values may be dicts, hardware types, a width (8: {'width': 8}) or a
    board's pin ('LED[7:0]': {'pin': 'LED[7:0]'}; an input without a width takes
    its pins', see pin_widths). A keyword's is a setting, no name's
    (attr['while'] = 24: as given)."""
    import keyword
    out = {}
    for k, v in (attr or {}).items():
        if isinstance(k, str) and keyword.iskeyword(k):
            out[k] = v
            continue
        if isinstance(v, HwType):
            v = v.attr()
        elif isinstance(v, int) and not isinstance(v, bool):
            v = {"width": v}
        elif isinstance(v, str):
            v = {"pin": v}
        out[k] = v
    return out


def pin_widths(attr: Dict[str, Dict[str, Any]], inputs) -> None:
    """In place: each input with a pin and no width, the pins' (an output's is
    its own, its pins checked against it)."""
    from .pins import pin_width
    for k in inputs:
        v = attr.get(k)
        if isinstance(v, dict) and "pin" in v and "width" not in v:
            attr[k] = dict(v, width=pin_width(v["pin"]))
