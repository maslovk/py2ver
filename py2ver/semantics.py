# semantics.py
"""
Make generated Verilog compute what Python computes.

Verilog sizes an expression to its widest operand (and the assigned signal),
and treats the whole expression as unsigned if any operand is unsigned. Python
integers are unbounded and signed. Left alone, that changes results for right
shifts, comparisons, division and anything mixing signed with unsigned values.

exact_ir() rewrites every assignment so that:
  - every operand is signed: an unsigned signal x is read through a view
    `x__s`, a signed wire one bit wider (signed signals are used as they are)
  - +, -, *, &, |, ^, ~ and << stay inline: their low bits never depend on
    high bits, so the surrounding width is always enough
  - operands whose high bits matter (division, right shift, comparisons,
    conditions, shift amounts) are first put in a wire of exactly the width
    their value needs (`<lhs>__t<k>`), and division results get one too, so
    those operators are no wider than necessary
  - division with both operands provably non-negative stays unsigned;
    otherwise Python's floor division (" // " from the visitor) becomes
    truncating division plus a correction
  - comparison / logical results used as numbers become 0/1 values
  - every expression is fully parenthesized

Each named signal still truncates the exact value to its declared width,
which is how py2ver already relates hardware to the Python reference.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from .ast_ir import Assignment, ModuleIR, Port, resolve_meta

log = logging.getLogger(__name__)

MAX_WIDTH = 256  # beyond this (e.g. huge variable left shifts) an assignment is left as is

_TOKEN = re.compile(
    r"\s*(?:"
    r"(?P<sized>\d+'[sS]?[bBoOdDhH][0-9a-fA-F_]+)"
    r"|(?P<num>\d+)"
    r"|(?P<sys>\$[A-Za-z_]\w*)"
    r"|(?P<id>[A-Za-z_]\w*)"
    r"|(?P<op>>>>|<<<|//|<=|>=|==|!=|&&|\|\||<<|>>|[-+*/%&|^~!<>?:()])"
    r")"
)

_BIN_PREC = {
    "||": 1, "&&": 2, "|": 3, "^": 4, "&": 5,
    "==": 6, "!=": 6,
    "<": 7, "<=": 7, ">": 7, ">=": 7,
    "<<": 8, ">>": 8, "<<<": 8, ">>>": 8,
    "+": 9, "-": 9,
    "*": 10, "/": 10, "//": 10, "%": 10,
}
_BOOL_BIN = {"||", "&&", "==", "!=", "<", "<=", ">", ">="}


class Unsupported(Exception):
    """Expression uses something this pass does not model (left unchanged)."""


@dataclass
class Node:
    kind: str                 # "id" | "num" | "un" | "bin" | "tern"
    op: str = ""
    name: str = ""
    value: int = 0
    kids: Tuple["Node", ...] = ()


# --------------------------------------------------------------------
# Parsing (Verilog operator precedence)
# --------------------------------------------------------------------

_TOKENS = re.compile(r"(?:" + _TOKEN.pattern + r")*\s*")
_KINDS = ("sized", "num", "sys", "id", "op")


def _tokenize(expr: str) -> List[Tuple[str, str]]:
    expr = expr.rstrip()
    if not _TOKENS.fullmatch(expr):                    # (every character a token's)
        pos = 0
        while pos < len(expr):
            m = _TOKEN.match(expr, pos)
            if not m or m.end() == pos:
                raise Unsupported(f"cannot tokenize at {expr[pos:pos + 20]!r}")
            pos = m.end()
    toks: List[Tuple[str, str]] = []
    for groups in _TOKEN.findall(expr):
        for kind, text in zip(_KINDS, groups):
            if text:
                if kind == "sys":
                    raise Unsupported(f"system function {text}")
                toks.append((kind, text))
                break
    return toks


def _sized_value(lit: str) -> int:
    size_s, rest = lit.split("'", 1)
    signed = rest[0] in "sS"
    if signed:
        rest = rest[1:]
    base = {"b": 2, "o": 8, "d": 10, "h": 16}[rest[0].lower()]
    width = int(size_s)
    v = int(rest[1:].replace("_", ""), base) & ((1 << width) - 1)
    if signed and width and v >> (width - 1):
        v -= 1 << width
    return v


def parse(expr: str) -> Node:
    toks = _tokenize(expr)
    pos = 0

    def peek() -> Optional[str]:
        return toks[pos][1] if pos < len(toks) else None

    def take(expected: Optional[str] = None) -> Tuple[str, str]:
        nonlocal pos
        if pos >= len(toks):
            raise Unsupported("unexpected end of expression")
        tok = toks[pos]
        if expected is not None and tok[1] != expected:
            raise Unsupported(f"expected {expected!r}, got {tok[1]!r}")
        pos += 1
        return tok

    def p_expr() -> Node:
        cond = p_bin(1)
        if peek() == "?":
            take("?")
            then = p_expr()
            take(":")
            other = p_expr()
            return Node("tern", kids=(cond, then, other))
        return cond

    def p_bin(min_prec: int) -> Node:
        left = p_unary()
        while True:
            op = peek()
            if pos >= len(toks) or toks[pos][0] != "op" or op not in _BIN_PREC:
                return left
            prec = _BIN_PREC[op]
            if prec < min_prec:
                return left
            take()
            right = p_bin(prec + 1)
            left = Node("bin", op=op, kids=(left, right))

    def p_unary() -> Node:
        if peek() in ("-", "+", "~", "!") and toks[pos][0] == "op":
            op = take()[1]
            return Node("un", op=op, kids=(p_unary(),))
        return p_primary()

    def p_primary() -> Node:
        kind, text = take()
        if text == "(":
            inner = p_expr()
            take(")")
            return inner
        if kind == "id":
            return Node("id", name=text)
        if kind == "num":
            return Node("num", value=int(text))
        if kind == "sized":
            return Node("num", value=_sized_value(text))
        raise Unsupported(f"unexpected token {text!r}")

    tree = p_expr()
    if pos != len(toks):
        raise Unsupported(f"trailing tokens from {toks[pos][1]!r}")
    return tree


# --------------------------------------------------------------------
# Width / sign analysis
# --------------------------------------------------------------------

@dataclass
class Info:
    bits: int          # signed two's-complement bits that hold every value
    nonneg: bool       # value is never negative
    boolean: bool = False


def _bits_for(v: int) -> int:
    return (v.bit_length() if v >= 0 else (-v - 1).bit_length()) + 1


def analyze(node: Node, meta) -> Info:
    """(Once per node: an expression's parts are asked about again and again.)"""
    info = node.__dict__.get("_info")
    if info is None:
        info = _analyze(node, meta)
        node.__dict__["_info"] = info
    return info


def _analyze(node: Node, meta) -> Info:
    k = node.kind
    if k == "id":
        m = meta(node.name)
        w, s = int(m.get("width", 1)), int(m.get("signed", 0)) == 1
        return Info(w if s else w + 1, not s)
    if k == "num":
        return Info(_bits_for(node.value), node.value >= 0)
    if k == "un":
        x = analyze(node.kids[0], meta)
        if node.op == "!":
            return Info(2, True, True)
        if node.op == "-":
            return Info(x.bits + 1, False)
        if node.op == "~":
            return Info(x.bits, False)
        return x
    if k == "tern":
        analyze(node.kids[0], meta)
        t, e = analyze(node.kids[1], meta), analyze(node.kids[2], meta)
        return Info(max(t.bits, e.bits), t.nonneg and e.nonneg)

    a, b = analyze(node.kids[0], meta), analyze(node.kids[1], meta)
    op = node.op
    if op in _BOOL_BIN:
        return Info(2, True, True)
    if op == "+":
        return Info(max(a.bits, b.bits) + 1, a.nonneg and b.nonneg)
    if op == "-":
        return Info(max(a.bits, b.bits) + 1, False)
    if op == "*":
        return Info(a.bits + b.bits, a.nonneg and b.nonneg)
    if op in ("/", "//"):
        return Info(a.bits + 1, a.nonneg and b.nonneg)
    if op == "%":
        # Python: |a % b| < |b|, and the result takes the sign of b
        return Info(b.bits, b.nonneg)
    if op in ("<<", "<<<"):
        rhs = node.kids[1]
        amount = rhs.value if rhs.kind == "num" else (1 << max(b.bits - 1, 0)) - 1
        return Info(a.bits + max(amount, 0), a.nonneg)
    if op in (">>", ">>>"):
        return Info(a.bits, a.nonneg)
    if op == "&":
        return Info(max(a.bits, b.bits), a.nonneg or b.nonneg)
    if op in ("|", "^"):
        return Info(max(a.bits, b.bits), a.nonneg and b.nonneg)
    raise Unsupported(f"operator {op!r}")


def max_bits(node: Node, meta) -> int:
    best = node.__dict__.get("_max_bits")
    if best is None:
        best = analyze(node, meta).bits
        for kid in node.kids:
            best = max(best, max_bits(kid, meta))
        node.__dict__["_max_bits"] = best
    return best


# --------------------------------------------------------------------
# Emission
# --------------------------------------------------------------------

def _literal(v: int) -> str:
    if 0 <= v < (1 << 31):
        return str(v)
    return f"{_bits_for(v)}'sd{v}" if v >= 0 else f"-{_bits_for(v)}'sd{-v}"


class _Emitter:
    """Emits one assignment; collects the exact-width temporaries it needs."""

    def __init__(self, lhs: str, meta):
        self.lhs = lhs
        self.meta = meta
        self.temps: List[Assignment] = []
        self.temp_attr: Dict[str, Dict[str, Any]] = {}
        self.views: Set[str] = set()

    def _temp(self, expr: str, bits: int, signed: bool = True) -> str:
        name = f"{self.lhs}__t{len(self.temps)}"
        self.temps.append(Assignment(name, expr, False))
        self.temp_attr[name] = {"width": max(bits, 1), "signed": int(signed), "type": "wire"}
        return name

    def _signed_name(self, name: str) -> str:
        if int(self.meta(name).get("signed", 0)) == 1:
            return name
        self.views.add(name)
        return f"{name}__s"

    # --- operands whose exact value matters ---
    def exact(self, node: Node) -> str:
        """A signed name or literal holding the node's exact value."""
        if node.kind == "id":
            return self._signed_name(node.name)
        if node.kind == "num":
            return _literal(node.value)
        return self._temp(self.value(node), analyze(node, self.meta).bits)

    def exact_unsigned(self, node: Node) -> str:
        """Unsigned name or literal for a provably non-negative node."""
        if node.kind == "id" and int(self.meta(node.name).get("signed", 0)) == 0:
            return node.name
        if node.kind == "num":
            return str(node.value) if node.value < (1 << 31) else (
                f"{node.value.bit_length()}'d{node.value}")
        bits = analyze(node, self.meta).bits - 1
        return self._temp(self.value(node), bits, signed=False)

    def truthy(self, node: Node) -> str:
        if analyze(node, self.meta).boolean:
            return self.cond(node)
        return self.exact(node)

    # --- expressions ---
    def cond(self, node: Node) -> str:
        """Boolean-valued node as a Verilog condition."""
        if node.kind == "un" and node.op == "!":
            return f"!({self.truthy(node.kids[0])})"
        a, b = node.kids
        if node.op in ("&&", "||"):
            return f"({self.truthy(a)}) {node.op} ({self.truthy(b)})"
        return f"({self.exact(a)}) {node.op} ({self.exact(b)})"

    def value(self, node: Node) -> str:
        """Signed expression, correct in any context at least as wide as its consumer."""
        k = node.kind
        if k == "id":
            return self._signed_name(node.name)
        if k == "num":
            return _literal(node.value)
        info = analyze(node, self.meta)
        if info.boolean:
            return f"(({self.cond(node)}) ? 1 : 0)"
        if k == "un":
            return f"{node.op}({self.value(node.kids[0])})"
        if k == "tern":
            c, t, e = node.kids
            return f"(({self.truthy(c)}) ? ({self.value(t)}) : ({self.value(e)}))"

        op = node.op
        a, b = node.kids
        if op in ("+", "-", "*", "&", "|", "^"):
            return f"(({self.value(a)}) {op} ({self.value(b)}))"
        if op in ("<<", "<<<"):
            return f"(({self.value(a)}) << ({self.exact(b)}))"
        if op in (">>", ">>>"):
            return f"(({self.exact(a)}) >>> ({self.exact(b)}))"
        if op in ("/", "//", "%"):
            return self._divide(node)
        raise Unsupported(f"operator {op!r}")

    def _divide(self, node: Node) -> str:
        op = node.op
        a, b = node.kids
        ia, ib = analyze(a, self.meta), analyze(b, self.meta)
        if ia.nonneg and ib.nonneg:
            # Unsigned divider, no wider than the operands
            ua, ub = self.exact_unsigned(a), self.exact_unsigned(b)
            vop = "%" if op == "%" else "/"
            q = self._temp(f"({ua}) {vop} ({ub})", ia.bits - 1, signed=False)
            return self._signed_name_of_temp(q)
        sa, sb = self.exact(a), self.exact(b)
        # Verilog truncates toward zero; Python floors. They differ exactly
        # when the remainder is nonzero and the operand signs differ.
        fix = f"((({sa}) % ({sb})) != 0) && ((({sa}) < 0) != (({sb}) < 0))"
        if op == "//":
            expr = f"(({sa}) / ({sb})) - (({fix}) ? 1 : 0)"
        elif op == "%":
            # Python's remainder takes the divisor's sign: add b when they differ
            expr = f"(({sa}) % ({sb})) + (({fix}) ? ({sb}) : 0)"
        else:
            expr = f"({sa}) / ({sb})"
        return self._temp(expr, max(ia.bits, ib.bits) + 1)

    def _signed_name_of_temp(self, name: str) -> str:
        self.views.add(name)
        return f"{name}__s"


# --------------------------------------------------------------------
# Module pass
# --------------------------------------------------------------------

def exact_ir(ir: ModuleIR, attr: Dict[str, Dict[str, Any]]) -> Tuple[ModuleIR, Dict[str, Dict[str, Any]]]:
    attr = {k: dict(v) for k, v in attr.items()}
    ports: Dict[str, Port] = {p.name: p for p in ir.inputs + ir.outputs}

    def meta(name: str) -> Dict[str, Any]:
        return resolve_meta(name, attr, ports)

    trees: List[Optional[Node]] = []
    for a in ir.assigns:
        try:
            tree = parse(a.right)
            analyze(tree, meta)
            if max_bits(tree, meta) > MAX_WIDTH:
                raise Unsupported(f"needs more than {MAX_WIDTH} bits")
        except Unsupported as e:
            log.warning("Leaving '%s = %s' as is: %s", a.left, a.right, e)
            tree = None
        trees.append(tree)

    views: Set[str] = set()
    assigns: List[Assignment] = []
    for a, tree in zip(ir.assigns, trees):
        if tree is None:
            assigns.append(Assignment(a.left, a.right.replace(" // ", " / "), a.is_reg))
            continue
        em = _Emitter(a.left, meta)
        rhs = em.value(tree)
        attr.update(em.temp_attr)
        assigns.extend(em.temps)
        assigns.append(Assignment(a.left, rhs, a.is_reg))
        views |= em.views

    view_assigns = []
    for name in sorted(views):
        m = meta(name)
        attr[f"{name}__s"] = {"width": int(m.get("width", 1)) + 1, "signed": 1, "type": "wire"}
        view_assigns.append(Assignment(f"{name}__s", name, False))

    new_ir = ModuleIR(
        name=ir.name,
        inputs=list(ir.inputs),
        outputs=list(ir.outputs),
        assigns=view_assigns + assigns,
        has_clk=ir.has_clk,
        items=list(ir.items),
    )
    return new_ir, attr
