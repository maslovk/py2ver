"""
Verilog expressions as people write them: the translator's passes write every
operand in parentheses (`((a) + ((3) * (b)))`), which keeps them simple; here
an expression is parsed and printed again with only the parentheses that
change its meaning or make it clearer (`a + 3 * b`).

Parentheses never change a Verilog expression's width or signedness, only its
grouping, so the printed expression is the same hardware. Every printed
expression is parsed again and must give the same tree; anything this parser
doesn't know (or a tree that doesn't come back the same) is left as it was.
"""

from __future__ import annotations

import re
from typing import Any, List, Tuple

_TOKEN = re.compile(r"""
    \s*(?:
      (?P<num>\d*'[sS]?[bBoOdDhH][0-9a-fA-FxXzZ_?]+|\d[\d_]*(?:\.\d+)?(?:[eE][-+]?\d+)?)
    | (?P<id>\$?[A-Za-z_][A-Za-z0-9_$]*)
    | (?P<op><<<|>>>|===|!==|\*\*|<<|>>|<=|>=|==|!=|&&|\|\||~&|~\||~\^|\^~|\+:|-:
             |[-+*/%<>!~&|^?:,()\[\]{}])
    )""", re.VERBOSE)

# Binary operators, tighter binding first (IEEE 1364 table 5-4)
_BINARY = {"**": 12, "*": 11, "/": 11, "%": 11, "+": 10, "-": 10,
           "<<": 9, ">>": 9, "<<<": 9, ">>>": 9, "<": 8, "<=": 8, ">": 8, ">=": 8,
           "==": 7, "!=": 7, "===": 7, "!==": 7, "&": 6, "^": 5, "~^": 5, "^~": 5,
           "|": 4, "&&": 3, "||": 2}
_UNARY = {"+", "-", "!", "~", "&", "|", "^", "~&", "~|", "~^", "^~"}
_GROUP = {12: "pow", 11: "arith", 10: "arith", 9: "shift", 8: "compare", 7: "compare",
          6: "bit", 5: "bit", 4: "bit", 3: "logic", 2: "logic"}
# A tighter child needs no parentheses under these parents (a + b * c, a + 1 < b,
# a < b && c == d); elsewhere they stay for the reader (a & (b << 2), (a & b) | c)
_CLEAR = {("arith", "arith"), ("compare", "arith"), ("logic", "compare"), ("logic", "arith")}

WIDTH = 100                     # longer lines are wrapped


class Unparsed(Exception):
    pass


def tokens(text: str) -> List[str]:
    out, pos = [], 0
    text = text.rstrip()
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m or m.end() == pos:
            raise Unparsed(text[pos:pos + 20])
        out.append(m.group(m.lastgroup))
        pos = m.end()
    return out


class _Parser:
    """Trees: ('atom', text), ('sel', base, [parts], sep), ('call', name, [args]),
    ('cat', [items]), ('rep', count, cat), ('un', op, x), ('bin', op, l, r),
    ('tern', c, a, b)."""

    def __init__(self, toks: List[str]):
        self.toks, self.i = toks, 0

    def peek(self) -> str | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self, want: str | None = None) -> str:
        t = self.peek()
        if t is None or (want is not None and t != want):
            raise Unparsed(f"expected {want!r}, got {t!r}")
        self.i += 1
        return t

    def expr(self) -> Any:
        c = self.binary(1)
        if self.peek() == "?":
            self.take()
            a = self.expr()
            self.take(":")
            return ("tern", c, a, self.expr())
        return c

    def binary(self, level: int) -> Any:
        left = self.unary()
        while True:
            op = self.peek()
            p = _BINARY.get(op)
            if p is None or p < level:
                return left
            self.take()
            left = ("bin", op, left, self.binary(p + 1))      # left-associative

    def unary(self) -> Any:
        t = self.peek()
        if t in _UNARY:
            self.take()
            return ("un", t, self.unary())
        return self.postfix(self.primary())

    def primary(self) -> Any:
        t = self.take()
        if t == "(":
            e = self.expr()
            self.take(")")
            return e
        if t == "{":
            first = self.expr()
            if self.peek() == "{":                       # {n{x}}
                self.take()
                inner = self.items("}")
                self.take("}")
                return ("rep", first, ("cat", inner))
            rest = [first]
            while self.peek() == ",":
                self.take()
                rest.append(self.expr())
            self.take("}")
            return ("cat", rest)
        if t[0] == "$" and self.peek() == "(":
            self.take()
            args = self.items(")") if self.peek() != ")" else []
            self.take(")")
            return ("call", t, args)
        if t[0].isalnum() or t[0] in "_$'":
            return ("atom", t)
        raise Unparsed(t)

    def items(self, end: str) -> List[Any]:
        out = [self.expr()]
        while self.peek() == ",":
            self.take()
            out.append(self.expr())
        return out

    def postfix(self, node: Any) -> Any:
        while self.peek() == "[":
            self.take()
            parts, sep = [self.expr()], None
            if self.peek() in (":", "+:", "-:"):
                sep = self.take()
                parts.append(self.expr())
            self.take("]")
            node = ("sel", node, parts, sep)
        return node


def parse(text: str) -> Any:
    p = _Parser(tokens(text))
    tree = p.expr()
    if p.peek() is not None:
        raise Unparsed(f"trailing {p.peek()!r}")
    return tree


def _plain(node: Any) -> bool:
    return node[0] in ("atom", "sel", "call", "cat", "rep")


def _paren_bin(child: Any, op: str, right: bool) -> bool:
    """Does a child of binary operator op need parentheses?"""
    kind = child[0]
    if _plain(child):
        return False
    if kind == "tern":
        return True
    if kind == "un":
        # (a negation, anything under ** and a unary operator on the right read
        #  better grouped: a - (-b), (-3) * a, a & (|b); ~ and ! stay: a & ~b)
        if op == "**" or child[1] in ("-", "+"):
            return True
        return right and child[1] not in ("~", "!")
    cp, pp = _BINARY[child[1]], _BINARY[op]
    if cp < pp or (cp == pp and right) or op == "**":
        return True
    if cp == pp:                                        # (a + b) - c: left-associative
        return _GROUP[cp] not in ("arith", "logic") and child[1] != op
    return (_GROUP[pp], _GROUP[cp]) not in _CLEAR


_shown: dict = {}       # id(node) -> show(node), during one tidy()


def show(node: Any) -> str:
    """The expression on one line."""
    key = id(node)
    if key not in _shown:
        _shown[key] = (node, _show(node))       # (the node kept: its id stays its own)
    return _shown[key][1]


def _show(node: Any) -> str:
    kind = node[0]
    if kind == "atom":
        return node[1]
    if kind == "sel":
        inner = show(node[2][0])
        if node[3]:
            inner += (":" if node[3] == ":" else f" {node[3]} ") + show(node[2][1])
        return f"{_wrap_base(node[1])}[{inner}]"
    if kind == "call":
        return f"{node[1]}({', '.join(show(a) for a in node[2])})"
    if kind == "cat":
        return "{" + ", ".join(show(a) for a in node[1]) + "}"
    if kind == "rep":
        return "{" + show(node[1]) + show(node[2]) + "}"
    if kind == "un":
        x = node[2]
        inner = show(x) if _plain(x) else f"({show(x)})"
        return node[1] + inner
    if kind == "bin":
        op, l, r = node[1], node[2], node[3]
        ls = f"({show(l)})" if _paren_bin(l, op, False) else show(l)
        rs = f"({show(r)})" if _paren_bin(r, op, True) else show(r)
        return f"{ls} {op} {rs}"
    c, a, b = node[1], node[2], node[3]
    cs = f"({show(c)})" if c[0] == "tern" else show(c)
    as_ = f"({show(a)})" if a[0] == "tern" else show(a)
    return f"{cs} ? {as_} : {show(b)}"


def _wrap_base(node: Any) -> str:
    return show(node) if _plain(node) else f"({show(node)})"


def _col(text: str, start: int) -> int:
    """The column after text, printed from column start."""
    nl = text.rfind("\n")
    return start + len(text) if nl < 0 else len(text) - nl - 1


def wrapped(node: Any, col: int, width: int = WIDTH) -> str:
    """The expression from column col, broken at its top-level operators where
    a line would pass width: a selection's branches one per line under it,
    a chain of a binary operator one operand per line."""
    flat = show(node)
    if col + len(flat) <= width or len(flat) < 40:
        return flat
    kind = node[0]
    if kind == "tern":
        # c1 ? a1          a chain of selections one per line; a branch that
        # : c2 ? a2        doesn't fit after its condition goes under it:
        # : e              c \n  ? a
        chain, rest = [], node
        while rest[0] == "tern":
            chain.append((rest[1], rest[2]))
            rest = rest[3]
        lines = []
        for i, (c, a) in enumerate(chain):
            start = col + (2 if i else 0)
            cs = f"({wrapped(c, start + 1, width)})" if c[0] == "tern" else wrapped(c, start, width)
            branch = f"({show(a)})" if a[0] == "tern" else show(a)
            if "\n" not in cs and _col(cs, start) + 3 + len(branch) <= width:
                line = f"{cs} ? {branch}"
            else:
                at = start + 4
                branch = (f"({wrapped(a, at + 1, width)})" if a[0] == "tern"
                          else wrapped(a, at, width))
                line = f"{cs}\n{' ' * (start + 2)}? {branch}"
            lines.append(line if i == 0 else ": " + line)
        lines.append(": " + wrapped(rest, col + 2, width))
        return ("\n" + " " * col).join(lines)
    if kind == "bin":
        op, l, r = node[1], node[2], node[3]
        ls = (f"({wrapped(l, col + 1, width)})" if _paren_bin(l, op, False)
              else wrapped(l, col, width))
        at = col + len(op) + 1
        rs = (f"({wrapped(r, at + 1, width)})" if _paren_bin(r, op, True)
              else wrapped(r, at, width))
        return ls + "\n" + " " * col + op + " " + rs
    if kind == "un" and not _plain(node[2]):
        return node[1] + "(" + wrapped(node[2], col + len(node[1]) + 1, width) + ")"
    return flat


def tidy(text: str, col: int = 0) -> str:
    """text, an expression, with only the parentheses it needs (and broken over
    lines from column col if long); as it was if it can't be parsed."""
    try:
        tree = parse(text)
        out = wrapped(tree, col)
        return out if parse(out) == tree else text
    except (Unparsed, RecursionError):
        return text
    finally:
        _shown.clear()


def text(node: Any) -> str:
    """A parsed expression on one line."""
    try:
        return show(node)
    finally:
        _shown.clear()


def tree_or_none(text: str) -> Any:
    try:
        return parse(text)
    except (Unparsed, RecursionError):
        return None


def literal(node: Any) -> Tuple[int, bool, int] | None:
    """(width, signed, value) of a number; None if it isn't a plain one."""
    if node[0] != "atom":
        return None
    t = node[1].replace("_", "")
    m = re.fullmatch(r"(\d*)'([sS]?)([bBoOdDhH])([0-9a-fA-F]+)", t)
    if m:
        base = {"b": 2, "o": 8, "d": 10, "h": 16}[m.group(3).lower()]
        return int(m.group(1) or 32), bool(m.group(2)), int(m.group(4), base)
    if t.isdigit():
        return 32, True, int(t)
    return None
