import random

import pytest
from hypothesis import given, settings, strategies as st

from py2ver import vprint
from py2ver.vprint import parse, tidy, wrapped

from icarus import assert_same_bits


@pytest.mark.parametrize("text, expected", [
    ("((arg1__s) + (arg2__s))", "arg1__s + arg2__s"),
    ("((acc__v2) + (((-(5)) * (window_1))))", "acc__v2 + (-5) * window_1"),
    ("rst ? 16'sd0 : (ce ? y : $signed(y_prev))", "rst ? 16'sd0 : ce ? y : $signed(y_prev)"),
    ("((((((y) << (21))) + (((209715) * (((((x) << (4))) - (y))))))) + (1048576))",
     "(y << 21) + 209715 * ((x << 4) - y) + 1048576"),
    ("((a) - ((b) - (c)))", "a - (b - c)"),
    ("(((a) - (b)) - (c))", "a - b - c"),
    ("((a) & ((b) << (2)))", "a & (b << 2)"),
    ("(((a) & (b)) | ((c) ^ (d)))", "(a & b) | (c ^ d)"),
    ("((((a) < (b))) && (((c) == (d))))", "a < b && c == d"),
    ("(((a) && (b)) || (c))", "(a && b) || c"),
    ("(ce && (__phase == 3'd7))", "ce && __phase == 3'd7"),
    ("~(a) & !(b)", "~a & !b"),
    ("-(-(a))", "-(-a)"),
    ("{{4{x[7]}}, x[7:0]}", "{{4{x[7]}}, x[7:0]}"),
    ("$signed({1'b0, (a)[3 +: 2]})", "$signed({1'b0, a[3 +: 2]})"),
    ("((c) ? ((d) ? (x) : (y)) : (z))", "c ? (d ? x : y) : z"),
    ("(((a) ** (2)) * (b))", "(a ** 2) * b"),
    ("((table1((i))) * (w))", "table1(i) * w"),
])
def test_tidy(text, expected):
    assert tidy(text) == expected


def test_unparsable_text_is_kept():
    for text in ("a + * b", "x[", "fn(a", "a /* c */ + b"):
        assert tidy(text) == text


def test_long_selection_chain_one_per_line():
    text = "((s) == (0)) ? (a) : (((s) == (1)) ? (b) : (((s) == (2)) ? (c) : (d)))"
    out = tidy(text.replace("a", "alpha_" * 6).replace("b", "beta_" * 6), col=10)
    assert all(len(line) <= vprint.WIDTH for line in out.splitlines())
    assert parse(out) == parse(text.replace("a", "alpha_" * 6).replace("b", "beta_" * 6))


# ---------------------------------------------------------------------------
# Random expressions: printed with every parenthesis, then tidied
# ---------------------------------------------------------------------------

_INPUTS = {"a": (8, True), "b": (8, False), "c": (5, True), "d": (3, False)}
# (weighted to the operators whose parentheses the printer drops: arithmetic,
#  comparisons and logic; the rest keep theirs)
_OPS = (["+", "-", "*", "/", "%"] * 6 + ["<", "<=", ">", ">=", "==", "!="] * 2 + ["&&", "||"] * 3
        + [op for op in vprint._BINARY if op != "**"])


def _full(node):
    """node with every operand in parentheses, as the translator writes it."""
    kind = node[0]
    if kind == "atom":
        return node[1]
    if kind == "sel":
        inner = _full(node[2][0]) + (f" {node[3]} {_full(node[2][1])}" if node[3] else "")
        return f"{_full(node[1])}[{inner}]"
    if kind == "call":
        return f"{node[1]}({', '.join(_full(a) for a in node[2])})"
    if kind == "cat":
        return "{" + ", ".join(f"({_full(a)})" for a in node[1]) + "}"
    if kind == "un":
        return f"{node[1]}({_full(node[2])})"
    if kind == "bin":
        return f"(({_full(node[2])}) {node[1]} ({_full(node[3])}))"
    return f"(({_full(node[1])}) ? ({_full(node[2])}) : ({_full(node[3])}))"


def _tree(rng, depth):
    if depth == 0 or rng.random() < 0.2:
        r = rng.random()
        if r < 0.6:
            return ("atom", rng.choice(sorted(_INPUTS)))
        if r < 0.8:
            return ("atom", str(rng.randrange(0, 20)))
        if r < 0.9:
            return ("atom", rng.choice(["4'sd5", "3'd6", "1'b1", "8'shF0"]))
        return ("sel", ("atom", "a"), [("atom", "5"), ("atom", "1")], ":")
    r = rng.random()
    if r < 0.6:
        return ("bin", rng.choice(_OPS), _tree(rng, depth - 1), _tree(rng, depth - 1))
    if r < 0.75:
        return ("un", rng.choice(["-", "~", "!", "&", "|", "^"]), _tree(rng, depth - 1))
    if r < 0.9:
        return ("tern", _tree(rng, depth - 1), _tree(rng, depth - 1), _tree(rng, depth - 1))
    if r < 0.95:
        return ("call", "$signed", [_tree(rng, depth - 1)])
    return ("cat", [("atom", rng.choice(sorted(_INPUTS))), ("atom", "b")])   # (sized operands)


@settings(max_examples=300, deadline=None)
@given(st.integers(0, 2 ** 32), st.integers(0, 60), st.integers(30, 120))
def test_printed_expression_parses_back_the_same(seed, col, width):
    tree = parse(_full(_tree(random.Random(seed), 5)))
    assert parse(wrapped(tree, col, width)) == tree
    vprint._shown.clear()


@pytest.mark.sim
def test_tidied_expressions_compute_the_same_in_icarus(tmp_path):
    """Icarus computes every tidied expression as the original: the printer's
    precedence is Verilog's."""
    rng = random.Random(1)
    pairs = []
    while len(pairs) < 300:
        full = _full(_tree(rng, 4))
        if tidy(full) != full or "(" not in full:
            pairs.append((32, False, full, tidy(full)))
    assert_same_bits(tmp_path, _INPUTS, pairs, samples=64)
