import importlib.util
import itertools

import pytest

from semantics import Unsupported, analyze, parse

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings, strategies as st  # noqa: E402


S8 = {"signed": 1, "width": 8}
U8 = {"signed": 0, "width": 8}


def mixed_sign_shift(r, a):
    y = (r >> 2) ^ a
    return y


def floor_div(a, b):
    y = a // b
    return y


def true_div(a, b):
    y = a / b
    return y


def neg_sum(a, b):
    y = -(a + b)
    return y


def invert_sum(a, b):
    y = ~(a + b)
    return y


def and_eq(a, b, c):
    y = a & b == c
    return y


def wide_shift(a, b):
    y = (a * b) >> 8
    return y


def signed_cmp(a, b):
    if a < b:
        y = a
    else:
        y = b
    return y


def cmp_as_number(a, b, c):
    y = (a > b) + c
    return y


# ----------------------------- regressions (simulated) -----------------------------

@pytest.mark.sim
@pytest.mark.parametrize(
    "func, attr, vectors",
    [
        # signed r with unsigned a: Verilog made the whole expression unsigned
        (mixed_sign_shift, {"r": S8, "a": U8, "y": {"signed": 1, "width": 16}},
         [(-44, 217), (-1, 3), (127, 0), (-128, 255)]),
        # Python // floors; Verilog / truncates
        (floor_div, {"a": S8, "b": S8, "y": S8},
         [(-7, 2), (7, -2), (-7, -2), (6, -3), (-128, 3), (0, -5)]),
        # Python / then int() truncates toward zero, like Verilog /
        (true_div, {"a": S8, "b": S8, "y": S8}, [(-7, 2), (7, -2), (-7, -2), (100, 7)]),
        # -(a + b) was emitted as -(a) + (b)
        (neg_sum, {"a": S8, "b": S8, "y": {"signed": 1, "width": 10}}, [(3, 4), (-5, 1), (-128, -128)]),
        (invert_sum, {"a": U8, "b": U8, "y": {"signed": 1, "width": 10}}, [(3, 4), (255, 255)]),
        # a & b == c is (a & b) == c in Python, a & (b == c) in Verilog
        (and_eq, {"a": U8, "b": U8, "c": U8, "y": U8}, [(6, 3, 2), (12, 10, 8), (1, 1, 0)]),
        # a * b was truncated to the assignment width before the shift
        (wide_shift, {"a": U8, "b": U8, "y": U8}, [(200, 100), (255, 255), (16, 16)]),
        # signed vs unsigned compare was done unsigned
        (signed_cmp, {"a": S8, "b": U8, "y": {"signed": 1, "width": 9}}, [(-3, 5), (4, 200), (-128, 0)]),
        # comparison result used as a number
        (cmp_as_number, {"a": S8, "b": S8, "c": S8, "y": {"signed": 1, "width": 9}},
         [(1, 0, -5), (0, 1, -5), (-1, -2, 127)]),
    ],
)
def test_python_semantics_regressions(assert_python_matches_sim, func, attr, vectors):
    assert_python_matches_sim(func, attr, vectors)


# ----------------------------- parser / analysis -----------------------------

def test_parser_uses_verilog_precedence():
    tree = parse("a & b == c")
    assert (tree.op, tree.kids[1].op) == ("&", "==")

    tree = parse("((a) + (b) > (c) * (d)) ? (x) : (y)")
    assert tree.kind == "tern" and tree.kids[0].op == ">"

    tree = parse("-(a) + (b)")
    assert tree.op == "+" and tree.kids[0].kind == "un"


def test_parser_sized_literals_and_unsupported():
    assert parse("16'sd0").value == 0
    assert parse("4'sb1111").value == -1
    assert parse("1'b1").value == 1
    with pytest.raises(Unsupported):
        parse("$sqrt((x)*(x))")


def test_width_analysis():
    meta = {"a": U8, "b": S8}.__getitem__
    assert analyze(parse("(a) * (a)"), meta).bits == 18
    assert analyze(parse("(a) + (b)"), meta).bits == 10
    assert analyze(parse("(a) << (26)"), meta).bits == 35
    assert analyze(parse("(a) >>> (3)"), meta).nonneg is True
    assert analyze(parse("(a) - (a)"), meta).nonneg is False
    assert analyze(parse("(b) & (255)"), meta).nonneg is True


# ----------------------------- random differential (simulated) -----------------------------

_LEAVES = ["a", "b", "c"]


def _expr():
    leaf = st.one_of(st.sampled_from(_LEAVES), st.integers(0, 20).map(str))

    def extend(sub):
        binop = st.tuples(sub, st.sampled_from(["+", "-", "*", "&", "|", "^"]), sub).map(
            lambda t: f"({t[0]} {t[1]} {t[2]})"
        )
        div = st.tuples(sub, sub).map(lambda t: f"({t[0]} // ({t[1]} | 1))")
        shift = st.tuples(sub, st.sampled_from([">>", "<<"]), st.integers(0, 5)).map(
            lambda t: f"({t[0]} {t[1]} {t[2]})"
        )
        unary = st.tuples(st.sampled_from(["-", "~"]), sub).map(lambda t: f"{t[0]}({t[1]})")
        mod = st.tuples(sub, sub).map(lambda t: f"({t[0]} % ({t[1]} | 1))")
        call = st.one_of(
            sub.map(lambda x: f"abs({x})"),
            st.tuples(st.sampled_from(["min", "max"]), sub, sub).map(
                lambda t: f"{t[0]}({t[1]}, {t[2]})"
            ),
        )
        ifexp = st.tuples(sub, st.sampled_from(["<", ">=", "=="]), sub, sub, sub).map(
            lambda t: f"({t[3]} if {t[0]} {t[1]} {t[2]} else {t[4]})"
        )
        return st.one_of(binop, div, shift, unary, mod, call, ifexp)

    return st.recursive(leaf, extend, max_leaves=6)


@st.composite
def _designs(draw):
    attr = {}
    for name in _LEAVES:
        attr[name] = {"signed": draw(st.integers(0, 1)), "width": draw(st.integers(1, 8))}
    attr["y"] = {"signed": 1, "width": 24}
    cond = f"{draw(_expr())} {draw(st.sampled_from(['<', '<=', '>', '>=', '==', '!=']))} {draw(_expr())}"
    then, other, extra = draw(_expr()), draw(_expr()), draw(_expr())
    aug = draw(st.sampled_from(["+=", "-=", "^=", "*="]))
    no_else = draw(st.booleans())

    def values(m):
        w = m["width"]
        lo, hi = (-(1 << (w - 1)), (1 << (w - 1)) - 1) if m["signed"] else (0, (1 << w) - 1)
        return st.integers(lo, hi)

    vectors = draw(st.lists(st.tuples(*[values(attr[n]) for n in _LEAVES]), min_size=4, max_size=16))
    if no_else:
        # if without else keeps y's earlier value; y is updated in place
        body = (
            f"    y = {other}\n"
            f"    if {cond}:\n"
            f"        y {aug} {then}\n"
        )
    else:
        body = (
            f"    if {cond}:\n"
            f"        y = {then}\n"
            "    else:\n"
            f"        y = {other}\n"
        )
    src = "def rand_expr(a, b, c):\n" + body + f"    y {aug} {extra}\n" + "    return y\n"
    return src, attr, vectors


_counter = itertools.count()


@pytest.mark.sim
@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(design=_designs())
def test_random_expressions_match_python(assert_python_matches_sim, tmp_path_factory, design):
    src, attr, vectors = design
    # Py2ver reads the function's source, so it must live in a real module file
    path = tmp_path_factory.mktemp("rand") / f"rand_expr_{next(_counter)}.py"
    path.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert_python_matches_sim(mod.rand_expr, attr, vectors)
