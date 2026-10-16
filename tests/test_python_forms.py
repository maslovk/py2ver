import random

import pytest

from py2ver import Py2ver

S8 = {"signed": 1, "width": 8}
U8 = {"signed": 0, "width": 8}
S16 = {"signed": 1, "width": 16}

MASK = 0xFF
GAIN = 3


# ----------------------------- designs -----------------------------

def reassign_from_self(a, b):
    y = a
    y = y + b
    y = y * 2
    return y


def augmented(a, b):
    y = a
    y += b
    y *= 3
    y -= a
    y >>= 1
    y ^= b
    return y


def tuple_swap(a, b):
    x, y = a, b
    x, y = y, x - y
    return x, y


def if_without_else(a, b):
    y = a
    if b > 0:
        y = y + b
    return y


def branch_sequential(a, b):
    if a > b:
        t = a - b
        y = t * 2
    else:
        t = b - a
        y = t + 1
    return y


def conditional_expr(a, b):
    y = a if a > b else b - 1
    return y


def modulo(a, b):
    y = a % (b | 1)
    return y


def builtins_minmax_abs(a, b, c):
    lo = min(a, b, c)
    hi = max(a, b)
    m = abs(c)
    return lo, hi, m


def undeclared_accumulator(x):
    acc = 0
    for i in range(1, 6):
        acc += x * i
    return acc


def return_expression(a, b):
    return a * b - 1


def _mac(acc, x, k):
    p = x * k
    return acc + p


def _clamp(v, lo, hi):
    return max(lo, min(v, hi))


def _divmod(a, b):
    return a // b, a % b


def helpers(a, b, c):
    y = _mac(a, b, GAIN) + 1
    y += _clamp(c, -10, 10)
    q, r = _divmod(a, b | 1)
    w = 0
    for i in range(3):
        w += _mac(w, i, c) % 7
    return y & MASK, q, r, w


def _fact(n):
    return n * _fact(n - 1)


def recursive(a):
    y = _fact(a)
    return y


def _early(a):
    if a > 0:
        return a
    return -a


def early_return(a):
    y = _early(a)
    return y


def only_some_branches(a, b):
    if a > b:
        y = a
    return y


def returns_argument(a, b):
    a = a + b
    return a, b


def returns_argument_unchanged(a, b):
    y = a - b
    return a, y


# ----------------------------- simulation vs Python -----------------------------

def _vectors(attrs, n=24, seed=0):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        row = []
        for m in attrs:
            w = m["width"]
            lo, hi = (-(1 << (w - 1)), (1 << (w - 1)) - 1) if m["signed"] else (0, (1 << w) - 1)
            row.append(rng.choice([lo, hi, 0, -1 if m["signed"] else 1, rng.randint(lo, hi)]))
        out.append(tuple(row))
    return out


@pytest.mark.sim
@pytest.mark.parametrize(
    "func, attr, ins",
    [
        (reassign_from_self, {"a": S8, "b": S8, "y": {"signed": 1, "width": 12}}, [S8, S8]),
        (augmented, {"a": S8, "b": S8}, [S8, S8]),
        (tuple_swap, {"a": S8, "b": S8}, [S8, S8]),
        (if_without_else, {"a": S8, "b": S8}, [S8, S8]),
        (branch_sequential, {"a": S8, "b": U8}, [S8, U8]),
        (conditional_expr, {"a": S8, "b": U8}, [S8, U8]),
        (modulo, {"a": S8, "b": S8}, [S8, S8]),
        (builtins_minmax_abs, {"a": S8, "b": U8, "c": S8}, [S8, U8, S8]),
        (undeclared_accumulator, {"x": S8}, [S8]),
        (return_expression, {"a": S8, "b": U8}, [S8, U8]),
        (helpers, {"a": S8, "b": S8, "c": {"signed": 1, "width": 6}},
         [S8, S8, {"signed": 1, "width": 6}]),
        (returns_argument, {"a": {"signed": 1, "width": 10}, "b": S8},
         [{"signed": 1, "width": 10}, S8]),
        (returns_argument_unchanged, {"a": S8, "b": S8}, [S8, S8]),
    ],
)
def test_python_forms_match_python(assert_python_matches_sim, func, attr, ins):
    assert_python_matches_sim(func, attr, _vectors(ins, seed=hash(func.__name__) % 1000))


# ----------------------------- generation details -----------------------------

def test_undeclared_temporaries_get_exact_widths(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    p = Py2ver(undeclared_accumulator, {"x": S8})

    # acc ends at x * 15: needs 12 signed bits for x in [-128, 127]
    assert p.attr["acc"]["signed"] == 1
    assert p.attr["acc"]["width"] >= 12


def test_helper_locals_are_prefixed_and_constants_folded(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    p = Py2ver(helpers, {"a": S8, "b": S8, "c": {"signed": 1, "width": 6}})

    lefts = {a.left for a in p.ir.assigns}
    assert any(name.startswith("___mac0_") for name in lefts)
    assert not any("GAIN" in a.right or "MASK" in a.right for a in p.ir.assigns)


@pytest.mark.sim
def test_early_return_in_a_helper_and_one_branch_values(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    attr = {"a": S8, "b": S8}
    p = Py2ver(early_return, {"a": S8})
    assert p.TB_batch()([(5,), (-7,), (0,)]) == [(5,), (7,), (0,)]
    # y exists only after the branch that sets it: Python reads it only then
    p = Py2ver(only_some_branches, attr)
    assert p.TB_batch()([(9, 2), (40, -3)]) == [(9,), (40,)]


@pytest.mark.parametrize("func, match", [(recursive, "Recursive call")])
def test_unsupported_forms_raise_clearly(monkeypatch, tmp_path, func, match):
    monkeypatch.chdir(tmp_path)
    attr = {"a": S8, "b": S8, "n": S8}
    with pytest.raises(NotImplementedError, match=match):
        Py2ver(func, attr)
