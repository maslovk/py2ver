import contextlib
import io
import random

import pytest

from py2ver import Py2ver

# Dicts with constant keys and lists of lists: rewritten into variables and
# flat lists before translation, with Python's results.

I16 = {"signed": 1, "width": 16}
U2 = {"signed": 0, "width": 2}
COEF = {"lo": 3, "hi": -2}
LUT = {0: 5, 1: 7, 3: -4}
_rng = random.Random(13)


def gains(x, y, sel):
    g = {"lo": x * COEF["lo"], "hi": y + COEF["hi"]}
    g["mid"] = g["lo"] - g["hi"]
    g["lo"] += 1
    total = 0
    for k, v in g.items():
        total += v
    has = 1 if "mid" in g else 0
    other = g.get("none", 9)
    pick = LUT[sel]                       # a run-time key into int keys
    sq = {i: i * x for i in range(3)}     # a dict comprehension
    s2 = sq[sel % 3]
    alias = g                             # the same dict, as in Python
    alias["hi"] = 0
    copied = dict(g)
    copied["hi"] = 99
    return total, has, other, pick, len(g), s2, g["hi"], copied["hi"]


def matvec(m, v):
    out = [0, 0, 0]
    for i in range(3):
        acc = 0
        for j in range(2):
            acc += m[i][j] * v[j]
        out[i] = acc
    return out


def transpose(m):
    t = [[0, 0, 0], [0, 0, 0]]
    for i in range(3):
        for j in range(2):
            t[j][i] = m[i][j]
    return t


def rows(m):
    sums = []
    for row in m:
        sums.append(sum(row))
    m2 = [list(r) for r in m]
    m2[0] = [7, 8]
    return sums, max(len(m), len(m[0])), m[1], m2


def life(x, cells):
    new = [[0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            n = cells[(i + 1) % 3][j] + cells[(i + 2) % 3][j] + cells[i][(j + 1) % 3]
            new[i][j] = (n + x) % 4
    return new[1][1], new


def mix(x, m, v):                        # (the README's example)
    acc = {"lo": x * COEF["lo"], "hi": 0}
    for i in range(3):
        acc["hi"] += m[i][0] * v[0] + m[i][1] * v[1]
    out = [[acc[k] * j for j in range(2)] for k in acc]
    return out


def dot_rows(m, w):
    return [sum(a * b for a, b in zip(row, w)) for row in m]


def fx_dict(x, y):
    k = {"a": 0.5 * x, "b": 0.25 * y}
    return k["a"] - k["b"] * x


def row_alias(m):
    row = m[0]
    row[1] = 5
    return m


def repeated(x):
    g = [[0] * 2] * 2
    return g


def missing(x):
    d = {"a": x}
    return d["b"]


def returns_dict(x):
    d = {"a": x}
    return d


def _flat(v):
    out = []
    for x in (v if isinstance(v, tuple) else (v,)):
        if isinstance(x, list):
            for y in x:
                out += y if isinstance(y, list) else [y]
        else:
            out.append(x)
    return tuple(out)


def _mat(r, c, top=9):
    return [[_rng.randint(-top, top) for _ in range(c)] for _ in range(r)]


CASES = [
    (gains, {"x": I16, "y": I16, "sel": U2},
     lambda: (_rng.randint(-50, 50), _rng.randint(-50, 50), _rng.choice([0, 1, 3]))),
    (matvec, {"m": dict(I16, shape=(3, 2)), "v": dict(I16, length=2)},
     lambda: (_mat(3, 2), [_rng.randint(-9, 9) for _ in range(2)])),
    (transpose, {"m": dict(I16, shape=(3, 2))}, lambda: (_mat(3, 2),)),
    (rows, {"m": dict(I16, shape=(3, 2))}, lambda: (_mat(3, 2),)),
    (mix, {"x": I16, "m": dict(I16, shape=(3, 2)), "v": dict(I16, length=2)},
     lambda: (_rng.randint(-50, 50), _mat(3, 2), [_rng.randint(-9, 9) for _ in range(2)])),
]


@pytest.mark.sim
@pytest.mark.parametrize("f, attr, make", CASES, ids=[c[0].__name__ for c in CASES])
def test_dicts_and_lists_of_lists_match_python(monkeypatch, tmp_path, f, attr, make):
    monkeypatch.chdir(tmp_path)
    vectors = [make() for _ in range(10)]
    p = Py2ver(f, attr)
    expected = p.PY_batch()(vectors)
    assert expected == [_flat(f(*v)) for v in vectors]
    assert p.TB_batch()(vectors) == expected


@pytest.mark.sim
def test_a_matrix_as_state(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    attr = {"x": U2, "cells": {"signed": 0, "width": 2, "shape": (3, 3), "next": "new"}}
    p = Py2ver(life, attr)
    xs = [_rng.randint(0, 3) for _ in range(12)]
    assert p.TB_stream()(xs) == p.PY_stream()(xs)
    # Folding the loop over the rows: one row per clock
    p = Py2ver(life, attr, fold="i")
    assert p.ram_phases == 3 and p.TB_stream()(xs) == p.PY_stream()(xs)


@pytest.mark.sim
def test_automatic_formats_find_the_shape(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    samples = [([[_rng.uniform(-1, 1) for _ in range(3)] for _ in range(2)],
                [_rng.uniform(-1, 1) for _ in range(3)]) for _ in range(40)]
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        p = Py2ver(dot_rows, samples=samples, width=18)
    assert "'shape': (2, 3)" in out.getvalue()
    assert p.TB_batch()(samples[:8]) == p.PY_batch()(samples[:8])
    assert max(e for e, _ in p.quantization_error(samples).values()) < 1e-3
    fx = [(_rng.uniform(-2, 2), _rng.uniform(-2, 2)) for _ in range(30)]
    with contextlib.redirect_stdout(out):
        p = Py2ver(fx_dict, samples=fx, width=16)
    assert p.TB_batch()(fx[:8]) == p.PY_batch()(fx[:8])
    assert max(e for e, _ in p.quantization_error(fx).values()) < 1e-2


def test_python_that_would_differ_is_refused(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    m = {"m": dict(I16, shape=(2, 2))}
    with pytest.raises(NotImplementedError, match="is a row of 'm'"):
        Py2ver(row_alias, m)
    with pytest.raises(NotImplementedError, match="comprehension"):
        Py2ver(repeated, {"x": I16})
    with pytest.raises(NotImplementedError, match="KeyError"):
        Py2ver(missing, {"x": I16})
    with pytest.raises(NotImplementedError, match="Returning the dict"):
        Py2ver(returns_dict, {"x": I16})
