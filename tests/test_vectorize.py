import ast
import contextlib
import copy
import io
import textwrap

import numpy as np
import pytest

from py2ver import Py2ver, vectorize
from py2ver.fixed import _tdiv, isqrt, wrap

# Loop programs run with numpy (vectorize.py): exactly what Python gives.


def _both(src, *args, env=None):
    """(Python's result and lists, numpy's)."""
    src = textwrap.dedent(src)
    fd = ast.parse(src).body[0]
    ns = dict(env or {})
    ns.update(__py2ver_wrap=wrap, __py2ver_tdiv=_tdiv, __py2ver_isqrt=isqrt,
              __py2ver_wrapl=lambda xs, w, s: [wrap(int(x), w, s) for x in xs])
    exec(compile(src, "<test>", "exec"), ns)
    py_args, np_args = copy.deepcopy(args), copy.deepcopy(args)
    want = ns[fd.name](*py_args)
    prog = vectorize.Program(fd, ns)
    got = prog(*np_args)
    return (want, py_args), (got, np_args), prog


def _same(a, b):
    assert type(a[0]) is type(b[0]) or isinstance(a[0], (int, float)) and isinstance(b[0], (int, float))
    assert a == b


def test_parallel_loops_and_carried_sums():
    a, b, prog = _both("""
        def f(x, w):
            out = [0] * 6
            for i in range(2):
                for j in range(3):
                    acc = 0
                    for k in range(4):
                        acc = acc + x[i * 4 + k] * w[k * 3 + j]
                    out[i * 3 + j] = acc
            best = out[0]
            arg = 0
            for t in range(6):
                if out[t] > best:
                    best = out[t]
                    arg = t
            return out, arg, best
        """, [1.5, -2.0, 0.25, 3.0, 0.5, 0.5, -1.0, 2.0], [0.1 * k - 0.4 for k in range(12)])
    _same(a, b)
    assert not prog._serial  # (the loops over i, j at once; k and the argmax in order)


def test_in_place_updates_repeated_indices_and_one_element():
    a, b, _ = _both("""
        def f(g, c, s):
            for i in range(4):                      # (overlapping windows: rounds)
                for j in range(3):
                    g[i + j] = g[i + j] + c[i * 3 + j]
            for i in range(6):                      # (in place, each element once)
                g[i] = g[i] * 0.5 - 1.0
            for i in range(5):                      # (one element: in order)
                x = s[0]
                x = x ^ x << 13 & 4294967295
                x = x ^ x >> 17
                s[0] = x
            return g[5]
        """, [0.0] * 6, [float(k) for k in range(12)], [12345])
    _same(a, b)


def test_integers_past_62_bits_and_the_model_s_helpers():
    a, b, _ = _both("""
        def f(x, y):
            out = [0] * 4
            for i in range(4):
                big = x[i] << 40
                p = big * y[i]                      # (past int64: Python's integers)
                out[i] = __py2ver_wrap(p >> 60, 24, True) + __py2ver_tdiv(x[i], y[i] - 3) \\
                    + __py2ver_isqrt(x[i] * x[i] + 7)
            return out
        """, [3, -7, 1 << 20, 123456], [1 << 30, 3, -5, 7])
    _same(a, b)


@pytest.mark.sim
def test_a_model_and_its_formats_are_the_same_with_numpy(monkeypatch, tmp_path):
    import test_numpy_loops as T
    monkeypatch.chdir(tmp_path)
    net = T.SmallCNN()
    x, y = T._images(24)
    steps = [(x[i:i + 4], y[i:i + 4]) for i in range(0, 24, 4)]
    built = {}
    for on in ("1", "0"):
        monkeypatch.setenv("PY2VER_VECTOR", on)
        with contextlib.redirect_stdout(io.StringIO()):
            p = Py2ver(copy.deepcopy(net).train_batch, samples=steps, width=24, saturate=True,
                       margin=2)
        built[on] = p
    fast, slow = built["1"], built["0"]
    assert isinstance(fast._model, vectorize.Program) and not isinstance(slow._model,
                                                                         vectorize.Program)
    assert fast.attr == slow.attr                         # (the formats: the same ranges)
    assert fast.PY_stream()(steps) == slow.PY_stream()(steps)   # (bit for bit)
