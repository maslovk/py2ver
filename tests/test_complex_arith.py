import importlib.util
import itertools
import random

import pytest

from py2ver import Py2ver

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings, strategies as st  # noqa: E402

S8 = {"signed": 1, "width": 8}


# ----------------------------- designs -----------------------------

def c_ops(ar, ai, br, bi):
    x = complex(ar, ai)
    y = complex(br, bi)
    s = x + y
    d = x - y
    p = x * y
    n = -x
    return s, d, p, n


def c_parts(ar, ai, br, bi):
    x = complex(ar, ai)
    y = complex(br, bi)
    z = x * y.conjugate()
    re = z.real
    im = z.imag
    return re, im, x.real * y.imag


def c_constants(ar, ai):
    x = complex(ar, ai)
    r = x * 1j + (2 - 3j)
    return r


def c_branches(ar, ai, sel):
    x = complex(ar, ai)
    if sel > 0:
        z = x * x
    else:
        z = x.conjugate()
    w = z if sel != 2 else -z
    return w


def dft4_bin1(x0, x1, x2, x3):
    # X[1] = sum x[n] * W^n with W = -j (integer twiddles of a 4-point DFT)
    w = complex(1, 0)
    acc = complex(0, 0)
    for xn in (x0, x1, x2, x3):
        acc = acc + xn * w
        w = w * -1j
    return acc


def _cmul_add(acc, x, k):
    return acc + x * k


def c_helpers(ar, ai, br, bi):
    x = complex(ar, ai)
    acc = complex(0, 0)
    for _ in range(3):
        acc = _cmul_add(acc, x, complex(br, bi))
    return acc


def c_return_expr(ar, ai):
    return complex(ar, ai) * complex(ai, ar)


def c_division(ar, ai):
    z = complex(ar, ai) / 2
    return z


def c_complex_arg(ar, ai):
    z = complex(complex(ar, ai), 1)
    return z


# ----------------------------- simulation vs Python -----------------------------

def _vectors(n_in, n=20, seed=0):
    rng = random.Random(seed)
    corners = [-128, 127, 0, -1, 1]
    return [tuple(rng.choice(corners + [rng.randint(-128, 127)]) for _ in range(n_in))
            for _ in range(n)]


@pytest.mark.sim
@pytest.mark.parametrize(
    "func, n_in",
    [(c_ops, 4), (c_parts, 4), (c_constants, 2), (c_branches, 3), (c_helpers, 4), (c_return_expr, 2)],
)
def test_complex_matches_python(monkeypatch, tmp_path, func, n_in):
    import shutil

    if shutil.which("iverilog") is None:
        pytest.skip("Icarus Verilog tools are not installed")
    monkeypatch.chdir(tmp_path)
    attr = {name: S8 for name in ("ar", "ai", "br", "bi", "sel")}
    p = Py2ver(func, attr)
    vectors = _vectors(n_in, seed=hash(func.__name__) % 1000)

    assert p.TB_batch()(vectors) == p.PY_batch()(vectors)


@pytest.mark.sim
def test_dft4_bin_with_integer_twiddles(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    p = Py2ver(dft4_bin1, {f"x{i}": S8 for i in range(4)})
    vectors = [(1, 0, 0, 0), (0, 1, 0, 0), (1, 2, 3, 4), (-128, 127, -128, 127)]

    assert p.output_args_list == ["acc_re", "acc_im"]
    # Known values: X[1] of [1,2,3,4] is -2 + 2j
    assert p.TB_batch()(vectors)[2] == (-2, 2)
    assert p.TB_batch()(vectors) == p.PY_batch()(vectors)


def test_complex_outputs_and_declared_widths(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    attr = {"ar": S8, "ai": S8, "br": S8, "bi": S8, "p": {"signed": 1, "width": 18}}
    p = Py2ver(c_ops, attr)

    assert p.output_args_list == ["s_re", "s_im", "d_re", "d_im", "p_re", "p_im", "n_re", "n_im"]
    # Declaring p covers p_re and p_im
    assert p.attr["p_re"]["width"] == p.attr["p_im"]["width"] == 18


@pytest.mark.parametrize(
    "func, match",
    [(c_division, "only \\+ - \\*"), (c_complex_arg, "must be real")],
)
def test_unsupported_complex_raise(monkeypatch, tmp_path, func, match):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(NotImplementedError, match=match):
        Py2ver(func, {"ar": S8, "ai": S8})


# ----------------------------- random complex expressions -----------------------------

def _cexpr():
    leaf = st.sampled_from(["x", "y", "a", "1j", "(2 - 1j)", "3"])

    def extend(sub):
        binop = st.tuples(sub, st.sampled_from(["+", "-", "*"]), sub).map(
            lambda t: f"({t[0]} {t[1]} {t[2]})"
        )
        unary = sub.map(lambda x: f"(-{x})")
        conj = sub.map(lambda x: f"({x}).conjugate()")
        return st.one_of(binop, unary, conj)

    return st.recursive(leaf, extend, max_leaves=5)


_counter = itertools.count()


@pytest.mark.sim
@settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(expr=_cexpr(), vectors=st.lists(
    st.tuples(*[st.integers(-128, 127)] * 5), min_size=3, max_size=8))
def test_random_complex_expressions(monkeypatch, tmp_path_factory, expr, vectors):
    src = (
        "def rand_cplx(ar, ai, br, bi, a):\n"
        "    x = complex(ar, ai)\n"
        "    y = complex(br, bi)\n"
        f"    z = {expr}\n"
        "    z = z + 0j\n"
        "    return z\n"
    )
    path = tmp_path_factory.mktemp("cplx") / f"rand_cplx_{next(_counter)}.py"
    path.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.chdir(path.parent)

    p = Py2ver(mod.rand_cplx, {n: S8 for n in ("ar", "ai", "br", "bi", "a")})
    assert p.TB_batch()(vectors) == p.PY_batch()(vectors), src
