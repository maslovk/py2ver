import importlib.util
import itertools
import random
import shutil

import pytest

from py2ver import Py2ver

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings, strategies as st  # noqa: E402

S8 = {"signed": 1, "width": 8}
U8 = {"signed": 0, "width": 8}

TABLE = (5, -3, 17, 0, 99, -128, 42, 7)
TAPS = (2, 8, 14, 16, 14, 8, 2)


def _gf_mul(a, b):
    r = 0
    while b:
        if b & 1:
            r ^= a
        a = ((a << 1) ^ 0x11B) if a & 0x80 else (a << 1)
        b >>= 1
    return r


def _aes_sbox():
    out = []
    for x in range(256):
        inv = next((y for y in range(1, 256) if _gf_mul(x, y) == 1), 0) if x else 0
        s = inv
        for i in range(1, 5):
            s ^= ((inv << i) | (inv >> (8 - i))) & 0xFF
        out.append(s ^ 0x63)
    return tuple(out)


SBOX = _aes_sbox()


# ----------------------------- designs -----------------------------

def list_return(a, b):
    data = []
    data.append(a + b)
    data.append(a - b)
    data.extend([a * b, a ^ b])
    return data


def rom_lookup(x):
    y = TABLE[x] + TABLE[-1]
    return y


def local_table(x, i):
    taps = [3, -1, 4, 1, -5]
    return taps[i] * x


def sub_word(w):
    out = 0
    for k in range(4):
        out |= SBOX[(w >> (8 * k)) & 0xFF] << (8 * k)
    return out


def var_index(xs, i, v):
    a = xs[i]
    xs[i] = v
    b = xs[-1] + xs[i - 1]
    return a, b, xs


def list_ops(a, b, c):
    xs = [a, b, c] * 2
    ys = xs[1:5] + [a - b]
    evens = ys[::2]
    total = sum(ys)
    return total, min(ys), max(xs), len(ys), evens, list(range(3))


def list_in_branches(a, b, sel):
    xs = [a, b, 0]
    if sel > 0:
        xs[2] = a * b
        xs[0] = -a
    else:
        xs = [b, a, a + b]
    acc = 0
    for v in xs:
        acc += v
    return acc, xs


def fir_step(x, delay):
    taps_in = [x] + delay
    acc = 0
    for k in range(len(TAPS)):
        acc += TAPS[k] * taps_in[k]
    y = acc >> 6
    new_delay = taps_in[:-1]
    return y, new_delay


def histogram(bin_, counts):
    counts[bin_] += 1
    return max(counts), counts


def alias(a):
    xs = [a, a]
    ys = xs
    return ys


def append_in_branch(a):
    xs = [a]
    if a > 0:
        xs.append(a)
    return xs


def nested(a):
    xs = [[a, a], [a]]
    return xs[0][0]


def out_of_range(a):
    xs = [a, a]
    return xs[2]


# ----------------------------- simulation vs Python -----------------------------

@pytest.fixture
def sim_dir(monkeypatch, tmp_path):
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("Icarus Verilog tools are not installed")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _r(width, signed, rng):
    lo, hi = (-(1 << (width - 1)), (1 << (width - 1)) - 1) if signed else (0, (1 << width) - 1)
    return rng.choice([lo, hi, 0, rng.randint(lo, hi)])


@pytest.mark.sim
def test_list_return_is_one_output_per_element(sim_dir):
    p = Py2ver(list_return, {"a": U8, "b": U8, "data": {"signed": 1, "width": 17}})
    vectors = [(4, 4), (200, 3), (17, 255), (0, 255)]

    assert p.output_args_list == ["data_0", "data_1", "data_2", "data_3"]
    assert p.attr["data_2"]["width"] == 17          # declaring data covers its elements
    assert p.TB_batch()(vectors) == p.PY_batch()(vectors)


@pytest.mark.sim
@pytest.mark.parametrize(
    "func, attr, make",
    [
        (rom_lookup, {"x": {"signed": 1, "width": 4}}, lambda r: (r.randint(-8, 7),)),
        (local_table, {"x": S8, "i": {"signed": 1, "width": 3}},
         lambda r: (_r(8, 1, r), r.randint(-4, 3))),   # i is 3-bit signed
        (var_index, {"xs": {"signed": 1, "width": 8, "length": 5},
                     "i": {"signed": 1, "width": 4}, "v": S8},
         lambda r: ([_r(8, 1, r) for _ in range(5)], r.randint(-4, 4), _r(8, 1, r))),
        (list_ops, {"a": S8, "b": S8, "c": S8}, lambda r: tuple(_r(8, 1, r) for _ in range(3))),
        (list_in_branches, {"a": S8, "b": S8, "sel": S8},
         lambda r: tuple(_r(8, 1, r) for _ in range(3))),
    ],
)
def test_lists_match_python(sim_dir, func, attr, make):
    rng = random.Random(func.__name__)
    vectors = [make(rng) for _ in range(24)]
    p = Py2ver(func, attr)
    assert p.TB_batch()(vectors) == p.PY_batch()(vectors)


@pytest.mark.sim
def test_aes_sbox_rom(sim_dir):
    assert SBOX[0x00] == 0x63 and SBOX[0x53] == 0xED and SBOX[0xFF] == 0x16
    p = Py2ver(sub_word, {"w": {"signed": 0, "width": 32}})
    rng = random.Random(2)
    vectors = [(rng.getrandbits(32),) for _ in range(32)] + [(0,), (0x53535353,), (0xFFFFFFFF,)]

    assert p.TB_batch()(vectors) == p.PY_batch()(vectors)


@pytest.mark.sim
def test_fir_delay_line_memory(sim_dir):
    attr = {
        "x": {"signed": 1, "width": 16},
        "delay": {"signed": 1, "width": 16, "length": 6, "next": "new_delay"},
        "y": {"signed": 1, "width": 16},
    }
    p = Py2ver(fir_step, attr)
    rng = random.Random(3)
    xs = [64] + [0] * 8 + [rng.randint(-2000, 2000) for _ in range(16)]
    sim = p.TB_stream()(xs)

    assert [row[0] for row in sim[:9]] == [2, 8, 14, 16, 14, 8, 2, 0, 0]   # impulse = taps
    assert sim == p.PY_stream()(xs)


@pytest.mark.sim
def test_histogram_updated_in_place(sim_dir):
    attr = {
        "bin_": {"signed": 0, "width": 3},
        "counts": {"signed": 0, "width": 8, "length": 8, "next": "counts",
                   "init": [0, 0, 0, 0, 0, 0, 0, 250]},
    }
    p = Py2ver(histogram, attr)
    rng = random.Random(5)
    bins = [7] * 7 + [rng.randint(0, 7) for _ in range(20)]   # bin 7 wraps past 255

    assert p.TB_stream()(bins) == p.PY_stream()(bins)


@pytest.mark.sim
@pytest.mark.board
def test_board_with_list_argument_and_list_state(sim_dir):
    p = Py2ver(var_index, {"xs": {"signed": 1, "width": 8, "length": 5},
                           "i": {"signed": 1, "width": 4}, "v": S8})
    vectors = [([1, -2, 3, -4, 5], -4, 9), ([-128, 127, 0, 1, -1], 3, -7)]
    assert p.HW_sim()(vectors) == p.PY_batch()(vectors)

    h = Py2ver(histogram, {"bin_": {"signed": 0, "width": 3},
                           "counts": {"signed": 0, "width": 8, "length": 8, "next": "counts"}})
    bins = [3, 3, 7, 0, 3]
    assert h.HW_sim()([(b,) for b in bins]) == h.PY_stream()(bins)


# ----------------------------- errors -----------------------------

@pytest.mark.parametrize(
    "func, match",
    [
        (alias, "copy it with"),
        (append_in_branch, "one path only"),
        (nested, "must be integers"),
        (out_of_range, "out of range"),
    ],
)
def test_unsupported_lists_raise_clearly(monkeypatch, tmp_path, func, match):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(NotImplementedError, match=match):
        Py2ver(func, {"a": S8})


def test_list_argument_length_is_checked(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    p = Py2ver(var_index, {"xs": {"signed": 1, "width": 8, "length": 5},
                           "i": {"signed": 1, "width": 4}, "v": S8})
    with pytest.raises(ValueError, match="needs 5 values"):
        p.gen_tb([([1, 2, 3], 0, 0)])


# ----------------------------- random list programs -----------------------------

_counter = itertools.count()


@st.composite
def _programs(draw):
    n = draw(st.integers(5, 7))      # indices go down to i - 1 = -5
    leaf = st.one_of(st.sampled_from(["a", "b", "c"]), st.integers(-9, 9).map(str))
    elems = draw(st.lists(leaf, min_size=n, max_size=n))
    read = draw(st.sampled_from(["i", "j", "-1", "i - 1", "(i + j) // 2"]))
    write = draw(st.sampled_from(["i", "j", "0", "-2"]))
    value = draw(st.sampled_from(["a", "b * c", "xs[j] + 1", "-a"]))
    tail = draw(st.sampled_from(["sum(xs)", "max(xs[1:])", "xs[i] - xs[j]", "len(xs) + a"]))
    src = (
        "def rand_list(a, b, c, i, j):\n"
        f"    xs = [{', '.join(elems)}]\n"
        f"    y = xs[{read}]\n"
        f"    xs[{write}] = {value}\n"
        f"    z = {tail}\n"
        "    return y, z, xs\n"
    )
    rng_vals = st.tuples(st.integers(-128, 127), st.integers(-128, 127), st.integers(-128, 127),
                         st.integers(-4, 3), st.integers(-4, 3))
    vectors = draw(st.lists(rng_vals, min_size=3, max_size=10))
    return src, vectors


@pytest.mark.sim
@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(program=_programs())
def test_random_list_programs(monkeypatch, tmp_path_factory, program):
    src, vectors = program
    path = tmp_path_factory.mktemp("lst") / f"rand_list_{next(_counter)}.py"
    path.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.chdir(path.parent)

    attr = {"a": S8, "b": S8, "c": S8, "i": {"signed": 1, "width": 3}, "j": {"signed": 1, "width": 3}}
    p = Py2ver(mod.rand_list, attr)
    assert p.TB_batch()(vectors) == p.PY_batch()(vectors), src
