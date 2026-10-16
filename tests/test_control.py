import contextlib
import io
import random

import pytest

from py2ver import Py2ver

# while loops, break / continue and early returns: rewritten into flags and
# guarded statements before translation, with Python's results.

I16 = {"signed": 1, "width": 16}
U16 = {"signed": 0, "width": 16}
_rng = random.Random(11)


def clip(x):
    if x > 255:
        return 255
    if x < 0:
        return 0
    y = x * 3
    return y >> 2


def first_big(xs, t):
    idx = -1
    for i in range(8):
        if xs[i] > t:
            idx = i
            break
    return idx


def sum_pos(xs):
    s = 0
    for x in xs:
        if x < 0:
            continue
        s += x
    return s


def find_in(xs, t):
    for i in range(6):
        if xs[i] == t:
            return i, 1
    return 0, 0


def popcount(x):
    c = 0
    i = 0
    while i < 16:
        c += (x >> i) & 1
        i += 1
    return c


def gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a


def cube_root(v):
    lo = 0.0
    hi = 2.0
    while hi - lo > 0.01:
        mid = (lo + hi) * 0.5
        if mid * mid * mid > v:
            hi = mid
        else:
            lo = mid
    return (lo + hi) * 0.5


def _sat(v):
    if v > 100:
        return 100
    return v


def use_helper(a, b):
    return _sat(a * b) + 1


def uneven(x):
    if x > 0:
        return x, 1
    return x


def endless(x):
    while x > 0:
        x = x - 3
    return x


def _small():
    return _rng.randint(-60, 60)


CASES = [
    (clip, {"x": I16}, lambda: (_rng.randint(-500, 500),)),
    (first_big, {"xs": dict(I16, length=8), "t": I16},
     lambda: ([_small() for _ in range(8)], _rng.randint(0, 60))),
    (sum_pos, {"xs": dict(I16, length=8)}, lambda: ([_small() for _ in range(8)],)),
    (find_in, {"xs": dict(I16, length=6), "t": I16},
     lambda: ([_rng.randint(0, 5) for _ in range(6)], _rng.randint(0, 7))),
    (popcount, {"x": U16}, lambda: (_rng.randint(0, 65535),)),
    (gcd, {"a": U16, "b": U16, "while": 24},
     lambda: (_rng.randint(1, 65535), _rng.randint(1, 65535))),
    (use_helper, {"a": I16, "b": I16}, lambda: (_rng.randint(-20, 20), _rng.randint(-20, 20))),
]


@pytest.mark.sim
@pytest.mark.parametrize("f, attr, make", CASES, ids=[c[0].__name__ for c in CASES])
def test_control_flow_matches_python(monkeypatch, tmp_path, f, attr, make):
    monkeypatch.chdir(tmp_path)
    vectors = [make() for _ in range(15)]
    p = Py2ver(f, attr)
    expected = p.PY_batch()(vectors)
    assert expected == [r if isinstance(r, tuple) else (r,) for r in (f(*v) for v in vectors)]
    assert p.TB_batch()(vectors) == expected


@pytest.mark.sim
def test_while_iterations_measured_on_the_examples(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    samples = [(_rng.uniform(0, 7.9),) for _ in range(40)]
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        p = Py2ver(cube_root, samples=samples, width=16)
    assert "'while': 8" in out.getvalue()
    assert p.TB_batch()(samples[:10]) == p.PY_batch()(samples[:10])
    # A while loop folds like a for loop: one iteration per clock
    with contextlib.redirect_stdout(out):
        p = Py2ver(cube_root, samples=samples, width=16, fold=True)
    assert p.ram_phases == 8 and p.TB_batch()(samples[:6]) == p.PY_batch()(samples[:6])


@pytest.mark.sim
def test_shared_multipliers_with_control_flow(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    vectors = [([_rng.randint(0, 5) for _ in range(6)], _rng.randint(0, 7)) for _ in range(10)]
    p = Py2ver(find_in, {"xs": dict(I16, length=6), "t": I16}, multipliers=1)
    assert p.TB_batch()(vectors) == p.PY_batch()(vectors)


def test_what_needs_more_is_explained():
    with pytest.raises(NotImplementedError, match=r"attr\['while'\] = N"):
        Py2ver(endless, {"x": I16})
    with pytest.raises(NotImplementedError, match="same number of values"):
        Py2ver(uneven, {"x": I16})
