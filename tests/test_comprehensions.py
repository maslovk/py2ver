import contextlib
import io
import random

import pytest

from py2ver import Py2ver

# List comprehensions and generator expressions are written out as lists
# (their lengths are known when translating).

I16 = {"signed": 1, "width": 16}
W = (3, -1, 4, 1)
_rng = random.Random(5)


def forms(xs, ys, g):
    scaled = [x * g for x in xs]                              # a list argument
    shifted = [scaled[i] + i for i in range(1, 4)]            # range(...)
    weighted = [w * x for w, x in zip(W, xs)]                 # zip with a module tuple
    indexed = [k * x for k, x in enumerate(ys) if k % 2 == 0]  # enumerate, an if on the index
    pairs = [a - b for a in xs[:2] for b in ys[2:]]           # two fors, slices
    total = sum(x * y for x, y in zip(xs, ys))                # a generator expression
    top = max(abs(v) for v in scaled + weighted)
    return shifted, weighted, indexed, pairs, total, top


FORMS = {"xs": dict(I16, length=4), "ys": dict(I16, length=4), "g": I16}


def dot(xs, ws):
    acc = sum(x * w for x, w in zip(xs, ws))
    scaled = [acc * w for w in ws]
    return scaled, acc


def by_value(xs):
    return [x for x in xs if x > 0]


def unknown(xs, n):
    return [x * 2 for x in range(n)]


@pytest.mark.sim
def test_comprehensions_match_python(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    p = Py2ver(forms, FORMS)
    vectors = [([_rng.randint(-99, 99) for _ in range(4)], [_rng.randint(-99, 99) for _ in range(4)],
                _rng.randint(-9, 9)) for _ in range(10)]
    expected = p.PY_batch()(vectors)
    assert p.TB_batch()(vectors) == expected
    xs, ys, g = vectors[0]
    shifted, weighted, indexed, pairs, total, top = forms(xs, ys, g)
    assert expected[0] == tuple(shifted + weighted + indexed + pairs + [total, top])


@pytest.mark.sim
def test_comprehensions_in_fixed_point(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    samples = [([_rng.uniform(-1, 1) for _ in range(4)], [_rng.uniform(-1, 1) for _ in range(4)])
               for _ in range(40)]
    with contextlib.redirect_stdout(io.StringIO()):
        p = Py2ver(dot, samples=samples, width=18)
    assert p.TB_batch()(samples[:10]) == p.PY_batch()(samples[:10])
    assert max(m for m, _ in p.quantization_error(samples).values()) < 1e-3


def test_lengths_must_be_known():
    with pytest.raises(NotImplementedError, match="depends on values"):
        Py2ver(by_value, {"xs": dict(I16, length=4)})
    with pytest.raises(NotImplementedError, match="Can't tell how many items 'range\\(n\\)'"):
        Py2ver(unknown, {"xs": dict(I16, length=4), "n": I16})
