import contextlib
import io
import random
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from py2ver import Py2ver
from py2ver.numpy_fe import NotTranslatable, translate

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from xor_network import X, Y, XORNetwork, sigmoid  # noqa: E402  (the original code, unchanged)


def _quiet():
    return contextlib.redirect_stdout(io.StringIO())


@pytest.fixture(scope="module")
def trained():
    net = XORNetwork()
    with _quiet():
        net.train(X, Y, lr=0.5)
    return net


def test_predict_translates_to_scalars(trained):
    tr = translate(trained.predict, list(X))
    assert tr.params == ["x"] and tr.shapes == {"x": (2,)}
    assert tr.tables == {"sigmoid"} and not tr.state                  # weights: constants
    assert "np.tanh(x[0] *" in tr.source and "sigmoid(" in tr.source
    assert "self" not in tr.source.replace("self_h", "").replace("self_o", "")
    out = tr.func(*tr.adapt(np.array([0.8, 0.3])))
    assert out[0] == pytest.approx(trained.predict([0.8, 0.3])[0, 0], rel=1e-12)


def test_train_translates_to_one_step():
    net = XORNetwork()
    tr = translate(net.train, [(X, Y)] * 3)
    assert tr.loop and tr.params == ["X", "Y"]                        # lr: its default, 0.5
    assert list(tr.state) == ["W1", "b1", "W2", "b2"]                 # read, then updated
    assert [n for n, _ in tr.outputs][0] == "r_L"
    assert "print" not in tr.source and "step" not in tr.source.split("def train_step")[1]
    assert np.array_equal(net.W1, XORNetwork().W1)                    # the object is untouched


EXTRA = '''
import numpy as np
A = np.array([[0.5, -1.0], [0.25, 2.0]])

def slices(x):
    return x[0, ::-1] @ A[:, 1:]

def indexing(x):
    return x[0.5]

def loops(x):
    s = 0
    while s < 1:
        s = s + x
    return s

def printing(x):
    print(x)
    return np.tanh(x @ A)

def fine(x):
    y = np.maximum(x @ A, 0) - abs(x).sum() / 2
    return y.T, np.mean(y ** 2, axis=0, keepdims=True)
'''


@pytest.fixture(scope="module")
def extra(tmp_path_factory):
    path = tmp_path_factory.mktemp("np") / "npextra.py"
    path.write_text(EXTRA)
    ns = {"__name__": "npextra"}
    exec(compile(EXTRA, str(path), "exec"), ns)
    return ns


def test_slices(extra):
    x = np.array([[0.1, 0.2]])
    tr = translate(extra["slices"], [x])                               # checked against numpy
    assert np.allclose(np.ravel(tr.func(*tr.adapt(x))), np.ravel(extra["slices"](x)))


@pytest.mark.parametrize("name, match", [("indexing", "constant integer indices"),
                                         ("loops", "Unsupported statement"),
                                         ("printing", "Unsupported statement")])
def test_untranslatable_code_says_why(extra, name, match):
    with pytest.raises(NotTranslatable, match=match):
        translate(extra[name], [np.array([[0.1, 0.2]])])


def test_a_numpy_function_with_reductions_and_broadcasting(extra):
    x = np.array([[0.3, -0.7]])
    tr = translate(extra["fine"], [x])                                 # checked against numpy
    got = tr.func(*tr.adapt(x))
    want = extra["fine"](x)
    assert np.allclose(np.ravel(got[0]), np.ravel(want[0]))
    assert np.allclose(np.ravel(got[1]), np.ravel(want[1]))


@pytest.mark.sim
def test_trained_inference_in_hardware(monkeypatch, tmp_path, trained):
    monkeypatch.chdir(tmp_path)
    rng = random.Random(0)
    points = list(X) + [np.array([rng.random(), rng.random()]) for _ in range(200)]
    with _quiet():
        p = Py2ver(trained.predict, samples=points, max_error=1e-2)
    assert p.input_args_list == ["x_0", "x_1"] and p.output_args_list == ["out_0"]
    assert [round(v[0]) for v in p.PY_batch()(list(X))] == [0, 1, 1, 0]
    assert p.quantization_error(points)["out_0"][0] <= 1e-2
    assert p.TB_batch()(points[:30]) == p.PY_batch()(points[:30])
    assert p.TB()(np.array([0.8, 0.3])) == p.PY_batch()([[0.8, 0.3]])[0]


@pytest.mark.sim
def test_training_in_hardware(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    net = XORNetwork()
    with _quiet():
        p = Py2ver(net.train, samples=[(X, Y)] * 500, width=24)
    steps = p.PY_stream()([(X, Y)] * 450)
    losses = [r[0] for r in steps]
    assert losses[0] > 0.5 and min(losses) < 0.01                   # trains like the original
    assert p.TB_stream()([(X, Y)] * 30) == steps[:30]
    w = p.state_values(steps[-1])
    assert w["W1"].shape == (2, 4) and w["b2"].shape == (1, 1)
    trained_hw = XORNetwork()
    for name, value in w.items():
        setattr(trained_hw, name, value)
    assert [round(float(trained_hw.predict(x)[0, 0])) for x in X] == [0, 1, 1, 0]
