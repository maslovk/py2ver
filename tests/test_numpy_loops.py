import contextlib
import io

import numpy as np
import pytest

from py2ver import Py2ver, numpy_loops

# numpy code with objects and lists of layers, kept as loops and sequenced.

rng = np.random.default_rng(3)


class Dense:
    def __init__(self, n_in, n_out):
        self.W = rng.normal(0, np.sqrt(2 / n_in), (n_in, n_out))
        self.b = rng.normal(0, 0.1, n_out)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b


class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class Net:
    def __init__(self, sizes=(12, 16, 5)):
        self.layers = []
        for i in range(len(sizes) - 1):
            self.layers.append(Dense(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                self.layers.append(ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return softmax(x)

    def predict(self, x):
        return self.forward(np.atleast_2d(x)).argmax(axis=1)

    def scores(self, x):
        return self.forward(np.atleast_2d(x))


NET = Net()
SAMPLES = [rng.uniform(0, 1, 12) for _ in range(60)]


def test_the_loop_program_agrees_with_numpy():
    tr = numpy_loops.translate(NET.predict, SAMPLES[:5])
    assert "for " in tr.source and len(tr.const_tables) == 4      # weights as tables
    for x in SAMPLES:
        assert tr.func(*tr.adapt(x))[0] == NET.predict(x)[0]
    tr = numpy_loops.translate(NET.scores, SAMPLES[:5])
    for x in SAMPLES[:10]:
        assert np.allclose(tr.func(*tr.adapt(x)), NET.scores(x).ravel())


@pytest.mark.sim
def test_a_network_too_varied_to_write_out_is_sequenced(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        p = Py2ver(NET.predict, samples=SAMPLES, width=16)     # (the list of layers: loops)
    assert p.ram_phases > 100
    model = [r[0] for r in p.PY_batch()(SAMPLES)]
    assert np.mean(np.array(model) == [NET.predict(x)[0] for x in SAMPLES]) > 0.95
    assert p.TB_batch()(SAMPLES[:6]) == p.PY_batch()(SAMPLES[:6])
    with contextlib.redirect_stdout(io.StringIO()):
        p = Py2ver(NET.scores, samples=SAMPLES, width=16)
    assert max(e for e, _ in p.quantization_error(SAMPLES[:20]).values()) < 0.02


@pytest.mark.sim
def test_the_digit_network(monkeypatch, tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    pytest.importorskip("sklearn")
    import digits_network as D
    monkeypatch.chdir(tmp_path)
    net = D.DigitNetwork()
    with contextlib.redirect_stdout(io.StringIO()):
        net.fit(D.X_train, D.y_train, epochs=5)
        p = Py2ver(net.predict, samples=list(D.X_train[:100]), width=16)
    images = list(D.X_test[:60])
    model = [r[0] for r in p.PY_batch()(images)]
    assert np.mean(np.array(model) == net.predict(np.array(images))) > 0.95
    assert p.TB_batch()(images[:4]) == [(m,) for m in model[:4]]
