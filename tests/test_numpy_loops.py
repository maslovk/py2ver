import contextlib
import copy
import io
import re

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


# ----------------------------- training: the object's arrays as state -----------------------------

class AdamDense:
    def __init__(self, n_in, n_out):
        self.W = rng.normal(0, np.sqrt(2 / n_in), (n_in, n_out))
        self.b = np.zeros(n_out)
        self.mW, self.vW = np.zeros_like(self.W), np.zeros_like(self.W)
        self.mb, self.vb = np.zeros_like(self.b), np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad):
        self.dW = self.x.T @ grad
        self.db = grad.sum(axis=0)
        return grad @ self.W.T

    def adam_step(self, lr, t, beta1=0.9, beta2=0.999, eps=1e-8):
        for p, g, m, v in ((self.W, self.dW, self.mW, self.vW), (self.b, self.db, self.mb, self.vb)):
            m[:] = beta1 * m + (1 - beta1) * g
            v[:] = beta2 * v + (1 - beta2) * g ** 2
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            p -= lr * m_hat / (np.sqrt(v_hat) + eps)


class Trainer:
    def __init__(self):
        self.layers = [AdamDense(6, 8), ReLU(), AdamDense(8, 3)]
        self.t = 0

    def train_batch(self, x, y, lr):
        z = x
        for layer in self.layers:
            z = layer.forward(z)
        probs = softmax(z)
        loss = -np.mean(np.sum(y * np.log(probs + 1e-9), axis=1))
        grad = (probs - y) / len(x)
        for layer in reversed(self.layers):
            if isinstance(layer, AdamDense):
                grad = layer.backward(grad)
            else:
                grad = grad * layer.mask
        self.t += 1
        for layer in self.layers:
            if isinstance(layer, AdamDense):
                layer.adam_step(lr, self.t)
        return loss


def _batches(n, seed=4):
    r = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        x = r.uniform(0, 1, (4, 6))
        out.append((x, np.eye(3)[x[:, :3].argmax(axis=1)], 0.01))
    return out


def test_a_training_step_keeps_the_changed_attributes_as_state():
    net = Trainer()
    before = net.layers[0].W.copy()
    tr = numpy_loops.translate(net.train_batch, _batches(3))
    assert np.array_equal(net.layers[0].W, before) and net.t == 0     # (the object is left as it is)
    labels = sorted(label for label, _, _ in tr.memories.values())
    assert labels == sorted(["AdamDense.W", "AdamDense.b", "AdamDense.mW", "AdamDense.vW",
                             "AdamDense.mb", "AdamDense.vb"] * 2 + ["Trainer.t"])
    assert all(tr.attr[m]["memory"] == "ram" for m in tr.memories)
    # Products: two forward, two weight gradients, one gradient back (the one for
    # the layer before the first is never used: not computed)
    assert len(re.findall(r"mm\d+ = \[0\]", tr.source)) == 5
    # Each Adam update of a parameter: one loop each for m, v and the parameter
    assert "beta" not in tr.source and "** t" not in tr.source


@pytest.mark.sim
def test_a_network_trains_in_hardware(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    net = Trainer()
    steps = _batches(60)
    with contextlib.redirect_stdout(io.StringIO()):
        p = Py2ver(net.train_batch, samples=steps, width=24, saturate=True)
    model = [r[0] for r in p.PY_stream()(steps)]
    twin = Trainer.__new__(Trainer)
    twin.__dict__ = copy.deepcopy(net.__dict__)
    numpy_losses = [twin.train_batch(*s) for s in steps]
    # Training in fixed point follows numpy's (the loss through a log table)
    assert abs(np.mean(model[-20:]) - np.mean(numpy_losses[-20:])) < 0.01
    assert np.mean(model[-20:]) < np.mean(model[:20]) - 0.1
    # The weights trained in fixed point, as the object's arrays
    state = p.state_values()
    W = state[(net.layers[0], "W")]
    assert W.shape == (6, 8) and np.abs(W - twin.layers[0].W).mean() < 0.005
    assert np.allclose(W, twin.layers[0].W, atol=0.1)
    assert state[(net, "t")] == 60
    assert p.TB_stream()(steps[:2]) == p.PY_stream()(steps[:2])


def _norm(x):
    return np.sqrt((x * x).sum()) / (x.max() - x.min())


@pytest.mark.sim
def test_square_roots_and_division_take_a_bit_a_clock(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    r = np.random.default_rng(6)
    xs = [r.uniform(-2, 2, 40) for _ in range(30)]
    with contextlib.redirect_stdout(io.StringIO()):
        p = Py2ver(_norm, samples=xs, width=20, sequential=True)
    model = [v for v, in p.PY_batch()(xs)]
    assert np.allclose(model, [_norm(x) for x in xs], rtol=1e-3)
    assert p.TB_batch()(xs[:3]) == p.PY_batch()(xs[:3])
