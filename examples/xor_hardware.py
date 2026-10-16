# The XOR network in xor_network.py, as it is, in hardware: its training
# step and its trained forward pass.
#
# py2ver translates the numpy methods into scalar code (the object's weights
# become state while training, constants once trained), chooses fixed-point
# formats from example runs and builds them.
#
#   python examples/xor_hardware.py           # training and inference, simulated
import contextlib
import io
import random

import numpy as np

from py2ver import Py2ver
from xor_network import X, Y, XORNetwork


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


if __name__ == "__main__":
    # Training in hardware: one step of net.train per call, the weights as state
    net = XORNetwork()
    with quiet():
        train = Py2ver(net.train, samples=[(X, Y)] * 600, width=24)
    steps = train.PY_stream()([(X, Y)] * 600)          # the bit-exact model of the hardware
    losses = [row[0] for row in steps]
    done = next(i for i, L in enumerate(losses) if L < 0.01)     # train()'s break
    print(f"training in fixed point: loss {losses[0]:.4f} -> {losses[done]:.4f} at step {done + 1}")
    sim = train.TB_stream()([(X, Y)] * 50)
    print("  hardware matches the model over 50 steps:", sim == steps[:50])

    # The weights trained in hardware, back on a network
    trained = XORNetwork()
    for name, value in train.state_values(steps[done]).items():
        setattr(trained, name, value)

    # Inference in hardware: the trained forward pass, the weights as constants
    rng = random.Random(0)
    points = list(X) + [np.array([rng.random(), rng.random()]) for _ in range(300)]
    with quiet():
        infer = Py2ver(trained.predict, samples=points, max_error=1e-2)
    print("inference in fixed point (max error 1e-2):")
    for x, y, p in zip(X, Y, infer.PY_batch()(list(X))):
        print(f"  {x} -> {p[0]:.3f}   (target {int(y[0])})")
    print("  hardware matches the model:", infer.TB_batch()(points[:40]) == infer.PY_batch()(points[:40]))
    print(f"  (0.8, 0.3) -> {infer.PY_batch()([[0.8, 0.3]])[0][0]:.3f}, "
          f"numpy {trained.predict([0.8, 0.3])[0, 0]:.3f}")
