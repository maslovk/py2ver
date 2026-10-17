# The convolutional network in cnn_network.py, as it is, trained in hardware:
# py2ver builds its training step, net.train_batch, unchanged.
#
# The step (forward pass, cross-entropy through the labels, backpropagation
# through the convolutions and the pooling, dropout, Adam) runs as loops on a
# small state machine (sequencer.py). What the step reads and then changes is
# kept in block RAM between steps: the weights and biases, Adam's step counter,
# and Adam's moments, which its dict makes on the first step (setdefault: they
# start at zero). Dropout draws from a random number generator in hardware
# (xorshift, its state kept too), not numpy's stream, so the numpy reference
# below drops other neurons and the losses agree on average, not step by step.
# Formats are chosen from a run over the first steps (24 bits), with room for
# the values to grow 4 times as large later on (margin=2).
#
# The state is about 1 Mbit at 24 bits, more than the DE0-Nano's block RAM, so
# this one is for simulation: the bit-exact model trains the network, and the
# hardware itself is simulated for a step (Verilator).
#
#   python examples/cnn_training.py              # 2 epochs in fixed point, a step simulated
#   python examples/cnn_training.py --epochs 1   # quicker
import contextlib
import copy
import io
import sys
import time

import numpy as np

import cnn_network as C
from cnn_hardware import data
from py2ver import Py2ver


def batches(X_train, y_train, epochs, batch_size=32):
    """Mini-batches as cnn_network.py's script makes them (shifted images), whole
    ones only (the hardware takes 32 images a step)."""
    out = []
    for _ in range(epochs):
        order = C.rng.permutation(len(X_train))
        out += [(C.random_shift(X_train[order[s:s + batch_size]]), y_train[order[s:s + batch_size]])
                for s in range(0, len(order) - batch_size + 1, batch_size)]
    return out


if __name__ == "__main__":
    epochs = int(sys.argv[sys.argv.index("--epochs") + 1]) if "--epochs" in sys.argv else 2
    X_train, y_train, X_test, y_test = data()
    net = C.CNN()
    steps = batches(X_train, y_train, epochs)
    per_epoch = len(steps) // epochs
    reference = copy.deepcopy(net)                     # numpy trains this one, as written

    t = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        train = Py2ver(net.train_batch, samples=steps[:8], width=24, saturate=True, margin=2)
    print(f"built in {time.time() - t:.0f} s: {train.ram_phases:,} clocks per step of "
          f"32 images ({train.ram_phases / 50e6 * 1e3:.0f} ms at 50 MHz)")

    t = time.time()
    hardware = [row[0] for row in train.PY_stream()(steps)]     # the bit-exact model
    print(f"trained in fixed point: {len(steps)} steps in {time.time() - t:.0f} s")
    numpy = [reference.train_batch(*s) for s in steps]
    for e in range(epochs):
        part = slice(e * per_epoch, (e + 1) * per_epoch)
        print(f"  epoch {e + 1}: loss {np.mean(hardware[part]):.4f}   (numpy {np.mean(numpy[part]):.4f})")

    # The weights trained in fixed point, back on a network
    trained = copy.deepcopy(net)
    twin = {id(a): b for a, b in zip([net] + net.layers, [trained] + trained.layers)}
    for (obj, name), value in train.state_values().items():
        if id(obj) in twin and "." not in name and "[" not in name:   # (the layers' arrays)
            setattr(twin[id(obj)], name, value)
    print(f"test accuracy: {np.mean(trained.predict(X_test) == y_test):.1%} "
          f"(numpy {np.mean(reference.predict(X_test) == y_test):.1%})")

    # The hardware itself, simulated, from the same start
    t = time.time()
    sim = [row[0] for row in train.TB_stream()(steps[:1])]
    print(f"simulated: a step in {time.time() - t:.0f} s, same as the model: {sim == hardware[:1]}")
