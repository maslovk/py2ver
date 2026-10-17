# The digit-recognising network in digits_network.py, as it is, trained in
# hardware: py2ver builds its training step, net.train_batch, unchanged.
#
# The step (forward pass, backpropagation, Adam) runs as loops on a small
# state machine (sequencer.py), one operation a clock. The weights, the
# biases and Adam's moments (51,678 numbers) and the step counter self.t are
# what the method reads and then changes: they are kept in block RAM between
# steps, starting from the network's values. Adam's beta ** t is a product of
# constants over t's bits, np.sqrt an exact square root, and divisions take a
# bit a clock. Formats are chosen from a run over the first epoch (24 bits),
# with room for the values to grow 4 times as large later on (margin=2).
#
# The state is about 1.2 Mbit at 24 bits, twice the DE0-Nano's block RAM, so
# this one is for simulation: the bit-exact model trains the network, and the
# hardware itself is simulated for a few steps (Verilator).
#
#   python examples/digits_training.py             # 3 epochs in fixed point, 2 steps simulated
#   python examples/digits_training.py --epochs 1  # quicker
import contextlib
import copy
import io
import sys
import time

import numpy as np

import digits_network as D
from py2ver import Py2ver


def batches(epochs, seed=1, batch_size=32, lr=0.001):
    """Mini-batches as DigitNetwork.fit makes them, but whole ones only (the
    hardware takes 32 images a step; fit's last batch of an epoch is short)."""
    rng = np.random.default_rng(seed)
    Y = D.one_hot(D.y_train)
    out = []
    for _ in range(epochs):
        order = rng.permutation(len(D.X_train))
        out += [(D.X_train[order[s:s + batch_size]], Y[order[s:s + batch_size]], lr)
                for s in range(0, len(order) - batch_size + 1, batch_size)]
    return out


if __name__ == "__main__":
    epochs = int(sys.argv[sys.argv.index("--epochs") + 1]) if "--epochs" in sys.argv else 3
    steps = batches(epochs)
    per_epoch = len(steps) // epochs
    net = D.DigitNetwork()
    reference = copy.deepcopy(net)                     # numpy trains this one, as written

    t = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        train = Py2ver(net.train_batch, samples=steps[:per_epoch], width=24, saturate=True,
                       margin=2)
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
        setattr(twin[id(obj)], name, value)
    print(f"test accuracy: {trained.accuracy(D.X_test, D.y_test):.1%} "
          f"(numpy {reference.accuracy(D.X_test, D.y_test):.1%})")

    # The hardware itself, simulated, from the same start
    n = 2
    t = time.time()
    sim = [row[0] for row in train.TB_stream()(steps[:n])]
    print(f"simulated: {n} steps in {time.time() - t:.0f} s, same as the model: {sim == hardware[:n]}")
