# The convolutional network in cnn_network.py, as it is, in hardware.
#
# It is trained in Python as cnn_network.py's own script trains it (40
# epochs, shifted images, dropout, Adam), then py2ver builds its forward pass,
# net.predict, unchanged: np.pad, sliding_window_view and the transposes are
# views or copies of block RAM, the convolutions and the dense layers loops
# of multiply-accumulates (sequencer.py), max pooling a reduction over the
# 2x2 blocks, and the weights ROMs. Formats are chosen from example images
# (16 bits).
#
#   python examples/cnn_hardware.py           # fixed point on the test set, hardware simulated
#   python examples/cnn_hardware.py --syn     # + Quartus area and timing (DE0-Nano's EP4CE22)
import contextlib
import io
import os
import sys
import time

import numpy as np

import cnn_network as C
from py2ver import Py2ver


def data():
    """The training and test images, as cnn_network.py's script splits them."""
    digits = C.load_digits()
    X = (digits.images / 16.0)[:, None, :, :]
    y = digits.target
    idx = C.rng.permutation(len(X))
    split = int(0.8 * len(X))
    return X[idx[:split]], y[idx[:split]], X[idx[split:]], y[idx[split:]]


def trained():
    """The network trained as cnn_network.py's script trains it, and the data."""
    X_train, y_train, X_test, y_test = data()
    net = C.CNN()
    for epoch in range(1, 41):
        if epoch == 30:
            net.opt.lr /= 4
        order = C.rng.permutation(len(X_train))
        for s in range(0, len(order), 32):
            b = order[s:s + 32]
            net.train_batch(C.random_shift(X_train[b]), y_train[b])
    return net, X_train, y_train, X_test, y_test


if __name__ == "__main__":
    net, X_train, y_train, X_test, y_test = trained()
    print(f"numpy:       test accuracy {np.mean(net.predict(X_test) == y_test):.1%}")

    t = time.time()
    images = [X_train[i:i + 1] for i in range(200)]           # one image a sample
    with contextlib.redirect_stdout(io.StringIO()):
        lanes = int(sys.argv[sys.argv.index("--lanes") + 1]) if "--lanes" in sys.argv else 1
        p = Py2ver(net.predict, samples=images, width=16, multipliers=lanes)
    print(f"hardware:    built in {time.time() - t:.0f} s, {p.ram_phases:,} clocks per image "
          f"({p.ram_phases / 50e6 * 1e3:.2f} ms at 50 MHz)")

    # The bit-exact model of the hardware, on every test image
    tests = [X_test[i:i + 1] for i in range(len(X_test))]
    model = np.array([r[0] for r in p.PY_batch()(tests)])
    print(f"fixed point: test accuracy {np.mean(model == y_test):.1%}, "
          f"same digit as numpy for {np.mean(model == net.predict(X_test)):.1%}")

    # The hardware itself, simulated (Verilator when it is installed), on a few
    n = 10
    t = time.time()
    hw = [r[0] for r in p.TB_batch()(tests[:n])]
    print(f"simulated:   {n} images in {time.time() - t:.0f} s, same as the model: "
          f"{hw == list(model[:n])}")

    if "--syn" in sys.argv[1:]:
        print(p.SYN({"QUARTUS_DIR": os.environ.get("QUARTUS_DIR",
                                                   "/opt/intelFPGA/22.1std/quartus/bin")}))
