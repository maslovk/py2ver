# The PyTorch MNIST network in pytorch_mnist_cnn.py, as it is, in hardware.
#
# py2ver builds the model's forward pass (DigitCNN, in eval mode) unchanged:
# nn.Sequential runs its layers, each Conv2d is im2col (its input padded and
# copied into its patches' order) and one matrix product with its filters,
# BatchNorm2d a scale and a shift per channel (from its running statistics),
# MaxPool2d a maximum over each window, Dropout nothing, Linear a matrix
# product; the 468,000 weights are ROMs, the activations block RAMs, and the
# loops a small state machine (sequencer.py). Formats are chosen from example
# test images (16 bits, rounded: a 576-term sum truncated term by term would
# lose half a bit per term).
#
# The model is pytorch_mnist_cnn.py's saved mnist_cnn.pt (run that script
# first), or, if there is none, the model trained for an epoch here as that
# script trains it. The weights are 7.5 Mbit at 16 bits, more than the
# DE0-Nano's block RAM, so this one is for simulation: the bit-exact model on
# test images, the hardware itself (Verilator) on one.
#
#   python examples/pytorch_mnist_hardware.py               # 100 test images in fixed point
#   python examples/pytorch_mnist_hardware.py --images 20   # quicker
#   python examples/pytorch_mnist_hardware.py --sim         # + one image simulated
#   python examples/pytorch_mnist_hardware.py --lanes 8     # 8 multiply-accumulates a clock
import contextlib
import io
import os
import sys
import time

import numpy as np
import torch

import pytorch_mnist_cnn as M
from py2ver import Py2ver

HERE = os.path.dirname(os.path.abspath(__file__))


def arg(name, default):
    return int(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def trained_model():
    """pytorch_mnist_cnn.py's saved model, or one trained for an epoch as it trains it."""
    saved = os.path.join(HERE, "mnist_cnn.pt")
    torch.manual_seed(0)
    model = M.DigitCNN()
    if os.path.exists(saved):
        model.load_state_dict(torch.load(saved))
        return model
    train_loader, _, _ = M.get_loaders(128, data_dir=os.path.join(HERE, "data"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-3, epochs=1, steps_per_epoch=len(train_loader))
    M.train_one_epoch(model, train_loader, optimizer, scheduler, torch.device("cpu"))
    torch.save(model.state_dict(), saved)
    return model


if __name__ == "__main__":
    model = trained_model()
    model.eval()                          # (dropout off, BatchNorm's saved statistics)
    _, _, test_set = M.get_loaders(1000, data_dir=os.path.join(HERE, "data"))
    n = arg("--images", 100)
    images = [test_set[i][0].unsqueeze(0) for i in range(n)]
    labels = np.array([test_set[i][1] for i in range(n)])
    with torch.no_grad():
        torch_digits = np.array([int(model(x).argmax(1)) for x in images])
    print(f"torch:       {np.mean(torch_digits == labels):.1%} of {n} test images")

    t = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        p = Py2ver(model, samples=images[:10], width=16, round=True,
                   multipliers=arg("--lanes", 1))
    print(f"hardware:    built in {time.time() - t:.0f} s, {p.ram_phases:,} clocks per image "
          f"({p.ram_phases / 50e6 * 1e3:.0f} ms at 50 MHz)")

    # The bit-exact model of the hardware: ten scores an image, the digit the largest
    t = time.time()
    scores = p.PY_batch()(images)
    digits = np.array([int(np.argmax(s)) for s in scores])
    print(f"fixed point: {np.mean(digits == labels):.1%}, the same digit as torch for "
          f"{np.mean(digits == torch_digits):.1%} ({time.time() - t:.0f} s)")
    with torch.no_grad():
        err = np.abs(np.array(scores) - np.array([model(x).numpy().ravel() for x in images]))
    print(f"             scores within {err.max():.3f} of torch's")

    if "--sim" in sys.argv[1:]:
        t = time.time()
        hw = p.TB_batch()(images[:1])
        print(f"simulated:   an image in {time.time() - t:.0f} s, the same as the model: "
              f"{hw == scores[:1]}")
