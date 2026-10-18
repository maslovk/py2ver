# The PyTorch MNIST network in pytorch_mnist_cnn.py, trained in hardware: py2ver
# builds one step of its train_one_epoch loop, unchanged.
#
# Py2ver(train_one_epoch, samples=[(model, batches, optimizer, scheduler,
# device)]) takes the function as written, with the objects it is given and
# the batches its loop goes over: the loop's body is the step (its inputs a
# batch's images and labels), what it changes is kept by the hardware from one
# step to the next, and the locals it adds up (total_loss, correct, seen) are
# the step's results:
#
#   - the forward pass in training mode: BatchNorm2d with the batch's mean and
#     variance (its running statistics updated), Dropout with numbers from a
#     generator in hardware
#   - F.cross_entropy, then loss.backward(): the gradients of every parameter
#     through the network, written out as loops (autograd.py): im2col's copy
#     added back into the padded input, the max pool's gradient to each
#     window's first maximum, as torch
#   - optimizer.step(): AdamW as torch computes it, its moments made on the
#     first step (zeros) and kept; scheduler.step(): OneCycleLR's learning
#     rate and beta1 for each step, tables read at its step count
#
# A step on MNIST is about 56 million multiply-accumulates per image, so this
# one is for the bit-exact model, on a few small batches: the model follows
# torch (given the hardware's dropout numbers, p.hardware_random()) step by
# step. The state (468,010 weights and their moments, 1.4 million numbers) is
# 34 Mbit at 24 bits.
#
#   python examples/pytorch_mnist_training.py              # 4 steps of 2 images
#   python examples/pytorch_mnist_training.py --steps 8    # more
import contextlib
import copy
import io
import itertools
import os
import sys
import time
import warnings

import numpy as np
import torch

import pytorch_mnist_cnn as M
from py2ver import Py2ver

HERE = os.path.dirname(os.path.abspath(__file__))


def arg(name, default):
    return int(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


if __name__ == "__main__":
    warnings.simplefilter("ignore")              # (torch's, about copies of its scheduler)
    n, batch = arg("--steps", 4), arg("--batch", 2)
    torch.manual_seed(0)
    train_loader, _, _ = M.get_loaders(batch, data_dir=os.path.join(HERE, "data"))
    # As pytorch_mnist_cnn.py's main() makes them, for an epoch of these batches
    model = M.DigitCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-3, epochs=1, steps_per_epoch=len(train_loader))
    batches = list(itertools.islice(iter(train_loader), n))   # (shifted, rotated images)
    device = torch.device("cpu")
    reference = copy.deepcopy((model, optimizer, scheduler))  # torch trains this one

    t = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        train = Py2ver(M.train_one_epoch, samples=[(model, batches[:2], optimizer, scheduler,
                                                    device)],
                       width=24, round=True, saturate=True, margin=2)
    print(f"built in {time.time() - t:.0f} s: {train.ram_phases:,} clocks per step of {batch} "
          f"images ({train.ram_phases / 50e6:.1f} s at 50 MHz)")

    # The bit-exact model, step by step: (total_loss, correct, seen) so far
    t = time.time()
    rows = train.PY_stream()(batches)
    print(f"trained in fixed point: {n} steps in {time.time() - t:.0f} s")
    losses = np.diff([0.0] + [r[0] for r in rows]) / batch

    # torch, from the same start, with the hardware's dropout numbers
    rm, ropt, rsched = reference
    torch_losses = []
    with train.hardware_random():
        for b in batches:
            loss, _ = M.train_one_epoch(rm, [b], ropt, rsched, device)
            torch_losses.append(loss)
    for k, (a, b) in enumerate(zip(losses, torch_losses)):
        print(f"  step {k + 1}: loss {a:.4f}   (torch {b:.4f})")
    state = train.state_values()
    w = state[(model.classifier[2], "weight")]
    print(f"Linear(3136, 128) weights after {n} steps: within "
          f"{np.abs(w - rm.classifier[2].weight.detach().numpy()).max():.2g} of torch's")
