# The diffusion model in diffusion_digits.py, as it is, in hardware.
#
# py2ver builds the U-Net's forward pass (UNet.forward, in eval mode: the noise
# it predicts in a 32x32 image at a timestep, for a digit or none) unchanged:
# the timestep embedding's sines and cosines one-period tables (an argument up
# to 1,000 radians indexed by its phase), the time MLP and the label embedding,
# the convolutions im2col and a matrix product (the 3x3 ones padded, the
# downsampling one strided), GroupNorm each group's mean and variance, FiLM's
# scale and shift a chunk of a linear layer, self-attention at 8x8 through
# F.scaled_dot_product_attention, nearest-neighbour upsampling a view, the skip
# connections a Python list, SiLU a table. The weights are ROMs, the loops a
# state machine (sequencer.py). Formats are chosen from inputs the network sees
# while drawing (DDIM steps of a few digits, with and without their label: 20
# bits, rounded, room for values twice as large: margin=2). The noise it
# predicts (to about 4) is within 1.03 of torch's at 16 bits (the drawing lost),
# 0.08 at 18, 0.02 at 20, 0.004 at 24.
#
# Drawing (Diffusion.sample: DDIM with classifier-free guidance) stays Python:
# each step asks the hardware's bit-exact model for the noise twice (with the
# digit's label and without), as diffusion_digits.py asks torch.
#
# The model is diffusion_digits.py's saved diffusion_digits.pt (run that script
# first: its cpu preset, 3 epochs, about 50 minutes on a CPU), or, if there is
# none, the model trained here for one epoch by its own main().
#
#   python examples/diffusion_hardware.py               # build, compare with torch, draw a 7
#   python examples/diffusion_hardware.py --digit 3     # another digit
#   python examples/diffusion_hardware.py --steps 20    # more DDIM steps (two calls each)
#   python examples/diffusion_hardware.py --sdram       # built for the board (weights in the SDRAM)
#
# With --sdram its 422 loop nests are 540 steps of 59 templates (steps.py), its
# products on a few multipliers, and the board's whole project fits the EP4CE22
# and meets 50 MHz: 20,531 logic elements (92 %), 51.8 MHz; 1,127,273,570 clocks
# a call (22.5 s at 50 MHz).
import contextlib
import io
import os
import sys
import time

import numpy as np
import torch

import diffusion_digits as D
from py2ver import Py2ver

HERE = os.path.dirname(os.path.abspath(__file__))


def arg(name, default):
    return int(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def trained():
    """diffusion_digits.py's saved model (its EMA weights), trained for an epoch by
    its main() if there is none."""
    saved = os.path.join(HERE, "diffusion_digits.pt")
    if not os.path.exists(saved):
        argv, cwd = sys.argv, os.getcwd()
        try:
            os.chdir(HERE)
            sys.argv = ["diffusion_digits.py", "--preset", "cpu", "--epochs", "1"]
            with contextlib.redirect_stdout(io.StringIO()):
                D.main()
        finally:
            sys.argv = argv
            os.chdir(cwd)
    ckpt = torch.load(saved, map_location="cpu")
    model = D.UNet(**ckpt["arch"])
    model.load_state_dict(ckpt["ema"])
    model.eval()
    return model


def ddim(eps_of, digit, steps, guidance=3.0, seed=0):
    """Diffusion.sample's DDIM (eta 0) for one digit, the noise from eps_of(x, t, y)
    with the label and without it; also every input the network was given."""
    diff = D.Diffusion()
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 1, D.IMG, D.IMG, generator=g)
    ts = torch.linspace(diff.T - 1, 0, steps).long()
    seen = []
    for i, t in enumerate(ts):
        tt = t.reshape(1)
        calls = [(x, tt, torch.tensor([digit])), (x, tt, torch.tensor([D.N_CLASSES]))]
        seen += calls
        eps_c, eps_u = (eps_of(*c) for c in calls)
        eps = eps_u + guidance * (eps_c - eps_u)
        ab = diff.alpha_bar[t]
        ab_prev = diff.alpha_bar[ts[i + 1]] if i + 1 < steps else torch.tensor(1.0)
        x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1)
        eps = (x - ab.sqrt() * x0) / (1 - ab).sqrt()
        x = ab_prev.sqrt() * x0 + (1 - ab_prev).clamp(min=0).sqrt() * eps
    return x.clamp(-1, 1), seen


if __name__ == "__main__":
    model = trained()

    def torch_eps(x, t, y):
        with torch.no_grad():
            return model(x, t, y)
    # Inputs the network sees while drawing: formats from 12, tested on 6
    seen = []
    for digit in (0, 3, 7):
        seen += ddim(torch_eps, digit, steps=6, seed=digit)[1]
    samples, tests = seen[::3][:12], seen[1::6][:6]

    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        p = Py2ver(model, samples=samples, width=20, round=True, margin=2,
                   attr={"np.exp": {"bits": 14}, "__py2ver_silu": {"bits": 14}},
                   **({"sdram": True} if "--sdram" in sys.argv[1:] else {}))
    print(f"hardware: built in {time.time() - t0:.0f} s, {p.ram_phases:,} clocks per call "
          f"({p.ram_phases / 50e6:.1f} s at 50 MHz)")

    # The bit-exact model against torch: the predicted noise
    t0 = time.time()
    got = np.array(p.PY_batch()(tests))
    want = np.array([torch_eps(*c).numpy().ravel() for c in tests])
    print(f"fixed point: {len(tests)} calls in {time.time() - t0:.0f} s; the noise within "
          f"{np.abs(got - want).max():.3f} of torch's (rms {np.sqrt(np.mean((got - want) ** 2)):.4f}, "
          f"values to {np.abs(want).max():.1f})")

    # A digit drawn with the hardware's noise, beside torch's from the same start
    digit, steps = arg("--digit", 7), arg("--steps", 10)

    def hw_eps(x, t, y):
        return torch.tensor(p.PY_batch()([(x, t, y)])[0], dtype=torch.float32).reshape(x.shape)
    t0 = time.time()
    hw, _ = ddim(hw_eps, digit, steps)
    ref, _ = ddim(torch_eps, digit, steps)
    print(f"\nA {digit} drawn in {steps} steps ({time.time() - t0:.0f} s), hardware | torch "
          f"(pixels within {float((hw - ref).abs().max()):.2f}):")
    for a, b in zip(D.ascii_digit(hw[0]).splitlines(), D.ascii_digit(ref[0]).splitlines()):
        print(f"{a}  |  {b}")
