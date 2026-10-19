# The tiny GPT in tiny_gpt.py, as it is, in hardware.
#
# py2ver builds the model's forward pass (TinyGPT.forward, in eval mode)
# unchanged: the token and position embeddings are gathers from their tables,
# each Transformer block's LayerNorm the mean and variance of each position's
# vector, the causal self-attention batched matrix products (a head a batch)
# with the mask a constant (masked_fill(-inf) and the softmax: the masked
# weights 0), the MLP's GELU a table, and the residual connections sums. The
# output layer shares the token embedding's matrix (weight tying): one table.
# The 1.8 million weights are ROMs, the loops a small state machine
# (sequencer.py); the four blocks, the same code with their own weights, are
# one loop over them (repeats.py). Formats are chosen from 12 windows of the
# text (16 bits, rounded, room for values twice as large: margin=2), the exp
# and GELU tables 16,384 entries each (attention's scores spread over 50 units).
#
# The model is tiny_gpt.py's saved tiny_gpt.pt (run that script first, e.g.
# python tiny_gpt.py --steps 300), or, if there is none, the model trained for
# 300 steps here by its own main(). The weights are 29 Mbit at 16 bits, more
# than the DE0-Nano's block RAM: with --sdram they are in its SDRAM (and the
# window and the scores too), and the design fits the board (p.SYN(...),
# p.HW(...)). The bit-exact model gives the next character's scores for a
# 128-character window, and writes text with them one character at a time,
# as TinyGPT.generate does with torch's.
#
#   python examples/tiny_gpt_hardware.py               # build, compare with torch, write 20 characters
#   python examples/tiny_gpt_hardware.py --chars 200   # more text
#   python examples/tiny_gpt_hardware.py --sdram       # built for the board (weights in the SDRAM)
#   python examples/tiny_gpt_hardware.py --sim         # + one window simulated (hours at this size)
import contextlib
import io
import os
import sys
import time

import numpy as np
import torch

import tiny_gpt as G
from py2ver import Py2ver

HERE = os.path.dirname(os.path.abspath(__file__))


def arg(name, default):
    return int(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def trained():
    """tiny_gpt.py's saved model and its text, trained for 300 steps by its main()
    if there is none."""
    saved, text_path = os.path.join(HERE, "tiny_gpt.pt"), os.path.join(HERE, "shakespeare.txt")
    if not os.path.exists(saved):
        argv, cwd = sys.argv, os.getcwd()
        try:
            os.chdir(HERE)
            sys.argv = ["tiny_gpt.py", "--steps", "300", "--eval-every", "100", "--chars", "100"]
            with contextlib.redirect_stdout(io.StringIO()):
                G.main()
        finally:
            sys.argv = argv
            os.chdir(cwd)
    text = G.load_text(text_path)
    ckpt = torch.load(saved, map_location="cpu")
    cfg = G.Config(**ckpt["cfg"])
    model = G.TinyGPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, G.CharData(text, cfg.block_size, "cpu")


if __name__ == "__main__":
    model, data = trained()
    T = model.cfg.block_size
    g = torch.Generator().manual_seed(1)
    windows = [data.val[i:i + T].unsqueeze(0)                  # (validation text, 128 characters)
               for i in torch.randint(len(data.val) - T - 1, (18,), generator=g).tolist()]
    samples, windows = windows[:12], windows[12:]               # (formats from 12, tested on 6)

    t = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        lanes = arg("--lanes", 1)
        p = Py2ver(model, samples=samples, width=16, round=True, margin=2, multipliers=lanes,
                   attr={"np.exp": {"bits": 14}, "__py2ver_gelu": {"bits": 14}},
                   **({"sdram": True} if "--sdram" in sys.argv[1:] else {}))
    print(f"hardware: built in {time.time() - t:.0f} s, {p.ram_phases:,} clocks per window of {T} "
          f"characters ({p.ram_phases / 50e6:.1f} s at 50 MHz)")

    # The bit-exact model against torch: the scores of every position's next character
    t = time.time()
    scores = np.array(p.PY_batch()(windows)).reshape(len(windows), T, -1)
    with torch.no_grad():
        want = np.array([model(w)[0][0].numpy() for w in windows])
    print(f"fixed point: {len(windows)} windows in {time.time() - t:.0f} s; scores within "
          f"{np.abs(scores - want).max():.3f} of torch's, the same likeliest next character "
          f"for {np.mean(scores.argmax(-1) == want.argmax(-1)):.1%} of the positions")

    # Text written by the hardware's scores, a character at a time (temperature 0.8,
    # the 20 likeliest, as TinyGPT.generate samples)
    n = arg("--chars", 20)
    idx = data.val[:T].tolist()
    rng = np.random.default_rng(0)
    t = time.time()
    for _ in range(n):
        logits = np.array(p.PY_batch()([torch.tensor([idx[-T:]])])[0]).reshape(T, -1)[-1] / 0.8
        keep = np.argsort(logits)[-20:]
        probs = np.exp(logits[keep] - logits[keep].max())
        idx.append(int(keep[rng.choice(len(keep), p=probs / probs.sum())]))
    print(f"\n{data.decode(data.val[:T].tolist())[-60:]!r} ... and the hardware wrote "
          f"({time.time() - t:.0f} s):\n{data.decode(idx[T:])!r}")

    if "--sim" in sys.argv[1:]:
        t = time.time()
        hw = p.TB_batch()(windows[:1])
        print(f"simulated: a window in {time.time() - t:.0f} s, the same as the model: "
              f"{hw == p.PY_batch()(windows[:1])}")
