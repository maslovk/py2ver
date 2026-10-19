# The mini Llama in mini_llama.py, as it is, in hardware.
#
# py2ver builds the model's forward pass (MiniLlama.forward, in eval mode, no KV
# cache: the next token's scores for a window) unchanged: the token embedding a
# gather from its table, each block's RMSNorm the mean square of each position's
# vector and 1 / sqrt of it, RoPE the even and odd channels rotated by constant
# tables (torch.stack, flatten), grouped-query attention's keys and values
# repeated as views (repeat_interleave) and F.scaled_dot_product_attention's
# products, scale and causal mask (a constant, the masked weights 0), SwiGLU's
# SiLU a table, and the output layer the tied embedding read at the last
# position (x[:, [-1], :]). The weights are ROMs, the loops a state machine
# (sequencer.py), the blocks one loop over them (repeats.py). Formats are chosen
# from 12 windows of the text (16 bits, rounded, room for values twice as
# large: margin=2), the exp and SiLU tables 16,384 entries each.
#
# The model is mini_llama.py's saved mini_llama.pt with its tokenizer.json (run
# python mini_llama.py --preset cpu first, e.g. with --steps 300), or, if there
# is none, the model trained here for 300 steps by its own main() (the cpu
# preset: 4 layers, 192 wide, a 128-token context, a 1,024-token vocabulary).
# With --sdram the weights are in the DE0-Nano's SDRAM. The bit-exact model
# gives the next token's scores for a window, and writes text with them a
# token at a time, as MiniLlama.generate does without its KV cache.
#
#   python examples/mini_llama_hardware.py               # build, compare with torch, write 20 tokens
#   python examples/mini_llama_hardware.py --tokens 100  # more text
#   python examples/mini_llama_hardware.py --sdram       # built for the board (weights in the SDRAM)
#   python examples/mini_llama_hardware.py --sim         # + one window simulated (hours at this size)
import contextlib
import io
import os
import sys
import time

import numpy as np
import torch

import mini_llama as L
from py2ver import Py2ver

HERE = os.path.dirname(os.path.abspath(__file__))


def arg(name, default):
    return int(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def trained():
    """mini_llama.py's saved model, its tokenizer and its text's tokens, trained
    for 300 steps (the cpu preset) by its main() if there is none."""
    saved = os.path.join(HERE, "mini_llama.pt")
    if not os.path.exists(saved):
        argv, cwd = sys.argv, os.getcwd()
        try:
            os.chdir(HERE)
            sys.argv = ["mini_llama.py", "--preset", "cpu", "--steps", "300", "--eval-every", "100",
                        "--tokens", "20"]
            with contextlib.redirect_stdout(io.StringIO()):
                L.main()
        finally:
            sys.argv = argv
            os.chdir(cwd)
    text = L.load_text(os.path.join(HERE, "shakespeare.txt"))
    tok = L.BPETokenizer.load(os.path.join(HERE, "tokenizer.json"))
    ckpt = torch.load(saved, map_location="cpu")
    model = L.MiniLlama(L.Config(**ckpt["cfg"]))
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, tok, L.TokenData(tok.encode(text), model.cfg.block_size, torch.device("cpu"))


if __name__ == "__main__":
    model, tok, data = trained()
    T = model.cfg.block_size
    g = torch.Generator().manual_seed(1)
    windows = [data.val[i:i + T].unsqueeze(0)                  # (validation text, 128 tokens)
               for i in torch.randint(len(data.val) - T - 1, (18,), generator=g).tolist()]
    samples, windows = windows[:12], windows[12:]               # (formats from 12, tested on 6)

    t = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        p = Py2ver(model, samples=samples, width=16, round=True, margin=2,
                   multipliers=arg("--lanes", 1),
                   attr={"np.exp": {"bits": 14}, "__py2ver_silu": {"bits": 14}},
                   **({"sdram": True} if "--sdram" in sys.argv[1:] else {}))
    print(f"hardware: built in {time.time() - t:.0f} s, {p.ram_phases:,} clocks per window of {T} "
          f"tokens ({p.ram_phases / 50e6:.1f} s at 50 MHz)")

    # The bit-exact model against torch: the next token's scores
    t = time.time()
    scores = np.array(p.PY_batch()(windows)).reshape(len(windows), -1)
    with torch.no_grad():
        want = np.array([model(w)[0].numpy().ravel() for w in windows])
    print(f"fixed point: {len(windows)} windows in {time.time() - t:.0f} s; scores within "
          f"{np.abs(scores - want).max():.3f} of torch's, the same likeliest next token "
          f"for {np.mean(scores.argmax(-1) == want.argmax(-1)):.0%} of the windows")

    # Text written by the hardware's scores, a token at a time (top-p sampling as
    # mini_llama.py's, temperature 0.8, p 0.9)
    n = arg("--tokens", 20)
    idx = data.val[:T].tolist()
    torch.manual_seed(0)
    t = time.time()
    for _ in range(n):
        logits = torch.tensor(p.PY_batch()([torch.tensor([idx[-T:]])])[0]).reshape(1, -1)
        idx.append(int(L.sample_top_p(logits, 0.8, 0.9)))
    print(f"\n{tok.decode(data.val[:T].tolist())[-80:]!r} ... and the hardware wrote "
          f"({time.time() - t:.0f} s):\n{tok.decode(idx[T:])!r}")

    if "--sim" in sys.argv[1:]:
        t = time.time()
        hw = p.TB_batch()(windows[:1])
        print(f"simulated: a window in {time.time() - t:.0f} s, the same as the model: "
              f"{hw == p.PY_batch()(windows[:1])}")
