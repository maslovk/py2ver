"""
A tiny GPT: a character-level Transformer language model in PyTorch.

Step up from the previous examples:
  1. XOR network      dense layers, 17 parameters
  2. Digit MLP        dense layers, 17k parameters
  3. CNN              convolutions see local patches of an image
  4. This Transformer self-attention lets every character look back at
                      every earlier character, and the model learns which
                      ones matter. It is the same architecture as GPT,
                      just much smaller (~1.8M parameters).

What it does: reads a text (the complete works of Shakespeare by default,
about 1 MB, public domain), learns to predict the next character, then
writes new text one character at a time.

New ideas compared with the CNN:
  * Embeddings: each character and each position becomes a learned vector
  * Causal self-attention with several heads (written out step by step)
  * Transformer blocks: LayerNorm, attention, MLP, residual connections
  * Autoregressive generation with temperature and top-k sampling
  * Training extras: learning-rate warmup + cosine decay, gradient
    clipping, weight decay only on weight matrices, a validation split

Install:  pip install torch
Run:      python tiny_gpt.py                     (train, then write a sample)
          python tiny_gpt.py --steps 5000        (train longer, better text)
          python tiny_gpt.py --data mytext.txt   (learn your own text file)
          python tiny_gpt.py --sample-only --prompt "ROMEO:"
"""
import argparse
import math
import os
import time
import urllib.request
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

SHAKESPEARE_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/"
                   "master/data/tinyshakespeare/input.txt")


# ====================================================================== config
@dataclass
class Config:
    vocab_size: int = 65        # number of distinct characters (set from data)
    block_size: int = 128       # how many characters the model can look back at
    n_embd: int = 192           # size of each character's vector
    n_head: int = 6             # attention heads per layer
    n_layer: int = 4            # number of Transformer blocks
    dropout: float = 0.1


# ======================================================================= model
class CausalSelfAttention(nn.Module):
    """Each position mixes in information from itself and earlier positions."""

    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)   # queries, keys, values
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        # Lower-triangular mask: position t may only look at positions <= t
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size),
                             persistent=False)

    def forward(self, x):
        B, T, C = x.shape                   # batch, time (characters), channels
        hs = C // self.n_head               # size of each head
        q, k, v = self.qkv(x).split(C, dim=2)
        # Split channels into heads: (B, T, C) -> (B, n_head, T, hs)
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)

        # How much should each position attend to each earlier one?
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hs)          # (B, h, T, T)
        att = att.masked_fill(~self.mask[:, :, :T, :T], float("-inf"))
        att = F.softmax(att, dim=-1)                             # rows sum to 1
        att = self.attn_drop(att)

        y = att @ v                                              # weighted mix of values
        y = y.transpose(1, 2).contiguous().view(B, T, C)         # merge heads again
        return self.resid_drop(self.proj(y))
        # (PyTorch 2 can do all of the above in one fused, faster call:
        #  F.scaled_dot_product_attention(q, k, v, is_causal=True))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """One Transformer block: communicate (attention), then compute (MLP)."""

    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))      # residual connections let gradients
        x = x + self.mlp(self.ln2(x))       # flow straight through deep stacks
        return x


class TinyGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)   # what character
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)   # where it is
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight    # weight tying: share the matrix
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))     # (B, T, C)
        x = self.blocks(x)
        logits = self.head(self.ln_f(x))                         # (B, T, vocab)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=20):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]             # crop to context size
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature              # last position only
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")      # keep top k choices
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)    # sample one character
            idx = torch.cat([idx, next_id], dim=1)
        return idx


# ======================================================================== data
def load_text(path):
    if not os.path.exists(path):
        print(f"Downloading Shakespeare to {path} ...")
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)
    with open(path, encoding="utf-8") as f:
        return f.read()


class CharData:
    def __init__(self, text, block_size, device):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        data = torch.tensor(self.encode(text), dtype=torch.long)
        n = int(0.9 * len(data))
        self.train, self.val = data[:n], data[n:]    # last 10% held out
        self.block_size, self.device = block_size, device

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids)

    def batch(self, split, batch_size):
        d = self.train if split == "train" else self.val
        ix = torch.randint(len(d) - self.block_size - 1, (batch_size,))
        x = torch.stack([d[i:i + self.block_size] for i in ix])
        y = torch.stack([d[i + 1:i + 1 + self.block_size] for i in ix])  # shifted by 1
        return x.to(self.device), y.to(self.device)


# ===================================================================== helpers
def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def lr_at(step, max_lr, warmup, total):
    if step < warmup:                                        # linear warmup
        return max_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)      # cosine decay to 10%
    return max_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def estimate_loss(model, data, batch_size, iters=50):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = [model(*data.batch(split, batch_size))[1].item() for _ in range(iters)]
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


def sample(model, data, prompt, n_chars, device, temperature, top_k):
    ids = data.encode(prompt) or data.encode("\n")
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, n_chars, temperature=temperature, top_k=top_k)
    return data.decode(out[0].tolist())


# ======================================================================== main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="shakespeare.txt")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--save", default="tiny_gpt.pt")
    p.add_argument("--prompt", default="ROMEO:")
    p.add_argument("--chars", type=int, default=600, help="characters to generate")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--sample-only", action="store_true",
                   help="skip training and generate from a saved model")
    args = p.parse_args()

    torch.manual_seed(1337)
    device = pick_device()
    print(f"Using device: {device}")

    text = load_text(args.data)
    cfg = Config()
    data = CharData(text, cfg.block_size, device)
    cfg.vocab_size = len(data.chars)
    print(f"Text: {len(text):,} characters, {cfg.vocab_size} distinct")

    if args.sample_only:
        ckpt = torch.load(args.save, map_location=device)
        cfg = Config(**ckpt["cfg"])
        model = TinyGPT(cfg).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        print(sample(model, data, args.prompt, args.chars, device,
                     args.temperature, args.top_k))
        return

    model = TinyGPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {cfg.n_layer} layers, {cfg.n_head} heads, {cfg.n_embd}-dim, "
          f"{n_params:,} parameters\n")

    model.eval()
    print("--- Sample BEFORE training (random weights) ---")
    print(sample(model, data, args.prompt, 200, device, 1.0, None), "\n")
    model.train()

    # Weight decay on weight matrices only, not on biases or LayerNorm
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.99))

    best_val, start = float("inf"), time.time()
    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            losses = estimate_loss(model, data, args.batch_size)
            print(f"step {step:5d}  train loss {losses['train']:.3f}  "
                  f"val loss {losses['val']:.3f}  ({time.time() - start:.0f}s)")
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save({"model": model.state_dict(), "cfg": asdict(cfg)}, args.save)
            if step == args.steps:
                break

        for g in optimizer.param_groups:
            g["lr"] = lr_at(step, args.lr, warmup=100, total=args.steps)

        x, y = data.batch("train", args.batch_size)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # avoid huge steps
        optimizer.step()

    print(f"\nBest validation loss {best_val:.3f}, saved to {args.save}")

    ckpt = torch.load(args.save, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print("\n--- Sample AFTER training ---")
    print(sample(model, data, args.prompt, args.chars, device,
                 args.temperature, args.top_k))


if __name__ == "__main__":
    main()
