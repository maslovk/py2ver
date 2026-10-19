"""
mini_llama.py: a small modern language model, built like Llama.

Step up from tiny_gpt.py (the character-level GPT):

  Tokens        characters (65)            ->  BPE word pieces (1,024),
                                               tokenizer written from scratch below
  Positions     learned position table     ->  RoPE: rotary position embeddings
  Normalisation LayerNorm                  ->  RMSNorm
  MLP           Linear-GELU-Linear         ->  SwiGLU (gated, as in Llama)
  Attention     multi-head, written out    ->  grouped-query attention (GQA)
                                               using PyTorch's fused kernel
  Generation    re-reads the whole context ->  KV cache: each new token only
                for every new character        computes its own attention row
  Sampling      top-k                      ->  top-p (nucleus) sampling
  Training      float32                    ->  bfloat16 mixed precision on GPU,
                                               gradient accumulation, optional
                                               torch.compile, scaled init

These are the same building blocks used in Llama, Mistral and Qwen models,
just at about 4.7 million parameters instead of billions.

Install:  pip install torch            (everything else is in this one file)
Run:      python mini_llama.py                      train, then write a sample
          python mini_llama.py --steps 4000         train longer
          python mini_llama.py --data mytext.txt    learn your own text
          python mini_llama.py --sample-only --prompt "JULIET:"
          python mini_llama.py --sample-only --compare-cache
On a CPU it automatically uses a smaller preset so it finishes in reasonable time.
"""
import argparse
import contextlib
import json
import math
import os
import re
import time
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


SHAKESPEARE_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/"
                   "master/data/tinyshakespeare/input.txt")


# ================================================================== tokenizer
# Byte-level BPE (byte pair encoding), the same idea GPT-2/4 and Llama use:
#   1. start from raw UTF-8 bytes (256 base tokens, so any text works)
#   2. find the most frequent pair of neighbouring tokens and merge it into
#      a new token ("t" + "h" -> "th", " t" + "he" -> " the")
#   3. repeat until the vocabulary is the size you asked for
# Text is pre-split into words/numbers/punctuation so merges never cross words.

# GPT-2 style pre-split: contractions, words (with leading space), numbers,
# punctuation runs, and whitespace.
SPLIT_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+"""
)


def _merge(ids, pair, new_id):
    """Replace every occurrence of `pair` in `ids` with `new_id`."""
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self, merges):
        # merges: list of (id_a, id_b) in the order they were learned;
        # the merged token gets id 256 + position in this list
        self.merges = {tuple(p): 256 + i for i, p in enumerate(merges)}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), idx in self.merges.items():
            self.vocab[idx] = self.vocab[a] + self.vocab[b]
        self._cache = {}

    @property
    def vocab_size(self):
        return 256 + len(self.merges)

    # ------------------------------------------------------------- training
    @classmethod
    def train(cls, text, vocab_size=1024, verbose=False):
        assert vocab_size >= 256
        # Count each distinct chunk once, with its frequency: much faster
        chunk_counts = Counter(SPLIT_PATTERN.findall(text))
        words = [list(c.encode("utf-8")) for c in chunk_counts]
        freqs = list(chunk_counts.values())

        merges = []
        for step in range(vocab_size - 256):
            pair_counts = Counter()
            for ids, f in zip(words, freqs):
                for pair in zip(ids, ids[1:]):
                    pair_counts[pair] += f
            if not pair_counts:
                break                               # nothing left to merge
            best = max(pair_counts, key=pair_counts.get)
            new_id = 256 + step
            words = [_merge(ids, best, new_id) if len(ids) > 1 else ids for ids in words]
            merges.append(best)
            if verbose and (step < 10 or (step + 1) % 100 == 0):
                tok = cls(merges)
                print(f"merge {step + 1:4d}: {tok.vocab[best[0]]!r} + "
                      f"{tok.vocab[best[1]]!r} -> {tok.vocab[new_id]!r} "
                      f"({pair_counts[best]:,} times)")
        return cls(merges)

    # ------------------------------------------------------------- encoding
    def _encode_chunk(self, chunk):
        if chunk in self._cache:
            return self._cache[chunk]
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # Apply the earliest-learned merge available in this chunk
            pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = _merge(ids, pair, self.merges[pair])
        self._cache[chunk] = ids
        return ids

    def encode(self, text):
        out = []
        # (the split pattern covers every character, so nothing is ever dropped)
        for chunk in SPLIT_PATTERN.findall(text):
            out.extend(self._encode_chunk(chunk))
        return out

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def token_str(self, i):
        return self.vocab[i].decode("utf-8", errors="replace")

    # ---------------------------------------------------------- save / load
    def save(self, path):
        merges = sorted(self.merges, key=self.merges.get)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"type": "byte-bpe", "merges": [list(p) for p in merges]}, f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f)["merges"])


# ====================================================================== config
@dataclass
class Config:
    vocab_size: int = 1024
    block_size: int = 256       # context length in tokens (~600 characters)
    n_embd: int = 256
    n_head: int = 8             # query heads
    n_kv_head: int = 4          # key/value heads (GQA: 2 query heads share each)
    n_layer: int = 6
    dropout: float = 0.1
    rope_base: float = 10000.0


PRESETS = {
    "gpu": dict(block_size=256, n_embd=256, n_head=8, n_kv_head=4, n_layer=6),
    "cpu": dict(block_size=128, n_embd=192, n_head=6, n_kv_head=2, n_layer=4),
}


# ================================================================ building blocks
class RMSNorm(nn.Module):
    """Scale each vector to unit root-mean-square, then by a learned gain.
    Cheaper than LayerNorm (no mean subtraction, no bias) and works as well."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.type_as(x) * self.weight


def rope_tables(seq_len, head_dim, base=10000.0):
    """Precompute the rotation angles for every position and frequency."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(seq_len).float(), inv_freq)   # (T, head_dim/2)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """Rotary position embedding: rotate each pair of channels by an angle that
    grows with the token's position. The dot product of a rotated query and key
    then depends only on how far apart they are, which is what attention needs.
    x: (B, heads, T, head_dim); cos, sin: (T, head_dim/2)"""
    x1, x2 = x[..., 0::2].float(), x[..., 1::2].float()
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return torch.stack((y1, y2), dim=-1).flatten(-2).type_as(x)


class Attention(nn.Module):
    """Causal self-attention with grouped-query attention and a KV cache."""

    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_head % cfg.n_kv_head == 0
        self.n_head, self.n_kv = cfg.n_head, cfg.n_kv_head
        self.hs = cfg.n_embd // cfg.n_head
        self.wq = nn.Linear(cfg.n_embd, self.n_head * self.hs, bias=False)
        self.wk = nn.Linear(cfg.n_embd, self.n_kv * self.hs, bias=False)  # fewer K/V
        self.wv = nn.Linear(cfg.n_embd, self.n_kv * self.hs, bias=False)  # heads (GQA)
        self.wo = nn.Linear(self.n_head * self.hs, cfg.n_embd, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin, cache=None):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.hs).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv, self.hs).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv, self.hs).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        if cache is not None:                  # KV cache: keep keys/values of
            if "k" in cache:                   # earlier tokens instead of
                k = torch.cat([cache["k"], k], dim=2)   # recomputing them
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        past = k.size(2) - T

        # GQA: each group of query heads shares one key/value head
        rep = self.n_head // self.n_kv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

        # Fused attention kernel (FlashAttention on supported GPUs).
        # With a cache, the single new token may attend to everything before it.
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=(past == 0),
            dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(y)


class SwiGLU(nn.Module):
    """Gated feed-forward layer used by Llama: w2( silu(w1 x) * w3 x )."""

    def __init__(self, cfg):
        super().__init__()
        hidden = int(8 * cfg.n_embd / 3)
        hidden = 64 * ((hidden + 63) // 64)      # round up to a multiple of 64
        self.w1 = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w3 = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w2 = nn.Linear(hidden, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.n_embd)
        self.ff = SwiGLU(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin, cache=None):
        x = x + self.drop(self.attn(self.norm1(x), cos, sin, cache))
        x = x + self.ff(self.norm2(x))
        return x


# ======================================================================= model
class MiniLlama(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight          # weight tying

        cos, sin = rope_tables(cfg.block_size, cfg.n_embd // cfg.n_head, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Scale down layers that write into the residual stream (GPT-2 trick),
        # so the stream doesn't grow with depth
        for name, p in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, caches=None):
        B, T = idx.shape
        start = caches[0]["k"].size(2) if caches and "k" in caches[0] else 0
        assert start + T <= self.cfg.block_size, "sequence longer than block_size"
        cos = self.rope_cos[start:start + T]
        sin = self.rope_sin[start:start + T]

        x = self.drop(self.tok_emb(idx))     # no position table: RoPE handles it
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, caches[i] if caches else None)
        x = self.norm(x)

        if targets is None:                  # generating: only the last position
            return self.head(x[:, [-1], :]), None
        logits = self.head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_p=0.9, use_cache=True):
        bs = self.cfg.block_size
        out = idx[:, -bs:]
        if not use_cache:                    # slow way: re-read the whole context
            for _ in range(max_new_tokens):
                logits, _ = self(out[:, -bs:])
                out = torch.cat([out, sample_top_p(logits[:, -1], temperature, top_p)], 1)
            return out

        caches = [{} for _ in self.layers]
        logits, _ = self(out, caches=caches)            # "prefill" the prompt
        for _ in range(max_new_tokens):
            next_id = sample_top_p(logits[:, -1], temperature, top_p)
            out = torch.cat([out, next_id], dim=1)
            if caches[0]["k"].size(2) >= bs:
                # Context window full: restart the cache from the last half
                caches = [{} for _ in self.layers]
                logits, _ = self(out[:, -bs // 2:], caches=caches)
            else:
                logits, _ = self(next_id, caches=caches)  # one token at a time
        return out


def sample_top_p(logits, temperature=0.8, top_p=0.9):
    """Nucleus sampling: pick from the smallest set of tokens whose
    probabilities add up to top_p, so the cut-off adapts to how sure the
    model is."""
    if temperature <= 0:
        return logits.argmax(-1, keepdim=True)
    probs = F.softmax(logits.float() / temperature, dim=-1)
    sorted_p, sorted_i = probs.sort(dim=-1, descending=True)
    cumulative = sorted_p.cumsum(dim=-1)
    sorted_p[(cumulative - sorted_p) > top_p] = 0.0
    sorted_p /= sorted_p.sum(dim=-1, keepdim=True)
    choice = torch.multinomial(sorted_p, num_samples=1)
    return sorted_i.gather(-1, choice)


# ======================================================================== data
def load_text(path):
    if not os.path.exists(path):
        print(f"Downloading Shakespeare to {path} ...")
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)
    with open(path, encoding="utf-8") as f:
        return f.read()


def get_tokenizer(text, path, vocab_size):
    if os.path.exists(path):
        tok = BPETokenizer.load(path)
        if tok.vocab_size == vocab_size:
            return tok
    print(f"Training a {vocab_size}-token BPE tokenizer (about 30 seconds)...")
    tok = BPETokenizer.train(text, vocab_size=vocab_size)
    tok.save(path)
    return tok


class TokenData:
    def __init__(self, ids, block_size, device):
        data = torch.tensor(ids, dtype=torch.long)
        n = int(0.9 * len(data))
        self.train, self.val = data[:n], data[n:]
        self.block_size, self.device = block_size, device

    def batch(self, split, batch_size):
        d = self.train if split == "train" else self.val
        ix = torch.randint(len(d) - self.block_size - 1, (batch_size,))
        x = torch.stack([d[i:i + self.block_size] for i in ix])
        y = torch.stack([d[i + 1:i + 1 + self.block_size] for i in ix])
        if self.device.type == "cuda":
            return (x.pin_memory().to(self.device, non_blocking=True),
                    y.pin_memory().to(self.device, non_blocking=True))
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
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return max_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def estimate_loss(model, data, batch_size, amp, iters=40):
    model.eval()
    out = {}
    for split in ("train", "val"):
        total = 0.0
        for _ in range(iters):
            x, y = data.batch(split, batch_size)
            with amp:
                total += model(x, y)[1].item()
        out[split] = total / iters
    model.train()
    return out


def write(model, tok, prompt, n_tokens, device, temperature, top_p, use_cache=True):
    ids = tok.encode(prompt) or tok.encode("\n")
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, n_tokens, temperature, top_p, use_cache=use_cache)
    return tok.decode(out[0].tolist())


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


# ======================================================================== main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="shakespeare.txt")
    p.add_argument("--vocab-size", type=int, default=1024)
    p.add_argument("--preset", choices=["gpu", "cpu"], default=None,
                   help="model size (default: gpu preset on a GPU, cpu preset otherwise)")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--grad-accum", type=int, default=1,
                   help="micro-batches per optimizer step (bigger effective batch)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--compile", action="store_true", help="use torch.compile (PyTorch 2+)")
    p.add_argument("--save", default="mini_llama.pt")
    p.add_argument("--tokenizer", default="tokenizer.json")
    p.add_argument("--prompt", default="ROMEO:")
    p.add_argument("--tokens", type=int, default=250, help="tokens to generate")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--sample-only", action="store_true")
    p.add_argument("--compare-cache", action="store_true",
                   help="time generation with and without the KV cache")
    args = p.parse_args()

    torch.manual_seed(1337)
    device = pick_device()
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if use_bf16 else contextlib.nullcontext())
    preset = args.preset or ("gpu" if device.type == "cuda" else "cpu")
    print(f"Device: {device}   preset: {preset}   bfloat16: {use_bf16}")

    text = load_text(args.data)
    tok = get_tokenizer(text, args.tokenizer, args.vocab_size)
    demo = tok.encode(args.prompt + " What light through yonder window breaks?")
    print("Example tokenisation:", " | ".join(repr(tok.token_str(i)) for i in demo))

    # ------------------------------------------------------- sample from a checkpoint
    if args.sample_only:
        ckpt = torch.load(args.save, map_location=device)
        cfg = Config(**ckpt["cfg"])
        model = MiniLlama(cfg).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        if args.compare_cache:
            for use_cache in (False, True):
                torch.manual_seed(0)
                sync(device)
                t = time.time()
                write(model, tok, args.prompt, args.tokens, device,
                      args.temperature, args.top_p, use_cache)
                sync(device)
                rate = args.tokens / (time.time() - t)
                print(f"KV cache {'on ' if use_cache else 'off'}: {rate:,.0f} tokens/s")
        torch.manual_seed(0)
        print("\n" + write(model, tok, args.prompt, args.tokens, device,
                           args.temperature, args.top_p))
        return

    # ------------------------------------------------------------------- training
    ids = tok.encode(text)
    print(f"Text: {len(text):,} characters -> {len(ids):,} tokens "
          f"({len(text) / len(ids):.2f} characters per token)")

    cfg = Config(vocab_size=tok.vocab_size, **PRESETS[preset])
    steps = args.steps or (3000 if preset == "gpu" else 1500)
    data = TokenData(ids, cfg.block_size, device)

    model = MiniLlama(cfg).to(device)
    raw_model = model                                   # uncompiled, for saving
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {cfg.n_layer} layers, {cfg.n_head} query / {cfg.n_kv_head} KV heads, "
          f"{cfg.n_embd}-dim, context {cfg.block_size} tokens, {n_params:,} parameters\n")
    if args.compile:
        model = torch.compile(model)

    decay = [p for p in raw_model.parameters() if p.dim() >= 2]
    no_decay = [p for p in raw_model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), fused=device.type == "cuda")

    tokens_per_step = args.batch_size * args.grad_accum * cfg.block_size
    best_val, t0 = float("inf"), time.time()
    for step in range(steps + 1):
        if step % args.eval_every == 0 or step == steps:
            losses = estimate_loss(model, data, args.batch_size, amp)
            elapsed = time.time() - t0
            speed = step * tokens_per_step / elapsed if step else 0
            print(f"step {step:5d}  train {losses['train']:.3f}  val {losses['val']:.3f}  "
                  f"{speed:,.0f} tokens/s  ({elapsed:.0f}s)")
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save({"model": raw_model.state_dict(), "cfg": asdict(cfg)}, args.save)
            if step == steps:
                break

        for g in optimizer.param_groups:
            g["lr"] = lr_at(step, args.lr, warmup=200, total=steps)

        for _ in range(args.grad_accum):              # gradient accumulation
            x, y = data.batch("train", args.batch_size)
            with amp:
                _, loss = model(x, y)
            (loss / args.grad_accum).backward()
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    print(f"\nBest validation loss {best_val:.3f} (saved to {args.save})")
    ckpt = torch.load(args.save, map_location=device)
    raw_model.load_state_dict(ckpt["model"])
    raw_model.eval()
    torch.manual_seed(0)
    print("\n--- Sample ---")
    print(write(raw_model, tok, args.prompt, args.tokens, device,
                args.temperature, args.top_p))


if __name__ == "__main__":
    main()
