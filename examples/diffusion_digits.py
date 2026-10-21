"""
diffusion_digits.py: a diffusion model that draws handwritten digits.

Step up from the earlier examples:
  CNN (pytorch_mnist_cnn.py)  looks at an image and says which digit it is
  This diffusion model        starts from pure noise and draws a digit you ask for

It is the same family of model as Stable Diffusion, Imagen and DALL-E 2,
at a small scale (28x28 MNIST digits, a few million parameters).

How diffusion works:
  Training:  take a real image, add a random amount of Gaussian noise, and
             teach the network to predict exactly which noise was added.
  Drawing:   start from pure noise and repeatedly ask the network "what noise
             is in this?", removing a little of it each step, until a clean
             image is left.

New ideas compared with the previous networks:
  * U-Net: an encoder that shrinks the image (32 -> 16 -> 8), a middle, and
    a decoder that grows it back, with skip connections between matching sizes
  * Residual blocks with GroupNorm, conditioned on the noise level and the
    digit label through a learned scale and shift (FiLM)
  * Self-attention over image positions at low resolution
  * Sinusoidal timestep embeddings (like Transformer position encodings)
  * Cosine noise schedule, noise-prediction loss
  * DDIM sampling: a good image in 50 steps instead of 1,000
  * Classifier-free guidance: trains with and without labels, then pushes
    samples towards the requested digit
  * EMA (exponential moving average) of the weights for cleaner samples

Install:  pip install torch torchvision
Run:      python diffusion_digits.py                         train, saving sample grids
          python diffusion_digits.py --epochs 30             train longer
          python diffusion_digits.py --sample-only --digit 7 draw a row of 7s
          python diffusion_digits.py --sample-only --guidance 5
Outputs:  samples/epoch_XX.png   a 10x10 grid, one row per digit 0-9
          samples/process.png    the drawing process, from noise to digit
          diffusion_digits.pt    the trained weights (EMA)
On a CPU it automatically uses a smaller model and fewer epochs.
"""
import argparse
import contextlib
import copy
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

N_CLASSES = 10          # digits 0-9; label 10 means "no label" (for guidance)
IMG = 32                # MNIST is 28x28, padded to 32x32 so it halves cleanly

PRESETS = {
    "gpu": dict(ch=64, mults=(1, 2, 2), n_res=2, epochs=20, batch_size=128),
    "cpu": dict(ch=32, mults=(1, 2, 2), n_res=1, epochs=3, batch_size=64),
}


# ============================================================== building blocks
def timestep_embedding(t, dim):
    """Turn each noise level t into a vector of sines and cosines at many
    frequencies, so the network can tell nearby noise levels apart."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
    args = t.float()[:, None] * freqs[None, :]
    return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock(nn.Module):
    """Two 3x3 convolutions with a shortcut. The noise level and digit label
    (`emb`) set a per-channel scale and shift in the middle (FiLM)."""

    def __init__(self, c_in, c_out, emb_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.emb = nn.Linear(emb_dim, 2 * c_out)
        self.norm2 = nn.GroupNorm(8, c_out)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)          # block starts as the identity,
        nn.init.zeros_(self.conv2.bias)            # which makes deep U-Nets stable
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb(F.silu(emb))[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(self.drop(F.silu(h)))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    """Self-attention across pixel positions: every position can look at every
    other one, so the network can make the whole digit consistent."""

    def __init__(self, c, heads=4):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(8, c)
        self.qkv = nn.Conv2d(c, 3 * c, 1)
        self.proj = nn.Conv2d(c, c, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x)).reshape(B, 3, self.heads, C // self.heads, H * W)
        q, k, v = (t.transpose(-1, -2) for t in qkv.unbind(1))     # (B, heads, HW, d)
        y = F.scaled_dot_product_attention(q, k, v)
        y = y.transpose(-1, -2).reshape(B, C, H, W)
        return x + self.proj(y)


class Level(nn.Module):
    """One ResBlock, optionally followed by attention."""

    def __init__(self, c_in, c_out, emb_dim, attn, dropout):
        super().__init__()
        self.res = ResBlock(c_in, c_out, emb_dim, dropout)
        self.attn = AttnBlock(c_out) if attn else nn.Identity()

    def forward(self, x, emb):
        return self.attn(self.res(x, emb))


class Downsample(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, stride=2, padding=1)       # halves H and W

    def forward(self, x, emb=None):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x, emb=None):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


# ======================================================================== U-Net
class UNet(nn.Module):
    """
    32x32 ──Level──Level──▶ skip ─────────────────────────▶ Level x3 ──▶ 32x32
      │ down                                                  ▲ up
    16x16 ──Level──Level──▶ skip ───────────────▶ Level x3 ───┘
      │ down                                        ▲ up
     8x8  ──Level+Attn x2 ─▶ skip ──▶ middle ──▶ Level+Attn x3
    """

    def __init__(self, ch=64, mults=(1, 2, 2), n_res=2, attn_res=(8,), dropout=0.1):
        super().__init__()
        emb_dim = 4 * ch
        self.ch = ch
        self.time_mlp = nn.Sequential(
            nn.Linear(ch, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.class_emb = nn.Embedding(N_CLASSES + 1, emb_dim)

        self.inp = nn.Conv2d(1, ch, 3, padding=1)

        # Encoder: record the channels of every output, for the skip connections
        self.down = nn.ModuleList()
        skip_ch, c, res = [ch], ch, IMG
        for i, m in enumerate(mults):
            for _ in range(n_res):
                self.down.append(Level(c, ch * m, emb_dim, res in attn_res, dropout))
                c = ch * m
                skip_ch.append(c)
            if i < len(mults) - 1:
                self.down.append(Downsample(c))
                skip_ch.append(c)
                res //= 2

        self.mid1 = ResBlock(c, c, emb_dim, dropout)
        self.mid_attn = AttnBlock(c)
        self.mid2 = ResBlock(c, c, emb_dim, dropout)

        # Decoder: each Level takes the current features + one skip connection
        self.up = nn.ModuleList()
        for i, m in reversed(list(enumerate(mults))):
            for _ in range(n_res + 1):
                self.up.append(Level(c + skip_ch.pop(), ch * m, emb_dim,
                                     res in attn_res, dropout))
                c = ch * m
            if i > 0:
                self.up.append(Upsample(c))
                res *= 2

        self.out = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, 1, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x, t, y):
        emb = self.time_mlp(timestep_embedding(t, self.ch)) + self.class_emb(y)

        h = self.inp(x)
        skips = [h]
        for layer in self.down:
            h = layer(h, emb)
            skips.append(h)

        h = self.mid2(self.mid_attn(self.mid1(h, emb)), emb)

        for layer in self.up:
            if isinstance(layer, Upsample):
                h = layer(h)
            else:
                h = layer(torch.cat([h, skips.pop()], dim=1), emb)
        return self.out(h)                       # predicted noise, same shape as x


# ==================================================================== diffusion
class Diffusion:
    def __init__(self, T=1000, device="cpu"):
        self.T = T
        # Cosine schedule (Nichol & Dhariwal 2021): alpha_bar[t] is how much of
        # the original image survives at step t, falling smoothly from 1 to ~0
        s = 0.008
        steps = torch.arange(T + 1, dtype=torch.float64) / T
        f = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = f / f[0]
        betas = (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(max=0.999)
        self.alpha_bar = torch.cumprod(1 - betas, dim=0).float().to(device)

    def add_noise(self, x0, t, noise):
        """Jump straight to noise level t: x_t = sqrt(ab) * x0 + sqrt(1 - ab) * noise"""
        ab = self.alpha_bar[t][:, None, None, None]
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    def loss(self, model, x0, labels, p_uncond=0.1):
        B = len(x0)
        t = torch.randint(0, self.T, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.add_noise(x0, t, noise)
        # Hide the label 10% of the time so the model also learns to draw
        # without one; classifier-free guidance needs both versions
        drop = torch.rand(B, device=x0.device) < p_uncond
        labels = torch.where(drop, torch.full_like(labels, N_CLASSES), labels)
        return F.mse_loss(model(xt, t, labels), noise)

    @torch.no_grad()
    def sample(self, model, labels, steps=50, guidance=3.0, eta=0.0, keep_every=None):
        """DDIM sampling with classifier-free guidance.
        guidance 0 = ignore the label, 1 = plain conditional, >1 = stronger.
        eta 0 = deterministic DDIM, 1 = random like the original DDPM."""
        device = labels.device
        B = len(labels)
        x = torch.randn(B, 1, IMG, IMG, device=device)
        ts = torch.linspace(self.T - 1, 0, steps, device=device).long()
        null = torch.full_like(labels, N_CLASSES)
        frames = []

        for i, t in enumerate(ts):
            tt = t.repeat(B)
            # One batched call for "with label" and "without label"
            eps_c, eps_u = model(torch.cat([x, x]), torch.cat([tt, tt]),
                                 torch.cat([labels, null])).float().chunk(2)
            eps = eps_u + guidance * (eps_c - eps_u)

            ab = self.alpha_bar[t]
            ab_prev = self.alpha_bar[ts[i + 1]] if i + 1 < steps else torch.tensor(1.0, device=device)
            x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1)   # current guess
            eps = (x - ab.sqrt() * x0) / (1 - ab).sqrt()                  # consistent noise
            sigma = eta * ((1 - ab_prev) / (1 - ab) * (1 - ab / ab_prev)).sqrt()
            x = ab_prev.sqrt() * x0 + (1 - ab_prev - sigma ** 2).clamp(min=0).sqrt() * eps
            if eta > 0 and i + 1 < steps:
                x = x + sigma * torch.randn_like(x)

            if keep_every and (i % keep_every == 0 or i == steps - 1):
                frames.append(x.clamp(-1, 1))
        return (x.clamp(-1, 1), frames) if keep_every else x.clamp(-1, 1)


# ====================================================================== helpers
def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def ema_update(ema, model, decay):
    for pe, p in zip(ema.parameters(), model.parameters()):
        pe.lerp_(p, 1 - decay)


def to_image(x):
    return (x + 1) / 2                        # [-1, 1] -> [0, 1] for saving


def ascii_digit(img):
    """Draw one 32x32 sample as 16 lines of text."""
    shades = " .:-=+*#%@"
    small = to_image(img[0, ::2, ::2]).clamp(0, 1)
    return "\n".join("".join(shades[int(v * 9)] * 2 for v in row) for row in small.tolist())


def save_grids(diff, model, device, path, steps, guidance, eta, digit=None, n_per_row=10):
    rows = [digit] if digit is not None else list(range(10))
    labels = torch.tensor([d for d in rows for _ in range(n_per_row)], device=device)
    imgs = diff.sample(model, labels, steps=steps, guidance=guidance, eta=eta)
    save_image(to_image(imgs), path, nrow=n_per_row, padding=2)
    return imgs


def save_process(diff, model, device, path, steps, guidance):
    """One row per digit, columns go from pure noise to the finished digit."""
    labels = torch.arange(10, device=device)
    _, frames = diff.sample(model, labels, steps=steps, guidance=guidance,
                            keep_every=max(1, steps // 9))
    grid = torch.stack(frames, dim=1).reshape(-1, 1, IMG, IMG)      # digit-major order
    save_image(to_image(grid), path, nrow=len(frames), padding=2)


# ========================================================================= main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=["gpu", "cpu"], default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--sample-steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=3.0)
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--save", default="diffusion_digits.pt")
    p.add_argument("--out", default="samples")
    p.add_argument("--sample-only", action="store_true")
    p.add_argument("--digit", type=int, default=None, choices=range(10))
    args = p.parse_args()

    torch.manual_seed(0)
    device = pick_device()
    os.makedirs(args.out, exist_ok=True)

    # ------------------------------------------------- draw from a saved model
    if args.sample_only:
        ckpt = torch.load(args.save, map_location=device)
        model = UNet(**ckpt["arch"]).to(device).eval()
        model.load_state_dict(ckpt["ema"])
        diff = Diffusion(device=device)
        path = os.path.join(args.out, "generated.png")
        imgs = save_grids(diff, model, device, path, args.sample_steps,
                          args.guidance, args.eta, digit=args.digit)
        save_process(diff, model, device, os.path.join(args.out, "process.png"),
                     args.sample_steps, args.guidance)
        print(ascii_digit(imgs[0]))
        print(f"\nSaved {path} and {os.path.join(args.out, 'process.png')}")
        return

    # ---------------------------------------------------------------- training
    preset = PRESETS[args.preset or ("gpu" if device.type == "cuda" else "cpu")]
    epochs = args.epochs or preset["epochs"]
    batch_size = args.batch_size or preset["batch_size"]
    arch = dict(ch=preset["ch"], mults=preset["mults"], n_res=preset["n_res"])
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp = (torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16
           else contextlib.nullcontext())

    tf = transforms.Compose([
        transforms.Pad(2),                       # 28x28 -> 32x32
        transforms.ToTensor(),                   # 0..1
        transforms.Normalize((0.5,), (0.5,)),    # -1..1, matching the noise scale
    ])
    train_set = datasets.MNIST("./data", train=True, download=True, transform=tf)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True,
                        num_workers=2, pin_memory=device.type == "cuda")

    model = UNet(**arch).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    diff = Diffusion(device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}   bfloat16: {use_bf16}")
    print(f"U-Net with {n_params:,} parameters, {epochs} epochs, batch {batch_size}\n")

    step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        for x0, labels in loader:
            x0, labels = x0.to(device), labels.to(device)
            lr = args.lr * min(1.0, (step + 1) / 500)          # short warmup
            for g in opt.param_groups:
                g["lr"] = lr
            with amp:
                loss = diff.loss(model, x0, labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema_update(ema, model, args.ema if step > 1000 else 0.0)   # copy early on
            total += loss.item()
            step += 1

        print(f"epoch {epoch:3d}   loss {total / len(loader):.4f}   ({time.time() - t0:.0f}s)")
        path = os.path.join(args.out, f"epoch_{epoch:02d}.png")
        save_grids(diff, ema, device, path, args.sample_steps, args.guidance, args.eta)
        torch.save({"ema": ema.state_dict(), "arch": arch}, args.save)

    save_process(diff, ema, device, os.path.join(args.out, "process.png"),
                 args.sample_steps, args.guidance)
    labels = torch.tensor([7], device=device)
    print("\nA freshly drawn 7:\n" + ascii_digit(diff.sample(ema, labels)[0]))
    print(f"\nSample grids in {args.out}/, weights in {args.save}")


if __name__ == "__main__":       # needed for DataLoader workers on Windows/macOS
    main()
