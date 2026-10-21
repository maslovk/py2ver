import ast
import contextlib
import copy
import io
import re
import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn                     # noqa: E402
import torch.nn.functional as F           # noqa: E402

from py2ver import Py2ver, numpy_loops    # noqa: E402

# PyTorch models and training loops, as written: modules, autograd, optimizers.


class Small(nn.Module):
    def __init__(self, c=3, side=6, n=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.25),
                                        nn.Linear(c * (side // 2) ** 2, n))

    def forward(self, x):
        return self.classifier(self.features(x))


def _model(seed=0, **kw):
    torch.manual_seed(seed)
    m = Small(**kw)
    m.train()
    with torch.no_grad():                         # (BatchNorm's running statistics)
        for _ in range(10):
            m(torch.randn(8, 1, 6, 6) + 0.3)
    return m


def _batches(n, size=4, seed=1):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        x = torch.randn(size, 1, 6, 6, generator=g)
        y = (x[:, 0, :3].sum((1, 2)) > 0).long() + (x[:, 0, :, :3].sum((1, 2)) > 0).long()
        out.append((x, y))
    return out


def train_epoch(model, loader, optimizer, scheduler, device):
    """(the shape of a usual PyTorch training loop)"""
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        seen += len(labels)
    return total_loss / seen, correct / seen


class Ops(nn.Module):
    """torch's tensor methods and functions with their own meanings."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(6, 4)

    def forward(self, x):
        h = x.view(x.size(0), -1)                         # (2, 6)
        z = self.lin(h).transpose(0, 1).permute(1, 0)     # (2, 4)
        p = F.softmax(z, dim=1) + F.log_softmax(z, 1).exp()
        best, where = z.max(dim=1)
        return torch.cat([p, best.unsqueeze(1), where.unsqueeze(1).float()], dim=1)


def test_a_torch_model_is_built_as_written():
    m = _model()
    m.eval()                                              # (BatchNorm: its running statistics)
    xs = [torch.randn(1, 1, 6, 6) for _ in range(6)]
    tr = numpy_loops.translate(m, xs[:3])
    with torch.no_grad():
        for x in xs:
            assert np.allclose(tr.func(*tr.adapt(x)), m(x).numpy().ravel(), atol=1e-4)
    assert not tr.memories                                # (eval: nothing changes)


def test_torch_s_own_meanings_of_methods():
    torch.manual_seed(3)
    m = Ops()
    xs = [torch.randn(2, 2, 3) for _ in range(4)]
    tr = numpy_loops.translate(m, xs[:2])
    with torch.no_grad():
        for x in xs:
            assert np.allclose(tr.func(*tr.adapt(x)), m(x).numpy().ravel(), atol=1e-4)


def _training(n=8, lr=0.05):
    m = _model()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, epochs=1, steps_per_epoch=n)
    return m, opt, sched


def test_a_training_loop_is_one_step_of_its_loop():
    m, opt, sched = _training()
    batches = _batches(8)
    before = m.features[0].weight.detach().clone()
    # (checked against torch's own training on two batches when translated)
    tr = numpy_loops.translate(train_epoch, [(m, batches, opt, sched, torch.device("cpu"))])
    assert torch.equal(m.features[0].weight, before) and not opt.state     # (left as they are)
    assert tr.params == ["images", "labels"] and len(tr.steps) == 8
    labels = {label for label, _, _ in tr.memories.values()}
    assert {"Conv2d.weight", "BatchNorm2d.running_mean", "BatchNorm2d.num_batches_tracked",
            "Linear.bias", "OneCycleLR.last_epoch", "optimizer.param_groups[0]['lr']",
            "optimizer.param_groups[0]['betas'][0]", "train_epoch.total_loss",
            "train_epoch.correct", "train_epoch.seen",
            "optimizer.state[model.features[0].weight]['exp_avg']"} <= labels
    # Adam's moments start at zero; the step counts are integers
    for name, (label, value, _) in tr.memories.items():
        if label.endswith("['exp_avg_sq']"):
            assert set(tr.attr[name]["init"]) == {0.0}
        if label.endswith("['step']"):
            assert tr.attr[name]["width"] == 32


@pytest.mark.sim
def test_a_torch_model_in_hardware(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    m = _model()
    m.eval()
    xs = [torch.randn(1, 1, 6, 6) for _ in range(30)]
    with contextlib.redirect_stdout(io.StringIO()):
        p = Py2ver(m, samples=xs, width=16, round=True)
    model = p.PY_batch()(xs)
    with torch.no_grad():
        want = [m(x).numpy().ravel() for x in xs]
    assert np.allclose(model, want, atol=0.02)
    assert p.TB_batch()(xs[:3]) == model[:3]


def _trainer(n=12):
    m, opt, sched = _training(n=n)
    batches = _batches(n)
    twin = copy.deepcopy((m, opt, sched))
    with contextlib.redirect_stdout(io.StringIO()):
        p = Py2ver(train_epoch, samples=[(m, batches, opt, sched, torch.device("cpu"))],
                   width=24, round=True, saturate=True, margin=2)
    return p, m, batches, twin


@pytest.mark.sim
def test_a_torch_training_loop_in_fixed_point(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    p, m, batches, twin = _trainer()
    rows = p.PY_stream()(batches)                 # (total_loss, correct, seen) after each step
    with p.hardware_random(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loss, acc = train_epoch(*twin[:1], batches, *twin[1:], torch.device("cpu"))
    total, correct, seen = rows[-1]
    assert seen == 48 and abs(total / seen - loss) < 0.01 and abs(correct / seen - acc) < 0.05
    state = p.state_values()
    assert np.allclose(state[(m.classifier[2], "weight")],
                       twin[0].classifier[2].weight.detach().numpy(), atol=0.01)


@pytest.mark.sim
@pytest.mark.slow
def test_a_torch_training_step_in_hardware(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    p, _, batches, _ = _trainer(n=4)
    assert p.TB_stream()(batches[:1]) == p.PY_stream()(batches[:1])


def test_max_pooling_s_gradient_goes_to_the_first_maximum():
    # Images on a flat background: windows of equal values after BatchNorm and ReLU,
    # whose gradient torch gives to the first only (checked against torch's training)
    m, opt, sched = _training()
    g = torch.Generator().manual_seed(5)
    batches = []
    for _ in range(4):
        x = torch.zeros(4, 1, 6, 6)
        x[:, :, 2:4, 1:4] = torch.randn(4, 1, 2, 3, generator=g)
        batches.append((x, torch.randint(0, 3, (4,), generator=g)))
    numpy_loops.translate(train_epoch, [(m, batches, opt, sched, torch.device("cpu"))])


# ----------------------------- a tiny GPT: embeddings, attention, LayerNorm, GELU -----------------------------

def _gpt(seed=0):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
    import tiny_gpt as G
    torch.manual_seed(seed)
    cfg = G.Config(vocab_size=11, block_size=8, n_embd=12, n_head=2, n_layer=2)
    return G, G.TinyGPT(cfg)


def gpt_steps(model, loader, optimizer):
    """(a training loop of the usual shape, with gradient clipping)"""
    model.train()
    total = 0.0
    for x, y in loader:
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total


def test_a_tiny_gpt_s_forward_pass_is_built_as_written():
    G, m = _gpt()
    m.eval()
    xs = [torch.randint(0, 11, (1, 8)) for _ in range(4)]
    tr = numpy_loops.translate(m, xs[:2])          # (logits; the loss None: no result)
    # (the two blocks' code is one loop over them: their weights one table each)
    assert "for rep0 in range(2):" in tr.source and "T_Linear_weight_7_all0[rep0 * 432" in tr.source
    with torch.no_grad():
        for x in xs:
            assert np.allclose(tr.func(*tr.adapt(x)), m(x)[0].numpy().ravel(), atol=1e-4)


def test_a_tiny_gpt_trains_as_torch_does():
    G, m = _gpt()
    decay = [p for p in m.parameters() if p.dim() >= 2]
    no_decay = [p for p in m.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                             {"params": no_decay, "weight_decay": 0.0}], lr=1e-2, betas=(0.9, 0.99))
    g = torch.Generator().manual_seed(2)
    batches = [(torch.randint(0, 11, (2, 8), generator=g), torch.randint(0, 11, (2, 8), generator=g))
               for _ in range(3)]
    # (checked against torch's training of a copy: dropout, attention, the tied
    # embedding, clipping, AdamW's two groups)
    tr = numpy_loops.translate(gpt_steps, [(m, batches, opt)])
    labels = {label for label, _, _ in tr.memories.values()}
    assert "Embedding.weight" in labels and "LayerNorm.weight" in labels
    assert not any(label.startswith("Linear.weight") and "head" in label for label in labels)


@pytest.mark.sim
@pytest.mark.slow
def test_a_tiny_gpt_in_hardware(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    G, m = _gpt()
    m.eval()
    xs = [torch.randint(0, 11, (1, 8)) for _ in range(20)]
    with contextlib.redirect_stdout(io.StringIO()):
        p = Py2ver(m, samples=xs, width=16, round=True)
    model = np.array(p.PY_batch()(xs))
    with torch.no_grad():
        want = np.array([m(x)[0].numpy().ravel() for x in xs])
    assert np.abs(model - want).max() < 0.01
    assert p.TB_batch()(xs[:2]) == p.PY_batch()(xs[:2])


# ----------------------------- a mini Llama: RMSNorm, RoPE, GQA, SwiGLU -----------------------------

class LlamaOps(nn.Module):
    """The operations a Llama adds, as torch means them."""

    def __init__(self):
        super().__init__()
        self.act = nn.SiLU()

    def forward(self, x, cache=None):                     # x: (1, 2, 4, 6)
        n = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)          # RMSNorm
        r = torch.stack((x[..., 0::2], -x[..., 1::2]), dim=-1).flatten(-2)   # RoPE's pairs
        k = x.repeat_interleave(2, dim=1)                                    # GQA: (1, 4, 4, 6)
        q = torch.cat([x, x * 0.5], dim=1)
        causal = F.scaled_dot_product_attention(q, k, k, is_causal=True)
        full = F.scaled_dot_product_attention(q, k, k, scale=0.3)
        last = x[:, :, [-1], :]                                              # (a slice)
        picked = x[:, :, [3, 0], :]                                          # (a gather)
        start = cache[0] if cache and cache[0] else 0                        # (Python's and)
        assert start == 0, "no cache"
        return (n.type_as(x), r, F.silu(x) + self.act(x), torch.pow(x, 3), x.pow(0.5 * 2),
                causal, full, last, picked)


def test_a_llama_s_operations_are_torch_s():
    torch.manual_seed(5)
    m = LlamaOps()
    xs = [torch.randn(1, 2, 4, 6) for _ in range(4)]
    tr = numpy_loops.translate(m, xs[:2])
    with torch.no_grad():
        for x in xs:
            got = np.concatenate([np.ravel(v) for v in tr.func(*tr.adapt(x))])
            want = np.concatenate([t.numpy().ravel() for t in m(x)])
            assert got.shape == want.shape and np.allclose(got, want, atol=1e-4)


def test_an_assert_is_checked_when_translating():
    class Checked(nn.Module):
        def forward(self, x):
            assert x.shape[-1] == 3, "three wide"
            return x * 2
    numpy_loops.translate(Checked(), [torch.randn(1, 3)])
    with pytest.raises(AssertionError, match="three wide"):
        numpy_loops.translate(Checked(), [torch.randn(1, 4)])


def _llama(seed=0, **size):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
    import mini_llama as L
    torch.manual_seed(seed)
    cfg = L.Config(**dict(dict(vocab_size=13, block_size=8, n_embd=16, n_head=4, n_kv_head=2,
                               n_layer=2), **size))
    m = L.MiniLlama(cfg)
    m.eval()
    return m


def test_a_norm_s_products_are_registered():
    """RMSNorm's x * rsqrt(...) * weight: a square root and a divider in its block,
    and a register after each of its two multipliers all the same (two in a row
    in a clock miss 50 MHz in Quartus)."""
    m = _llama()
    xs = [torch.randint(0, 13, (1, 8)) for _ in range(6)]
    with contextlib.redirect_stdout(io.StringIO()):
        p = Py2ver(m, samples=xs, width=16, round=True, margin=2, write=False)
    held = re.findall(r"\breg\b[^;\n]*\b(r_b\d+_mul\w*);", p.verilog)
    assert len(held) == 6                          # (two products in each of the 3 norms)


@pytest.mark.long
def test_a_mini_llama_in_the_sdram_meets_50_mhz(quartus):
    """The board's build (lists and tables in the SDRAM: sdram=0) meets the
    DE0-Nano's clock: a norm's two products a clock apart (registered), the
    SDRAM's write data and its controller. (At this size, 57 MHz; without the
    products' registers, 46: smaller, both meet it.)"""
    m = _llama(vocab_size=256, block_size=32, n_embd=64)
    g = torch.Generator().manual_seed(1)
    xs = [torch.randint(0, 256, (1, 32), generator=g) for _ in range(6)]
    with contextlib.redirect_stdout(io.StringIO()):
        p = Py2ver(m, samples=xs, width=16, round=True, margin=2, sdram=0)
    r = p.SYN(quartus)
    assert r.meets_timing, str(r)


def _unbounded_products(real):
    """share._interval, raising for a product of a block's values (b5_x * b5_y: as a
    value py2ver can't bound; the program's own, before, as they are)."""
    from py2ver.share import NotShareable

    def interval(node, rng, attr):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult) and any(
                isinstance(x, ast.Name) and re.match(r"(r_)?b\d+_", x.id)
                for x in (node.left, node.right)):
            raise NotShareable("a product (in the test)")
        return real(node, rng, attr)
    return interval


def test_a_product_left_without_its_register_is_said(monkeypatch, caplog):
    import py2ver.sequencer as S
    monkeypatch.setattr(S, "_interval", _unbounded_products(S._interval))
    m = _llama()
    xs = [torch.randint(0, 13, (1, 8)) for _ in range(6)]
    with caplog.at_level("WARNING", logger="py2ver.sequencer"), \
            contextlib.redirect_stdout(io.StringIO()):
        p = Py2ver(m, samples=xs, width=16, round=True, margin=2, write=False)
    assert not re.findall(r"\breg\b[^;\n]*\br_b\d+_mul\w*;", p.verilog)
    assert caplog.text.count("no register after") == 3 and "a product (in the test)" in caplog.text


def test_a_loop_left_unpipelined_is_said(monkeypatch, caplog):
    import py2ver.sequencer_blocks as B
    import py2ver.share as share
    real = B.pipeline_block

    def pipeline_block(*args):                     # (its products can't be bounded)
        saved = share._interval
        share._interval = _unbounded_products(saved)
        try:
            return real(*args)
        finally:
            share._interval = saved
    monkeypatch.setattr(B, "pipeline_block", pipeline_block)
    m = _llama()
    xs = [torch.randint(0, 13, (1, 8)) for _ in range(6)]
    with caplog.at_level("WARNING", logger="py2ver.sequencer_blocks"), \
            contextlib.redirect_stdout(io.StringIO()):
        Py2ver(m, samples=xs, width=16, round=True, margin=2, write=False)
    assert "not pipelined: can't bound" in caplog.text and "a product (in the test)" in caplog.text


def test_a_value_the_visitor_can_t_size_is_said(caplog):
    from py2ver.visitor import FunctionVisitor
    v = FunctionVisitor({"a": {"width": 8, "signed": 0}})
    with caplog.at_level("WARNING", logger="py2ver.visitor"):
        assert v._infer_meta("foo(a)") == {"width": 1, "signed": 0}
    assert "Can't size foo(a)" in caplog.text and "declare it in attr" in caplog.text


def test_a_mini_llama_s_forward_pass_is_built_as_written():
    m = _llama()
    xs = [torch.randint(0, 13, (1, 8)) for _ in range(4)]
    tr = numpy_loops.translate(m, xs[:2])          # (the last position's scores)
    with torch.no_grad():
        for x in xs:
            assert np.allclose(tr.func(*tr.adapt(x)), m(x)[0].numpy().ravel(), atol=1e-4)


class Repeated(nn.Module):
    """A parameter read through a view of stride 0 (GQA's keys repeated)."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(1, 2, 3))

    def forward(self, x):
        return self.w.repeat_interleave(3, dim=1) * x


def sgd_steps(model, loader, optimizer):
    total = 0.0
    for x, y in loader:
        loss = (model(x) * y).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total


def test_a_repeated_view_s_gradient_is_the_sum_of_its_copies():
    torch.manual_seed(1)
    m = Repeated()
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    batches = [(torch.randn(1, 6, 3), torch.randn(1, 6, 3)) for _ in range(3)]
    numpy_loops.translate(sgd_steps, [(m, batches, opt)])     # (checked against torch's steps)


def test_a_mini_llama_trains_as_torch_does():
    m = _llama()
    decay = [p for p in m.parameters() if p.dim() >= 2]
    no_decay = [p for p in m.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                             {"params": no_decay, "weight_decay": 0.0}], lr=1e-2, betas=(0.9, 0.95))
    g = torch.Generator().manual_seed(2)
    batches = [(torch.randint(0, 13, (2, 8), generator=g), torch.randint(0, 13, (2, 8), generator=g))
               for _ in range(3)]
    # (checked against torch's training of a copy: dropout, attention's dropout_p
    # included, the repeated keys and values, the tied embedding, clipping, AdamW)
    tr = numpy_loops.translate(gpt_steps, [(m, batches, opt)])
    labels = {label for label, _, _ in tr.memories.values()}
    assert "Embedding.weight" in labels and "RMSNorm.weight" in labels


@pytest.mark.sim
@pytest.mark.slow
@pytest.mark.long
def test_a_mini_llama_trains_in_fixed_point_and_in_hardware(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    # (smaller than _llama's: a step simulated takes minutes at that size)
    m = _llama(vocab_size=11, block_size=4, n_embd=8, n_head=2, n_kv_head=1)
    decay = [p for p in m.parameters() if p.dim() >= 2]
    no_decay = [p for p in m.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                             {"params": no_decay, "weight_decay": 0.0}], lr=1e-2, betas=(0.9, 0.95))
    g = torch.Generator().manual_seed(2)
    batches = [(torch.randint(0, 11, (1, 4), generator=g), torch.randint(0, 11, (1, 4), generator=g))
               for _ in range(12)]
    twin = copy.deepcopy((m, opt))
    with contextlib.redirect_stdout(io.StringIO()):
        p = Py2ver(gpt_steps, samples=[(m, batches, opt)], width=24, round=True, saturate=True,
                   margin=2)
    # (24 bits: the loss of each step and the weights after 12 as torch's float training)
    totals = [r[0] if isinstance(r, tuple) else r for r in p.PY_stream()(batches)]
    with p.hardware_random(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        want = [gpt_steps(twin[0], [b], twin[1]) for b in batches]
    assert np.abs(np.diff([0.0] + totals) - want).max() < 0.05
    assert np.abs(np.asarray(p.state_values()[(m.tok_emb, "weight")])
                  - twin[0].tok_emb.weight.detach().numpy()).max() < 0.05
    # (a step simulated: the same as the bit-exact model)
    assert p.TB_stream()(batches[:1]) == p.PY_stream()(batches[:1])


@pytest.mark.sim
@pytest.mark.slow
def test_a_mini_llama_in_hardware(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    m = _llama()
    xs = [torch.randint(0, 13, (1, 8)) for _ in range(20)]
    with contextlib.redirect_stdout(io.StringIO()):
        p = Py2ver(m, samples=xs, width=16, round=True, margin=2)
    model = np.array(p.PY_batch()(xs))
    with torch.no_grad():
        want = np.array([m(x)[0].numpy().ravel() for x in xs])
    assert np.abs(model - want).max() < 0.01
    assert p.TB_batch()(xs[:2]) == p.PY_batch()(xs[:2])
