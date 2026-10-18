import contextlib
import copy
import io
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
