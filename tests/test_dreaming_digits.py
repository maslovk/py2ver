import contextlib
import io
import sys
from pathlib import Path

import pytest

# examples/dreaming_digits.py: a VAE's decoder on the FPGA drawing digits that
# melt into one another. Importing it trains the network the first time (about
# 20 seconds), so these tests are slow, and import it only when they run.

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def dreaming():
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    import dreaming_digits
    return dreaming_digits


def test_the_fixed_point_decoder_draws_as_torch_s(dreaming):
    import torch
    model = dreaming._vae()
    model.load_state_dict(torch.load(dreaming.MODEL))
    torch.manual_seed(3)
    for z in torch.randn(4, dreaming.LATENT):
        with torch.no_grad():
            logits = model["decoder"](z).view(28, 28)
        want = (logits * 16 + 128).clamp(0, 255)          # (torch's answer as the levels)
        z_fixed = [round(float(v) * 2 ** dreaming.Z_FRAC) for v in z]
        got = torch.tensor(dreaming.decode_levels(dreaming.NET, z_fixed), dtype=torch.float32)
        # (8-bit weights: within a level on average, 16 levels a unit of the answer)
        assert (got - want).abs().mean() < 1.5 and (got - want).abs().max() < 12


@pytest.mark.parametrize("sim", [False, pytest.param(True, marks=pytest.mark.sim)])
def test_the_digits_melt_from_one_to_the_next(dreaming, sim, monkeypatch, tmp_path):
    from py2ver import Py2ver
    from py2ver.devices.ili9341 import ILI9341, ILI9341Model
    monkeypatch.chdir(tmp_path)
    w, h, count = 32, 32, 4
    lcd = ILI9341(wake_clocks=50)
    net = dreaming.NET
    with contextlib.redirect_stdout(io.StringIO()):
        p = Py2ver(dreaming.show_on_screen(lcd, net, w, h, zoom=0, morph_bits=1, hold=1))
    assert "Table in a memory block: 50176 x 8 bits" in p.verilog          # (the weights)
    screen = ILI9341Model(lcd, width=w, height=h)
    shots = p.check(dreaming.frames(screen, count, w, h), verilog=sim).value
    digit = [[row[2:30] for row in shot[2:30]] for shot in shots]
    assert digit[0] == dreaming.decode(net, [0] * dreaming.LATENT)       # (the point 0)
    assert digit[1] != digit[0]                                          # (on its way)
    # Then at a digit point, and held there a frame
    first = dreaming.random_pick(None, 1, len(net.digits) // dreaming.LATENT)
    next(first)
    with pytest.raises(StopIteration) as done:
        next(first)
    pick = done.value.value[0]
    goal = net.digits[pick * dreaming.LATENT:(pick + 1) * dreaming.LATENT]
    assert digit[2] == digit[3] == dreaming.decode(net, goal)
    assert all(c == 0 for c in shots[0][0] + shots[0][-1])              # (black around)


def test_the_digit_points_are_digits_of_every_kind(dreaming):
    import torch
    import torch.nn.functional as F
    from torchvision import datasets
    net = dreaming.NET
    assert len(net.digits) == 10 * 8 * dreaming.LATENT                  # (8 styles of each digit)
    # A judge: a small classifier trained on MNIST (about 98% right)
    torch.manual_seed(0)
    data = datasets.MNIST(str(Path(dreaming.HERE) / "data"), train=True, download=True)
    images, labels = data.data.float().div(255).view(-1, 784), data.targets
    judge = torch.nn.Sequential(torch.nn.Linear(784, 256), torch.nn.ReLU(), torch.nn.Linear(256, 10))
    optimizer = torch.optim.Adam(judge.parameters(), lr=1e-3)
    for _ in range(3):
        for i in range(0, len(images), 128):
            loss = F.cross_entropy(judge(images[i:i + 128]), labels[i:i + 128])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    # Each point as the hardware draws it: read as its own digit, and surely
    read = 0
    for d in range(80):
        point = net.digits[d * dreaming.LATENT:(d + 1) * dreaming.LATENT]
        levels = torch.tensor(dreaming.decode_levels(net, point), dtype=torch.float32).view(1, 784)
        with torch.no_grad():
            p = F.softmax(judge(torch.sigmoid((levels - 128) / 16)), dim=1)[0]
        read += p.argmax().item() == d // 8 and p.max().item() > 0.9
    assert read >= 0.95 * 80
