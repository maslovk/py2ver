# Dreaming digits: a neural network on the FPGA draws handwritten digits that
# melt into one another, on an ILI9341 display.
#
# The network is the decoder of a small VAE (a variational autoencoder) trained
# on MNIST: from 8 numbers it draws a 28 x 28 digit, through a layer of 64
# neurons. Every point of those 8 numbers is some drawing, and points close
# together are similar drawings. Most points are clearly a digit, but many
# fall between digits, so before building, 80 points are picked that are real
# digits: 8 styles of each, from where the network puts MNIST's own digits.
# The hardware picks one of them at random, glides to it a frame at a time,
# and stays there a moment: a 3 slowly turns into an 8, then into a 5, and so on.
#
#   dream --color/valid/ready--> ili9341_display
#
# Each frame: the 64 neurons' values for this frame's point (8 multiplies
# each), then each of the 784 pixels from them (64 multiplies each), its color
# from how sure the network is that the pixel is ink. The weights (8 bits,
# fixed point) are tables in memory blocks. Each pixel is drawn as an 8 x 8
# block: the digit 224 pixels wide, at about 17 frames a second.
#
# The first run trains the network (about 20 seconds; MNIST is downloaded to
# examples/data if it isn't there) and saves it as dreaming_digits.pt.
#
#   python examples/dreaming_digits.py          # a few small frames: test them, save dreaming_digits.png
#   python examples/dreaming_digits.py --sim    # also simulate the generated Verilog
#   python examples/dreaming_digits.py --pins   # build it for the board (wiring: mandelbrot_lcd.py)
import functools
import math
import os
import sys
from typing import NamedTuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mandelbrot_lcd import PINS  # noqa: E402
from py2ver import connect  # noqa: E402
from py2ver.devices.ili9341 import HEIGHT, ILI9341, WIDTH, ili9341_display, rgb565  # noqa: E402
from py2ver.protocol import SPI, stream_send  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "dreaming_digits.pt")
SIDE = 28                               # the digits are 28 x 28 pixels
LATENT = 8                              # the numbers a digit is drawn from
HIDDEN = 64                             # the decoder's neurons
Z_FRAC = 5                              # a latent number's fraction bits (1.0 is 32)


# ----------------------------- the network, trained in torch -----------------------------

def train(epochs=10, path=MODEL):
    """Train the VAE on MNIST and save it (the decoder is what the hardware runs)."""
    import torch
    from torchvision import datasets
    torch.manual_seed(0)
    data = datasets.MNIST(os.path.join(HERE, "data"), train=True, download=True)
    images = data.data.float().div(255).view(-1, SIDE * SIDE)
    model = _vae()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    for epoch in range(epochs):
        order = torch.randperm(len(images))
        for i in range(0, len(images), 128):
            x = images[order[i:i + 128]]
            mean, log_var = model["encoder"](x).chunk(2, dim=1)
            z = mean + torch.randn_like(mean) * (0.5 * log_var).exp()
            logits = model["decoder"](z)
            # (how far the drawing is from the digit, and how far the points
            # are from a normal spread: what a VAE learns from)
            drawing = torch.nn.functional.binary_cross_entropy_with_logits(logits, x, reduction="sum")
            spread = -0.5 * torch.sum(1 + log_var - mean ** 2 - log_var.exp())
            loss = (drawing + spread) / len(x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"training the network: epoch {epoch + 1} of {epochs}, loss {loss.item():.1f}",
              flush=True)
    torch.save(model.state_dict(), path)


def _vae():
    import torch.nn as nn
    return nn.ModuleDict({
        "encoder": nn.Sequential(nn.Linear(SIDE * SIDE, 400), nn.ReLU(), nn.Linear(400, 2 * LATENT)),
        "decoder": nn.Sequential(nn.Linear(LATENT, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, SIDE * SIDE)),
    })


class Decoder(NamedTuple):
    """The decoder in fixed point, 8-bit weights, as tables:

    w1[j * LATENT + k], b1[j]: neuron j's weight for latent number k, its bias
    w2[p * HIDDEN + j], b2[p]: pixel p's weight for neuron j, its bias
    frac1, frac2:              each layer's fraction bits (its weights times 2**frac)
    digits[d * LATENT + k]:    the points to glide between, each a clear digit
                               (latent number k of point d; 1.0 is 32)
    """
    w1: tuple
    b1: tuple
    w2: tuple
    b2: tuple
    frac1: int
    frac2: int
    digits: tuple


def _fixed(values, bits=8):
    """values as integers of `bits` bits (signed), all scaled by the same power of
    two, as large as fits: (the integers, their fraction bits)."""
    largest = max(abs(v) for v in values)
    frac = bits - 1 - math.ceil(math.log2(largest))
    low, high = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return [min(high, max(low, round(v * 2 ** frac))) for v in values], frac


def digit_points(model, styles=8):
    """Points that are clear digits: for each digit, `styles` points, in fixed
    point (1.0 is 32). Each is the middle of a group of MNIST's images of that
    digit, as the encoder places them (k-means), so each is a typical way of
    writing it. (A random point is clearly a digit only about 40% of the time;
    these, about 98%, as a small classifier judges them.)"""
    import torch
    from torchvision import datasets
    data = datasets.MNIST(os.path.join(HERE, "data"), train=True, download=True)
    with torch.no_grad():
        means = model["encoder"](data.data.float().div(255).view(-1, SIDE * SIDE)).chunk(2, dim=1)[0]
    generator = torch.Generator().manual_seed(0)
    points = []
    for digit in range(10):
        mine = means[data.targets == digit]
        centers = mine[torch.randperm(len(mine), generator=generator)[:styles]]
        for _ in range(20):                       # (k-means: each center to the middle of its group)
            nearest = torch.cdist(mine, centers).argmin(1)
            centers = torch.stack([mine[nearest == j].mean(0) if (nearest == j).any() else centers[j]
                                   for j in range(styles)])
        points += [min(127, max(-128, round(float(v) * 2 ** Z_FRAC))) for v in centers.flatten()]
    return tuple(points)


def decoder(path=MODEL):
    """The trained decoder in fixed point (trained first if it hasn't been)."""
    import torch
    if not os.path.exists(path):
        train(path=path)
    model = _vae()
    model.load_state_dict(torch.load(path))
    first, second = model["decoder"][0], model["decoder"][2]
    layer1, frac1 = _fixed(first.weight.flatten().tolist() + first.bias.tolist())
    layer2, frac2 = _fixed(second.weight.flatten().tolist() + second.bias.tolist())
    n1, n2 = LATENT * HIDDEN, HIDDEN * SIDE * SIDE
    return Decoder(tuple(layer1[:n1]), tuple(layer1[n1:]), tuple(layer2[:n2]), tuple(layer2[n2:]),
                   frac1, frac2, digit_points(model))


def decoder_widths(net):
    """The widths of the sums in attr. py2ver can't follow a sum over a loop's
    rounds, so they come from the weights: the largest each sum can be."""
    z_max = 128                         # (a latent number: 8 bits)
    neuron = [sum(abs(net.w1[j * LATENT + k]) * z_max for k in range(LATENT))
              + (abs(net.b1[j]) << Z_FRAC) for j in range(HIDDEN)]
    pixel = max(sum(abs(net.w2[p * HIDDEN + j]) * neuron[j] for j in range(HIDDEN))
                + (abs(net.b2[p]) << (Z_FRAC + net.frac1)) for p in range(SIDE * SIDE))
    return {"neuron_sum": {"width": max(neuron).bit_length() + 1, "signed": 1},
            "pixel_sum": {"width": pixel.bit_length() + 1, "signed": 1}}


def ink_colors(stops=((0, 0, 0), (40, 10, 90), (200, 40, 120), (255, 160, 40), (255, 255, 220))):
    """A color for each of 256 levels of the network's answer (from -8 to 8: how
    sure it is that a pixel is ink), through the sigmoid: black paper, glowing ink."""
    colors = []
    for level in range(256):
        ink = 1 / (1 + math.exp(-(level - 128) / 16))
        t = ink * (len(stops) - 1)
        k = min(int(t), len(stops) - 2)
        a, b = stops[k], stops[k + 1]
        colors.append(rgb565(*(round(x + (y - x) * (t - k)) for x, y in zip(a, b))))
    return tuple(colors)


NET = decoder()
INK = ink_colors()


# ----------------------------- the hardware -----------------------------

def random_pick(port, seed, count):
    """A random number from 0 to count - 1, and the next seed (a step of xorshift)."""
    seed = seed ^ ((seed << 7) & 0xFFFF)
    seed = seed ^ (seed >> 9)
    seed = seed ^ ((seed << 8) & 0xFFFF)
    yield
    return (seed * count) >> 16, seed


def dream(port, net=NET, width=WIDTH, height=HEIGHT, zoom=3, morph_bits=5, hold=24, colors=INK):
    """Digits melting into one another, frame after frame, sending each pixel's color.

    zoom:       each of the digit's pixels is 2**zoom pixels square on the screen
    morph_bits: 2**morph_bits frames from one digit to the next
    hold:       then this many frames on the digit, so that it can be read
    Output stream: color (RGB565), valid; input: ready (see protocol.stream_send).
    """
    port.valid = 0
    side = SIDE << zoom                          # the digit on the screen
    top = (height - side) // 2
    left = (width - side) // 2
    seed = 1
    start = [0] * LATENT                         # the point the digit comes from,
    goal = [0] * LATENT                          # the one it goes to,
    z = [0] * LATENT                             # and this frame's, between them
    neurons = [0] * HIDDEN
    steps = 1 << morph_bits
    frame = 0
    while True:
        if frame == 0:                           # a new goal, a clear digit picked at random;
            pick, seed = yield from random_pick(port, seed, len(net.digits) // LATENT)
            for k in range(LATENT):              # the old goal, the start
                start[k] = goal[k]
                value = net.digits[pick * LATENT + k]
                yield
                goal[k] = value
        # This frame's point: from start to goal, slow at both ends (smoothstep),
        # then at the goal for `hold` frames
        t = min(frame, steps)
        ease = (t * t * (3 * steps - 2 * t)) >> (2 * morph_bits)
        for k in range(LATENT):
            z[k] = start[k] + (((goal[k] - start[k]) * ease) >> morph_bits)
            yield
        # The neurons: each a weighted sum of the point, or 0 if it is negative (ReLU)
        for j in range(HIDDEN):
            neuron_sum = 0
            for k in range(LATENT):
                weight1 = net.w1[j * LATENT + k]
                yield
                neuron_sum = neuron_sum + weight1 * z[k]
            bias1 = net.b1[j]
            yield
            neurons[j] = max(0, neuron_sum + (bias1 << Z_FRAC))
        # The picture: black around the digit; each of its pixels a weighted sum
        # of the neurons, drawn as a square
        for row in range(height):
            if row < top or row >= top + side:
                for col in range(width):
                    yield from stream_send(port, 0, data="color")
            else:
                y = (row - top) >> zoom
                for col in range(left):
                    yield from stream_send(port, 0, data="color")
                for x in range(SIDE):
                    p = y * SIDE + x
                    pixel_sum = 0
                    for j in range(HIDDEN):
                        weight2 = net.w2[p * HIDDEN + j]
                        yield
                        pixel_sum = pixel_sum + weight2 * neurons[j]
                    bias2 = net.b2[p]
                    yield
                    pixel_sum = pixel_sum + (bias2 << (Z_FRAC + net.frac1))
                    # (the network's answer, 16 a unit, from -8 to 8, as a color)
                    level = (pixel_sum >> (Z_FRAC + net.frac1 + net.frac2 - 4)) + 128
                    color = colors[min(255, max(0, level))]
                    yield
                    for _ in range(1 << zoom):
                        yield from stream_send(port, color, data="color")
                for col in range(width - left - side):
                    yield from stream_send(port, 0, data="color")
        frame = 0 if frame == steps + hold - 1 else frame + 1


def dream_attr(net=NET):
    return {"ready": 1, **decoder_widths(net)}


def decode_levels(net, z):
    """The digit for the point z (LATENT numbers, 1.0 is 32), worked out directly
    as the hardware does: its 28 x 28 levels (0 to 255, from -8 to 8: how sure
    the network is that a pixel is ink)."""
    neurons = [max(0, sum(net.w1[j * LATENT + k] * z[k] for k in range(LATENT))
                   + (net.b1[j] << Z_FRAC)) for j in range(HIDDEN)]
    shift = Z_FRAC + net.frac1 + net.frac2 - 4
    rows = []
    for y in range(SIDE):
        line = []
        for x in range(SIDE):
            p = y * SIDE + x
            total = (sum(net.w2[p * HIDDEN + j] * neurons[j] for j in range(HIDDEN))
                     + (net.b2[p] << (Z_FRAC + net.frac1)))
            line.append(min(255, max(0, (total >> shift) + 128)))
        rows.append(line)
    return rows


def decode(net, z, colors=INK):
    """The digit for the point z, as decode_levels, in its colors."""
    return [[colors[level] for level in row] for row in decode_levels(net, z)]


# SCK at 25 MHz, as in mandelbrot_zoom.py (if the picture is noisy: SPI(2, ...))
LCD = ILI9341(spi=SPI(1, mode=0, miso=""))


def show_on_screen(lcd=LCD, net=NET, width=WIDTH, height=HEIGHT, zoom=3, morph_bits=5, hold=24):
    """The dreaming network feeding the display."""
    source = functools.partial(dream, net=net, width=width, height=height, zoom=zoom,
                               morph_bits=morph_bits, hold=hold)
    display = functools.partial(ili9341_display, lcd=lcd, width=width, height=height)
    return connect(dream=(source, dream_attr(net)), display=(display, {"color": 16, "valid": 1}),
                   name="dreaming_digits",
                   wires={"display.color": "dream.color", "display.valid": "dream.valid",
                          "dream.ready": "display.ready"})


def frames(screen, count, width, height):
    """A testbench: run until the display has received `count` frames; a copy of
    the picture after each."""
    from py2ver.protocol import alongside
    shots = []

    def watch(dut):
        while len(shots) < count:
            if screen.written >= (len(shots) + 1) * width * height:
                shots.append([list(row) for row in screen.pixels])
            yield

    def tb(dut):
        yield from alongside(watch(dut), screen.run(dut))
        return shots
    return tb


if __name__ == "__main__":
    from mandelbrot_zoom import film_strip
    from py2ver import Py2ver
    from py2ver.devices.ili9341 import ILI9341Model, rgb888
    from py2ver.devices.spi_target import save_png

    args = sys.argv[1:]
    # A small screen for the simulation: the digit's own 28 x 28 pixels on 32 x
    # 32; 2 frames to the next digit, then a frame on it
    w, h, count = 32, 32, 4
    quick = ILI9341(wake_clocks=50)
    screen = ILI9341Model(quick, width=w, height=h)
    p = Py2ver(show_on_screen(quick, NET, w, h, zoom=0, morph_bits=1, hold=1))
    run = p.check(frames(screen, count, w, h), verilog="--sim" in args)
    digit = [[row[2:30] for row in shot[2:30]] for shot in run.value]
    assert digit[0] == decode(NET, [0] * LATENT)     # (the first frame: the point 0)
    points = [NET.digits[d * LATENT:(d + 1) * LATENT] for d in range(len(NET.digits) // LATENT)]
    assert any(digit[2] == decode(NET, point) for point in points)   # (the third: a clear digit)
    save_png("dreaming_digits.png", film_strip(run.value, scale=4), rgb888)
    print(f"{count} frames of {w} x {h} in {len(run.inputs):,} clock cycles: dreaming_digits.png")
    if "--pins" in args:
        board = Py2ver(show_on_screen(), pins=PINS)
        print(board.PINS({"QUARTUS_DIR": os.environ.get(
            "QUARTUS_DIR", "/opt/intelFPGA/22.1std/quartus/bin")}))
