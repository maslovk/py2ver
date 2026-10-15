# A small receiver chain: downconvert -> decimate by 4 -> power detector.
#
# Each stage is an ordinary Python function, and the chain is just Python
# calling them: the downconverter runs on each sample of a block of 4, the
# decimating filter on the block. py2ver chooses fixed-point formats for an
# error target and prints them. Built as one design ('serial' feeds it one
# sample per handshake), or with --stages as three stages connected by
# streams: one downconverter at the sample rate, and the filter folded to use
# the 4 clocks it has per block.
import cmath
import math
import random

from py2ver import Py2ver

from _run import run

ROT = cmath.exp(-2j * math.pi * 0.125)        # carrier at 1/8 of the sample rate
TAPS = (0.02, 0.06, 0.15, 0.27, 0.27, 0.15, 0.06, 0.02)


def downconvert(x, z):              # one sample: mix with the oscillator
    y = x * z
    zn = z * ROT
    return y, zn


def decimate(xs, delay):            # 4 samples in, 1 out
    window = xs[::-1] + delay
    acc = 0
    for k in range(len(TAPS)):
        acc += TAPS[k] * window[k]
    new_delay = window[:len(TAPS) - 1]
    return acc, new_delay


def power(v):                       # detector: |v|^2
    return v.real * v.real + v.imag * v.imag


def receiver(xs, z, delay):
    ys = []
    for x in xs:
        y, z = downconvert(x, z)
        ys.append(y)
    acc, delay = decimate(ys, delay)
    p = power(acc)
    return p, z, delay


state = {'xs': {'serial': True},
         'z': {'next': 'z', 'init': 1, 'complex': True},
         'delay': {'length': len(TAPS) - 1, 'next': 'delay', 'complex': True}}


if __name__ == "__main__":
    rng = random.Random(0)
    am = [0.5 + 0.3 * math.sin(2 * math.pi * n / 64) for n in range(256)]     # an AM signal
    xs = [a * math.cos(2 * math.pi * 0.125 * n) + 0.02 * rng.uniform(-1, 1) for n, a in enumerate(am)]
    blocks = [(xs[i:i + 4],) for i in range(0, len(xs), 4)]
    import sys
    staged = "--stages" in sys.argv
    if staged:
        sys.argv.remove("--stages")
    p = Py2ver(receiver, state, samples=blocks, max_error=1e-3, round=True, stages=staged)
    run(p, blocks[:24], stream=True)
