# A decimate-by-4 low-pass filter: four samples in, one out.
#
# The float model takes a block of 4 new samples per call. 'serial': True
# makes the stream wrapper take that list one sample per handshake, so the
# hardware is fed one sample at a time; the filter loop is folded onto 4
# multipliers, which take 4 clocks per block: about one input sample per clock.
import math

from py2ver import Py2ver

from _run import run

M, N = 4, 16
TAPS = tuple(math.sin(math.pi * (k + 1) / (N + 1)) / 10 for k in range(N))


def decimate(xs, delay):
    window = xs[::-1] + delay          # newest sample first
    acc = 0.0
    for k in range(N):
        acc += TAPS[k] * window[k]
    y = acc
    new_delay = window[:N - 1]
    return y, new_delay


Q15 = {'signed': 1, 'width': 16, 'frac': 15}
attr = {
    'xs':    dict(Q15, length=M, serial=True),
    'delay': dict(Q15, length=N - 1, next='new_delay'),
    'acc':   {'signed': 1, 'width': 20, 'frac': 17},
    'y':     dict(Q15, round=True, saturate=True),
}


if __name__ == "__main__":
    tone = [0.8 * math.sin(2 * math.pi * 0.01 * n) for n in range(64)]
    blocks = [(tone[i:i + M],) for i in range(0, len(tone), M)]
    run(Py2ver(decimate, attr, fold='k', unroll=4), blocks, stream=True)
