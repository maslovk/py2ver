# A 32-tap FIR filter with its loop folded: 4 multipliers instead of 32.
#
# The float model loops over the taps. fold='k' runs the loop body over
# several clocks instead of unrolling it, and unroll=4 does 4 taps per clock:
# 8 clocks per sample with 4 multipliers. The results are the same as the
# unfolded filter's; only the hardware and the timing change.
import math

from py2ver import Py2ver

from _run import run

N = 32
TAPS = tuple(math.sin(math.pi * (k + 1) / (N + 1)) / 16 for k in range(N))


def fir_step(x, delay):
    window = [x] + delay
    acc = 0.0
    for k in range(N):
        acc += TAPS[k] * window[k]
    y = acc
    new_delay = window[:-1]
    return y, new_delay


Q15 = {'signed': 1, 'width': 16, 'frac': 15}
attr = {
    'x':     Q15,
    'delay': dict(Q15, length=N - 1, next='new_delay'),
    'acc':   {'signed': 1, 'width': 22, 'frac': 18},
    'y':     dict(Q15, round=True, saturate=True),
}


if __name__ == "__main__":
    step = [0.5] * 40
    run(Py2ver(fir_step, attr, fold='k', unroll=4), step, stream=True)
