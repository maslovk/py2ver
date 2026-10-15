# A 7-tap low-pass FIR filter: an ordinary float model, built in fixed point.
#
# The function is plain Python on floats, with coefficients designed in
# Python (a Hamming-windowed sinc). attr gives the samples Q15 (16 bits, 15
# of them fraction: -1 .. 1). Products and sums inside are exact; the output
# is rounded back to Q15 and saturated. The coefficients are constants,
# rounded to 18 significant bits (const_bits).
import math

from py2ver import Py2ver

from _run import run

N, CUTOFF = 7, 0.2


def _tap(k):
    t = k - (N - 1) / 2
    sinc = 2 * CUTOFF * (math.sin(2 * math.pi * CUTOFF * t) / (2 * math.pi * CUTOFF * t) if t else 1)
    return sinc * (0.54 - 0.46 * math.cos(2 * math.pi * k / (N - 1)))


TAPS = tuple(_tap(k) for k in range(N))


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
    'delay': dict(Q15, length=N - 1, next='new_delay'),   # state, reset to 0
    'y':     dict(Q15, round=True, saturate=True),
}


if __name__ == "__main__":
    impulse = [0.5] + [0] * 8                         # the response is the taps / 2
    tone = [0.9 * math.sin(2 * math.pi * 0.4 * n) for n in range(12)]   # mostly filtered out
    run(Py2ver(fir_step, attr), impulse + [0.9] * 10 + tone, stream=True)
