# A 7-tap FIR filter whose delay line is a list kept in hardware registers.
#
# 'delay' is a list argument ('length') that is also state ('next'): each
# sample, the hardware loads it from the returned new_delay, so it is a small
# memory that shifts by one per clock. The taps are a constant table.
from py2ver import Py2ver

from _run import run

TAPS = (2, 8, 14, 16, 14, 8, 2)   # Q6 coefficients, sum = 64


def fir_step(x, delay):
    window = [x] + delay
    acc = 0
    for k in range(len(TAPS)):
        acc += TAPS[k] * window[k]
    y = acc >> 6
    new_delay = window[:-1]
    return y, new_delay


attr = {
    'x':     {'signed': 1, 'width': 16},
    'delay': {'signed': 1, 'width': 16, 'length': len(TAPS) - 1, 'next': 'new_delay'},
    'y':     {'signed': 1, 'width': 16},
}


if __name__ == "__main__":
    impulse = [64] + [0] * 8           # the response is the taps themselves
    step = [1000] * 10
    run(Py2ver(fir_step, attr), impulse + step, stream=True)
