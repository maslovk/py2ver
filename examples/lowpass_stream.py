# A first-order low-pass filter with its state in a hardware register.
#
# The function is one step of the filter: it takes the previous output as an
# argument and returns the new one. 'next' tells py2ver that y_prev is loaded
# from y on every clock, so the hardware keeps the state itself and each
# sample is one clock (one UART transaction on the board).
from py2ver import Py2ver

from _run import run


def lpf_step(x, y_prev):
    err = x - y_prev
    y = y_prev + (err >> 3)
    return y


attr = {
    'x':      {'signed': 1, 'width': 12},
    'y_prev': {'signed': 1, 'width': 16, 'next': 'y'},   # state, reset to 0
    'y':      {'signed': 1, 'width': 16},
}


if __name__ == "__main__":
    step = [0] * 3 + [1000] * 12 + [-1000] * 8
    run(Py2ver(lpf_step, attr), step, stream=True)
