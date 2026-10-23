# The smallest py2ver design: a plain Python function of two 8-bit numbers.
#
# The function stays exactly as it is. Everything about the hardware, such as
# how many bits each value has, goes in `attr`.
#
#   python examples/basic.py               # simulate the Verilog and compare with Python
#   python examples/basic.py --help        # more: the simulated board, Quartus, the real board
from py2ver import Py2ver

from _run import run


def arithmetic(x, y):
    total = x + y
    difference = x - y
    product = x * y
    quotient = x // y
    return total, difference, product, quotient


attr = {
    "x": 8,                                     # unsigned, 8 bits: 0 .. 255
    "y": 8,
    "total": 9,                                 # up to 255 + 255 = 510, so 9 bits
    "difference": {"signed": 1, "width": 9},    # -255 .. 255
    "product": 16,                              # up to 255 * 255
    "quotient": 8,
}


if __name__ == "__main__":
    run(Py2ver(arithmetic, attr), [(4, 4), (200, 3), (17, 255), (255, 1), (0, 7)])
