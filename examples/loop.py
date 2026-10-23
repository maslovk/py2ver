# A for loop with constant bounds: py2ver writes it out (unrolls it), so all
# ten additions happen within one clock cycle.
#
#   python examples/loop.py                # simulate the Verilog and compare with Python
from py2ver import Py2ver

from _run import run


def sum_loop(x):
    acc = 0
    for i in range(10):
        acc = acc + (x + i)
    return acc


attr = {
    "x": {"signed": 1, "width": 8},             # -128 .. 127
    "acc": {"signed": 1, "width": 16},
}


if __name__ == "__main__":
    run(Py2ver(sum_loop, attr), [1, -1, 0, 127, -128])
