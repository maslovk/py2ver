# if / elif / else in hardware: a tiny ALU that does one of four operations.
#
# Every branch is built, and the condition chooses which result comes out (a
# multiplexer), all within one clock cycle.
#
#   python examples/branches.py            # simulate the Verilog and compare with Python
from py2ver import Py2ver

from _run import run


def alu(op, x, y):
    if op == 0:
        result = x + y
    elif op == 1:
        result = x - y
    elif op == 2:
        result = x * y
    else:
        result = max(x, y)
    return result


attr = {
    "op": 2,                                    # 2 bits: the operations 0 to 3
    "x": 8,
    "y": 8,
    "result": {"signed": 1, "width": 17},       # -255 (0 - 255) up to 65025 (255 * 255)
}


if __name__ == "__main__":
    run(Py2ver(alu, attr), [(0, 200, 100), (1, 3, 250), (2, 255, 255), (3, 7, 9), (3, 9, 7)])
