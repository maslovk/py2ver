# Registers: 'type': 'reg' in attr puts a register on a value.
#
# A registered value comes out one clock cycle later, but it splits the logic
# into shorter pieces, so the design can run at a higher clock. py2ver works out
# how many cycles each output is late, and the testbench waits for it.
#
#   python examples/registers.py           # simulate the Verilog and compare with Python
#   python examples/registers.py --syn     # see the speed in Quartus (needs Quartus)
from py2ver import Py2ver

from _run import run


def multiply_add(x, y, z):
    product = x * y
    result = product + z
    return product, result


attr = {
    "x": 8,
    "y": 8,
    "z": 16,
    "product": {"width": 16, "type": "reg"},    # registered: one cycle late
    "result": {"width": 17, "type": "reg"},     # computed from product: two cycles late
}


if __name__ == "__main__":
    run(Py2ver(multiply_add, attr), [(4, 4, 1), (200, 3, 1000), (255, 255, 65535)])
