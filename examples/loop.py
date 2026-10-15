from py2ver import Py2ver

from _run import run



# ==========================================================
# Example HDL-like Python function with for-loop unrolling
# ==========================================================
def sum_loop(x):
    acc = 0
    for i in range(0, 10):       # unrolled by FunctionVisitor
        acc = acc + (x + i)
    return acc


# ----------------------------------------------------------
# Attribute definitions (signal widths, signedness)
# ----------------------------------------------------------
attr = {
    'x':   {'signed': 1, 'width': 8},   # input
    'acc': {'signed': 1, 'width': 16},  # accumulator output
}

# Optional synthesis / Quartus environment attributes


if __name__ == "__main__":
    run(Py2ver(sum_loop, attr), [1, -1, 0, 127, -128])
