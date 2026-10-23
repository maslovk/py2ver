# Testing with many inputs: basic.py's function on a few corner cases and 40 random ones.
#
# All the vectors go through one simulation run, so testing many costs little.
#
#   python examples/regression.py          # simulate the Verilog and compare with Python
import random

from py2ver import Py2ver

from _run import run
from basic import arithmetic, attr


if __name__ == "__main__":
    corners = [(0, 1), (1, 1), (5, 8), (4, 4), (255, 1), (0, 255), (255, 255)]
    rng = random.Random(0)
    randomized = [(rng.randint(0, 255), rng.randint(1, 255)) for _ in range(40)]   # never / 0
    run(Py2ver(arithmetic, attr), corners + randomized)
