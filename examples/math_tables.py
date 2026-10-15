# A soft clipper with math.tanh, in fixed point with automatic formats.
#
# math functions become tables (ROMs) over the range of their argument.
# Given example inputs and an error target, py2ver measures every range,
# chooses the formats and table sizes, and prints the attr it chose.
import math
import random

from py2ver import Py2ver

from _run import run

DRIVE = 2.5


def clip(x):
    y = 0.8 * math.tanh(DRIVE * x) + 0.2 * x
    return y


if __name__ == "__main__":
    rng = random.Random(0)
    examples = [rng.uniform(-1, 1) for _ in range(300)]
    p = Py2ver(clip, samples=examples, max_error=2e-3)
    run(p, [(x,) for x in examples[:30]])
