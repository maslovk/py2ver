# A float model whose fixed-point formats are chosen automatically.
#
# A second-order low-pass (biquad) filter, written with floats and nothing
# else. auto_fixed() runs it on example inputs, measures the range of every
# value, and searches the fewest fraction bits that keep the output within
# 1e-3 of the float model. The result is an ordinary attr dict.
import math
import random
from pprint import pprint

from py2ver import Py2ver, auto_fixed

from _run import run

# Butterworth low-pass, cutoff at 0.1 x the sample rate
B = (0.0674552738890719, 0.1349105477781438, 0.0674552738890719)
A = (-1.1429805025399011, 0.4128015980961886)


def biquad(x, s1, s2):
    y = B[0] * x + s1
    n1 = B[1] * x - A[0] * y + s2
    n2 = B[2] * x - A[1] * y
    return y, n1, n2


state = {'s1': {'next': 'n1'}, 's2': {'next': 'n2'}}


if __name__ == "__main__":
    rng = random.Random(0)
    examples = [math.sin(0.05 * n) + 0.3 * rng.uniform(-1, 1) for n in range(400)]
    attr = auto_fixed(biquad, examples, state, width=16, max_error={'y': 1e-3})
    pprint(attr)
    run(Py2ver(biquad, attr), examples[:40], stream=True)
