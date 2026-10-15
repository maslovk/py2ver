import random

from py2ver import Py2ver

from _run import run

def foo(arg1, arg2):
    a = arg1 + arg2
    b = arg1 - arg2
    c = arg1 * arg2
    d = arg1 / arg2
    return a, b, c, d

attr = {
    'arg1': {'signed': 0, 'width': 8},
    'arg2': {'signed': 0, 'width': 8},
    'a':    {'signed': 0, 'width': 9},
    'b':    {'signed': 1, 'width': 9},
    'c':    {'signed': 0, 'width': 9},
    'd':    {'signed': 0, 'width': 9},
}

import random


if __name__ == "__main__":
    # Directed corner cases plus random vectors (divisor never 0)
    directed = [(0, 1), (1, 1), (5, 8), (4, 4), (255, 1)]
    rng = random.Random(0)
    randomized = [(rng.randint(0, 255), rng.randint(1, 255)) for _ in range(40)]
    run(Py2ver(foo, attr), directed + randomized)
