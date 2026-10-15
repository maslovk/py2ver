from py2ver import Py2ver

from _run import run


def foo(arg1, arg2):
    a = arg1 + arg2
    b = arg1 - arg2
    c = arg1 * arg2
    d = arg1 / arg2
    e = d
    return a, b, c, d, e

attr = {
    'arg1': {'signed': 0, 'width': 8},
    'arg2': {'signed': 0, 'width': 8},
    'a':    {'signed': 0, 'width': 9},
    'b':    {'signed': 1, 'width': 9},
    'c':    {'signed': 0, 'width': 9},
    'd':    {'signed': 0, 'width': 9, "type": "reg"},
    'e':    {'signed': 0, 'width': 9, "type": "reg"},
}


if __name__ == "__main__":
    run(Py2ver(foo, attr), [(4, 4), (200, 3), (17, 255)])
