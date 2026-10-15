from py2ver import Py2ver

from _run import run


def foo(arg1, arg2, arg3, arg4):
    if arg3 == 0:
        if arg4 == 0:
            a = arg1 + arg2
            b = arg1 - arg2
            c = arg1 * arg2
            d = arg1 / arg2
        else:
            a = arg2 + arg1
            b = arg2 - arg1
            c = arg2 * arg1
            d = arg2 / arg1
    elif arg3 == 1:
        b = arg1 + arg2
        c = arg1 - arg2
        d = arg1 * arg2
        a = arg1 / arg2
    else:
        d = arg1 + arg2
        c = arg1 - arg2
        b = arg1 * arg2
        a = arg1 / arg2
    return a, b, c, d

attr = {
    'arg1': {'signed': 0, 'width': 8},
    'arg2': {'signed': 0, 'width': 8},
    'arg3': {'signed': 0, 'width': 8},
    'arg4': {'signed': 0, 'width': 8},
    'a':    {'signed': 0, 'width': 9},
    'b':    {'signed': 1, 'width': 9},
    'c':    {'signed': 0, 'width': 9},
    'd':    {'signed': 0, 'width': 9},
}


if __name__ == "__main__":
    run(Py2ver(foo, attr), [(4, 3, 2, 0), (4, 3, 1, 0), (4, 3, 1, 1), (4, 3, 0, 0), (9, 200, 0, 5)])
