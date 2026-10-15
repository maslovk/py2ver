# Lists: build a list with append and return it. A returned list is one
# output per element (data_0, data_1, ...); declaring 'data' in attr sets the
# width of every element.
from py2ver import Py2ver

from _run import run


def foo_vec(arg1, arg2):
    data = []
    data.append(arg1 + arg2)
    data.append(arg1 - arg2)
    data.append(arg1 * arg2)
    data.append(arg1 ^ arg2)
    return data


attr = {
    'arg1': {'signed': 0, 'width': 8},
    'arg2': {'signed': 0, 'width': 8},
    'data': {'signed': 1, 'width': 17},
}


if __name__ == "__main__":
    run(Py2ver(foo_vec, attr), [(4, 4), (200, 3), (17, 255), (255, 255)])
