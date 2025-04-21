from py2ver import Py2ver

def foo(arg1, arg2):
    a = arg1 + arg2
    b = arg1 - arg2
    c = arg1 * arg2
    d = arg1 / arg2
    return a, b, c

e = foo(3,1)
print (e)

attr = {'arg1': {'signed': 0, 'width': 8},
        'arg2': {'signed': 0, 'width': 8},
        'a': {'signed': 0, 'width': 9},
        'b': {'signed': 1, 'width': 9},
        'c': {'signed': 0, 'width': 9},}

foo_tb = Py2ver(foo, attr).tb()

g = foo_tb(1,3)
print(g)

