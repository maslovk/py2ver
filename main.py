import py2ver

def foo(arg1, arg2):
    a = arg1 + arg2
    return a

c = foo(1,2)
print (c)

attr = {'arg1': {'signed': 0, 'width': 8},
        'arg2': {'signed': 0, 'width': 8},
        'a': {'signed': 0, 'width': 9},}
py2ver.Py2ver(foo, attr)

