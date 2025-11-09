from py2ver import Py2ver
import logging

logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

def foo(arg1, arg2, arg3):
    if arg3 == 0:
        a = arg1 + arg2
        b = arg1 - arg2
        c = arg1 * arg2
        d = arg1 / arg2
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
    'a':    {'signed': 0, 'width': 9},
    'b':    {'signed': 1, 'width': 9},
    'c':    {'signed': 0, 'width': 9},
    'd':    {'signed': 0, 'width': 9},
}

syn_attr = {
    'QUARTUS_DIR': '/opt/intelFPGA/22.1std/quartus/bin'
}

p = Py2ver(foo, attr)

foo_tb = p.TB()
foo_hw = p.HW(syn_attr)

g = foo_tb(4, 4, 2)   # simulation result
h = foo_hw(4, 4, 2)   # hardware
print("TB:", g)
print("HW:", h)

g = foo_tb(4, 4, 1)   # simulation result
h = foo_hw(4, 4, 1)   # hardware
print("TB:", g)
print("HW:", h)

g = foo_tb(4, 4, 0)   # simulation result
h = foo_hw(4, 4, 0)   # hardware
print("TB:", g)
print("HW:", h)
