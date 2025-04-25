# py2ver
## Simple verilog code generator from python source

### Example:

```

from py2ver import Py2ver

#Standard Python function
def foo(arg1, arg2):
    a = arg1 + arg2
    b = arg1 - arg2
    c = arg1 * arg2
    d = arg1 / arg2
    return a, b, c, d

e = foo(3,1)
print (e)

#Attributes for verilog generation
attr = {'arg1': {'signed': 0, 'width': 8},
        'arg2': {'signed': 0, 'width': 8},
        'a': {'signed': 0, 'width': 9, 'type' : 'reg'},
        'b': {'signed': 1, 'width': 9},
        'c': {'signed': 0, 'width': 9},
        'd': {'signed': 0, 'width': 9},}

foo_tb = Py2ver(foo, attr).tb()

#Result from cocotb simulation
g = foo_tb(3,1)
print(g)
```