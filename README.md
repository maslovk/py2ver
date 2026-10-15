# py2ver
## Simple Verilog code generator from Python source

py2ver turns a plain Python function into a Verilog module and gives you back
callables with the same signature that run it in simulation (cocotb + Icarus)
or on an FPGA (DE0-Nano, over UART).

### Setup

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Simulation also needs [Icarus Verilog](https://steveicarus.github.io/iverilog/)
(`iverilog`, `vvp`) on `PATH`. The hardware path needs Intel Quartus and a
DE0-Nano board with a USB-UART adapter.

### Example

```python
from py2ver import Py2ver

# Standard Python function
def foo(arg1, arg2):
    a = arg1 + arg2
    b = arg1 - arg2
    c = arg1 * arg2
    d = arg1 / arg2
    return a, b, c, d

# Attributes for Verilog generation
attr = {'arg1': {'signed': 0, 'width': 8},
        'arg2': {'signed': 0, 'width': 8},
        'a': {'signed': 0, 'width': 9, 'type': 'reg'},
        'b': {'signed': 1, 'width': 9},
        'c': {'signed': 0, 'width': 9},
        'd': {'signed': 0, 'width': 9}}

p = Py2ver(foo, attr)      # writes hdl/main.v

foo_tb = p.TB()            # cocotb + Icarus simulation
print(foo_tb(3, 1))

# Many input vectors in a single simulation run (much faster than
# calling foo_tb in a loop, which rebuilds and reruns the sim each time)
foo_batch = p.TB_batch()
print(foo_batch([(3, 1), (7, 2), (255, 255)]))   # list of output tuples

foo_hw = p.HW({'QUARTUS_DIR': '/opt/intelFPGA/22.1std/quartus/bin'})
print(foo_hw(3, 1))        # synthesizes, programs the board, runs over UART
```

Generated files (`hdl/`, `sim_build/`, `tb.py`, `results/`, `hw/`) are written
to the current working directory.

### Attributes

Each name maps to `{'signed': 0|1, 'width': <bits>, 'type': 'wire'|'reg'}`, plus
`'next'` and `'init'` for state registers (see below).
`type: 'reg'` registers the signal; the testbench accounts for the resulting
pipeline latency. Widths of undeclared temporaries are inferred.

### State registers (streaming designs)

Write one step of a stateful design as a function that takes its state as
arguments and returns the next state. Mark each state argument with `'next'`
(the returned value that loads it every clock) and optionally `'init'`
(its reset value, default 0):

```python
def lpf_step(x, y_prev):
    err = x - y_prev
    delta = err >> 3
    y = y_prev + delta
    return y

attr = {'x':      {'signed': 1, 'width': 12},
        'y_prev': {'signed': 1, 'width': 16, 'next': 'y'},
        'err':    {'signed': 1, 'width': 16},
        'delta':  {'signed': 1, 'width': 16},
        'y':      {'signed': 1, 'width': 16}}

p = Py2ver(lpf_step, attr)    # y_prev becomes a register; clk and rst are added
xs = [0, 256, 256, 512, -2048]
sim = p.TB_stream()(xs)       # one sample per clock, single simulation
ref = p.PY_stream()(xs)       # Python reference with the same register wrapping
assert sim == ref
```

State arguments disappear from the module ports and become registers with a
synchronous active-high `rst`. All returned values stay output ports. A stream
returns every output for each sample, read before the clock edge that loads
the next state. State cannot yet be combined with `'type': 'reg'` signals,
and `HW()` does not support stateful designs yet.

`HW()` options: `QUARTUS_DIR` (required), `UART_PORT` (default `/dev/ttyUSB0`),
`UART_BAUD` (default `115200`), `UART_TIMEOUT` (default `1.0` s).

### Supported Python subset

- Integer arithmetic `+ - * / // & | ^ << >>` (`>>` is arithmetic, like Python),
  unary `- + ~`, comparisons, `and`/`or`
- `if` / `elif` / `else`, nested (every chain needs a final `else`, and all
  branches must assign the same variables)
- `for i in range(...)` with constant bounds (fully unrolled)
- `complex(re, im)` and `abs()` of a complex variable
- A single `return` of one or more values

Not supported: `while`, calls to other functions, subscripts, list literals,
multiple `return` statements.

### Tests

```
.venv/bin/python -m pytest -q
```

Simulation tests compare the Python function against the cocotb + Icarus
result and are skipped if Icarus or cocotb is missing.
