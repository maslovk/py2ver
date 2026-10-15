# py2ver

[![tests](https://github.com/maslovk/py2ver/actions/workflows/tests.yml/badge.svg)](https://github.com/maslovk/py2ver/actions/workflows/tests.yml)

## Simple Verilog code generator from Python source

py2ver turns a plain Python function into a Verilog module and gives you back
callables with the same signature that run it in simulation (cocotb + Icarus)
or on an FPGA (DE0-Nano, over UART).

### Setup

```
python3 -m venv .venv
.venv/bin/pip install "py2ver[hw] @ git+https://github.com/maslovk/py2ver"
```

For development, from a clone: `.venv/bin/pip install -e ".[test,hw]"`.
The `hw` extra (pyserial) is only needed to talk to a physical board.

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

Generated files (`hdl/`, `sim_build/`, `tb.py`, `results/`, `hw/`, `syn/`) are written
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
synchronous active-high `rst` and a clock enable `ce`. All returned values stay
output ports. A stream returns every output for each sample, read before the
clock edge that loads the next state. On the board, each `HW()` call is one
sample and advances the state once; KEY0 or power-up resets it. State cannot
yet be combined with `'type': 'reg'` signals.

### Automatic pipelining

`Py2ver(func, attr, pipeline=N)` splits a combinational design into `N + 1`
stages of roughly equal estimated delay. It adds registers where values cross
a stage boundary, plus delay registers for values that skip stages, so every
output appears exactly `N` clocks after its inputs and a new input can be
applied every clock. `TB()`/`TB_batch()` simulate it that way (one vector per
clock) and compare as usual.

```python
p = Py2ver(sha4_round_core, attr, pipeline=3)
p.TB_batch()(vectors)    # same results as unpipelined, 3 clocks later
p.SYN(syn_attr)          # SHA-256 4 rounds on EP4CE22: 31 MHz -> 88 MHz
```

Delay is estimated per assignment from its operators, and one assignment is
never split, so break long expressions into temporaries for finer stages.
Constants are not registered. Pipelining can't yet be combined with state
registers or `'type': 'reg'` signals.

### Valid/ready streaming interface

`p.stream_wrapper()` writes `hdl/<name>_stream.v`: the module behind a
valid/ready handshake, for connecting it to other hardware.

```
<name>_stream ( clk, rst,
                s_valid, s_ready, s_<input>...,     // inputs, taken when both high
                m_valid, m_ready, m_<output>... )   // results, in input order
```

A pipelined design takes one input per clock and stalls as a whole when the
consumer isn't ready; a design with state advances once per input taken; a
design with block RAM takes an input every other clock. Designs using `'reg'`
signals are rejected (use `pipeline=N`). `TB_handshake()` checks the wrapper
in simulation with inputs and readiness that come and go at random:

```python
got = p.TB_handshake()(vectors, valid=0.6, ready=0.5, seed=1)
assert got == p.PY_batch()(vectors)        # PY_stream for designs with state
```

### Area and timing (`SYN()`)

`SYN()` compiles the generated module alone with Quartus (no board needed,
nothing is programmed) and reports what it costs:

```python
r = p.SYN({'QUARTUS_DIR': '/opt/intelFPGA/22.1std/quartus/bin'})
print(r)
# Device:          EP4CE22F17C6
# Combinational:   48
# Registers:       16
# 9-bit mults:     0
# Memory bits:     0
# Fmax:            194.21 MHz
# Timing @ 50 MHz: met (worst setup slack 14.851 ns)
r.fmax_mhz, r.meets_timing, r.fit_summary   # fields for scripting
```

The module is wrapped with registers on all inputs and outputs, so Fmax is
for register-to-register paths through your logic, and its I/O uses virtual
pins, so wide designs fit. Logic, multiplier and memory counts are the
module's alone; registers are the module's own (state or `reg` signals).
Options: `QUARTUS_DIR` (default: `quartus_sh` on `PATH`), `CLK_PERIOD_NS`
(default 20), `DEVICE` (default `EP4CE22F17C6`), `FAMILY` (default
`Cyclone IV E`). The project is written to `syn/<name>/`.

### On the board (`HW()`) and the virtual board (`HW_sim()`)

`HW()` builds the DE0-Nano project (UART link plus your module), programs the
FPGA and returns a function that sends inputs over the serial port and reads
the outputs back. The board waits for your module's latency (`reg` signals,
`pipeline=N`) before replying, and advances state once per call.

`HW()` raises instead of returning simulation results when the board can't be
used, and refuses a build that misses 50 MHz timing (results could be wrong;
use `pipeline=N`). Options: `QUARTUS_DIR` (required), `UART_PORT` (default
`/dev/ttyUSB0`), `UART_BAUD` (default `115200`), `UART_TIMEOUT` (default
`1.0` s), `FALLBACK_TO_SIM` (return simulation results instead of raising),
`ALLOW_TIMING_FAILURE` (run anyway).

`HW_sim()` needs no board or Quartus: it simulates the whole board project
and talks to it over its UART pins with the exact bytes `HW()` would send,
so the board path can be checked against Python before touching hardware:

```python
p.HW_sim()(vectors) == p.TB_batch()(vectors)   # one reply per input vector
```

It runs at real UART speed in simulation (a few seconds per handful of
vectors).

### Supported Python subset

- Integer arithmetic `+ - * / // % & | ^ << >>` with Python semantics,
  unary `- + ~`, comparisons, `and`/`or`, `x if c else y`
- `abs`, `min`, `max` on integers
- Assignment, reassignment (`y = y + b`), augmented assignment (`y += b`,
  `y >>= 1`, ...) and tuple assignment (`a, b = b, a - b`)
- `if` / `elif` / `else`, nested; a branch may leave out a variable that
  already had a value (it keeps it), and statements in a branch see earlier
  ones in the same branch
- `for i in range(...)` with constant bounds, and `for v in (a, b, c)`
  (fully unrolled)
- Calls to your own helper functions (inlined; helpers may call helpers,
  must end with a single `return`, and may return a tuple for
  `x, y = helper(...)`), and module-level integer constants
- Complex values from integer parts: `complex(re, im)`, `1j` constants,
  `+ - *`, unary `-`, `.real`, `.imag`, `.conjugate()`; a complex variable
  `z` is held as `z_re` / `z_im`, a returned one is two outputs, and declaring
  `z` in `attr` sets both widths. `abs()` of a complex value uses `$sqrt`, so
  it only works in simulation. Python computes complex numbers in floating
  point, so results match exactly while they stay below 2**53.
- A single `return` of variables or expressions (an expression output is
  named `ret`, or `ret0`, `ret1`, ... for several)

- Lists, tables and memories (see below)
- Returning an argument: it gets its own output port, `<name>_out`

Not supported: `while`, recursion, early `return`, keyword arguments in
helper calls, dicts and sets, nested lists, lists of complex values.

Python names that are Verilog keywords (`buf`, `time`, `table`, `type`, ...)
get a `_` suffix in the HDL; keep using the Python names in `attr` and calls.

Variables you don't list in `attr` get the exact width their values need, so
they never overflow. A listed variable keeps its declared width for every
reassignment.

### Lists, tables and memories

A list has a length fixed at translation time (like an unrolled loop); list
`xs` of length N is held as signals `xs_0` ... `xs_{N-1}`. Supported:
literals, `[v] * n`, `append`/`extend`, slices with constant bounds, `+`,
`len`, `sum`/`min`/`max` over a list, `list(...)`, `for v in xs`, indexing
and assignment (`xs[i] = v`, `xs[i] += v`), including Python's negative
indices. Returning a list gives one output per element, and declaring `xs`
in `attr` sets every element's width.

- **Tables / ROMs**: a module-level tuple or list of integers (e.g. an
  S-box, computed in Python if you like) or a constant local list, indexed by
  a variable, becomes a ROM built from logic; indexed by a constant it is just
  that constant.
- **Variable index**: reading `xs[i]` is a balanced mux, writing `xs[i] = v`
  updates the element that matches. An out-of-range index (an `IndexError` in
  Python) reads an unspecified element and writes nothing.
- **List arguments**: `'length': N` in `attr` makes an argument a list of N
  input ports; call `TB_batch()`, `PY_batch()`, `HW()`, ... with a list there.
- **Memories**: a list argument with `'next'` is state, a register file that
  keeps its contents between samples (`'init'` may be one value or a list).
  `'next'` can name the argument itself when it is updated in place and
  returned:

```python
def histogram(bin_, counts):
    counts[bin_] += 1
    return max(counts), counts

attr = {'bin_':   {'signed': 0, 'width': 3},
        'counts': {'signed': 0, 'width': 8, 'length': 8, 'next': 'counts'}}
```

Lists are copied by value in hardware, so `ys = xs` for a list is rejected
(Python would make both names the same list); write `ys = list(xs)`. Memories
made this way are register files, which suits small tables and buffers; for
larger ones use block RAM:

- **Block RAM**: `'memory': 'ram'` on a list argument keeps it as a RAM array
  (FPGA block RAM: a 256 x 16 histogram is 4096 RAM bits and ~60 logic
  elements, against ~11,000 as registers). The function updates it in place
  and doesn't return it. Per sample it may be read at one address and written
  once, reading before writing. A sample then takes two clocks (read, then
  compute and write); `TB_stream`, `HW()` and the stream wrapper account for
  that. `'init'` sets the power-up contents (KEY0 does not clear a RAM).

```python
def histogram(bin_, counts):
    c = counts[bin_] + 1
    counts[bin_] = c
    return c

attr = {'bin_':   {'signed': 0, 'width': 8},
        'counts': {'signed': 0, 'width': 16, 'length': 256, 'memory': 'ram'}}
```

### How results relate to Python

Every named signal holds the exact Python value of its expression, truncated
to the signal's declared width and sign (so declare widths large enough for
what you need to keep). Within an expression, the generated Verilog computes
what Python does, including signed/unsigned mixing, right shifts of negative
values, comparisons and `//` (floor division); `/` matches `int(a / b)`.
Operations whose result depends on high bits (division, right shift,
comparisons) get exactly-sized intermediate wires (`<name>__t<k>`), and
unsigned signals are read through signed views (`<name>__s`), so this costs no
extra hardware. `and`/`or` give 0/1, not Python's operand values.

### Examples

`examples/` has runnable scripts (basic arithmetic, branches, loops, `reg`
signals, complex arithmetic, lists, an AES S-box ROM, a randomized regression,
a streaming low-pass filter, a FIR filter with its delay line in a memory, a
histogram in block RAM, and pipelined SHA-256 rounds). Each checks the hardware against Python in simulation, and
optionally on the virtual board, in Quartus, or on the real board:

```
python examples/sha256_pipeline.py              # simulation
python examples/sha256_pipeline.py --board-sim  # + virtual DE0-Nano
python examples/sha256_pipeline.py --handshake  # + valid/ready wrapper, random stalls
python examples/sha256_pipeline.py --syn        # + Quartus area/timing
python examples/sha256_pipeline.py --board      # + connected DE0-Nano
```

### Tests

```
.venv/bin/python -m pytest -q
QUARTUS_DIR=/opt/intelFPGA/22.1std/quartus/bin .venv/bin/python -m pytest -q   # also runs the Quartus test
```

Simulation tests compare the Python function against the cocotb + Icarus
result and are skipped if Icarus or cocotb is missing.
