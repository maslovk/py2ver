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
`'next'` and `'init'` for state registers, `'length'` and `'memory'` for lists
and memories, and `'frac'`, `'round'` and `'saturate'` for fixed point (see
below). `type: 'reg'` registers the signal; the testbench accounts for the
resulting pipeline latency. Widths of undeclared temporaries are inferred.

### Annotations (optional)

The function is used as written, with everything about the hardware in
`attr`. If you write a function for py2ver anyway, you can put the same
information in annotations instead, and leave `attr` out:

```python
from py2ver import Py2ver, Reg, S, U

def foo(arg1: U[8], arg2: U[8]) -> (Reg(U[9]), S[9], U[9], U[9]):
    ...

p = Py2ver(foo)
```

| Annotation | `attr` equivalent |
|---|---|
| `U[8]`, `S[12]` | `{'signed': 0, 'width': 8}` |
| `Reg(U[9])` | `'type': 'reg'` |
| `Array(U[8], 4)` | `'length': 4` |
| `State(S[16], next='y', init=0)` | `'next': 'y', 'init': 0` |
| `Ram(U[16], 256, init=0)` | `'length': 256, 'memory': 'ram'` |

Parameters (`x: S[12]`), variables (`t: S[16] = ...`, or a bare `t: S[16]`)
and the return (one type per returned value; a returned expression is the
output `ret` / `ret<i>`) can be annotated; other annotations (`x: int`) are
ignored. An `attr` entry is merged over an annotation of the same name, and
`attr` values may be types too (`{'x': S[12]}`).

### Fixed point

A float model needs no changes: give its values a fixed-point format in
`attr` and py2ver builds it in fixed point. `'frac'` is the number of the
`'width'` bits after the binary point (`{'signed': 1, 'width': 16, 'frac':
15}` is Q15, -1 .. 1 - 2**-15); the hardware holds `value * 2**frac` as an
integer.

```python
def lpf(x, y):                      # the float model, as written
    yn = y + 0.1 * (x - y)
    return yn

attr = {'x': {'signed': 1, 'width': 12, 'frac': 10},
        'y': {'signed': 1, 'width': 16, 'frac': 14, 'next': 'yn', 'round': True}}

p = Py2ver(lpf, attr)
p.TB_stream()(xs) == p.PY_stream()(xs)     # hardware == bit-exact model
p.quantization_error(xs)                   # {'yn': (max error, rms error)} vs. the float model
```

What the hardware computes:

- **Assignments to a variable with a format quantize**: floor to its
  fraction bits (`'round': True`: round half up) and wrap to its width
  (`'saturate': True`: clamp). This applies to parameters, state (a state's
  `next` value takes the state's format), list elements, memories and
  returned expressions (`'ret'`, `'ret0'`, ... in attr).
- **Everything else is exact**: `+`/`-` align the binary points, `*` adds the
  fraction bits, and comparisons, `min`, `max`, `abs` and `x if c else y`
  align. A variable without a format holds the widest value assigned to it
  (one whose format would grow on every loop iteration, like `y = y * a`,
  needs a format in attr).
- **`/` by a constant**: a power of two only moves the binary point; any
  other constant multiplies by its reciprocal. Division by a variable isn't
  supported.
- **Float constants** (literals and module-level floats or tables of floats,
  like filter coefficients) are exact if they fit in `const_bits`
  significant bits, else rounded to that many: `Py2ver(f, attr,
  const_bits=18)` (the default).
- `int(x)` truncates toward zero, `math.floor` / `math.ceil` work, `float(x)`
  is `x`, and `math.pi` and other module-level numbers are constants, as are
  `math` / `cmath` functions of constants (`cmath.exp(0.3j)`). `//`,
  `%`, `round()` and bitwise operators aren't supported on fixed-point
  values. `/` between two integers keeps its integer meaning (`int(a / b)`).
- Complex values and math functions work too, see below.

The Python references:

- `PY_batch()` / `PY_stream()` are the **bit-exact model**: your function
  rewritten the way the hardware computes it, run in Python. Simulation, the
  virtual board and the board must match it exactly.
- `PY_float()` runs **your function as written** on the same (quantized)
  inputs: the ideal. `quantization_error(vectors)` gives the maximum and RMS
  error of each output against it.

Inputs are numbers (floats too), quantized to each input's format like an
assignment. Fixed-point outputs come back as floats, and a fixed-point
state's or memory's `'init'` is a number too.

**Complex values.** A complex value is a pair of fixed-point values (`z_re`,
`z_im`), with the format of `z` in attr for both. A complex parameter or
state needs `'complex': True` (it becomes two ports); you pass and get back
Python complex numbers, and a returned complex value is two outputs. `+`,
`-`, `*`, `/` by a real value, `.real`, `.imag`, `.conjugate()`,
`complex(a, b)` and complex constants work, including module-level tables of
them (FFT twiddle factors) indexed by a variable. Lists of complex values
work too (`ys.append(y)`, indexing, slices, `+`, `sum`, `for z in zs`; a
complex list parameter has `'complex': True` and `'length'`). `abs()` of a
complex value (a square root) and division by a complex value aren't
supported.

```python
ROT = cmath.exp(-2j * math.pi * 0.125)

def downconvert(x, z):             # a real signal times a complex oscillator
    y = x * z
    zn = z * ROT
    return y, zn

attr = {'x': {'signed': 1, 'width': 12, 'frac': 11},
        'z': {'signed': 1, 'width': 18, 'frac': 16, 'complex': True, 'next': 'zn', 'init': 1},
        'y': {'signed': 1, 'width': 14, 'frac': 12}}
```

**Math functions.** A call to a `math` function of one argument (`math.sin`,
`math.exp`, `math.sqrt`, `math.tanh`, ...; `sin` if imported that way)
becomes a table (a ROM) of the function over a range of its argument,
configured in attr under the call as written:

```python
attr = {..., 'math.tanh': {'range': (-4, 4), 'bits': 10, 'signed': 1, 'width': 16, 'frac': 14}}
```

The table has up to `2**bits` entries (default 10) at a power-of-two step
covering `range`; each holds the function at the middle of its step, rounded
to the format given. The argument picks its entry (arguments outside the
range use the end entries). Every call of that function shares the
configuration. With `samples=` / `auto_fixed`, the ranges, formats and (with
`max_error`) the table sizes are chosen for you.


### Choosing fixed-point formats automatically

Without `attr`, give example inputs and py2ver chooses the formats itself,
then prints the `attr` it chose (to keep, or adjust and pass next time):

```python
p = Py2ver(lpf, samples=examples)     # vectors, as for PY_batch()
# py2ver: formats chosen for lpf (pass this as attr to keep them):
# attr = {
#     'x': {'signed': 1, 'width': 16, 'frac': 14},
#     'y': {'signed': 1, 'width': 16, 'frac': 14},
#     'yn': {'signed': 1, 'width': 16, 'frac': 14},
# }
```

Without `attr` every parameter is an input. For state, pass the part that
can't be guessed and the rest is filled in: `Py2ver(lpf, {'y': {'next':
'yn'}}, samples=xs, max_error=1e-3)`. The same is available as a function,
`auto_fixed()`, which returns the attr instead of printing it:

```python
from py2ver import Py2ver, auto_fixed

state = {'y': {'next': 'yn'}}
attr = auto_fixed(lpf, examples, state, width=16)                 # 16 bits everywhere
attr = auto_fixed(lpf, examples, state, width=16, max_error=1e-3) # fewest bits for 1e-3
p = Py2ver(lpf, attr)
```

It runs the float function on the example inputs, measures the range of
every parameter and variable, and gives each value that holds floats a
format with enough integer bits for its range (plus `margin=1`):

- `width=W`: every value gets `W` bits (16 for `Py2ver(..., samples=...)`
  unless given); what its integer part leaves over is fraction.
- `max_error=E`: the fraction bits are searched instead, value by value: the
  fewest that keep every output of the bit-exact model within `E` of the
  float function on the examples (`quantization_error`). `E` may be a dict
  `{output: error}`; other outputs are then free. Inputs are your interface,
  so they aren't searched: they keep the formats you give in `attr`, or get
  `width` bits.

The result is an ordinary `attr` dict to read, adjust and keep. Entries you
pass with a `'width'` are kept as they are, and `'next'`, `'init'`,
`'length'` and `'memory'` are kept; a state's next value takes the state's
format, and states loaded from one another (`d2` from `d1` from `x`) share
one. Integer inputs get integer formats (just wide enough), integer
variables keep their exact widths, and float values are signed. `round=True`
/ `saturate=True` add those options to every chosen format.

The ranges come only from the examples, so give inputs that cover the real
range: a value they never reach can still overflow (add `margin` bits or use
`saturate=True` if unsure). Trial designs for the search are built in a
temporary directory.

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
Constants are not registered. Pipelining can't be combined with `'type':
'reg'` signals or block-RAM memories.

In a pipelined design without state, function tables (a fixed-point
`math.tanh` or `np.tanh`, a `sigmoid` helper: tables of 64 entries or more)
are ROMs in block RAM instead of logic. A table read is a stage of its own (the
RAM latches the index at the end of one stage and gives the entry in the next),
so `pipeline` must be at least the number of tables one after another on a
path. The XOR network's forward pass (`examples/xor_hardware.py`) goes from
6624 logic cells at 23 MHz unpipelined to 1736 cells and 5 block RAMs at 61 MHz
with `pipeline=3`.

With state (`'next'`), the state loop (state -> ... -> next state) must finish
in one clock to keep a sample per clock, so it stays in one stage: logic that
only depends on inputs is pipelined before it, logic from the state to the
outputs after it. A valid bit travels with each sample (`vin` -> `vout` on
the module) so gaps in the input never update the state; `TB_stream()`,
`HW()` and the stream wrapper drive it for you. A FIR filter whose delay line
is state goes from 84 to 126 MHz with `pipeline=2`.

### Folding loops (fewer multipliers)

By default every loop is unrolled: a 64-tap FIR becomes 64 multipliers that
take a new sample every clock. `fold` runs a loop's body over several clocks
instead, trading speed for area:

```python
p = Py2ver(fir_step, attr, fold='k')             # 1 copy of the body, N clocks per sample
p = Py2ver(fir_step, attr, fold='k', unroll=4)   # 4 copies, N / 4 clocks per sample
```

`fold` names the loop variable of a top-level `for ... in range(...)` loop
(`fold=True` takes the longest one, and builds a design without loops
unfolded), and `unroll` (dividing the number of
iterations) is how many iterations run per clock. Variables the loop updates
(an accumulator, a running maximum) become registers as wide as their widest
value in the unrolled design, so the results don't change: the Python
references are the same, and simulation must match them exactly. The design
then takes N / unroll clocks per sample, which `TB_batch()` / `TB_stream()`,
`HW()`, `HW_sim()` and the stream wrapper account for (`p.ram_phases`).

A 16-tap integer FIR on the DE0-Nano's EP4CE22 (Quartus, `SYN()`):

| | logic elements | 9-bit multipliers | Fmax | samples / s |
|---|---|---|---|---|
| unrolled | 1996 | 0 | 38 MHz | 38 M |
| `fold='k'` | 551 | 2 | 68 MHz | 4.2 M |
| `fold='k', unroll=4` | 637 | 8 | 63 MHz | 16 M |

A folded loop may assign list elements (`ys[i] = ...`): the list becomes a
register per element. Its bounds may be constant expressions
(`range(len(TAPS))`). One loop per design is folded, and folding can't be
combined with `pipeline`, block RAM or `'reg'` signals.

### Sharing multipliers (`multipliers=N`)

`multipliers=N` runs a design over as many clocks as it needs on N shared
multipliers. Unlike `fold`, it takes any design without `while`: numpy code
written out element by element, several loops one after another, loops with
`if`s inside:

```python
p = Py2ver(train_step, attr, multipliers=4)     # 4 multipliers, p.ram_phases clocks per sample
```

- `for` loops over constants or lists are unrolled, and `if` / `else` becomes
  a selection: both branches are computed and each variable they assign is
  chosen by the condition
- every product runs on one of N shared multipliers (a product by a constant
  too, unless the constant is a power of two), and a sum of products
  accumulates in the multipliers' adders (multiply-accumulate); a long sum is
  split over the multipliers and the parts added at the end
- every function table is a ROM in block RAM, read once per clock
- the rest (additions, shifts, comparisons, clamps) is built as it is, chained
  within a clock up to an estimated 12 ns
- a value used in a later clock than the one making it is kept in a register;
  values from the same multiplier or ROM share registers
- the multipliers' inputs are registered, so a product comes a clock after
  its operands; a few wide products share one wide multiplier

Every operation keeps the width the unshared design gives it, so the results
don't change: the Python references are the same, and simulation must match
them exactly. Like a folded design it takes `p.ram_phases` clocks per sample,
which the testbenches, `HW()` and the stream wrapper account for. Not
supported: `while`, `break` / `continue`, a `return` inside a loop or an
`if`; it can't be combined with `fold` or `pipeline`.

For one loop, `fold` is smaller; `multipliers=` registers its multipliers'
inputs, so it clocks faster. The 16-tap FIR from the folding section:

| | logic elements | 9-bit multipliers | Fmax | clocks per sample |
|---|---|---|---|---|
| `fold='k'` | 551 | 2 | 68 MHz | 16 |
| `multipliers=1` | 791 | 2 | 115 MHz | 17 |
| `multipliers=4` | 1063 | 8 | 88 MHz | 8 |

The XOR network's training step (`examples/xor_hardware.py`) on the EP4CE22:

| | logic elements | 9-bit multipliers | block RAM bits | Fmax | clocks per step |
|---|---|---|---|---|---|
| unshared | doesn't fit: 152 multipliers | | | | 1 |
| `multipliers=4` | 10503 (47 %) | 39 | 62592 | 56 MHz | 45 |

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

### Multi-rate streams (decimation, interpolation)

A decimator takes several input samples per output, an interpolator gives
several outputs per input. Write one call of the function for one block: a
decimate-by-4 filter takes a list of 4 new samples, an interpolate-by-4
filter returns a list of 4. Then mark the list `'serial': True` in attr, and
the stream wrapper moves it one element per handshake:

```python
def decimate(xs, delay):                 # xs: 4 new samples, oldest first
    window = xs[::-1] + delay
    acc = 0.0
    for k in range(N):
        acc += TAPS[k] * window[k]
    new_delay = window[:N - 1]
    return acc, new_delay

attr = {'xs': dict(Q15, length=4, serial=True), 'delay': dict(Q15, length=N - 1, next='new_delay'), ...}
p = Py2ver(decimate, attr, fold='k', unroll=N // 4)
```

`<name>_stream` then has a single `s_xs` port: 4 transfers make one call,
element 0 first (other inputs are taken with the last element). For a
`'serial'` returned list, each result is that many transfers on one `m_<name>`
port (other outputs repeat on each). The core itself and the Python
references are per block as before (`PY_stream()` takes blocks);
`TB_handshake()` feeds and collects the wrapper one element at a time and
gives the results per block. Folding the filter so it takes as many clocks
per block as samples per block (`unroll=N // 4` above) keeps up with one
sample per clock while sharing the multipliers.

### Signal chains

A chain of blocks is plain Python: a function that calls the blocks'
functions. py2ver inlines them into one design, so the whole chain gets one
set of formats, one Python reference and one stream interface. State passes
in and out of each block as usual, and a rate change is a loop over the
samples of a block:

```python
def downconvert(x, z):              # one sample
    y = x * z
    zn = z * ROT
    return y, zn

def decimate(xs, delay):            # 4 samples in, 1 out
    ...
    return acc, new_delay

def receiver(xs, z, delay):         # the chain, on a block of 4 samples
    ys = []
    for x in xs:
        y, z = downconvert(x, z)
        ys.append(y)
    acc, delay = decimate(ys, delay)
    return acc.real * acc.real + acc.imag * acc.imag, z, delay

state = {'xs': {'serial': True},
         'z': {'next': 'z', 'init': 1, 'complex': True},
         'delay': {'length': 4, 'next': 'delay', 'complex': True}}
p = Py2ver(receiver, state, samples=blocks, max_error=1e-3, round=True)
```

`examples/receiver.py` is this chain: its formats are chosen for the error
target, it matches the bit-exact model in simulation and takes one sample per
handshake through the stream wrapper. `fold` applies to the chain too (the
longest loop, here the filter's taps).

**As separate stages.** Built as one design, the chain runs once per block,
so the downconverter, called for each of the 4 samples, is built 4 times.
`stages=True` builds each call as its own core behind a stream wrapper, and
`hdl/<chain>_stages.v` connects them with valid/ready handshakes: one
downconverter running at the sample rate, the filter taking its 4 samples one
per transfer (a serial input), and the detector. A stage that gets several
clocks per call (here 4 per block) has its longest loop folded to use them.

```python
p = Py2ver(receiver, attr, stages=True)
p.TB_stages()(blocks) == p.PY_stages()(blocks)     # the staged hardware vs. the chain
p.SYN_stages(q)                                    # its area and timing
```

Each stage is a generated function around the call, whose parameters and
results are the chain's variables, so the chain's attr applies unchanged and
the stages compute exactly what the fused design does. `TB_stages()` feeds
the top module one sample per transfer and gives each block's results like
`PY_stages()` (the chain's `PY_stream()` without its state).
`stage_options={'decimate': {'fold': ..., 'unroll': ..., 'pipeline': ...}}`
overrides a stage's folding. The 8-tap receiver on the EP4CE22, both taking
one sample per clock:

| | logic elements | 9-bit multipliers | Fmax |
|---|---|---|---|
| one design | 3569 | 20 | 20 MHz |
| `stages=True` | 1071 | 24 | 64 MHz |

To be split, the chain function must be a list of stages: calls of your
functions at its top level (once per block), `for v in xs:` loops holding one
call and appends of its results (once per sample), `ys = []` for the lists
they fill, and a return. Data flows only through the calls' arguments and
results, each stage takes its inputs from the stage before it (the first from
the chain's one input), state belongs to one stage, and the results come out
of the last stage; anything else is reported (`NotSplittable`), and the chain
still builds as one design.

Helpers get copies of list arguments, so a helper can't change a list
argument in place (copy it first and return it).

### numpy and classes

Code written with numpy arrays, and methods of objects, works as it is
(install the `numpy` extra). `examples/xor_network.py` is a small numpy
neural network, unchanged; `examples/xor_hardware.py` builds its training
step and its trained forward pass:

```python
net = XORNetwork()
train = Py2ver(net.train, samples=[(X, Y)] * 600, width=24,    # one training step, on 4
               multipliers=4)                                   # shared multipliers
steps = train.PY_stream()([(X, Y)] * 600)                       # loss < 0.01 at step 397
for name, value in train.state_values(steps[396]).items():      # {'W1': array, ...}
    setattr(net, name, value)                                   # the weights trained in hardware

infer = Py2ver(net.predict, samples=points, max_error=1e-2,     # weights: constants,
               pipeline=3)                                      # one input per clock
infer.TB()(np.array([0.8, 0.3]))                                # -> 0.875 (numpy: 0.876)
```

py2ver writes such a function out as scalar Python, checks it against the
original in float on the examples, and builds that:

- calls of the object's methods (`self.forward(x)`) and of your functions are
  inlined; a function of one value that uses `np.exp` & co. or divides by a
  variable (`sigmoid`) is built as a table of the whole function
- the object's attributes: read and never written are constants (trained
  weights), read and then written (`self.W1 -= ...`) are state starting from
  the object's values, written before read (`self.h = ...`) are variables
- arrays take the shapes of the example inputs, so every operation is written
  out element by element: `@`, `+ - * /` and `**` with broadcasting,
  indexing and slices with constants, `.T`, `.sum(axis, keepdims)`,
  `np.mean`, `np.sum`, `len`, `np.atleast_2d`, `np.zeros` / `np.ones`,
  `np.maximum` / `np.minimum`, `abs`, and elementwise `np.tanh` / `np.exp` /
  `np.log` / ... (tables)
- a method that is a loop with `print` / `break` on a condition (a training
  loop) is built as one step of the loop: the variables its condition uses
  (the loss) are the step's results, and the loop itself stays in Python

Calls take the arguments the original does (numpy arrays); results come out
flat, in row-major order. `samples=` is needed (for the shapes), and the
examples should cover the inputs' real range: formats and tables are chosen
from them. Written out, numpy code is large and has no loops: build a
training step with `multipliers=` (see Sharing multipliers) and a forward
pass with `pipeline=` (its tables then go to block RAM).

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
  `z` in `attr` sets both widths. Complex parameters need `'complex': True`
  in attr. `abs()` of a complex value uses `$sqrt`, so it only works in
  simulation. Python computes complex numbers in floating point, so results
  match exactly while they stay below 2**53. Complex fixed point: see above.
- `return` of variables or expressions (an expression output is named `ret`,
  or `ret0`, `ret1`, ... for several), also early inside an `if` or a loop:
  every return must give the same number of values
- `break` and `continue` in loops
- `while` loops, built as their largest number of iterations, each guarded by
  the loop's test: worked out for a counting loop (`i = 0` before it, `while
  i < 10:` and one `i += 1` in it), else given as `attr['while'] = N` (for
  the function's while loops), or measured on the examples with `samples=`
  (and printed with the formats). If Python would need more iterations, the
  hardware stops early. A while loop can be folded (`fold=True`) to run one
  iteration per clock, e.g. a bisection or a GCD
- A variable first assigned in only one branch of an `if` takes that
  branch's value (Python can only read it after that branch)

- Lists, tables and memories (see below)
- List comprehensions and generator expressions (`[x * g for x in xs]`,
  `sum(x * y for x, y in zip(xs, ys))`), written out as lists: over lists of
  known length, `range(...)` of constants, `enumerate` and `zip` of those,
  with several `for`s; an `if` may test the index but not the values (the
  list's length must be known when translating)
- Returning an argument: it gets its own output port, `<name>_out`

Not supported: recursion, keyword arguments in helper calls, dicts and sets,
nested lists, `try` / `with`, a loop's `else`.

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
  and doesn't return it. A sample may read it at several addresses (each read
  address gets its own copy of the RAM, like Quartus' own read-port
  replication), including addresses that come from another read
  (`mem[mem[i]]`), and write it several times (one write per clock, in program
  order, so the last write to an address wins); all reads must come before the
  writes. A sample takes (read levels) + (writes) clocks, e.g. 2 for read then
  write, 3 for a swap; `TB_stream`, `HW()` and the stream wrapper account for
  that (`p.ram_phases`). `'init'` sets the power-up contents (KEY0 does not
  clear a RAM).

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
fixed-point FIR filter from a float model, a biquad filter with automatically
chosen formats, a complex downconverter, a soft clipper using a `math.tanh`
table, a 32-tap FIR folded onto 4 multipliers, a numpy XOR network trained on 4
shared multipliers and run pipelined in hardware, a decimate-by-4 filter fed one
sample per handshake, a receiver chain built from plain Python functions, a
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
result and are skipped if Icarus or cocotb is missing. `tests/test_combinations.py`
builds a set of designs (state, lists, complex values, tables, block RAM,
helpers, keyword names) with each option (pipelining, folding, automatic
formats, serial ports, stages, annotations) and checks every result against
the references.
