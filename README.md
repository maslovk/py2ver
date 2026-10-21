# py2ver

[![tests](https://github.com/maslovk/py2ver/actions/workflows/tests.yml/badge.svg)](https://github.com/maslovk/py2ver/actions/workflows/tests.yml)

## Simple Verilog code generator from Python source

py2ver turns a plain Python function into a Verilog module and gives you back
callables with the same signature that run it in simulation (cocotb + Icarus)
or on an FPGA (DE0-Nano, over UART). Code written with numpy, and PyTorch
models and training loops, are built as they are written too.

### Setup

```
python3 -m venv .venv
.venv/bin/pip install "py2ver[hw] @ git+https://github.com/maslovk/py2ver"
```

For development, from a clone: `.venv/bin/pip install -e ".[test,hw]"`.
The `hw` extra (pyserial) is only needed to talk to a physical board; the
`numpy` and `torch` extras for designs written with numpy or PyTorch.

Simulation also needs [Icarus Verilog](https://steveicarus.github.io/iverilog/)
(`iverilog`, `vvp`) on `PATH`; with [Verilator](https://www.veripool.org/verilator/)
installed too, long simulations of large designs run on it (see "Simulation
speed"). The hardware path needs Intel Quartus and a DE0-Nano board with a
USB-UART adapter.

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
to the current working directory. With `write=False` nothing is written until a
simulation or a project needs it: the Verilog is in `p.verilog` (every file for
`hdl/` in `p.hdl_files`), and `p.write_hdl()` writes it.

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
  other constant multiplies by its reciprocal. **By a variable**, the quotient
  keeps the dividend's fraction bits, truncated toward zero (`a / b` is
  `int((a_raw << fb) / b_raw)` on the raw values; a constant dividend has
  `const_bits` significant bits, so `1.0 / x` keeps 17 fraction bits); a
  sequenced design divides a bit per clock (and by 0 gives 0), others with a
  combinational divider.
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
range use the end entries). `sin` and `cos` (`math`'s, numpy's, torch's)
over a range wider than a period are a table of one period instead, its
`2**bits` entries over `2 pi`, indexed by the argument's phase (a product by a
constant, a shift and a mask): every argument, beyond the range too, gets its
own value, at a step that doesn't grow with the range (a timestep embedding's
`sin(t * f)` for `t` up to 1000). Every call of that function shares the
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
format with enough integer bits for its range (plus `margin=1`), and each
math function a table over the arguments seen (each end's magnitude `2 **
margin` times as large):

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
/ `saturate=True` add those options to every chosen format, and `margin=`
integer bits (1 by default) are added above every range seen (`Py2ver(...,
samples=..., margin=2)` too).

The ranges come only from the examples, so give inputs that cover the real
range: a value they never reach can still overflow (add `margin` bits or use
`saturate=True` if unsure). With state that changes over a long run (a
network's weights and gradients while it trains), the examples should cover
that run, not just its start. Trial designs for the search are built in a
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
- a value a multiplier reads two or more clocks after it was made waits in
  that multiplier input's register file, a block RAM written when the value
  is made; one made a clock before comes from the multiplier's or table's
  last-result register. Anything else read in a later clock is kept in a
  register (values from the same multiplier or table share one)
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
| `multipliers=1` | 786 | 2 | 120 MHz | 17 |
| `multipliers=4` | 1087 | 8 | 96 MHz | 8 |

The XOR network's training step (`examples/xor_hardware.py`) on the EP4CE22:

| | logic elements | 9-bit multipliers | block RAM bits | Fmax | clocks per step |
|---|---|---|---|---|---|
| unshared | doesn't fit: 152 multipliers | | | | 1 |
| `multipliers=4` | 9072 (41 %) | 39 | 64823 | 57 MHz | 45 |
| `multipliers=2` | 7378 (33 %) | 25 | 64931 | 56 MHz | 78 |

### Sequenced loops (`sequential=True`)

`sequential=True` builds a design's loops as they are written: a small state
machine runs the program a step at a time, for designs far too big to unroll
(a network's forward pass, thousands of multiply-accumulates over arrays):

- the program is for-range loops over straight-line blocks; each block is a
  state, run once per iteration of the loops around it, and afterwards the
  loop counters choose the next one
- loop counters and scalars are registers; lists of 32 or more elements are
  block RAMs (each read in a step a port), shorter ones registers; constant
  tables of 64 or more entries are ROMs in block RAM
- a block takes a few clocks: block-RAM and ROM reads give their values a
  clock after their address, so they are issued level by level (a read whose
  address needs another read waits for it); then everything is computed, and
  the registers and the RAMs written (one write per RAM per clock). A product
  of two computed values gets a register after its multiplier (a clock later:
  two multipliers in a row don't make 50 MHz). A division by a variable takes
  a clock per quotient bit, an exact square root (see below) a clock per two
  bits of its argument
- identical layers one after the other (a Transformer's blocks: the same code,
  each with its own weights) are one loop over them, the hardware of one
  layer: each layer's weight tables are one table, read at the layer's part,
  and the lists the next layer reads (its input) copied at the end of each
  turn. A value in it gets the integer bits of all the layers and the fraction
  bits of the one with the smallest values (its formats are chosen per layer:
  as precise as each written out, a few bits wider). `PY2VER_FOLD=0` keeps
  the layers written out
- a list in block RAM made by `[0] * n` is filled by a loop, unless the loop
  after it writes every element before anything reads it
- an innermost loop whose body is one block is pipelined: an iteration
  starts every clock, each value computed at its stage of the iteration and
  carried to later stages by delay registers, and a value carried from one
  iteration to the next (`acc = acc + x[i] * w[i]`) kept at one stage, so the
  next iteration finds it a clock later. A loop of `N` iterations then takes
  `N + stages - 1` clocks instead of `N * stages` (a matrix product's
  multiply-accumulates: one a clock instead of one every two). A loop that
  divides or takes square roots is pipelined with them written out a step a
  stage (a quotient bit, a root bit), a register stage per step, when that
  saves at least 5 % of a sample's clocks (Adam's update of a large weight
  matrix: yes; softmax's ten divisions: no). A loop isn't pipelined if it
  writes a list held in registers, writes one RAM twice, reads a RAM it writes
  anywhere but at the element it writes (a different one each iteration: `m[i]
  = 0.9 * m[i] + g[i]` is fine, `a[i] = a[i - 1] + x[i]` isn't), or carries a
  value through a RAM read. Which loops weren't pipelined, and why, is logged
  (INFO: `-v` in the examples); a product left without its register or a loop
  left unpipelined because py2ver can't bound a value is a WARNING
- `multipliers=N` runs the multiply-accumulate loops (an innermost loop that
  multiplies computed values and writes no list: a matrix product's sums) `N`
  iterations a clock: the body is written once per lane, each lane with its
  own read ports, the copies in order so a sum carried through them is
  Python's. The lanes divide the loop's length (a loop of 10 asked for 4 runs
  on 2). A table the lanes read at their own entries (`W[i * 128 + o]`, `i`
  the loop's variable, `o` an outer one's) is banked: each lane gets a ROM of
  the entries it reads (rows `u`, `u + N`, `u + 2N`, ... of `W`), so the lanes
  together hold the table once, however many there are. Another table (read
  at an index that isn't a sum of loop variables times constants) is read by
  two lanes through the two ports of one block RAM, a copy per two lanes.
  `rate=` picks the fewest lanes that keep up
- a memory argument (`'memory'` in attr) is state: its block RAM (or its
  registers, if it is short) keeps its values from one sample to the next,
  starting from its `'init'`, and isn't a port
- `sdram=True` (or a number of bits) puts the largest lists and tables in the
  DE0-Nano's SDRAM until the rest fits its block RAM (counted in whole M9K
  blocks, 9,216 bits each: a short list takes one all the same): the module gets the
  SDRAM's pins and a controller (`templates/hw/sdram`), one access a clock,
  a read's data 6 clocks after its address, and its clock enable stops while
  the SDRAM opens a row or refreshes. A pipelined loop starts an iteration
  every as many clocks as it makes SDRAM accesses, each on its own clock (a
  product of two lists in the SDRAM: a multiply-accumulate every 2 clocks).
  The simulations include a model of the chip, loaded with the tables: the
  PyTorch MNIST network with `sdram=True` keeps 16 Mbit in the SDRAM and takes
  39,569,650 clocks an image (0.79 s at 50 MHz; 19.8 million all in block
  RAM), simulated in Verilator as its bit-exact model. An input list of 64
  values or more is in the SDRAM too (no ports: each sample's values are
  written there before it runs), and so is a returned list of 64 values or
  more (no ports either: read back after each sample, in its place among the
  outputs). A list or table of values up to 32 bits wide takes two words
  each (two accesses, the low word first), an input or returned list too. Lists not needed at the same time
  (a layer's activations, done with when a later layer's are written) share
  addresses: a list's from the program's first top-level step using it to
  its last. `SYN()` gives the SDRAM's pins their
  DE0-Nano places and timing: the MNIST network fits the EP4CE22, 7,663
  logic elements (34 %), 297 kbit of block RAM (49 %), 16 multipliers, 58 MHz
  (timing met at 50 MHz). `HW()` and `HW_sim()` load the SDRAM over the
  UART (see "On the board" below); the whole board project for MNIST fits
  too: 8,096 logic elements (36 %), 297 kbit, 56 pins, timing met at 50 MHz.

The number of clocks is fixed by the loop bounds (`p.ram_phases` per sample),
and register widths come from following the program with intervals, so the
results are exactly the Python program's. The loops need constant bounds, and
a loop can't be inside an `if`.

### Choosing the structure from a rate (`rate=`)

Instead of picking `fold`, `unroll` or `multipliers` yourself, give the rate
the design must keep up with: `rate=` samples per second at `clock=` Hz (50
MHz, the DE0-Nano's, by default). Each sample may then take `clock / rate`
clocks, and py2ver builds the smallest structure that finishes in time:

- one clock per sample: every operation its own hardware (unrolled)
- a loop holding most of the products: that loop folded, onto the fewest
  copies of its body (`unroll`) that finish in time
- else: the fewest shared multipliers (`multipliers=N`) whose schedule fits
- a sequenced design (`sequential=True`, or numpy code too big to write out):
  the fewest lanes for its multiply-accumulate loops (`multipliers=N`, see
  "Sequenced loops") that finish in time

```python
p = Py2ver(fir_step, attr, rate=10e6)
# py2ver: 1e+07 samples/s at 50 MHz leaves 5 clocks per sample: fold=True, unroll=4
#         (4 clocks per sample; the loop folded onto 4 copies of its body)
train = Py2ver(net.train, samples=[(X, Y)] * 600, width=24, rate=1e6)
# py2ver: 1e+06 samples/s at 50 MHz leaves 50 clocks per sample: multipliers=4
#         (45 clocks per sample; 152 products on 4 multipliers)
```

The choice is printed and kept in `p.choice` (pass its options to keep it).
When no shared schedule is fast enough (a chain of dependent steps longer than
the clocks allowed), the design is built unrolled and the message says how
many clocks the fastest shared schedule needs. Whether the clock itself is
met is for `SYN()` to tell (add `pipeline=` to an unrolled design to speed it
up).

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

#### Networks too big to write out: loops

A network of real size can't be written out element by element: the
64 -> 128 -> 64 -> 10 digit recogniser in `examples/digits_network.py` (a list
of `Dense` / `ReLU` layer objects, softmax, argmax, unchanged) is 17,000
multiply-accumulates per image. When the scalar translation can't take a
design (here: the loop over `self.layers`), or with `sequential=True`, py2ver
keeps its loops instead:

- the method runs at translation time, the way Python would, on a copy of
  the object: the object, its lists of layers, `isinstance()`, `hasattr()`,
  loops over lists (with `break` / `continue`), ifs on them, comprehensions
  and dicts are real, and each method called is inlined with its own `self`
- each array is a view (shape, strides, offset) of a flat list or a constant
  table (an attribute read and never written: trained weights), so `.T`,
  `.transpose(...)`, `np.atleast_2d`, `sliding_window_view`, constant slices
  and indices with `None` and `...` cost nothing (`.reshape` of a view in
  another order is a copy); each numpy operation is a loop nest writing a new
  list: `@`, `+ - * / **` and comparisons with broadcasting, `np.exp` /
  `np.log` / `np.tanh` / ... (tables), `np.sqrt` (exact: an integer square
  root of the raw value), `.sum` / `.max` / `.mean` / `.argmax` with `axis`
  (or a tuple of axes) and `keepdims`, `np.maximum` / `np.minimum`, `abs`,
  `np.pad` (zeros), `np.concatenate`
- indexing with arrays of integers (`probs[np.arange(n), labels]`, the labels
  an input) reads the elements they pick, and assigning to it
  (`grad[np.arange(n), labels] -= 1`) writes them, one after another
- elementwise operations are computed where they are used: a chain of them
  (`m[:] = beta1 * m + (1 - beta1) * g`) is one loop, each operation's
  result still a variable with a format of its own, as each numpy result
  would be; work whose result nothing uses is left out
- an array divided by a computed scalar is multiplied by its reciprocal
  (computed once), and `c ** n` with a computed integer `n` (Adam's
  `beta1 ** t`) is the product of the constants `c ** 2 ** k` over `n`'s bits
- the weight tables get formats like the variables (from their values)
- the loops are built by the sequencer (see "Sequenced loops")

```python
net = DigitNetwork()
net.fit(X_train, y_train, epochs=30)                        # in Python, as written
p = Py2ver(net.predict, samples=list(X_train[:200]), width=16)
p.PY_batch()(list(X_test))                                  # the bit-exact model
p.TB_batch()(list(X_test[:20]))                             # the hardware (Verilator)
```

`examples/digits_hardware.py` (needs scikit-learn for the dataset), on the
DE0-Nano's EP4CE22:

| | test accuracy | logic elements | block RAM | 9-bit multipliers | Fmax | per image |
|---|---|---|---|---|---|---|
| numpy (float) | 98.6 % | | | | | |
| hardware, 16 bits | 98.9 % (same digit as numpy for 99.7 %) | 3217 (14 %) | 270 kbit (44 %) | 6 | 74 MHz | 18,377 clocks: 0.37 ms at 50 MHz |
| 2 lanes (`multipliers=2`) | the same | 3321 (15 %) | 273 kbit (45 %) | 12 | 71 MHz | 9,865 clocks: 0.20 ms at 50 MHz |
| 4 lanes | the same | 3528 (16 %) | 279 kbit (46 %) | 24 | 68 MHz | 5,609 clocks: 0.11 ms at 50 MHz |
| 8 lanes | the same | 3964 (18 %) | 287 kbit (47 %) | 48 | 62 MHz | 3,481 clocks: 0.07 ms at 50 MHz |

The weight tables are banked, so lanes cost multipliers, not weight memory
(the few extra kbit are the activations, a copy per lane's read port). 16
lanes take 2,417 clocks and reach 54 MHz, the least: a sum is carried through
all the lanes in a clock, rounded at each, so its additions can't be regrouped.

#### Training: the object's arrays as state

The training step, `net.train_batch(x, y_onehot, lr)` (forward pass,
backpropagation through the list of layers, Adam), is built the same way, as
it is. What the method reads and then changes is state:

- the weights and biases and Adam's moments (`p -= ...`, `m[:] = ...`, in
  place or reassigned) and the step counter `self.t` (`self.t += 1`) become
  memory arguments of the loop program (`'memory'` in attr, `'init'` the
  object's values), which the sequencer keeps in block RAM (registers if
  short) from one step to the next; the counter gets 32 bits
- attributes written before they are read (`self.x`, `self.mask`,
  `self.dW`) are variables, and the gradient computed for the layer before
  the first is never used, so it isn't computed
- an array the object holds in a dict or a list (not as an attribute) is
  state too if the method changes it, and so is one the method makes with
  `np.zeros` / `np.ones` that Python code it calls keeps in the object: an
  optimiser's moments made on its first step by `dict.setdefault(id(p),
  ...)` start at zero and are kept
- `rng.random(size)` of a numpy `Generator` draws from a random number
  generator in hardware (xorshift32, its state kept between steps): dropout
  works, with numbers of its own rather than numpy's stream (the translation
  is checked against the original with these numbers in place of `rng`'s)
- attributes read and never written are constants, built in: changing one
  between steps (`net.opt.lr /= 4`) needs another build
- the object itself is left as it is (the translation runs on a copy);
  `p.state_values()` gives the state after the last `PY_stream()` as
  `{(object, attribute): array}`, to put back on a network (an array held in
  a dict: `(network, 'opt.state[id(.layers[0].W)][0]')`, its path)

```python
steps = batches(epochs=3)                                    # [(x, y_onehot, lr)], 32 images each
train = Py2ver(net.train_batch, samples=steps[:44], width=24, saturate=True, margin=2)
losses = train.PY_stream()(steps)                            # training in fixed point (bit-exact)
for (obj, name), value in train.state_values().items():     # the trained weights
    setattr(obj, name, value)
train.TB_stream()(steps[:2])                                 # the hardware, simulated
```

Formats come from a run over the first epoch: activations, gradients and
Adam's moments grow and shrink as the network learns (activations to 2.7
times epoch 1's, softmax's `np.exp` arguments to 3 times), so the examples
should cover training, or `margin=` leave room for it. `examples/digits_training.py` (3 epochs):

| | loss, epochs 1 / 2 / 3 | test accuracy | per step (32 images) |
|---|---|---|---|
| numpy (float) | 1.863 / 0.838 / 0.419 | 93.3 % | |
| fixed point, 24 bits (`margin=2`) | 1.865 / 0.840 / 0.418 | 93.1 % | 1,688,064 clocks: 34 ms at 50 MHz |

The bit-exact model trains the three epochs in 23 seconds; the hardware,
simulated in Verilator, gives the model's losses bit for bit (2 steps). The
step is mostly its matrix products, a multiply-accumulate a clock (about
1.4 million); Adam's division and square root are pipelined, an element a
clock, for the two large weight matrices. With `multipliers=4` the products
run four a clock: 664,016 clocks a step (13 ms at 50 MHz).

The state is 1.2 Mbit at 24 bits, twice the DE0-Nano's block RAM, so this
one is for simulation. The batch's 2,048 inputs are ports: a list argument
that long read by index is an array of its ports, and the testbench reads
a vector that wide in parts.

#### A convolutional network

`examples/cnn_network.py` is a CNN in numpy, conv -> pool -> conv -> pool ->
dense -> dense (13,706 parameters), with dropout, shifted training images
and an Adam that keeps its moments in a dict. Its forward pass and its
training step are built as they are:

- `Conv2D.forward`'s im2col (`np.pad`, `sliding_window_view`, a transpose
  and a reshape) is a copy of the padded image into the patches' order, and
  the convolution a matrix product of the patches and the filters
- `MaxPool2x2` is a reduction over the 2x2 blocks of a 6-dimensional view,
  and its mask a comparison with the maxima broadcast back (`None` indices)
- the backward pass adds each patch's gradient back with `+=` on 3x3 shifted
  views (col2im), and the gradient of the softmax picks each image's label
  (`grad[np.arange(n), labels] -= 1`)

`examples/cnn_hardware.py` (inference), on the DE0-Nano's EP4CE22:

| | test accuracy | logic elements | block RAM | 9-bit multipliers | Fmax | per image |
|---|---|---|---|---|---|---|
| numpy (float) | 98.3 % | | | | | |
| hardware, 16 bits | 98.3 % (same digit as numpy for 100 %) | 4648 (21 %) | 355 kbit (58 %) | 8 | 77 MHz | 110,641 clocks: 2.2 ms at 50 MHz |
| 2 lanes (`multipliers=2`) | the same | 5132 (23 %) | 380 kbit (63 %) | 14 | 70 MHz | 69,361 clocks: 1.39 ms at 50 MHz |

On 4 lanes the activations read by the lanes (the patches) are copied per
lane, which takes more block RAMs (74) than the EP4CE22 has (66).

`examples/cnn_training.py` builds `net.train_batch(x, labels)` (24 bits,
formats from the first 8 steps, `margin=2`) and trains 2 epochs of the
script's shifted images in fixed point. Its dropout draws other numbers than
numpy's, so the losses agree on average rather than step by step:

| | loss, epochs 1 / 2 | test accuracy | per step (32 images) |
|---|---|---|---|
| numpy (float) | 2.398 / 1.845 | 78.1 % | |
| fixed point, 24 bits | 2.362 / 1.805 | 78.6 % | 10,360,533 clocks: 207 ms at 50 MHz |

A step is 8.5 million multiply-accumulates (the convolutions' products
forward, for the filters' gradients and back), most of its clocks. Verilator
simulates a step in 3 minutes, the same as the model's; the state (weights,
moments: 41,120 numbers, 1 Mbit at 24 bits) is for simulation.

### PyTorch

PyTorch code is built the same way, as it is (the `torch` extra): a
model's forward pass, and a training loop's step with autograd, an optimizer
and a learning-rate schedule.

```python
model.eval()
p = Py2ver(model, samples=images, width=16, round=True)      # its forward(), in eval mode
p.PY_batch()(images)                                         # ten scores an image

train = Py2ver(train_one_epoch,                              # the function, as written
               samples=[(model, batches, optimizer, scheduler, device)],
               width=24, round=True, saturate=True, margin=2)
train.PY_stream()(batches)                                   # (total_loss, correct, seen) a step
```

- calling a module: `nn.Sequential` runs its children; `Conv2d` (im2col, as
  above, then one matrix product), `BatchNorm1d/2d`, `MaxPool2d`, `Linear`,
  `Embedding` (a gather of its rows), `LayerNorm` (each row's mean and
  variance), `GroupNorm` (each group's), `GELU` and `SiLU` (tables), `ReLU`,
  `Flatten`, `Dropout`, `Identity`, `Softmax`, `LogSoftmax`, `Tanh` are built
  from the translator's operations;
  a module of your own runs its `forward()` as written. Parameters and buffers
  are tables, or state when a training step changes them; a parameter two
  modules share (weight tying) is one table
- `torch` / `torch.nn.functional` functions (`relu`, `gelu`, `silu`,
  `softmax`, `log_softmax`, `cross_entropy`, `layer_norm`, `group_norm`,
  `embedding`, `conv2d`, `linear`, `max_pool2d`, `dropout`, `flatten`, `cat`,
  `stack`, `argmax`, `exp`, `log`, `sin`, `cos`, `rsqrt`, `pow`,
  `interpolate` (`mode='nearest'` by a whole factor: a view), ...) and tensor
  methods with torch's meanings (`view`, `reshape`, `flatten`, `transpose(d0,
  d1)`, `permute`, `unsqueeze`, `squeeze`, `split`, `chunk`, `unbind`,
  `repeat_interleave` (a view),
  `masked_fill`, `size()`, `max(dim)` giving values and indices, `sum` /
  `mean` / `argmax` with `dim` and `keepdim`, `pow`, `item()`, `to()`,
  `type_as()`); `@` broadcasts its leading dimensions (attention's heads), a
  reshape that only splits or merges contiguous dimensions stays a view, and
  `x[:, [-1], :]` (a list of indices) is a slice when they are evenly spaced
- `F.scaled_dot_product_attention` as torch's reference computes it (the
  scale `1 / sqrt(E)` unless given; `is_causal=True` its lower-triangular
  mask, as `masked_fill(-inf)` below); `examples/mini_llama_hardware.py`
  builds a Llama-style model (RMSNorm, RoPE, grouped-query attention, SwiGLU)
  this way, `examples/diffusion_hardware.py` a diffusion U-Net (GroupNorm, FiLM,
  attention at low resolution, upsampling, skip connections)
- a Python list of tensors (a U-Net's skip connections) is a list:
  `append`, `pop`, `extend`, `insert` run as Python
- `masked_fill(mask, -inf)` with a constant mask (causal attention), then a
  softmax: the mask stays a mask, the softmax's exponentials multiplied by it,
  so the masked weights are 0 as with torch's -inf, with no infinities in
  fixed point
- in eval mode BatchNorm is a scale and a shift per channel, computed from its
  running statistics when translating, and Dropout nothing; in training mode
  BatchNorm uses the batch's mean and variance and updates its running
  statistics (state), and Dropout draws from the generator in hardware
- a function whose body loops over an argument given as batches (a training
  epoch) is built as one step of that loop: the statements before it run as
  Python (`model.train()`, `total_loss = 0.0`), the loop's body is the step,
  its target (`images, labels`) the inputs, and the locals it changes are
  state and the step's results. Its objects (the model, the optimizer, the
  scheduler) are copied together, so the optimizer's parameters are the
  model's
- `loss.backward()`: reverse-mode gradients, as loops (`autograd.py`): each
  operation on values that depend on a parameter is recorded, and backward
  writes each one's gradient code in reverse; a list's gradient is a list of
  its length, so a view's contribution (a transpose, a convolution's
  overlapping windows) is added at its positions, as col2im does. Max
  pooling's gradient goes to each window's first maximum, as torch's does
- `optimizer.step()`: `torch.optim.Adam`, `AdamW` and `SGD` as torch computes
  them (the single-tensor algorithm), their state (moments, made as zeros on
  the first step; the step count, an integer) kept by the hardware;
  `optimizer.zero_grad()` and `model.train()` run as Python
- `torch.nn.utils.clip_grad_norm_`: the gradients' total norm, each gradient
  scaled by `min(1, max_norm / (norm + 1e-6))`, as torch does
- `scheduler.step()`: `OneCycleLR`'s learning rate and momentum (Adam's beta1)
  depend only on the step's number, so torch's values for every step are
  tables, read at the scheduler's step count; they are the optimizer's
  `param_groups[0]['lr']` and `['betas'][0]`, state
- the translation is checked against torch on the examples: a training step
  against torch's own training over the first two batches (on copies), with
  the hardware's random numbers. `p.hardware_random()` gives the original
  those numbers, to compare it with the hardware step by step

Deep networks want `round=True`: a sum of hundreds of products truncated term
by term loses half a bit each time, and four convolutions of that shrank the
MNIST network's scores by a quarter. The examples' ranges are all the formats
know, so give samples that cover the inputs (a GPT: windows with the rare
characters too) or more `margin`; and a table spread wide (attention's
exponentials, whose arguments reach -50) wants more entries than the default
1,024: `attr={'np.exp': {'bits': 14}}` (a table's `bits` and `range` in attr
are kept when the formats are chosen).

`examples/pytorch_mnist_cnn.py` is a PyTorch CNN for MNIST (28x28 digits, two
blocks of two 3x3 convolutions with BatchNorm, 468,010 parameters, AdamW and
a one-cycle schedule), unchanged. `examples/pytorch_mnist_hardware.py` builds
its trained forward pass (16 bits, rounded; the model of one epoch):

| | test images | scores | per image |
|---|---|---|---|
| torch (float) | 20 of 20 | | |
| fixed point, 16 bits | 20 of 20, the same digits as torch | within 0.034 of torch's | 19,798,182 clocks: 0.40 s at 50 MHz |

Verilator simulates an image in 33 s, the same as the model's. The weights
(7.5 Mbit) are ROMs: a table longer than 4,096 entries is written to a file
next to `main.v` and read with `$readmemh`, which simulators and Quartus
take, rather than as an assignment an entry. They are more than the
DE0-Nano's block RAM, so this one is for simulation.

`examples/pytorch_mnist_training.py` builds one step of its
`train_one_epoch` loop (24 bits, batches of 2 of its augmented images) and
trains 4 steps in fixed point, torch the same 4 with the hardware's dropout
numbers:

| step | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| torch (float) | 2.7769 | 2.2949 | 2.0928 | 2.2563 |
| fixed point, 24 bits | 2.7784 | 2.2950 | 2.0960 | 2.2553 |

After the 4 steps the weights of `Linear(3136, 128)` are within 0.0004 of
torch's. A step is 135,357,363 clocks (2.7 s at 50 MHz): 56 million
multiply-accumulates an image forward and back. It builds in about 2 minutes
and the model runs a step in 10 s (see "The models, with numpy" below); its
state (the weights, AdamW's moments: 1.4 million numbers, 34 Mbit) is for
simulation.

`examples/tiny_gpt.py` is a character-level GPT in PyTorch (4 Transformer
blocks of causal self-attention with 6 heads and a GELU MLP, LayerNorm,
residual connections, token and position embeddings, the output layer tied to
the token embedding: 1.8 million parameters), unchanged. Its training loop is
written inside `main()`, so the hardware is its forward pass;
`tests/test_torch.py` trains a small one with a loop function of the usual
shape (dropout, the tied embedding, gradient clipping, AdamW with two groups),
checked against torch. `examples/tiny_gpt_hardware.py` builds the model
`tiny_gpt.py` saves (300 steps of training; 16 bits, rounded, `margin=2`,
16,384-entry exp and GELU tables, formats from 12 windows of the text):

| | 6 other windows of 128 characters |
|---|---|
| scores against torch's | within 0.158 (mean 0.011) |
| the likeliest next character | torch's at 99.5 % of the positions |
| per window | 271,157,254 clocks: 5.4 s at 50 MHz |
| per window, `--sdram` (for the board) | 804,532,889 clocks: 16.1 s at 50 MHz |

The four blocks are one loop over them (identical layers, see "Sequenced
loops"), their values a few bits wider than 16 where the layers' ranges
differ. It builds in about 2 minutes, and the bit-exact model runs a window in
7 s, so the example also writes text with the hardware's scores, a character
at a time. With `--sdram` the weights (29 Mbit), the window and the scores are
in the SDRAM, the activations too (values wider than 16 bits in two words),
and the DE0-Nano's whole project (UART commands, the design, the SDRAM's
controller) fits the EP4CE22 and meets 50 MHz: 12,549 logic elements (56 %),
5,228 registers, 272 kbit of block RAM (45 %), 69 of 132 9-bit multipliers.
The small GPT of the tests is simulated (Verilog), with and without the
SDRAM, the same as its model. (Not yet run on a real board.)

`examples/mini_llama.py` is a step up, built like Llama, unchanged: a byte-level
BPE tokenizer (1,024 tokens), RMSNorm, rotary position embeddings (RoPE),
grouped-query attention through `F.scaled_dot_product_attention`, SwiGLU, the
output layer tied to the embedding, a KV cache and top-p sampling for
generation. `examples/mini_llama_hardware.py` builds its forward pass (no KV
cache: the next token's scores for a window) for the cpu preset `mini_llama.py`
saves (4 blocks, 192 wide, 6 query and 2 key/value heads, a 128-token context:
1.8 million parameters; 300 steps of training; 16 bits, rounded, `margin=2`,
16,384-entry exp and SiLU tables, formats from 12 windows of the text):

| | 6 other windows of 128 tokens |
|---|---|
| scores against torch's | within 0.106 |
| the likeliest next token | torch's in all 6 |
| per window | 257,324,550 clocks: 5.1 s at 50 MHz |
| per window, `--sdram` (for the board) | 587,791,770 clocks: 11.8 s at 50 MHz |

It builds in about 70 s. With `--sdram` the core (`SYN()`: the SDRAM's pins and
timing, its controller) fits the EP4CE22 and meets 50 MHz (Fmax 54.8 MHz):
12,301 logic elements' functions, 5,432 registers, 383 kbit of block RAM (64 %),
60 of 132 9-bit multipliers. `tests/test_torch.py` trains a small one with a loop
function of the usual shape (dropout, attention's `dropout_p` included, the
repeated keys and values, the tied embedding, gradient clipping, AdamW with two
groups), checked against torch; builds that training step in 24-bit fixed
point (rounded, saturating, `margin=2`: 1.9 million clocks a step), whose loss
over 12 steps stays within 0.05 of torch's float training and its weights
too, a step simulated (Verilog) the same as its model; and simulates the
forward pass, the same as its model. (Not yet run on a real board.)

`examples/diffusion_digits.py` draws digits: a class-conditional U-Net
(GroupNorm, FiLM from the timestep's sine embedding and the label,
self-attention at 8x8, skip connections, nearest upsampling) trained to predict
noise, sampled by DDIM with classifier-free guidance.
`examples/diffusion_hardware.py` builds its U-Net's forward pass, unchanged, for
the cpu preset (32x32 images, 1.2 million parameters; 20 bits, rounded,
`margin=2`; formats from DDIM steps of three digits), the sines and cosines
one-period tables. Drawing stays Python, asking the bit-exact model for the
noise twice a step:

| | 6 other calls while drawing |
|---|---|
| the noise against torch's (values to 4.2) | within 0.021 (rms 0.0027) |
| a 7 drawn in 10 DDIM steps | each pixel within 0.27 of torch's drawing |
| per call | 268,802,472 clocks: 5.4 s at 50 MHz |

16 bits lose the drawing (the noise within 1.03), 18 bits give 0.08, 24 bits
0.004. It builds in about 4 minutes. With `--sdram` the weights (two words
each) and the activations go to the SDRAM, the activations of layers done with
written over (19.8 million words needed at once otherwise, more than the
chip's 16.8 million), 1,096,539,628 clocks a call, but it doesn't fit the
EP4CE22: the translation makes about a thousand loop nests (the pads, im2col
copies, products, GroupNorm's passes, the adds), each its own logic, about
138,000 combinational nodes in Quartus for the device's 22,320. The tiny GPT and
the mini Llama fit because their identical blocks are one loop; the U-Net's
layers differ in their sizes. `tests/test_torch.py` simulates a small U-Net
(8x8, two levels) the same as its model.

### Simulation speed (Icarus, Verilator)

`TB()`, `TB_batch()` and `TB_stream()` run a plain Verilog testbench: it reads
the input vectors from a file, steps the design as described above, and writes
the outputs to a file, with no Python in the loop. A long run on a large design
goes to Verilator when it is installed (any version, 4.x included); the rest
to Icarus. A compiled simulation is kept (in `native_build/`) and reused until
the design changes, so a second run costs only the simulation itself:

| | Icarus | Verilator, first run | Verilator, again | cocotb + Icarus (before) |
|---|---|---|---|---|
| XOR training step on 4 multipliers, 600 steps (27000 clocks) | 80 s | 9.3 s | 0.1 s | 67 s |
| 16-tap FIR, 20000 samples | 3.6 s | 3.8 s | 0.5 s | 7.5 s |

On a small design the time went to cocotb's Python round trip per clock; on
a large one it is Icarus's own evaluation, which only Verilator cuts.

`PY2VER_SIM=icarus`, `verilator` or `cocotb` picks one (cocotb runs the older
Python testbenches). A design Verilator can't build runs on Icarus, with a
warning. `TB_handshake()`, `HW_sim()` and designs with `'reg'`
signals use cocotb.

#### The models, with numpy

A design built as loops (numpy or PyTorch code, `numpy_loops.py`) is a loop
program, and four things run it in Python: the check of the translation, the
run that chooses the formats, `PY_float()` and the bit-exact model
(`PY_batch()`, `PY_stream()`). They run it with numpy (`vectorize.py`), with
exactly Python's results:

- a loop whose iterations don't depend on each other runs all of them at
  once, its variable an array of its values; a loop carrying a value (a sum)
  runs an iteration at a time, its body over all the iterations of the loops
  around it (a matrix product: every element's sum a step at a time)
- a store whose index repeats (col2im adding overlapping windows back) runs
  in rounds, each element's updates in Python's order; an if is a mask
- integers are int64, or Python's integers wherever an operation could pass
  62 bits, so the fixed-point model stays bit-exact (the hardware's
  simulations agree with it)

| | in Python | with numpy |
|---|---|---|
| PyTorch MNIST forward pass: build | 413 s | 13 s |
| its bit-exact model, an image | 13 s | 1 s |
| PyTorch MNIST training step (2 images): build | 1000 s | 135 s |
| its bit-exact model, a step | 100 s | 10 s |
| numpy CNN training step: build | 205 s | 16 s |
| its bit-exact model, 2 epochs (88 steps) | 585 s | 127 s |
| digit network's training, 3 epochs in the model | 3 min | 23 s |

(the builds also lost some slow spots that had nothing to do with Python's
speed: width analysis of long expressions done once per part, the tables'
long literals shared instead of copied.) `PY2VER_VECTOR=0` runs them as
Python.

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

With `sdram=` the board project has a small command set on the UART instead:
`'W'`, a 3-byte address, a 3-byte count and that many 16-bit words (high byte
first) writes them into the SDRAM, and the board answers `'W'`; `'R'` and the
sample's port inputs runs a sample (the design pausing whenever the SDRAM is
busy) and sends its outputs back (a byte at least); `'X'`, an address and a
count reads that many words back (the returned lists in the SDRAM). `HW()`
writes the tables once, on its first
call (the MNIST network's 16 Mbit: about 3 minutes at 115200 baud; this
board project is built for `UART_BAUD`, so a faster serial adapter helps),
then for each sample its input lists
in the SDRAM, `'R'` and `'X'`. `HW_sim()` runs the same bytes against the project
with a model of the SDRAM chip on its pins. (Checked in simulation and
compiled by Quartus; not yet run on a real board.)

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

- Lists, tables and memories (see below), lists of lists, and dicts with
  constant keys (see "Dicts and lists of lists")
- List comprehensions and generator expressions (`[x * g for x in xs]`,
  `sum(x * y for x, y in zip(xs, ys))`), written out as lists: over lists of
  known length, `range(...)` of constants, `enumerate` and `zip` of those,
  with several `for`s; an `if` may test the index but not the values (the
  list's length must be known when translating)
- Returning an argument: it gets its own output port, `<name>_out`

Not supported: recursion, keyword arguments in helper calls, sets, `try` /
`with`, a loop's `else`, a dict as an argument or a result.

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
  a variable, becomes a ROM; indexed by a constant it is just that constant.
  A table of 64 entries or more (a fixed-point function table, too) is written
  as an array read by its index (padded to a power of two), which Quartus
  builds compactly and any simulator reads; in a pipelined design it is in
  block RAM. Smaller ones, and tables whose index might be negative, are
  balanced muxes. The XOR network's forward pass, unpipelined: 6624 logic
  cells at 23 MHz as muxes, 2550 at 30 MHz as arrays.
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

### Dicts and lists of lists

A dict whose keys are known when translating is one variable per key, and a
list of lists (a matrix, any depth) one flat list in row-major order:

```python
GAIN = {"lo": 3, "hi": -2}

def mix(x, m, v):                       # m: a 3 x 2 matrix, v: a 2-vector
    acc = {"lo": x * GAIN["lo"], "hi": 0}
    for i in range(3):
        acc["hi"] += m[i][0] * v[0] + m[i][1] * v[1]
    out = [[acc[k] * j for j in range(2)] for k in acc]
    return out                          # 4 outputs, row-major

attr = {"x": I16, "m": dict(I16, shape=(3, 2)), "v": dict(I16, length=2)}
p = Py2ver(mix, attr)
p.TB_batch()([(5, [[1, 2], [3, 4], [5, 6]], [7, 8])])
```

- **Dicts**: `d[k]`, `d[k] = v`, `d[k] += v` with a constant key (a literal, a
  module constant, or the variable of a loop over the dict); `k in d`,
  `len(d)`, `d.get(k, default)`, `d.update(...)`, loops over `d`, `d.keys()`,
  `d.values()`, `d.items()` (unrolled), dict comprehensions, `d2 = d` (the
  same dict) and `dict(d)` / `d.copy()` (a copy), module-level constant dicts.
  With int keys a key computed at run time reads, or updates, the existing
  keys. `attr['d']` gives every entry its format. A dict can't be an argument
  or a result (return its values).
- **Lists of lists**: literals and comprehensions, `m[i][j]` with constant or
  variable indices, writes, `m[i]` as a row, `for row in m` (a loop over the
  rows, which `fold` can fold), `len(m)`, `len(m[0])`. An argument declares
  its shape, `'shape': (R, C)` in attr, and takes nested lists (automatic
  formats find it from the examples); a returned one gives its elements in
  row-major order. Writing through a row (`row = m[0]; row[1] = 5`) would
  change `m` in Python, so it is refused (write `m[0][1] = 5`), and so is
  `[[0] * C] * R`, whose rows are one list in Python.

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

### Reading the generated Verilog

`hdl/main.v` is laid out to be read: the ports in an ANSI list (`clk`,
`rst`, `ce` first), declarations aligned in the order the logic assigns them,
registers updated in `if (rst) ... else if (ce) ...` blocks, and expressions
with only the parentheses they need. Each Python statement's logic sits under
a comment quoting it with its line in your file (a helper's logic under its
call):

```verilog
// line 30: acc += TAPS[k] * window[k]
assign acc__v2 = acc__v1__s + 3 * window_0;
assign acc__v3 = acc__v2 + (-5) * window_1;
```

A design built with `multipliers=` or `sequential=True` is a schedule rather
than your statements in order, so it has no such comments.

A list read at an index only known in hardware (`TAPS[k]` in a folded loop)
is a `case` table. A table of numbers is a function, written once however
many times it is read; a list of signals is a reg set by its own
`always @*` block:

```verilog
assign acc__v3__t0 = (acc__v2 << 19) + table1(__t1__s) * window__at1;
```

A ROM table read in several places is one array.

### Examples

`examples/` has runnable scripts (basic arithmetic, branches, loops, `reg`
signals, complex arithmetic, lists, an AES S-box ROM, a randomized regression,
a streaming low-pass filter, a FIR filter with its delay line in a memory, a
fixed-point FIR filter from a float model, a biquad filter with automatically
chosen formats, a complex downconverter, a soft clipper using a `math.tanh`
table, a 32-tap FIR folded onto 4 multipliers, a numpy XOR network trained on 4
shared multipliers and run pipelined in hardware, a 64 -> 128 -> 64 -> 10 numpy
digit recogniser built with its loops sequenced, on up to 8 lanes, and its
training step with Adam (simulated), a convolutional network with pooling and
dropout, its forward pass and its training step, a PyTorch MNIST network's
forward pass and a step of its training loop (autograd, AdamW, a one-cycle
schedule), a decimate-by-4 filter fed one sample per handshake, a receiver chain built from plain Python functions,
a histogram in block RAM, and pipelined SHA-256 rounds). Each checks the
hardware against Python in simulation, and optionally on the virtual board, in
Quartus, or on the real board:

```
python examples/sha256_pipeline.py              # simulation
python examples/sha256_pipeline.py --board-sim  # + virtual DE0-Nano
python examples/sha256_pipeline.py --handshake  # + valid/ready wrapper, random stalls
python examples/sha256_pipeline.py --syn        # + Quartus area/timing
python examples/sha256_pipeline.py --board      # + connected DE0-Nano
```

### Tests

Tests run in parallel (`-n auto`, pytest-xdist from the `test` extra) and are
marked by what they need, so a quick check after a change doesn't wait for
everything:

```
.venv/bin/python -m pytest -q -n auto -m "not sim and not board and not slow"   # no simulator: ~30 s
.venv/bin/python -m pytest -q -n auto -m "not board and not slow"  # + simulation: ~2.5 min
.venv/bin/python -m pytest -q -n auto                              # all but the long ones: ~5.5 min
.venv/bin/python -m pytest -q -n auto --long                       # everything: ~10 min
.venv/bin/python -m pytest -q -n auto tests/test_share.py          # one feature's tests
.venv/bin/python -m pytest -q --lf                                 # what failed last time
.venv/bin/python -m pytest -q -n auto -m "not quartus"             # everything but Quartus
```

- `sim`: compares the Python function against a simulation (the native
  testbench on Icarus or Verilator, or cocotb); skipped if Icarus is missing.
  A test that isn't marked `sim` (or `board`) and starts a simulator fails and
  says so, which keeps the quick run quick.
- `board`: the virtual DE0-Nano (`HW_sim()`): the whole project, UART
  included, simulated.
- `slow`: a large design simulated (a torch training step: minutes, most of
  them compiling its Verilog), or a Quartus compile.
- `quartus`: a real Quartus compile (area, timing at 50 MHz: an SDRAM design
  of the mini Llama's must meet it), with Quartus from `QUARTUS_DIR`,
  `quartus_sh` on `PATH`, or its usual install
  (`/opt/intelFPGA/22.1std/quartus/bin`); skipped if none of them has it.
- `long`: minutes each (the mini Llama's training step simulated, its SDRAM
  design and the stages in Quartus): skipped unless pytest runs with `--long`.

The SDRAM controller is tested against py2ver's model of the chip, and also
against Micron's MT48LC16M16A2 model (organized as the DE0-Nano's chip; not in
the repository) when `PY2VER_MICRON_MODEL` gives the path of its
`mt48lc16m16a2.v`.

Every test builds in its own temporary directory, so they can run in parallel.
`tests/test_combinations.py` builds a set of designs (state, lists, complex
values, tables, block RAM, helpers, keyword names) with each option
(pipelining, folding, automatic formats, serial ports, stages, annotations)
and checks every result against the references.

A change to how the Verilog is written (the same hardware, other text) is
checked with `tests/icarus.py`: `assert_same_bits` runs pairs of expressions
on random inputs in Icarus and says which differ, and `assert_same_modules`
runs two modules with the same ports side by side, from reset if they are
clocked:

```python
p = Py2ver(f, attr, write=False)
assert_same_modules(tmp_path, before, p.verilog, "f", inputs, outputs)   # mark it sim
```
