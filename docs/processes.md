# Protocols as Python generators: a guide

This guide shows how to go from a protocol's specification to hardware on the
DE0-Nano with py2ver. You write the protocol as a Python generator in which
each `yield` is one clock. py2ver turns it into a state machine that does
exactly what the generator does, clock for clock, and the generator itself
stays the reference you test against.

The running example is a UART (8N1, 115200 baud at the board's 50 MHz clock):
a transmitter, a receiver, then both connected as an echo that sends back
every byte it receives. Every `python` block below runs as written, in order
(`tests/test_docs.py` runs them).

## 1. Write the protocol

A UART frame is: the line idles high; a start bit (low); eight data bits,
least significant first; a stop bit (high). Each bit lasts `BIT` clocks.

```python
import functools

from py2ver import Py2ver, connect, process

BIT = 50_000_000 // 115_200          # 434 clocks per bit


def uart_tx(port, bit=BIT):
    """Send port.data when port.start is high for a clock; port.busy is high
    while a frame goes out."""
    port.tx = 1                      # the line idles high
    while True:
        port.busy = 0
        while not port.start:        # wait for a byte to send
            yield
        data = port.data
        port.busy = 1
        port.tx = 0                  # the start bit
        yield bit
        for i in range(8):           # eight data bits, least significant first
            port.tx = (data >> i) & 1
            yield bit
        port.tx = 1                  # the stop bit
        yield bit
```

The rules:

- **The port.** The generator's first argument is its port. An attribute it
  only reads (`port.start`, `port.data`) is an input pin. An attribute it
  writes (`port.tx`, `port.busy`) is an output pin.
- **A `yield` is one clock.** `yield n` waits `n` clocks; `n` can be an
  expression. All the code between two yields runs within one clock.
- **Outputs are registers.** A value written to an output appears on the pin
  from the next clock on (see section 4, "When things happen").
- **Locals keep their values** from one clock to the next. Each one becomes a
  register.
- **Loops and ifs.** A `while`, a `for ... in range(...)` or an `if` that
  contains a yield becomes states of the machine; `break` and `continue`
  work. A loop with no yield in it is logic within one clock, and its bounds
  must be constants.
- **Parameters.** Arguments after the port (`bit=BIT`) are constants: their
  default values, or the ones a `functools.partial` gives.

## 2. Build it

```python
tx = Py2ver(uart_tx, {"start": 1, "data": 8, "tx": {"init": 1}})
```

The second argument, attr, gives each input its width: `8` is short for
`{"width": 8}`. You don't give widths to outputs or locals; py2ver follows
the values written to them (see "Widths" below). An output's `"init"` is its
value after reset (here the line is high from the start).

Run it a clock at a time. Each clock's inputs are a dict; an input left out
is 0. Each clock's outputs come back as a dict:

```python
clocks = [{"start": 1, "data": 0x55}] + [{"start": 0}] * (10 * BIT + 2)
rows = tx.GEN_process()(clocks)          # the generator itself
assert rows[0] == {"tx": 1, "busy": 0}   # clock 0: start is seen...
assert rows[1] == {"tx": 0, "busy": 1}   # ...and the start bit is on the line at clock 1
assert tx.PY_process()(clocks) == rows   # the machine's bit-exact model, the same
```

There are three ways to run a process, and they must always agree:

| method | runs |
|---|---|
| `GEN_process()` | the generator itself (the reference) |
| `PY_process()` | the state machine py2ver built, as a bit-exact Python model |
| `TB_process()` | the generated Verilog, in a simulator (Icarus or Verilator) |

### Widths

py2ver works out the width of every local and output by following the
values assigned to it through each state, taking the `if` tests into
account:

- `if pos < 255: pos = pos + 1` keeps `pos` at 8 bits, and so does
  `pos = min(pos + 1, 255)`.
- A counter reset at a bound (`n = n + 1`, then `if n == 50_000: n = 0`) is
  as wide as the bound: 16 bits.
- A `for i in range(8)` variable is 4 bits wide: it reaches 8 when the loop
  ends.

When a value really can grow without end (`total = total + port.x`), py2ver
stops with an error naming every such variable and the line where it is
first assigned. Give those widths in attr: `{"total": 16}`.

Use the names that `explain()` shows (section 6). Two kinds of local are
renamed:

- A local with the same name as a pin is renamed with `_v`: `data` becomes
  `data_v`, because `port.data` is a pin.
- A local of a step is prefixed with the step's name: `n` in `edge` becomes
  `edge_n` (section 3).

## 3. Steps, parameters and the protocol library

A step that a protocol takes more than once can be written as a generator of
its own and used with `yield from`. The step is built into the process that
uses it, so going in and out of it costs no clock. A value it `return`s is
the value of `x = yield from ...`. A step can return several values with
`return a, b`, and the caller takes them as `x, y = yield from ...`. They're
assigned all at once, as in Python, and every `return` in the step gives as
many as the caller assigns:

```py
def divides(port, n, d):
    q = 0
    while n >= d:            # a subtraction a clock
        n = n - d
        q = q + 1
        yield
    return q, n

...
    q, r = yield from divides(port, port.n, port.d)
```

A step's parameters, after the port, are constants (an int, a string, bytes)
or the values it's given where it's used. A **record** is a constant too: a
`NamedTuple` or a dataclass whose fields are constants, such as a bus's
settings. Its fields are read where the step is built (`bus.q`, `bus.scl`),
it can be given on to other steps, and it can be made in a process from
constants (`SPI(half, mode=3)`, with `half` a parameter). Its fields can't be
assigned, and it isn't a value of its own (`x = bus` is refused).

`py2ver.protocol` has common steps. Each takes the port, then a pin's name (a
constant string) or a bus's record: `SPI(half, mode=0, sclk="sclk",
mosi="mosi", miso="miso")` or `I2C(q=125, scl="scl", sda="sda")`:

| step | does |
|---|---|
| `wait_until(port, pin, value=1)` | waits until the pin has the value (no clock if it already has) |
| `wait_edge(port, pin, rising=True)` | waits for the pin's next rising (or falling) edge |
| `wait_cycles(n)` | waits `n` clocks |
| `shift_out(port, pin, value, bits, clocks, msb_first=False)` | puts `value`'s bits on the pin, `clocks` clocks each |
| `shift_in(port, pin, bits, clocks, msb_first=False)` | reads `bits` bits from the pin, one every `clocks` clocks, and returns the value |
| `uart_send(port, pin, byte, bit)` | sends a UART frame |
| `uart_receive(port, pin, bit)` | receives a UART frame and returns its byte |
| `spi_transfer(port, bus, word, bits)` | an SPI word out on the bus's MOSI and in from its MISO (the result), `bus.half` clocks a half of SCLK, in SPI mode `bus.mode` (0 to 3); the chip select is yours |
| `i2c_start(port, bus)`, `i2c_stop(port, bus)` | an I2C start (or repeated start), a stop; `bus.q` a quarter of SCL's period in clocks |
| `i2c_write_byte(port, bus, byte)` | a byte out; returns 1 if it wasn't acknowledged |
| `i2c_read_byte(port, bus, last)` | a byte in, acknowledged unless `last` |
| `i2c_write_register(port, bus, address, register, value)` | a device's register written (returns 1 without an acknowledge) |
| `i2c_read_register(port, bus, address, register)` | a device's register read: `value, nack = ...` (`nack` 1 without an acknowledge) |
| `i2c_read_registers(port, bus, address, register, n)` | `n` registers read in one transfer: `value, nack = ...`, the first register's byte the lowest of `value` |
| `jtag_send(port, byte)` | sends a byte to the PC over the JTAG UART (section 9) |
| `jtag_print(port, b"text")` | sends a constant message to the PC: a debug console |
| `jtag_print_hex(port, value, digits)` | sends a number to the PC in hexadecimal |
| `together(*steps)` | in testbenches only: runs steps at the same time, and returns when every one has returned, with a list of their values (section 5) |

`debounced(port, clocks)` is a whole process rather than a step: connect it
in front of a switch or a slow line (section 7) and its `port.clean` follows
`port.raw` once `raw` has held a new value for `clocks` clocks.

The receiver, written with two of these steps:

```python
from py2ver.protocol import shift_in, wait_until


def uart_rx(port, bit=BIT):
    """port.rx is the line. For each byte received: port.data, and port.valid
    high for one clock (port.error too, if the stop bit was low)."""
    while True:
        port.valid = 0
        yield from wait_until(port, "rx", 0)            # the start bit's falling edge
        yield bit // 2                                  # to the middle of the start bit
        if port.rx:                                     # high again: a glitch, not a start bit
            continue
        data = yield from shift_in(port, "rx", 8, bit)  # each bit at its middle
        yield bit                                       # the middle of the stop bit
        port.data = data
        port.error = 1 - port.rx                        # the stop bit should be high
        port.valid = 1
        yield


rx = Py2ver(uart_rx, {"rx": 1})
```

Writing your own step is the same as writing a process: a generator whose
first argument is the port. Inside a step, `getattr(port, name)` and
`setattr(port, name, value)` with a constant name read and write that pin.

Parameters make variants without editing the code:
`functools.partial(uart_rx, bit=217)` is the same receiver at twice the
baud rate.

## 4. When things happen

Within one clock, a process reads its inputs' values for that clock and runs
until its next `yield`. What it writes to an output appears on the pin at the
next clock:

| clock | input `a` | the process | output `y` |
|---|---|---|---|
| 0 | 1 | reads `a = 1`, writes `y = 1` | 0 (its initial value) |
| 1 | 0 | reads `a = 0`, writes `y = 0` | 1 |
| 2 | 0 | ... | 0 |

```python
def follow(port):
    while True:
        port.y = port.a
        yield


f = Py2ver(follow, {"a": 1})
assert [r["y"] for r in f.GEN_process()([{"a": 1}, {"a": 0}, {"a": 0}])] == [0, 1, 0]
```

Two `wait_until`s in a row follow an edge: `wait_until(port, "clk", 1)` then
`wait_until(port, "clk", 0)` waits for the next falling edge (that is what
`wait_edge(port, "clk", rising=False)` does). This works because an input
keeps one value through a clock: once the first wait has passed, the pin is
known to be 1 for the rest of that clock, so the second one waits at least
a clock. A 1-bit input that isn't 0 is 1, so `while port.clk == 0:` and
`while port.clk != 1:` mean the same.

A loop that could go round without a clock passing is an error. The message
shows the way round: `ps2.py:46 -> ps2.py:34 -> ps2.py:46`. Add a `yield` on
that path, or a test that the inputs rule out.

## 5. Test it

A testbench is a generator too. Its one argument is the design's pins. It
writes the design's inputs (a value stays until written again), reads the
design's outputs, and each `yield` is one clock, as in a process.

The timing follows the table above. Each time the testbench resumes after
a `yield`, it is at a new clock: reading `dut.y` gives the output's value
during that clock, which is what the design wrote at the clock before. The
inputs the testbench sets before its next `yield` are the ones the design
reads during this clock.

```python
def tb(dut):
    dut.a = 1             # clock 0: the design reads a = 1
    yield
    assert dut.y == 1     # clock 1: what the design wrote at clock 0
    dut.a = 0
    yield
    assert dut.y == 0


process.drive(follow, tb, {"a": 1})     # runs tb against the generator
```

So a handshake needs no extra clocks. A testbench that sets `start = 1`,
yields once, then sets `start = 0` and calls `wait_until(dut, "busy", 0)`,
already sees `busy` high: the transmitter saw `start` during the first clock
and wrote `busy = 1` then.

`check()` is the one call to test a design. It works in three steps:

1. It runs the testbench against the generator, a clock at a time.
2. It feeds each clock's inputs from that run to the model, and
3. to the Verilog in a simulator (skip this with `verilog=False`).

Each must give the generator's outputs at every clock. Because the outputs
are registers, a testbench that reacts to them (waits for `busy`, reads a
reply) runs exactly the same against all three.

The echo: the receiver's byte goes to the transmitter. A testbench plays the
other end of the line. `together()` runs the sending and the receiving steps
at the same time, because the echo starts answering before the host has
finished sending:

```python
from py2ver.protocol import together, uart_receive, uart_send


def echo(bit=BIT):
    """Every byte received on rx, sent back on tx."""
    return connect(rx=(functools.partial(uart_rx, bit=bit), {"rx": 1}),
                   tx=(functools.partial(uart_tx, bit=bit), {"tx": {"init": 1}}),
                   name="uart_echo",
                   wires={"tx.data": "rx.data", "tx.start": "rx.valid"})


def host(text, bit=BIT):
    """A testbench: send each byte of text and receive what comes back."""
    def tb(dut):
        dut.rx = 1                           # the line idles high
        yield 10
        got = []
        for b in text:
            _, back = yield from together(uart_send(dut, "rx", b, bit),
                                          uart_receive(dut, "tx", bit))
            got.append(back)
        return bytes(got)
    return tb


e = Py2ver(echo())
run = e.check(host(b"hi"))
assert run.value == b"hi"                    # what the testbench returned
print(len(run.inputs), "clocks, Verilog run in", run.simulator)
```

The `Run` that `check()` returns has each clock's `inputs` and `outputs`,
the testbench's return `value`, and the `simulator` that ran the Verilog.
The simulator is Icarus, or Verilator for long runs when it is installed;
set `PY2VER_SIM=icarus` or `PY2VER_SIM=verilator` to choose.

When something is wrong:

- A failed `assert` in the testbench raises `process.TestbenchFailed`,
  giving the clock and the line.
- A model or Verilog that differs from the generator raises
  `process.CheckFailed` with the first clock and pin that differ, for example
  `the model differs from the generator at clock 1,000: busy 0 / 1`. It
  writes the waveforms of both runs next to your script
  (`uart_echo_generator.vcd`, `uart_echo_model.vcd`) for GTKWave or Surfer.
- A testbench that never returns stops after `limit=` clocks (10 million by
  default) and says where it was waiting.

Real protocol timing makes long runs: a few frames at 115200 baud are tens
of thousands of clocks, and the Python side is the slow part. A parameter
makes a fast variant for testing:

```python
fast = Py2ver(echo(bit=8))
assert fast.check(host(b"hello", bit=8), verilog=False).value == b"hello"
```

Other tools: `GEN_process(vcd="file.vcd")` (and the same for `PY_process`
and `TB_process`) writes a run's waveforms, and
`process.difference(rows_a, rows_b)` gives the first clock where two runs
differ, or `None`.

## 6. See what was built

```python
print(rx.explain())
```

prints each state with the line where it waits, and every register's
width (here the guide's code was run as `guide.py`):

```text
uart_rx: 6 states. Each state is a place where the generator waits: a yield (one clock cycle) or a yield n (n cycles).
  0  guide.py:40  def uart_rx(port, bit=BIT):  (where it starts)
  1  py2ver/protocol.py:40  yield  (in wait_until, called from guide.py:45)
  2  guide.py:46  yield bit // 2                                  # to the middle of the start bit
  3  py2ver/protocol.py:97  yield clocks  (in shift_in, called from guide.py:49)
  4  guide.py:50  yield bit                                       # the middle of the stop bit
  5  guide.py:54  yield
Widths in bits (the outputs, the local variables and the control are registers):
  inputs   rx 1
  outputs  valid 1, data 8, error 1
  locals   data_v 8, shift_in_i 4
  control  __pc 3, __wait 9  (__pc is the current state; __wait counts down a yield n)
  Note: data_v is the local variable data, renamed so that it doesn't clash with the pin port.data.
```

A state inside a step shows the step and where it was called from. For a
connected design, `explain()` shows each process in turn.

## 7. Connect processes

`connect()` makes one design from several processes. Name each process with
a keyword and give it as `(generator, its attr)`. `wires` connects an input
of one process to an output of another, as `"process.pin"`:

```python
assert [q.name for q in e.ir.inputs] == ["rx", "clk", "rst", "ce"]
assert [q.name for q in e.ir.outputs] == ["valid", "data", "error", "tx", "busy"]
```

- **What a wire carries.** A wire carries the output's register: the value
  its process wrote before this clock, exactly as a pin would. Processes
  therefore can't loop back on each other within a clock, and the order you
  list them in doesn't matter.
- **Widths.** A wired input takes its width from the output feeding it, so
  it needs no width in attr.
- **The design's pins.** They are the inputs that aren't wired, plus every
  output, each under its own name. If two processes have a pin of the same
  name, both get their process's name in front (`rx_data`, `tx_data`).
- **The design's attr.** `Py2ver(design, attr)` takes those pin names. Use
  `"process.name"` for a process's own local (`{"enc.count": 16}`).

## 8. Open-drain pins: I2C and other shared lines

Some buses share a line between several devices. Each device either pulls
the line low or lets go of it, and a pull-up resistor holds it high when
nobody pulls. I2C works this way (both SCL and SDA), and so do 1-Wire and a
PS/2 host. Mark such a pin `open_drain` in attr:

- writing `port.sda = 0` pulls the line low, and `port.sda = 1` lets go;
- reading `port.sda` gives the line's level, which is low while anyone pulls
  it, this process included.

```python
def pulls(port):
    """Pull the line low two clocks out of four; port.seen is the line as read."""
    while True:
        port.sda = 0
        port.seen = port.sda
        yield 2
        port.sda = 1                         # let go: high, unless someone else pulls it
        port.seen = port.sda
        yield 2


od = Py2ver(pulls, {"sda": {"open_drain": True}})
assert [q.name for q in od.ir.outputs] == ["sda", "seen"]   # sda: what it drives
assert "sda_in" in [q.name for q in od.ir.inputs]          # sda_in: the line


def other_device(dut):
    """A testbench: it pulls the line low too, for clocks 8 to 11."""
    seen = []
    for k in range(16):
        dut.sda = 0 if 8 <= k < 12 else 1
        seen.append(dut.sda)                 # the line: low if either side pulls it
        yield
    return seen


assert od.check(other_device, verilog=False).value == [1, 0, 0, 1, 1, 0, 0, 1,
                                                       0, 0, 0, 0, 1, 0, 0, 1]
```

In hardware, an open-drain pin is an output `sda` (what this design drives;
1 after reset, let go) and an input `sda_in` (the line). In a testbench the
pin works like the real bus. The testbench writes its own drive with
`dut.sda = 0` or `1`, and reading `dut.sda` gives the line: low if either
side pulls it. `check()` replays the line's values to the model and the
Verilog.

On the board, the pin is pulled low or left floating. Its input comes from
the same pin, through the usual two registers. Use a GPIO pin, or the
board's own `I2C_SCLK` and `I2C_SDAT`. An open-drain pin gets the FPGA's
weak pull-up, so its line is high when nobody pulls it. That is enough for
the board's own I2C bus at 100 kHz; a longer bus on GPIO wants pull-up
resistors as well (a few kΩ to 3.3 V).

**Chips' drivers** are in `py2ver.devices`, a module for each chip: its
registers, the steps that work it, and a model of it for testbenches. A
driver step takes the port, then the bus's record:

| module | steps | model |
|---|---|---|
| `adxl345` (accelerometer, I2C) | `adxl345_start(port, bus)` → `nack`; `adxl345_id(port, bus)` → `id, nack`; `adxl345_read_xyz(port, bus)` → `x, y, z, nack` | `adxl345_model(dut, registers(x, y, z), transfers)` |
| `adc128s022` (ADC, SPI) | `adc128s022_read(port, adc, channel)` → its 16 bits (4 zeros, 12 of result); `adc128s022_frame(...)` | `adc128s022_model(values, frames)` |
| `eeprom24` (24LC02, I2C) | `eeprom_read(port, bus, offset)` → `value, nack`; `eeprom_write(port, bus, offset, value)` → `nack`, its write cycle waited for | `eeprom_model(dut, mem, transfers, write_clocks=5000)` |
| `i2c_target` | | `i2c_memory(dut, mem, transfers, address)`: any chip of registers on I2C |

A step can be called through its module (`adxl345.adxl345_read_xyz(port,
bus)`), and a module's constants read the same way (`adxl345.POWER_CTL`).
A chip with a pin of its own carries it in its record: `ADC128S022(spi=SPI(10,
mode=3), cs_n="cs_n")`. `tests/test_devices.py` checks each driver against
its model, in the Verilog too.

**The board's own I2C bus** (`I2C_SCLK`, `I2C_SDAT`) has its ADXL345
accelerometer, at address 0x1D once `G_SENSOR_CS_N` is held high (its I2C
mode). Something else acknowledges from 0x50 to 0x57 and reads `0xFF`: its
24LC02 EEPROM, erased. `examples/accelerometer.py` probes the bus, and
reads the accelerometer's X, Y and Z five times a second onto the JTAG
console, all six bytes in one transfer so they belong together:
`x, y, z, nack = yield from adxl345_read_xyz(port, bus)`, with `bus`
`I2C(q=125)`, 100 kHz. On a board lying flat it reads about `x=0x0003 y=0x000F z=0x011F`:
Z is 1 g, about 256 counts at its ±2 g range.

**The board's ADC** (an ADC128S022: 8 inputs, 12 bits) is on `ADC_CS_N`,
`ADC_SCLK`, `ADC_SADDR` and `ADC_SDAT`, in SPI mode 3. `examples/adc.py`
reads all 8 inputs with `adc128s022_read(port, ADC128S022(), channel)`
(2.5 MHz) five times a second onto the JTAG console. Each frame sends back
4 zeros and then the result of the channel chosen in the frame before.
Inputs left open read whatever they drift to, about 0x500.

`py2ver.protocol`'s I2C steps (`i2c_start`, `i2c_write_byte`,
`i2c_read_register`, ...) are the bus's spec written as generators:

- start and stop conditions;
- bits set while SCL is low and taken while it's high;
- acknowledges;
- a device that holds SCL low makes them wait (clock stretching).

Their pins are `scl` and `sda` unless the bus names others
(`I2C(scl="i2c_clock")`).
`examples/i2c.py` is a master built from them that writes and reads one
register of a device. Its testbench plays a
24LC02-style memory chip on the same two lines (`py2ver.devices.i2c_target`'s
`i2c_memory`), acknowledging its own
address and stretching the clock, and the master's write, read-back and
missing acknowledge all check against the generator, the model and the
Verilog. At 100 kHz it compiles to 285 LUTs and 79 registers (166 MHz).

Open-drain pins can't be wired between processes in `connect()` yet: play
the other devices on the bus in a testbench.

## 9. Put it on the board

A port's attr can name a DE0-Nano pin. A string is short for `{"pin": ...}`:

```python
board = Py2ver(echo(), {"rx": "GPIO_0_RX", "tx": "GPIO_0_TX", "data": "LED"})
print(board.board_top())        # the board's top module, with the design inside
```

```py
board.PINS({"QUARTUS_DIR": "/opt/intelFPGA/22.1std/quartus/bin"})
# Device:          EP4CE22F17C6
# Combinational:   143
# Registers:       58
# Fmax:            181.59 MHz
# Timing @ 50 MHz: met (worst setup slack 15.541 ns)
```

`PINS()` compiles the board project with Quartus and reports its area and
timing. It does not program the board: load `syn/<name>_pins/DE0_Nano.sof`
with Quartus's programmer.

**Pin names** are those of the board's assignments
(`py2ver/templates/hw/de0_nano_qsf.txt`):

| pins | names |
|---|---|
| the LEDs (outputs) | `LED[7:0]`; `'LED'` is all 8 |
| the keys and switches (inputs) | `KEY[1:0]`, `SW[3:0]` |
| the GPIO_0 header | `GPIO_0[33:2]` (inputs or outputs); `GPIO_0_IN[1:0]` (inputs only) |
| the serial pins, where the USB-UART adapter goes | `GPIO_0_RX` (FPGA pin D3), `GPIO_0_TX` (C3): GPIO_0's first two pins |
| GPIO_1, GPIO_2 | `GPIO_1[33:0]`, `GPIO_1_IN[1:0]`, `GPIO_2[12:0]`, `GPIO_2_IN[2:0]` |
| the JTAG UART (bytes to and from the PC, below) | `JTAG_RX_DATA`, `JTAG_RX_VALID`, `JTAG_RX_READY`, `JTAG_TX_DATA`, `JTAG_TX_VALID`, `JTAG_TX_READY` |

The FPGA pin of every name is in the assignments file, and the board's user
manual maps FPGA pins to header pins. `GPIO_0[0]` and `GPIO_0[1]` don't
exist under those names: they are `GPIO_0_RX` and `GPIO_0_TX`.

**Widths and pins.** An input without a width gets as many bits as it has
pins. Any other port keeps its width, and pins that don't match it are an
error as soon as the design is built: an 8-bit `data` on `"LED[7:1]"` is
"7 pins for a port of 8 bits". Not every port needs a pin. Inputs without
one are tied to 0, outputs without one are left unconnected, and LEDs no
output drives are off.

**The keys read 0 while pressed** (they are active-low). A process that
waits for a press waits for the key to go low:
`wait_edge(port, "key", rising=False)`.

**Feeding an input from inside the board.** An input with no pin is 0 for
good, so something on the board has to drive it. Connect a process in front
of it. Here, each press of `KEY[1]` sends a count of presses out of the
serial pin, which a terminal on the USB-UART adapter shows. `debounced`
cleans up the key's bounces:

```python
from py2ver.protocol import debounced, wait_edge


def presses(port):
    """A one-clock pulse on port.pulse at each press of the key, and the
    count of presses in port.count."""
    count = 0
    port.pulse = 0
    port.count = 0
    while True:
        yield from wait_edge(port, "key", rising=False)    # pressed: the key goes low
        count = (count + 1) & 255
        port.count = count
        port.pulse = 1
        yield
        port.pulse = 0


button = connect(db=(functools.partial(debounced, clocks=500_000), {"raw": 1}),   # 10 ms
                 key=(presses, {}),
                 tx=(uart_tx, {"tx": {"init": 1}}),
                 name="button_tx",
                 wires={"key.key": "db.clean", "tx.start": "key.pulse",
                        "tx.data": "key.count"})
on_board = Py2ver(button, {"raw": "KEY[1]", "tx": "GPIO_0_TX", "count": "LED"})
assert "assign GPIO_0_TX = tx_w;" in on_board.board_top()
```

**What the board's top adds:**

- the 50 MHz clock (`CLOCK_50`);
- a reset at power-up and while `KEY[0]` is pressed (unless a port uses
  `KEY[0]`);
- two registers in front of every input, because a pin can change at any
  moment. On the board, the design therefore sees an input two clocks
  later than in simulation.

There is no glitch filter: put `debounced` in front of a switch or a slow
line.

**Electrical.**

- The GPIO pins are 3.3-V LVTTL. They are not 5-V tolerant: a 5-V device
  such as a PS/2 keyboard needs a level shifter.
- Pull-ups are enabled only on open-drain pins: the FPGA's weak ones.
- A GPIO pin that no output drives is left floating. An open-drain pin
  (section 8) is pulled low or left floating.

**Fmax.** Quartus reports the *restricted* Fmax. On this device it is at
most 250 MHz (the I/O toggle rate), even when the logic alone could run
faster.

### Talking to the PC: the JTAG UART

The JTAG UART carries bytes between the design and the PC over the same
USB-Blaster cable that programs the board, so no serial adapter is needed.
Its six pins connect it to the design's ports:

| pin | the design's | means |
|---|---|---|
| `JTAG_RX_DATA` | input, 8 bits | a byte from the PC |
| `JTAG_RX_VALID` | input | high for the clock a byte is given, and taken |
| `JTAG_RX_READY` | output (optional) | bytes are given only while this is high; without it, as they come |
| `JTAG_TX_DATA` | output, 8 bits | a byte to the PC |
| `JTAG_TX_VALID` | output | high one clock per byte to send |
| `JTAG_TX_READY` | input (optional) | high while there's room to send |

```py
def jtag_echo(port):                           # examples/jtag_echo.py
    port.ready = 1
    port.send = 0
    while True:
        yield from wait_until(port, "valid")   # a byte from the PC, taken this clock
        b = port.byte
        port.ready = 0                         # no more until it's sent back
        port.out = b
        port.leds = b
        yield from wait_until(port, "room")
        port.send = 1                          # one clock: one byte to the PC
        yield
        port.send = 0
        port.ready = 1

board = Py2ver(jtag_echo, {"byte": "JTAG_RX_DATA", "valid": "JTAG_RX_VALID",
                           "ready": "JTAG_RX_READY", "out": "JTAG_TX_DATA",
                           "send": "JTAG_TX_VALID", "room": "JTAG_TX_READY", "leds": "LED"})
board.PINS({"QUARTUS_DIR": "/opt/intelFPGA/22.1std/quartus/bin"})
```

The board's top adds Intel's JTAG UART core (`alt_jtag_atlantic`), with a
FIFO of 8 bytes from the PC and 16 to it. These pins are inside the FPGA, on
the design's clock, so they don't go through the two input registers. On
the PC, after programming the board, run `juart-terminal` (in Quartus's
`bin` directory) and type. The JTAG UART and SignalTap can be used
together; both connect to the board through the same JTAG hub.

From Python, the same design can be programmed, talked to, and checked
against its generator:

```py
board.PINS(q)                                  # compile for the board
board.PROGRAM(q)                               # load it into the FPGA (its RAM: the flash is left alone)
expected = board.jtag_reference(b"hello")      # what the generator sends back: b'hello'
with board.jtag_uart(q) as link:               # Quartus's JTAG library (or juart-terminal)
    link.write(b"hello")
    assert link.read(len(expected), timeout=5) == expected
```

`jtag_reference(data)` runs the generator with a testbench that behaves
like the board's JTAG UART: a byte is given while the design is ready,
every byte sent back is taken, and there's always room. It returns what the
board must answer. `PROGRAM()` and `jtag_uart()` find Quartus's tools in
`QUARTUS_DIR`, on `PATH`, or in `/opt/intelFPGA/22.1std/quartus/bin`.
`jtag_uart()` talks through Quartus's own JTAG library (`libjtag_atlantic`,
the one its terminals use): a round trip in about 25 ms, hundreds of KB/s.
Where Quartus has no such library it runs `juart-terminal` instead, with a
round trip of about 0.2 s and 1.3 KB/s to the board.
`tests/test_hardware.py` does all this on a board plugged in: run
`pytest --hardware`.

### A debug console

A design can tell the PC what it is doing, like `printf`, with the
`jtag_print` steps. They use the ports `tx_data`, `tx_send` and `tx_room`,
on the pins `JTAG_TX_DATA`, `JTAG_TX_VALID` and `JTAG_TX_READY`.
`examples/jtag_console.py` prints a banner at reset, counts each press of
`KEY[1]` (through `debounced`), and prints the switches when you type `s`:

```py
def console(port):
    port.rx_ready = 0
    yield from jtag_print(port, b"py2ver console: press KEY1, or type s\n")
    count = 0
    was = 0
    while True:
        port.rx_ready = 1
        key = port.key
        if was == 1 and key == 0:                    # pressed (a key reads 0)
            port.rx_ready = 0
            count = (count + 1) & 255
            yield from jtag_print(port, b"presses: 0x")
            yield from jtag_print_hex(port, count, 2)
            yield from jtag_print(port, b"\n")
        elif port.rx_valid and port.rx_data == KEY_S:
            ...
        was = key
        yield
```

A message is a bytes constant, built as a table of its bytes. Printing takes
a clock or two per byte, so like `printf` it slows the design while it
prints. Text sent at reset waits in the JTAG UART until the PC connects (up
to about 80 bytes), so `juart-terminal` started later still shows the banner.
A port to `JTAG_TX_DATA` may be narrower than 8 bits; text only needs 7.

### Debugging on the board with SignalTap

SignalTap is Quartus's logic analyzer inside the FPGA. It records chosen
signals at every clock into the device's memory while the design runs, and
you read the record back over the USB-Blaster. `PINS(..., signaltap=True)`
builds it into the board's project:

```py
board = Py2ver(echo(), {"rx": "GPIO_0_RX", "tx": "GPIO_0_TX", "data": "LED"}, probes=True)
board.PINS({"QUARTUS_DIR": "/opt/intelFPGA/22.1std/quartus/bin"}, signaltap=True)
# Memory bits:     249856        (61 signals' bits x 4,096 samples)
# Fmax:            147.54 MHz    (the design's clock)
```

- **What it records.** By default, every input on a pin as the design gets
  it (after its two registers), and every output. `probes=True` also puts
  the process's state on ports: each process's state number (`__pc`), its
  wait counter (`__wait`) and its locals, all as they will be at the next
  clock. To record fewer signals, list them:
  `signaltap={"signals": ["rx", "tx", "rx___pc_out"]}`.
- **How long.** By default, the most samples that fit in 256 Kbit of the
  device's 594 Kbit of memory: 4,096 samples (82 µs at 50 MHz) for the echo's
  61 bits, and more for fewer bits. `"depth": 16384` sets it, as a power of
  two.
- **Which bit is which.** `syn/<name>_pins/signaltap.txt` lists each
  signal's bits in the recording.

To capture from Python, load the build and call `CAPTURE()`:

```py
board = Py2ver(jtag_echo, pins, probes=True)
board.PINS(q, signaltap=True)
board.PROGRAM(q)
with board.jtag_uart(q) as link:
    cap = board.CAPTURE(q, trigger={"valid": 1, "byte": ord("Z")},
                        during=lambda: link.write(b"Z"))
print(cap.show())
#    -512  byte=65, valid=0, ..., __pc_out=1   [__pc_out: ... (in wait_until, called from examples/jtag_echo.py:27)]
#       0  byte=90, valid=1, __pc_out=3   [__pc_out: examples/jtag_echo.py:34  yield]
#       1  byte=65, valid=0, ready=0, send=1, __pc_out=1   [...]
#       2  ready=1, send=0
cap.write_vcd("capture.vcd")                   # for GTKWave or Surfer
```

- **The trigger.** `trigger` gives each port's value (every bit of it), or
  `"rising"` / `"falling"` for a 1-bit port's edge. All of them must hold
  at once. Without a trigger, the capture starts at once.
- **What happens meanwhile.** `CAPTURE()` arms SignalTap, and a few seconds
  later calls `during()`, which should make the trigger happen. Here it
  sends a byte. If the trigger doesn't happen within `timeout` seconds
  (30 by default), it raises `TimeoutError`.
- **The record.** `cap.rows` has every sample's values by port name.
  `cap.trigger` is the trigger's sample: by default an eighth of the way
  into the record (512 of 4,096). `cap.changes()` lists only the samples
  where something changes, counted from the trigger.
- **States.** `cap.show()` names each state probe's value by its line in
  the generator, as `explain()` lists it. `__pc_out` is the state the
  process goes to next.

Behind it, `quartus_stp` makes a SignalTap file (`.stp`) from the design's
SignalTap instance, puts the trigger in it, runs it, and exports the record.
Quartus's SignalTap window works on the same build too: open
`syn/<name>_pins/DE0_Nano.qpf`, then Tools > Signal Tap Logic Analyzer,
then File > Create/Update > Create SignalTap File from Design Instance(s).

SignalTap's logic adds about 700 LUTs and its memory. The design's clock
keeps meeting 50 MHz. Quartus also reports SignalTap's JTAG clock
(`altera_reserved_tck`), which `PINS()` leaves out of its Fmax and slack.

## 10. Error messages

| message | meaning, and what to do |
|---|---|
| `give the input 'x' its width in attr` | every input needs a width: `{"x": 8}`, or a pin |
| `'x': its values grow without end as followed: give its width in attr` | nothing bounds `x`; give its width, or test it against a bound |
| `a loop that doesn't yield: in one clock it can go a.py:46 -> a.py:34 -> a.py:46` | that path never reaches a `yield`; add one, or a test the inputs rule out |
| `a while loop that doesn't yield (it would wait without a clock passing)` | a `while` without a `yield` in it can't wait; add a `yield` |
| `the port 'data' (8 bits): LED[7:1]: 7 pins for a port of 8 bits` | give 8 pins, or give the port 7 bits in attr |
| `the board has no GPIO_0[0]` | see the pin names above |
| `LED[0] can't float: not for the open-drain pin sda` | an open-drain pin needs a GPIO pin, or `I2C_SCLK` / `I2C_SDAT` |
| `a.sda is an open-drain pin: a bus of them isn't wired yet` | play the other devices on the bus in a testbench |
| `yield from a generator function's call` | `yield from` takes a call of a step, `yield from step(port, ...)` |
| `the testbench at clock 1,234 (tb.py:20): ...` | the testbench's `assert` failed at that clock and line |
| `the model differs from the generator at clock ...` | a width given in attr too small for the values (Python's numbers don't wrap; the hardware's do), or a py2ver bug: please report it, with the waveforms it wrote |
