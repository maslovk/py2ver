# py2ver

[![tests](https://github.com/maslovk/py2ver/actions/workflows/tests.yml/badge.svg)](https://github.com/maslovk/py2ver/actions/workflows/tests.yml)

**Turn Python into hardware, and check the hardware against the Python.**

py2ver turns a plain Python function into Verilog for an FPGA. So it does a
generator that follows a protocol one clock cycle at a time, numpy code, and
PyTorch models. Your Python stays the reference: py2ver checks the hardware
against it in a simulator, in Intel Quartus, and on a DE0-Nano board.

## Install

```
python3 -m venv .venv
.venv/bin/pip install "py2ver @ git+https://github.com/maslovk/py2ver"
sudo apt install iverilog                # a Verilog simulator, to run what py2ver writes
.venv/bin/python -m py2ver doctor        # what py2ver can do on this computer
```

`doctor` lists what's installed and what each tool is for. Only Icarus
Verilog is needed to start. Optional extras: `py2ver[numpy]`, `py2ver[torch]`,
`py2ver[hw]` (pyserial). For the board you need Intel Quartus (the free Lite
edition) and a DE0-Nano.

## Your first design

```python
from py2ver import Py2ver

# A plain Python function: py2ver uses it exactly as written.
def arithmetic(x, y):
    total = x + y
    difference = x - y
    product = x * y
    quotient = x // y
    return total, difference, product, quotient

# Everything about the hardware goes here: how many bits each value has.
attr = {
    "x": 8,                                     # unsigned, 8 bits: 0 .. 255
    "y": 8,
    "total": 9,                                 # up to 255 + 255 = 510
    "difference": {"signed": 1, "width": 9},    # -255 .. 255
    "product": 16,
    "quotient": 8,
}

p = Py2ver(arithmetic, attr)          # the Verilog is in p.verilog, and in hdl/main.v
outputs = p.check([(3, 1), (7, 2), (255, 255), (2, 5)])
print(outputs)                        # [(4, 2, 3, 3), (9, 5, 14, 3), (510, 0, 65025, 1), (7, -3, 10, 0)]
```

`check()` runs the inputs through your function, through py2ver's model of
the hardware, and through the generated Verilog in the simulator, and makes
sure all three agree. If a width is too small, it says so. With
`"product": 9` the check stops with:

```text
arithmetic(255, 255): Python gives product = 65025, but its 9 bits hold only 0 to 511,
so the hardware gives 1. Give it more bits in attr: {'product': 16} holds every value it takes here.
```

## Hardware that follows a protocol

Protocols are usually described as a sequence in time: "the line goes low
for one bit, then eight data bits follow, then it goes high". Write exactly
that as a Python generator in which each `yield` is one clock cycle, and
py2ver builds a state machine that does the same, cycle for cycle:

```python
BIT = 50_000_000 // 115_200             # one bit of a 115200-baud UART, in 50 MHz cycles

def uart_tx(port, bit=BIT):
    port.tx = 1                         # the line is high when idle
    while True:
        port.busy = 0
        while not port.start:           # wait for a byte to send
            yield
        data = port.data
        port.busy = 1
        port.tx = 0                     # the start bit
        yield bit
        for i in range(8):              # eight data bits, least significant first
            port.tx = (data >> i) & 1
            yield bit
        port.tx = 1                     # the stop bit
        yield bit

tx = Py2ver(uart_tx, {"start": 1, "data": 8})     # the inputs' widths; py2ver works out the rest
print(tx.explain())                               # its states, each at a line of uart_tx
```

What the generator reads from `port` becomes the module's inputs, and what
it writes becomes its outputs. `py2ver.protocol` has ready-made steps (UART,
SPI, I2C, waiting for a pin, printing to the PC), and `py2ver.devices` has
drivers for chips (an accelerometer, an ADC, an EEPROM, a TFT display), each
with a model of the chip to test against. The guide is
**[docs/processes.md](docs/processes.md)**.

## On the board

py2ver builds for the DE0-Nano (an Intel Cyclone IV FPGA), and talks to it
over the USB-Blaster cable that programs it:

```py
q = {"QUARTUS_DIR": "/opt/intelFPGA/22.1std/quartus/bin"}

# A process on the board's own pins: the UART echo on GPIO_0, the byte on the LEDs
board = Py2ver(echo(), pins={"rx": "GPIO_0_RX", "tx": "GPIO_0_TX", "data": "LED"})
board.PINS(q)                  # compile with Quartus: size and speed
board.PROGRAM(q)               # load it into the FPGA

# A function on the board, many inputs at once, over the JTAG cable
p.HW(dict(q, LINK="jtag"))
print(p.HW_batch()([(3, 1), (7, 2), (255, 255)]))
```

## Where to go next

| | |
|---|---|
| [docs/processes.md](docs/processes.md) | Protocols as generators: steps, testbenches, `connect()`, chips' drivers, the board's pins, the JTAG console, SignalTap |
| [docs/reference.md](docs/reference.md) | Functions: attr, fixed point, state, pipelining, folding, shared multipliers, sequenced loops, numpy, PyTorch, simulation speed, Quartus, the board, the supported Python |
| [examples/README.md](examples/README.md) | Runnable examples, from a function of two numbers to a PyTorch network |
| [docs/testing.md](docs/testing.md) | Running py2ver's own tests |
