"""
Steps of protocols, each a generator: used with `yield from` in a process (built
in it, see process.py) or in a testbench (drive(), check(): run as Python).
A step's pin is given by its name, a constant:

    yield from uart_send(port, "tx", data, BIT)       # a process sending a byte
    yield from uart_send(dut, "rx", 0x41, BIT)         # a testbench, to the design

    byte = yield from uart_receive(port, "rx", BIT)

debounced() is a process of its own, to connect() before a switch or a slow
line: connect(db=(functools.partial(debounced, clocks=50_000), {'raw': 1}), ...).

A bus's steps take its settings as one value, a record (SPI, I2C: its timing,
its pins' names), right after the port:

    ADC = SPI(half=10, mode=3, sclk="adc_sclk")
    got = yield from spi_transfer(port, ADC, channel << 11, 16)
"""
from typing import NamedTuple


def wait_until(port, pin, value=1):
    """Clocks pass until the pin is value (none, if it already is)."""
    while getattr(port, pin) != value:
        yield


def wait_edge(port, pin, rising=True):
    """Clocks pass until the pin's next edge: rising (low, then high) or, with
    rising=False, falling. A clock-following receiver's step (PS/2, SPI)."""
    yield from wait_until(port, pin, 0 if rising else 1)
    yield from wait_until(port, pin, 1 if rising else 0)


def debounced(port, clocks):
    """A process (connect() it before what reads the pin): port.clean follows
    port.raw once raw has kept its new value clocks clocks (a switch's bounces,
    a slow edge's: none get through)."""
    level = 0
    n = 0
    port.clean = 0
    while True:
        if port.raw == level:
            n = 0
        else:
            n = n + 1
            if n == clocks:
                level = port.raw
                n = 0
                port.clean = level
        yield


def jtag_send(port, byte, data="tx_data", send="tx_send", room="tx_room"):
    """One byte to the PC over the JTAG UART: once there's room (the pin room,
    on JTAG_TX_READY), data (on JTAG_TX_DATA) and send (JTAG_TX_VALID) high a
    clock."""
    yield from wait_until(port, room)
    setattr(port, data, byte)
    setattr(port, send, 1)
    yield
    setattr(port, send, 0)


def jtag_print(port, text, data="tx_data", send="tx_send", room="tx_room"):
    """A message to the PC over the JTAG UART (a debug console): text a constant,
    bytes (b"presses: "), a byte at a time."""
    for i in range(len(text)):
        yield from jtag_send(port, text[i], data, send, room)


def jtag_print_hex(port, value, digits, data="tx_data", send="tx_send", room="tx_room"):
    """A number to the PC over the JTAG UART, in hexadecimal: digits of it, the
    most significant first."""
    for i in range(digits):
        d = (value >> (4 * (digits - 1 - i))) & 15
        yield from jtag_send(port, d + 48 if d < 10 else d + 55, data, send, room)


# ----------------------------- SPI (a controller's side) -----------------------------

class SPI(NamedTuple):
    """An SPI bus's settings: half, the clocks of each half of SCLK; mode, SPI's 0
    to 3 (CPOL, the clock's idle level, its high bit; CPHA its low: 0, data taken
    on the leading edge, 1 on the trailing); its pins' names."""
    half: int
    mode: int = 0
    sclk: str = "sclk"
    mosi: str = "mosi"
    miso: str = "miso"


def spi_transfer(port, bus, word, bits):
    """bits bits out of word on the bus's MOSI and as many in from its MISO, most
    significant first; the word read. The chip select is the caller's, and SCLK
    should be at its idle level when it starts."""
    got = 0
    for i in range(bits):
        out = (word >> (bits - 1 - i)) & 1
        if bus.mode & 1:                         # (CPHA 1: set on leading, taken on trailing)
            setattr(port, bus.sclk, 1 - (bus.mode >> 1))
            setattr(port, bus.mosi, out)
            yield bus.half
            setattr(port, bus.sclk, bus.mode >> 1)
            got = got | (getattr(port, bus.miso) << (bits - 1 - i))
            yield bus.half
        else:                                    # (CPHA 0: set before leading, taken on it)
            setattr(port, bus.mosi, out)
            yield bus.half
            setattr(port, bus.sclk, 1 - (bus.mode >> 1))
            got = got | (getattr(port, bus.miso) << (bits - 1 - i))
            yield bus.half
            setattr(port, bus.sclk, bus.mode >> 1)
    return got


# ----------------------------- I2C (a controller's side) -----------------------------
# SCL and SDA open drain ({'scl': {'open_drain': True}, ...}). Checked on a
# DE0-Nano's ADXL345 (examples/accelerometer.py).

class I2C(NamedTuple):
    """An I2C bus's settings: q, a quarter of SCL's period in clocks (125: 100 kHz
    at 50 MHz); its pins' names."""
    q: int = 125
    scl: str = "scl"
    sda: str = "sda"


def i2c_scl_high(port, bus):
    """SCL let go, then waited for (a device may hold it low: clock stretching),
    then held high half a period."""
    setattr(port, bus.scl, 1)
    yield from wait_until(port, bus.scl)
    yield 2 * bus.q


def i2c_send_bit(port, bus, bit):
    """A bit on SDA: set while SCL is low, taken by the device while it's high."""
    setattr(port, bus.sda, bit)
    yield bus.q
    yield from i2c_scl_high(port, bus)
    setattr(port, bus.scl, 0)
    yield bus.q


def i2c_receive_bit(port, bus):
    """A bit from SDA: let go, and taken at the end of SCL's high half."""
    setattr(port, bus.sda, 1)
    yield bus.q
    yield from i2c_scl_high(port, bus)
    bit = getattr(port, bus.sda)
    setattr(port, bus.scl, 0)
    yield bus.q
    return bit


def i2c_start(port, bus):
    """A start (or a repeated start): SDA falls while SCL is high."""
    setattr(port, bus.sda, 1)
    yield bus.q
    yield from i2c_scl_high(port, bus)
    setattr(port, bus.sda, 0)
    yield 2 * bus.q
    setattr(port, bus.scl, 0)
    yield bus.q


def i2c_stop(port, bus):
    """A stop: SDA rises while SCL is high; then the bus is free."""
    setattr(port, bus.sda, 0)
    yield bus.q
    yield from i2c_scl_high(port, bus)
    setattr(port, bus.sda, 1)
    yield 2 * bus.q


def i2c_write_byte(port, bus, byte):
    """A byte out, most significant bit first; 1 if no one acknowledged it."""
    for i in range(8):
        yield from i2c_send_bit(port, bus, (byte >> (7 - i)) & 1)
    nack = yield from i2c_receive_bit(port, bus)
    return nack


def i2c_read_byte(port, bus, last):
    """A byte in, most significant bit first, then acknowledged (not if last: the
    last one read)."""
    data = 0
    for i in range(8):
        b = yield from i2c_receive_bit(port, bus)
        data = data | (b << (7 - i))
    yield from i2c_send_bit(port, bus, last)
    return data


def i2c_write_register(port, bus, address, register, value):
    """A register of the device at address (7 bits) written; 1 if an acknowledge
    was missing."""
    yield from i2c_start(port, bus)
    n1 = yield from i2c_write_byte(port, bus, address << 1)
    n2 = yield from i2c_write_byte(port, bus, register)
    n3 = yield from i2c_write_byte(port, bus, value)
    yield from i2c_stop(port, bus)
    return n1 | n2 | n3


def i2c_read_register(port, bus, address, register):
    """A register of the device at address read (its number written, then a
    repeated start): value, nack (1 if an acknowledge was missing)."""
    value, nack = yield from i2c_read_registers(port, bus, address, register, 1)
    return value, nack


def i2c_read_registers(port, bus, address, register, n):
    """n registers of the device at address read from register on, in one
    transfer (so they belong together: an accelerometer's X, Y and Z): value
    (the first register's byte its least significant), nack (1 if an
    acknowledge was missing)."""
    yield from i2c_start(port, bus)
    n1 = yield from i2c_write_byte(port, bus, address << 1)
    n2 = yield from i2c_write_byte(port, bus, register)
    yield from i2c_start(port, bus)
    n3 = yield from i2c_write_byte(port, bus, (address << 1) | 1)
    value = 0
    for i in range(n):
        b = yield from i2c_read_byte(port, bus, 1 if i == n - 1 else 0)
        value = value | (b << (8 * i))
    yield from i2c_stop(port, bus)
    return value, n1 | n2 | n3


def wait_cycles(n):
    """n clocks pass."""
    yield n


def shift_out(port, pin, value, bits, clocks, msb_first=False):
    """value's bits on the pin, each for clocks clocks: the least significant
    first, or (msb_first) the most."""
    for i in range(bits):
        if msb_first:
            setattr(port, pin, (value >> (bits - 1 - i)) & 1)
        else:
            setattr(port, pin, (value >> i) & 1)
        yield clocks


def shift_in(port, pin, bits, clocks, msb_first=False):
    """bits bits from the pin, each taken clocks clocks after the one before (the
    first: clocks from now); the value, its first bit the least significant or
    (msb_first) the most."""
    value = 0
    for i in range(bits):
        yield clocks
        if msb_first:
            value = value | (getattr(port, pin) << (bits - 1 - i))
        else:
            value = value | (getattr(port, pin) << i)
    return value


def together(*steps):
    """Steps at once, a clock at a time, in a testbench (a process is one sequence:
    there, connect() processes). It returns when every step has returned, with
    a list of their values, in order:

        _, got = yield from together(uart_send(dut, "rx", b, BIT),
                                     uart_receive(dut, "tx", BIT))

    A step that watches the pins forever keeps it from returning: give such a
    step its own end (n bytes seen, say).
    """
    waits = [0] * len(steps)
    values = [None] * len(steps)
    done = [False] * len(steps)
    while True:
        for k, step in enumerate(steps):
            if not done[k] and waits[k] == 0:
                try:
                    n = next(step)
                    waits[k] = 1 if n is None else int(n)
                except StopIteration as e:
                    done[k], values[k] = True, e.value
        if all(done):
            return values
        yield
        waits = [w - 1 for w in waits]


def uart_send(port, pin, byte, bit):
    """A UART frame (8N1) of byte on the pin, bit clocks a bit: the start bit
    low, eight bits least significant first, the stop bit high."""
    setattr(port, pin, 0)
    yield bit
    yield from shift_out(port, pin, byte, 8, bit)
    setattr(port, pin, 1)
    yield bit


def uart_receive(port, pin, bit):
    """A UART frame's byte from the pin: from its start bit's edge, each bit taken
    at its middle; it returns at the stop bit's middle."""
    yield from wait_until(port, pin, 0)
    yield bit // 2                                   # (the start bit's middle)
    byte = yield from shift_in(port, pin, 8, bit)    # (each bit's middle)
    yield bit                                        # (the stop bit's middle)
    return byte
