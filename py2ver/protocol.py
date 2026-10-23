"""
Ready-made protocol steps: waiting for a pin, shifting bits, UART, SPI, I2C and
printing to the PC over the JTAG UART.

Each step is a generator. Use it with `yield from`, either inside a process
(py2ver builds the step into the hardware) or inside a testbench (where it runs
as ordinary Python against the design):

    yield from uart_send(port, "tx", data, BIT)       # a process sends a byte
    yield from uart_send(dut, "rx", 0x41, BIT)        # a testbench sends one to the design
    byte = yield from uart_receive(port, "rx", BIT)   # a step can return a value

Pins are given by name, as strings. The SPI and I2C steps take the bus's
settings as one record instead, right after the port:

    ADC_BUS = SPI(half=10, mode=3)                    # 2.5 MHz SCLK at 50 MHz
    reading = yield from spi_transfer(port, ADC_BUS, channel << 11, 16)

debounced() is not a step but a whole process: connect() it in front of a
switch so that bounces never reach the rest of the design.
"""
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Waiting
# ---------------------------------------------------------------------------

def wait_cycles(n):
    """Wait n clock cycles."""
    yield n


def wait_until(port, pin, value=1):
    """Wait until the pin has the given value.

    Takes no time at all if the pin already has it.
    """
    while getattr(port, pin) != value:
        yield


def wait_edge(port, pin, rising=True):
    """Wait for the pin's next rising edge (or falling edge, if rising=False)."""
    yield from wait_until(port, pin, 0 if rising else 1)
    yield from wait_until(port, pin, 1 if rising else 0)


def debounced(port, clocks):
    """A process that cleans up a bouncing input such as a push button.

    port.clean follows port.raw, but only once raw has held its new value for
    `clocks` clock cycles in a row. Shorter glitches never get through.
    Use it with connect(), in front of the logic that reads the button.
    """
    level = 0
    stable_for = 0
    port.clean = 0
    while True:
        if port.raw == level:
            stable_for = 0
        else:
            stable_for = stable_for + 1
            if stable_for == clocks:
                level = port.raw
                stable_for = 0
                port.clean = level
        yield


# ---------------------------------------------------------------------------
# Shifting bits
# ---------------------------------------------------------------------------

def shift_out(port, pin, value, bits, clocks, msb_first=False):
    """Put the bits of `value` on the pin one after another.

    Each bit stays on the pin for `clocks` clock cycles. The least significant
    bit goes first, unless msb_first=True.
    """
    for i in range(bits):
        if msb_first:
            setattr(port, pin, (value >> (bits - 1 - i)) & 1)
        else:
            setattr(port, pin, (value >> i) & 1)
        yield clocks


def shift_in(port, pin, bits, clocks, msb_first=False):
    """Read `bits` bits from the pin, one every `clocks` clock cycles.

    The first bit is read `clocks` cycles from now. Returns the bits as a
    number; the first bit read is the least significant, unless msb_first=True.
    """
    value = 0
    for i in range(bits):
        yield clocks
        if msb_first:
            value = value | (getattr(port, pin) << (bits - 1 - i))
        else:
            value = value | (getattr(port, pin) << i)
    return value


# ---------------------------------------------------------------------------
# UART (8 data bits, no parity, 1 stop bit)
# ---------------------------------------------------------------------------

def uart_send(port, pin, byte, bit):
    """Send one byte on the pin. `bit` is the length of one bit in clock cycles
    (434 for 115200 baud at 50 MHz)."""
    setattr(port, pin, 0)                            # start bit
    yield bit
    yield from shift_out(port, pin, byte, 8, bit)    # data, least significant bit first
    setattr(port, pin, 1)                            # stop bit
    yield bit


def uart_receive(port, pin, bit):
    """Receive one byte from the pin and return it.

    Waits for the start bit, then reads each bit in its middle, where the
    line is most stable. Returns in the middle of the stop bit.
    """
    yield from wait_until(port, pin, 0)              # the start bit begins
    yield bit // 2                                   # move to the middle of the start bit
    byte = yield from shift_in(port, pin, 8, bit)    # then to the middle of each data bit
    yield bit                                        # and to the middle of the stop bit
    return byte


# ---------------------------------------------------------------------------
# SPI (the controller's side)
# ---------------------------------------------------------------------------

class SPI(NamedTuple):
    """The settings of an SPI bus.

    half: the length of half an SCLK period, in clock cycles.
    mode: the SPI mode, 0 to 3. Its high bit (CPOL) is SCLK's idle level; its
          low bit (CPHA) says when data is read: 0 on the first edge of each
          clock pulse, 1 on the second.
    sclk, mosi, miso: the names of the bus's pins.
    """
    half: int
    mode: int = 0
    sclk: str = "sclk"
    mosi: str = "mosi"
    miso: str = "miso"


def spi_transfer(port, bus, word, bits):
    """Send `bits` bits of `word` on MOSI while reading as many from MISO.

    Both go most significant bit first. Returns the word that was read.
    Selecting the chip (pulling its CS low) is up to the caller, and SCLK
    should be at its idle level when the transfer starts.
    """
    idle = bus.mode >> 1
    received = 0
    for i in range(bits):
        bit = (word >> (bits - 1 - i)) & 1
        if bus.mode & 1:
            # CPHA 1: change the data on the first edge, read on the second.
            setattr(port, bus.sclk, 1 - idle)
            setattr(port, bus.mosi, bit)
            yield bus.half
            setattr(port, bus.sclk, idle)
            received = received | (getattr(port, bus.miso) << (bits - 1 - i))
            yield bus.half
        else:
            # CPHA 0: set the data before the first edge, read on it.
            setattr(port, bus.mosi, bit)
            yield bus.half
            setattr(port, bus.sclk, 1 - idle)
            received = received | (getattr(port, bus.miso) << (bits - 1 - i))
            yield bus.half
            setattr(port, bus.sclk, idle)
    return received


# ---------------------------------------------------------------------------
# I2C (the controller's side)
#
# SCL and SDA are open drain: give them {'open_drain': True} in attr. Writing 1
# lets the line go (a pull-up makes it high), writing 0 pulls it low.
# Tested on the DE0-Nano's ADXL345 accelerometer (examples/accelerometer.py).
#
# The register steps return `nack`: 0 if the device acknowledged every byte,
# 1 if it didn't (usually: no device answers at that address).
# ---------------------------------------------------------------------------

class I2C(NamedTuple):
    """The settings of an I2C bus.

    q: a quarter of the SCL period, in clock cycles (125 gives 100 kHz at 50 MHz).
    scl, sda: the names of the bus's pins.
    """
    q: int = 125
    scl: str = "scl"
    sda: str = "sda"


def i2c_scl_high(port, bus):
    """Release SCL and keep it high for half a period.

    A device may hold SCL low to make the controller wait (clock
    stretching), so this waits until SCL really is high before counting.
    """
    setattr(port, bus.scl, 1)
    yield from wait_until(port, bus.scl)
    yield 2 * bus.q


def i2c_send_bit(port, bus, bit):
    """Send one bit: set SDA while SCL is low; the device reads it while SCL is high."""
    setattr(port, bus.sda, bit)
    yield bus.q
    yield from i2c_scl_high(port, bus)
    setattr(port, bus.scl, 0)
    yield bus.q


def i2c_receive_bit(port, bus):
    """Receive one bit: release SDA and read it at the end of SCL's high time."""
    setattr(port, bus.sda, 1)
    yield bus.q
    yield from i2c_scl_high(port, bus)
    bit = getattr(port, bus.sda)
    setattr(port, bus.scl, 0)
    yield bus.q
    return bit


def i2c_start(port, bus):
    """Send a start condition (SDA falls while SCL is high).

    Also used for a repeated start, in the middle of a transfer.
    """
    setattr(port, bus.sda, 1)
    yield bus.q
    yield from i2c_scl_high(port, bus)
    setattr(port, bus.sda, 0)
    yield 2 * bus.q
    setattr(port, bus.scl, 0)
    yield bus.q


def i2c_stop(port, bus):
    """Send a stop condition (SDA rises while SCL is high). The bus is then free."""
    setattr(port, bus.sda, 0)
    yield bus.q
    yield from i2c_scl_high(port, bus)
    setattr(port, bus.sda, 1)
    yield 2 * bus.q


def i2c_write_byte(port, bus, byte):
    """Send one byte, most significant bit first.

    Returns nack: 0 if the device acknowledged the byte, 1 if not.
    """
    for i in range(8):
        yield from i2c_send_bit(port, bus, (byte >> (7 - i)) & 1)
    nack = yield from i2c_receive_bit(port, bus)
    return nack


def i2c_read_byte(port, bus, last):
    """Receive one byte, most significant bit first, and return it.

    The byte is acknowledged unless last=1, which tells the device that this
    was the last byte the controller wants.
    """
    byte = 0
    for i in range(8):
        bit = yield from i2c_receive_bit(port, bus)
        byte = byte | (bit << (7 - i))
    yield from i2c_send_bit(port, bus, last)
    return byte


def i2c_write_register(port, bus, address, register, value):
    """Write `value` into a register of the device at `address` (7 bits).

    Returns nack: 0 if the device acknowledged everything, 1 if not.
    """
    yield from i2c_start(port, bus)
    nack_address = yield from i2c_write_byte(port, bus, address << 1)
    nack_register = yield from i2c_write_byte(port, bus, register)
    nack_value = yield from i2c_write_byte(port, bus, value)
    yield from i2c_stop(port, bus)
    return nack_address | nack_register | nack_value


def i2c_read_register(port, bus, address, register):
    """Read one register of the device at `address` (7 bits).

    Returns (value, nack); nack is 1 if the device didn't acknowledge.
    """
    value, nack = yield from i2c_read_registers(port, bus, address, register, 1)
    return value, nack


def i2c_read_registers(port, bus, address, register, n):
    """Read `n` registers in a row, starting at `register`, in one transfer.

    Reading them in one transfer matters when they belong together, such as
    an accelerometer's X, Y and Z: they are then all from the same moment.

    Returns (value, nack). The bytes are packed into one number, the first
    register in the lowest byte. nack is 1 if the device didn't acknowledge.
    """
    # Write the register's number, then restart the transfer to read from it.
    yield from i2c_start(port, bus)
    nack_address = yield from i2c_write_byte(port, bus, address << 1)
    nack_register = yield from i2c_write_byte(port, bus, register)
    yield from i2c_start(port, bus)
    nack_read = yield from i2c_write_byte(port, bus, (address << 1) | 1)
    value = 0
    for i in range(n):
        byte = yield from i2c_read_byte(port, bus, 1 if i == n - 1 else 0)
        value = value | (byte << (8 * i))
    yield from i2c_stop(port, bus)
    return value, nack_address | nack_register | nack_read


# ---------------------------------------------------------------------------
# Printing to the PC over the JTAG UART (a debug console)
#
# The design's ports tx_data, tx_send and tx_room go to the board's JTAG UART
# pins (JTAG_TX_DATA, JTAG_TX_VALID, JTAG_TX_READY). On the PC, the text shows
# in juart-terminal, or in Python through Py2ver.jtag_uart().
# ---------------------------------------------------------------------------

def jtag_send(port, byte, data="tx_data", send="tx_send", room="tx_room"):
    """Send one byte to the PC, once the JTAG UART has room for it."""
    yield from wait_until(port, room)
    setattr(port, data, byte)
    setattr(port, send, 1)
    yield
    setattr(port, send, 0)


def jtag_print(port, text, data="tx_data", send="tx_send", room="tx_room"):
    """Send a fixed message to the PC, such as b"button pressed\\n"."""
    for i in range(len(text)):
        yield from jtag_send(port, text[i], data, send, room)


def jtag_print_hex(port, value, digits, data="tx_data", send="tx_send", room="tx_room"):
    """Send a number to the PC as `digits` hexadecimal digits, such as 00FF."""
    for i in range(digits):
        digit = (value >> (4 * (digits - 1 - i))) & 15
        # The ASCII code of '0'..'9' is 48..57, and of 'A'..'F' is 65..70.
        yield from jtag_send(port, digit + 48 if digit < 10 else digit + 55, data, send, room)


# ---------------------------------------------------------------------------
# For testbenches
# ---------------------------------------------------------------------------

def together(*steps):
    """Run several testbench steps at the same time, one clock cycle at a time.

    Returns a list of the steps' return values, once all of them have
    returned. For example, send a byte to the design while listening to its
    answer:

        _, answer = yield from together(uart_send(dut, "rx", byte, BIT),
                                        uart_receive(dut, "tx", BIT))

    Every step must end on its own; one that watches the pins forever keeps
    together() from ever returning. (Inside a design, run things at the same
    time with connect() instead.)
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
                except StopIteration as stop:
                    done[k], values[k] = True, stop.value
        if all(done):
            return values
        yield
        waits = [w - 1 for w in waits]
