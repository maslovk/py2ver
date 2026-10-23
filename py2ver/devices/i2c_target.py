"""
A model of an I2C chip, for testbenches.

i2c_memory() plays a chip made of registers, as most I2C chips are: it watches
SCL and SDA one clock cycle at a time, answers its own address, and lets the
design write and read its registers. The device models in this package are
built on it.
"""
from ..protocol import I2C, wait_until


def wait_start(dut, bus=I2C()):
    """Wait for a start condition: SDA falling while SCL is high."""
    while True:
        yield from wait_until(dut, bus.sda, 1)
        yield from wait_until(dut, bus.sda, 0)
        if getattr(dut, bus.scl):
            return


def take_bit(dut, bus=I2C()):
    """Read the next bit from the bus.

    Returns 0 or 1, or "start" / "stop" if SDA changed while SCL was high.
    """
    yield from wait_until(dut, bus.scl, 0)          # right after a start, SCL is still high
    yield from wait_until(dut, bus.scl, 1)
    first = getattr(dut, bus.sda)
    while getattr(dut, bus.scl) == 1:
        if getattr(dut, bus.sda) != first:
            return "start" if getattr(dut, bus.sda) == 0 else "stop"
        yield
    return first


def take_byte(dut, bus=I2C()):
    """Read the next byte from the bus (or "start" / "stop" if one comes first)."""
    byte = 0
    for _ in range(8):
        bit = yield from take_bit(dut, bus)
        if bit in ("start", "stop"):
            return bit
        byte = (byte << 1) | bit
    return byte


def acknowledge(dut, stretch=0, bus=I2C()):
    """Acknowledge a byte: hold SDA low through the next SCL pulse.

    With stretch, first hold SCL low for that many clock cycles, as a slow
    chip does to make the controller wait (clock stretching).
    """
    setattr(dut, bus.sda, 0)
    if stretch:
        setattr(dut, bus.scl, 0)
        yield stretch
        setattr(dut, bus.scl, 1)
    yield from wait_until(dut, bus.scl, 1)
    yield from wait_until(dut, bus.scl, 0)
    setattr(dut, bus.sda, 1)


def send_byte(dut, byte, bus=I2C()):
    """Send a byte to the controller, one bit per SCL pulse.

    Returns the controller's acknowledge bit: 0 means it wants another byte.
    """
    for i in range(8):
        setattr(dut, bus.sda, (byte >> (7 - i)) & 1)
        yield from wait_until(dut, bus.scl, 1)
        yield from wait_until(dut, bus.scl, 0)
    setattr(dut, bus.sda, 1)
    ack = yield from take_bit(dut, bus)
    return ack


def i2c_memory(dut, mem, transfers, address=0x50, stretch=0, bus=I2C(), write_clocks=0):
    """A testbench step that plays an I2C chip of registers at `address`.

    mem holds the registers (number -> value); a register that isn't there
    reads 0xFF, like an erased EEPROM. The chip works the usual way:

    - a write sends a register number, then values for it and the registers after it;
    - a read, after a write of just the register number, gets values from there on.

    Options:
        stretch       hold SCL low this many clock cycles after each acknowledge
        write_clocks  after each write, ignore the bus this long (an EEPROM storing it)

    Stops after `transfers` bytes have been written or read, and returns a log
    of them: ("write" or "read", register, value) for each.
    """
    log = []
    register = 0
    setattr(dut, bus.scl, 1)
    setattr(dut, bus.sda, 1)
    while len(log) < transfers:
        yield from wait_start(dut, bus)
        while len(log) < transfers:                 # after each start or repeated start
            first = yield from take_byte(dut, bus)
            if first == "start":
                continue
            if first == "stop" or first >> 1 != address:
                break                               # not for us: don't acknowledge
            yield from acknowledge(dut, stretch, bus)

            if first & 1:                           # a read: send registers until told to stop
                ack = 0
                while ack == 0:
                    value = mem.get(register, 0xFF)
                    log.append(("read", register, value))
                    ack = yield from send_byte(dut, value, bus)
                    register = (register + 1) & 255
                break

            number = yield from take_byte(dut, bus)  # a write: first the register's number
            if number in ("start", "stop"):
                break
            register = number
            yield from acknowledge(dut, stretch, bus)
            wrote = False
            while True:                             # then values, until a stop or a restart
                value = yield from take_byte(dut, bus)
                if value in ("start", "stop"):
                    break
                mem[register] = value
                log.append(("write", register, value))
                wrote = True
                register = (register + 1) & 255
                yield from acknowledge(dut, stretch, bus)
            if value == "stop":
                if wrote and write_clocks:
                    yield write_clocks              # busy storing: deaf to the bus
                break
    return log
