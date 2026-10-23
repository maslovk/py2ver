"""
A chip on an I2C bus, for testbenches (run as Python against the design, see
drive() and check()): what an I2C chip of registers does, a clock at a time,
on the bus's two open-drain lines.
"""
from ..protocol import I2C, wait_until


def wait_start(dut, bus=I2C()):
    """Until a start: SDA falling while SCL is high."""
    while True:
        yield from wait_until(dut, bus.sda, 1)
        yield from wait_until(dut, bus.sda, 0)
        if getattr(dut, bus.scl):
            return


def take_bit(dut, bus=I2C()):
    """The next bit on the bus: SDA at SCL's next rising edge, once SCL falls; or
    'start' / 'stop' if SDA changed while SCL was high."""
    yield from wait_until(dut, bus.scl, 0)          # (after a start SCL is still high)
    yield from wait_until(dut, bus.scl, 1)
    first = getattr(dut, bus.sda)
    while getattr(dut, bus.scl) == 1:
        if getattr(dut, bus.sda) != first:
            return "start" if getattr(dut, bus.sda) == 0 else "stop"
        yield
    return first


def take_byte(dut, bus=I2C()):
    byte = 0
    for _ in range(8):
        b = yield from take_bit(dut, bus)
        if b in ("start", "stop"):
            return b
        byte = (byte << 1) | b
    return byte


def acknowledge(dut, stretch=0, bus=I2C()):
    """SDA low through the next SCL pulse, SCL held low stretch clocks before it
    (clock stretching: the controller waits)."""
    setattr(dut, bus.sda, 0)
    if stretch:
        setattr(dut, bus.scl, 0)
        yield stretch
        setattr(dut, bus.scl, 1)                    # (the line high at once: the pulse)
    yield from wait_until(dut, bus.scl, 1)
    yield from wait_until(dut, bus.scl, 0)
    setattr(dut, bus.sda, 1)


def send_byte(dut, byte, bus=I2C()):
    """A byte out, a bit each SCL pulse; the controller's acknowledge (0: one more)."""
    for i in range(8):
        setattr(dut, bus.sda, (byte >> (7 - i)) & 1)
        yield from wait_until(dut, bus.scl, 1)
        yield from wait_until(dut, bus.scl, 0)
    setattr(dut, bus.sda, 1)
    ack = yield from take_bit(dut, bus)
    return ack


def i2c_memory(dut, mem, transfers, address=0x50, stretch=0, bus=I2C(), write_clocks=0):
    """A testbench step playing a chip of registers at address (mem: register ->
    value), as a 24LC02 or an ADXL345 does: a write sends a register's number,
    then values; a read (after a write of the number) gets the values from it
    on. It acknowledges what's its own; holds SCL low stretch clocks after each
    byte it acknowledges; after a write's stop, takes write_clocks clocks to
    write (an EEPROM's write cycle: it doesn't answer meanwhile). Returns after
    transfers bytes written or read: their log, (kind, register, value) each."""
    log, reg = [], 0
    setattr(dut, bus.scl, 1)
    setattr(dut, bus.sda, 1)
    while len(log) < transfers:
        yield from wait_start(dut, bus)
        while len(log) < transfers:                 # (after a start)
            b = yield from take_byte(dut, bus)
            if b == "start":
                continue
            if b == "stop" or b >> 1 != address:    # (not ours: no acknowledge)
                break
            yield from acknowledge(dut, stretch, bus)
            if b & 1:                               # read: values from reg on
                ack = 0
                while ack == 0:
                    log.append(("read", reg, mem.get(reg, 0xFF)))
                    ack = yield from send_byte(dut, mem.get(reg, 0xFF), bus)
                    reg = (reg + 1) & 255
                break
            r = yield from take_byte(dut, bus)      # write: the register's number
            if r in ("start", "stop"):
                break
            reg = r
            yield from acknowledge(dut, stretch, bus)
            wrote = False
            while True:                             # values, or a repeated start
                v = yield from take_byte(dut, bus)
                if v in ("start", "stop"):
                    break
                mem[reg] = v
                log.append(("write", reg, v))
                wrote = True
                reg = (reg + 1) & 255
                yield from acknowledge(dut, stretch, bus)
            if v == "stop":
                if wrote and write_clocks:
                    yield write_clocks              # (writing: deaf to the bus)
                break
    return log
