"""
A 24LC02-style I2C EEPROM: 256 bytes that keep their values without power.

The DE0-Nano has one at address 0x50. An erased byte reads 0xFF. After a
write the chip needs up to 5 ms to store the byte, and ignores the bus until
it's done; eeprom_write() waits for that by asking the chip until it answers
again (acknowledge polling).

    value, nack = yield from eeprom_read(port, bus, 0x10)
    nack = yield from eeprom_write(port, bus, 0x10, 0x42)   # permanent!
"""
from ..protocol import (I2C, i2c_read_register, i2c_start, i2c_stop, i2c_write_byte,
                        i2c_write_register)
from .i2c_target import i2c_memory

ADDRESS = 0x50
MAX_POLLS = 255     # about 25 ms at 100 kHz; a write takes at most 5 ms


def eeprom_read(port, bus, offset, address=ADDRESS):
    """Read the byte at `offset`. Returns (value, nack); nack is 1 if the chip didn't answer."""
    value, nack = yield from i2c_read_register(port, bus, address, offset)
    return value, nack


def eeprom_write(port, bus, offset, value, address=ADDRESS):
    """Write `value` at `offset`, and wait until the chip has stored it.

    Returns nack: 1 if the chip didn't take the write, or never finished it.
    """
    nack = yield from i2c_write_register(port, bus, address, offset, value)
    # While the chip is busy storing the byte, it doesn't answer its address.
    busy = 1 - nack
    polls = 0
    while busy and polls < MAX_POLLS:
        yield from i2c_start(port, bus)
        busy = yield from i2c_write_byte(port, bus, address << 1)
        yield from i2c_stop(port, bus)
        polls = polls + 1
    return nack | busy


# ---------------------------------------------------------------------------
# A model of the chip, for testbenches
# ---------------------------------------------------------------------------

def eeprom_model(dut, mem, transfers, address=ADDRESS, write_clocks=5000, bus=I2C()):
    """A testbench step that plays the EEPROM on the bus.

    mem holds its bytes (offset -> value; a missing one reads 0xFF). After
    each write the model ignores the bus for `write_clocks` clock cycles, as
    the real chip does. It stops after `transfers` bytes have been written or
    read, and returns a log of them.
    """
    log = yield from i2c_memory(dut, mem, transfers, address, bus=bus,
                                write_clocks=write_clocks)
    return log
