"""
A 24LC02-like I2C EEPROM: 256 bytes (2 Kbit), at 0x50 (the DE0-Nano's; a
24LC02B answers 0x50 to 0x57 alike). Erased, it reads 0xFF. A write takes the
chip up to 5 ms, and it doesn't answer meanwhile: eeprom_write() waits for it
by asking (acknowledge polling).

    value, nack = yield from eeprom_read(port, bus, 0x10)
    nack = yield from eeprom_write(port, bus, 0x10, 0x42)   # (written: the chip keeps it)
"""
from ..protocol import (I2C, i2c_read_register, i2c_start, i2c_stop, i2c_write_byte,
                        i2c_write_register)
from .i2c_target import i2c_memory

ADDRESS = 0x50
POLLS = 255                     # (asked at most: 255 at 100 kHz, some 25 ms; its write, 5 at most)


def eeprom_read(port, bus, offset, address=ADDRESS):
    """The byte at offset; nack, 1 if the chip didn't answer."""
    value, nack = yield from i2c_read_register(port, bus, address, offset)
    return value, nack


def eeprom_write(port, bus, offset, value, address=ADDRESS):
    """value written at offset, and the chip's write waited for (it answers its
    address again once done); nack, 1 if it didn't answer the write, or never
    after."""
    nack = yield from i2c_write_register(port, bus, address, offset, value)
    busy = 1 - nack
    polls = 0
    while busy and polls < POLLS:
        yield from i2c_start(port, bus)
        busy = yield from i2c_write_byte(port, bus, address << 1)
        yield from i2c_stop(port, bus)
        polls = polls + 1
    return nack | busy


def eeprom_model(dut, mem, transfers, address=ADDRESS, write_clocks=5000, bus=I2C()):
    """A testbench step playing the EEPROM (mem: offset -> byte; missing, 0xFF):
    each write taking write_clocks clocks, the chip deaf to the bus meanwhile;
    until transfers bytes have gone either way; their log."""
    log = yield from i2c_memory(dut, mem, transfers, address, bus=bus,
                                write_clocks=write_clocks)
    return log
