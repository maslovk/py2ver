"""
The ADXL345 accelerometer on I2C (its CS high: on the DE0-Nano, G_SENSOR_CS_N;
its SDO, ALT ADDRESS, high there: at 0x1D, else 0x53). At its +-2 g range,
1 g is about 256 counts.

    yield from adxl345_start(port, bus)               # measuring
    x, y, z, nack = yield from adxl345_read_xyz(port, bus)

X, Y and Z come as 16 bits each, two's complement (signed16() in Python).
"""
from ..protocol import I2C, i2c_read_register, i2c_read_registers, i2c_write_register
from .i2c_target import i2c_memory

ADDRESS = 0x1D                  # (ALT ADDRESS high, as on the DE0-Nano; low: 0x53)
DEVID = 0x00                    # its id, 0xE5
ID = 0xE5
POWER_CTL = 0x2D
MEASURE = 0x08                  # (POWER_CTL's bit 3: measuring, not standby)
DATA_FORMAT = 0x31              # (0: +-2 g, 10 bits)
DATAX0 = 0x32                   # X, Y, Z: 6 bytes from here, each axis's low first


def adxl345_id(port, bus, address=ADDRESS):
    """Its id (DEVID: 0xE5), nack (1 if it didn't answer)."""
    value, nack = yield from i2c_read_register(port, bus, address, DEVID)
    return value, nack


def adxl345_start(port, bus, address=ADDRESS):
    """Measuring (out of standby, as it starts); nack, 1 if it didn't answer."""
    nack = yield from i2c_write_register(port, bus, address, POWER_CTL, MEASURE)
    return nack


def adxl345_read_xyz(port, bus, address=ADDRESS):
    """X, Y, Z (16 bits each, two's complement), read in one transfer so they're
    of one sample; nack, 1 if it didn't answer."""
    xyz, nack = yield from i2c_read_registers(port, bus, address, DATAX0, 6)
    return xyz & 0xFFFF, (xyz >> 16) & 0xFFFF, xyz >> 32, nack


def signed16(v):
    """A reading as a number: 16 bits of two's complement."""
    return v - 65536 if v >= 32768 else v


def registers(x=0, y=0, z=256):
    """The chip's registers as its model starts (its id, standby, a reading of
    x, y, z: lying flat, 1 g on Z)."""
    regs = {DEVID: ID, POWER_CTL: 0, DATA_FORMAT: 0}
    for k, v in enumerate((x, y, z)):
        regs[DATAX0 + 2 * k] = v & 0xFF
        regs[DATAX0 + 2 * k + 1] = (v >> 8) & 0xFF
    return regs


def adxl345_model(dut, regs, transfers, address=ADDRESS, bus=I2C()):
    """A testbench step playing the ADXL345 (regs: its registers, registers()
    to start from), until transfers bytes have gone either way; their log."""
    log = yield from i2c_memory(dut, regs, transfers, address, bus=bus)
    return log
