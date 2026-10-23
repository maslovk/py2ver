"""
The ADXL345 accelerometer, on I2C.

On the DE0-Nano it sits at address 0x1D, and talks I2C only while its CS pin
(G_SENSOR_CS_N) is held high. In its default range of +-2 g, 1 g reads as
about 256.

    yield from adxl345_start(port, bus)                     # start measuring
    x, y, z, nack = yield from adxl345_read_xyz(port, bus)

X, Y and Z are 16-bit two's complement numbers; signed16() turns one into a
Python int.
"""
from ..protocol import I2C, i2c_read_register, i2c_read_registers, i2c_write_register
from .i2c_target import i2c_memory

ADDRESS = 0x1D          # its address when its SDO pin is high, as on the DE0-Nano (low: 0x53)

# Registers
DEVID = 0x00            # the chip's id, always 0xE5
POWER_CTL = 0x2D        # power control: write MEASURE here to start measuring
DATA_FORMAT = 0x31      # range and resolution (0 at power-up: +-2 g)
DATAX0 = 0x32           # the first of 6 data bytes: X, Y, Z, low byte first

ID = 0xE5
MEASURE = 0x08


def adxl345_id(port, bus, address=ADDRESS):
    """Read the chip's id. Returns (id, nack); id should be 0xE5."""
    value, nack = yield from i2c_read_register(port, bus, address, DEVID)
    return value, nack


def adxl345_start(port, bus, address=ADDRESS):
    """Wake the chip up so that it measures. Returns nack (1: no answer)."""
    nack = yield from i2c_write_register(port, bus, address, POWER_CTL, MEASURE)
    return nack


def adxl345_read_xyz(port, bus, address=ADDRESS):
    """Read the acceleration. Returns (x, y, z, nack).

    All three come from one transfer, so they belong to the same sample.
    Each is a 16-bit two's complement number.
    """
    xyz, nack = yield from i2c_read_registers(port, bus, address, DATAX0, 6)
    return xyz & 0xFFFF, (xyz >> 16) & 0xFFFF, xyz >> 32, nack


def signed16(value):
    """Turn a 16-bit two's complement reading into a Python int (0xFFF1 -> -15)."""
    return value - 65536 if value >= 32768 else value


# ---------------------------------------------------------------------------
# A model of the chip, for testbenches
# ---------------------------------------------------------------------------

def registers(x=0, y=0, z=256):
    """The chip's registers for its model, holding one reading of x, y, z.

    The default is a board lying flat: 1 g on Z.
    """
    regs = {DEVID: ID, POWER_CTL: 0, DATA_FORMAT: 0}
    for k, value in enumerate((x, y, z)):
        regs[DATAX0 + 2 * k] = value & 0xFF
        regs[DATAX0 + 2 * k + 1] = (value >> 8) & 0xFF
    return regs


def adxl345_model(dut, regs, transfers, address=ADDRESS, bus=I2C()):
    """A testbench step that plays the ADXL345 on the bus.

    regs holds its registers (start from registers()); a write changes them.
    The model stops after `transfers` bytes have been written or read, and
    returns a log of them: ("write" or "read", register, value) for each.
    """
    log = yield from i2c_memory(dut, regs, transfers, address, bus=bus)
    return log
