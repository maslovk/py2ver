# The DE0-Nano's own I2C bus: its ADXL345 accelerometer (and what else answers),
# read with py2ver.protocol's I2C steps, the answers printed on the JTAG console
# (see docs/processes.md, the guide).
#
#   python examples/accelerometer.py             # the testbench (a memory on the bus)
#   python examples/accelerometer.py --pins      # compiled for the board (not programmed)
#
# On the board: program syn/i2c_probe_pins/DE0_Nano.sof, run juart-terminal.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py2ver.protocol import (i2c_read_byte, i2c_read_register, i2c_start,  # noqa: E402
                             i2c_stop, i2c_write_byte, i2c_write_register, jtag_print,
                             jtag_print_hex)

Q = 50_000_000 // 100_000 // 4           # a quarter of a 100 kHz SCL period: 125 clocks
ADDRESSES = (0x53, 0x1D, 0x50)           # the ADXL345 (its ALT ADDRESS low, or high); an EEPROM


def i2c_probe(port, q=Q):
    """At reset: register 0 of each address in ADDRESSES read and printed (an
    ADXL345's: its id, 0xE5), with its acknowledge."""
    port.scl = 1
    port.sda = 1
    port.cs_n = 1                        # (the ADXL345's CS high: I2C)
    yield 50_000                         # (1 ms: the bus settled)
    yield from jtag_print(port, b"i2c probe\n")
    for k in range(len(ADDRESSES)):
        address = ADDRESSES[k]
        got = yield from i2c_read_register(port, address, 0, q)
        yield from jtag_print(port, b"0x")
        yield from jtag_print_hex(port, address, 2)
        yield from jtag_print(port, b": register 0 = 0x")
        yield from jtag_print_hex(port, got & 255, 2)
        if got >> 8:
            yield from jtag_print(port, b" (no acknowledge)\n")
        else:
            yield from jtag_print(port, b" (acknowledged)\n")
    yield from jtag_print(port, b"done\n")
    while True:
        yield


ADXL345 = 0x1D                           # (on the DE0-Nano: its ALT ADDRESS high)
POWER_CTL, DATAX0 = 0x2D, 0x32           # (measure: 0x08; X, Y, Z: 6 bytes from DATAX0)
PERIOD = 10_000_000                      # 0.2 s at 50 MHz between readings


def print_axis(port, name, value):
    yield from jtag_print(port, name)
    yield from jtag_print_hex(port, value, 4)


def accelerometer(port, q=Q, period=PERIOD):
    """The ADXL345 measuring (+-2 g: 1 g about 256), its X, Y and Z read (one
    transfer of 6 bytes, so they belong together) and printed every period."""
    port.scl = 1
    port.sda = 1
    port.cs_n = 1
    yield 50_000
    nack = yield from i2c_write_register(port, ADXL345, POWER_CTL, 0x08, q)
    yield from jtag_print(port, b"adxl345: ")
    if nack:
        yield from jtag_print(port, b"no answer\n")
    else:
        yield from jtag_print(port, b"measuring\n")
    while True:
        yield from i2c_start(port, q)
        yield from i2c_write_byte(port, ADXL345 << 1, q)
        yield from i2c_write_byte(port, DATAX0, q)
        yield from i2c_start(port, q)
        yield from i2c_write_byte(port, (ADXL345 << 1) | 1, q)
        x0 = yield from i2c_read_byte(port, 0, q)
        x1 = yield from i2c_read_byte(port, 0, q)
        y0 = yield from i2c_read_byte(port, 0, q)
        y1 = yield from i2c_read_byte(port, 0, q)
        z0 = yield from i2c_read_byte(port, 0, q)
        z1 = yield from i2c_read_byte(port, 1, q)
        yield from i2c_stop(port, q)
        yield from print_axis(port, b"x=0x", x0 | (x1 << 8))
        yield from print_axis(port, b" y=0x", y0 | (y1 << 8))
        yield from print_axis(port, b" z=0x", z0 | (z1 << 8))
        yield from jtag_print(port, b"\n")
        yield period


ATTR = {"scl": {"open_drain": True}, "sda": {"open_drain": True}, "tx_room": 1}
PINS = dict(ATTR, scl=dict(ATTR["scl"], pin="I2C_SCLK"), sda=dict(ATTR["sda"], pin="I2C_SDAT"),
            cs_n="G_SENSOR_CS_N", tx_data="JTAG_TX_DATA", tx_send="JTAG_TX_VALID",
            tx_room="JTAG_TX_READY")


def with_a_memory(mem, address=0x50, transfers=1, until=b"done\n"):
    """A testbench: the design on a bus with an I2C memory at address
    (examples/i2c.py's model: an ADXL345's registers can be its bytes); what it
    printed, up to until."""
    from i2c import i2c_memory
    from py2ver.protocol import together

    def printed(dut):
        dut.tx_room = 1
        text = []
        while bytes(text[-len(until):]) != until:
            if dut.tx_send:
                text.append(dut.tx_data)
            yield
        return bytes(text)

    def tb(dut):
        text, _ = yield from together(printed(dut), i2c_memory(dut, mem, transfers, address))
        return text
    return tb


def signed(v):
    return v - 65536 if v >= 32768 else v


def readings(text):
    """The (x, y, z) of each line an accelerometer printed."""
    out = []
    for line in text.decode(errors="replace").splitlines():
        if line.startswith("x=0x"):
            parts = dict(p.split("=0x") for p in line.split())
            out.append(tuple(signed(int(parts[a], 16)) for a in "xyz"))
    return out


if __name__ == "__main__":
    import functools
    from py2ver import Py2ver

    p = Py2ver(functools.partial(i2c_probe, q=4), ATTR)
    print(p.check(with_a_memory({0: 0x42}), verilog=False).value.decode())
    if "--pins" in sys.argv[1:]:
        board = Py2ver(i2c_probe, PINS)
        print(board.PINS({"QUARTUS_DIR": os.environ.get(
            "QUARTUS_DIR", "/opt/intelFPGA/22.1std/quartus/bin")}))
