# The DE0-Nano's own I2C bus: its ADXL345 accelerometer (py2ver.devices.adxl345's
# driver) and what else answers (py2ver.protocol's I2C steps), printed on the
# JTAG console (see docs/processes.md, the guide).
#
#   python examples/accelerometer.py             # the testbench (a memory on the bus)
#   python examples/accelerometer.py --pins      # compiled for the board (not programmed)
#
# On the board: program syn/i2c_probe_pins/DE0_Nano.sof, run juart-terminal.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py2ver.devices.adxl345 import (ADDRESS as ADXL345, POWER_CTL,  # noqa: E402
                                    adxl345_read_xyz, adxl345_start, signed16)
from py2ver.protocol import I2C, i2c_read_register, jtag_print, jtag_print_hex  # noqa: E402

BUS = I2C(q=50_000_000 // 100_000 // 4)  # 100 kHz at 50 MHz, on the pins scl and sda
ADDRESSES = (0x53, 0x1D, 0x50)           # where an ADXL345 can be (two choices), and the EEPROM


def i2c_probe(port, bus=BUS):
    """Look for devices on the bus: read register 0 at each address in ADDRESSES,
    and print what came back and whether anyone answered.

    (An ADXL345 answers with its id, 0xE5.)
    """
    port.scl = 1
    port.sda = 1
    port.cs_n = 1                        # the ADXL345 talks I2C while its CS is high
    yield 50_000                         # give the bus 1 ms to settle
    yield from jtag_print(port, b"i2c probe\n")
    for k in range(len(ADDRESSES)):
        address = ADDRESSES[k]
        got, nack = yield from i2c_read_register(port, bus, address, 0)
        yield from jtag_print(port, b"0x")
        yield from jtag_print_hex(port, address, 2)
        yield from jtag_print(port, b": register 0 = 0x")
        yield from jtag_print_hex(port, got, 2)
        if nack:
            yield from jtag_print(port, b" (no acknowledge)\n")
        else:
            yield from jtag_print(port, b" (acknowledged)\n")
    yield from jtag_print(port, b"done\n")
    while True:
        yield


PERIOD = 10_000_000                      # 0.2 s at 50 MHz between readings


def print_axis(port, name, value):
    """Print a label and a 4-digit hexadecimal reading, such as x=0x0003."""
    yield from jtag_print(port, name)
    yield from jtag_print_hex(port, value, 4)


def accelerometer(port, bus=BUS, period=PERIOD):
    """Start the ADXL345, then print its X, Y and Z every period.

    At its +-2 g range, 1 g reads as about 256 (0x0100).
    """
    port.scl = 1
    port.sda = 1
    port.cs_n = 1
    yield 50_000
    nack = yield from adxl345_start(port, bus)
    yield from jtag_print(port, b"adxl345: ")
    if nack:
        yield from jtag_print(port, b"no answer\n")
    else:
        yield from jtag_print(port, b"measuring\n")
    while True:
        x, y, z, nack = yield from adxl345_read_xyz(port, bus)
        yield from print_axis(port, b"x=0x", x)
        yield from print_axis(port, b" y=0x", y)
        yield from print_axis(port, b" z=0x", z)
        yield from jtag_print(port, b"\n")
        yield period


# scl and sda are open drain; for the board, each port is put on its pin
ATTR = {"scl": {"open_drain": True}, "sda": {"open_drain": True}, "tx_room": 1}
PINS = dict(ATTR, scl=dict(ATTR["scl"], pin="I2C_SCLK"), sda=dict(ATTR["sda"], pin="I2C_SDAT"),
            cs_n="G_SENSOR_CS_N", tx_data="JTAG_TX_DATA", tx_send="JTAG_TX_VALID",
            tx_room="JTAG_TX_READY")


def with_a_memory(mem, address=0x50, transfers=1, until=b"done\n"):
    """A testbench: the design on a bus with a chip of registers at `address`.

    mem holds the chip's registers (an ADXL345's, say). Returns everything the
    design printed, up to and including `until`.
    """
    from py2ver.devices.i2c_target import i2c_memory
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


def readings(text):
    """The (x, y, z) of each line an accelerometer printed."""
    out = []
    for line in text.decode(errors="replace").splitlines():
        if line.startswith("x=0x"):
            parts = dict(p.split("=0x") for p in line.split())
            out.append(tuple(signed16(int(parts[a], 16)) for a in "xyz"))
    return out


if __name__ == "__main__":
    import functools
    from py2ver import Py2ver

    p = Py2ver(functools.partial(i2c_probe, bus=I2C(q=4)), ATTR)
    print(p.check(with_a_memory({0: 0x42}), verilog=False).value.decode())
    if "--pins" in sys.argv[1:]:
        board = Py2ver(i2c_probe, PINS)
        print(board.PINS({"QUARTUS_DIR": os.environ.get(
            "QUARTUS_DIR", "/opt/intelFPGA/22.1std/quartus/bin")}))
