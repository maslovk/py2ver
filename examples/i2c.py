# I2C, from its spec to the FPGA: a master that writes and reads a register of a
# device, with py2ver.protocol's I2C steps (written as the bus's spec says, as
# Python generators in which a yield is a clock: see docs/processes.md, the guide),
# checked by a testbench that plays an I2C memory chip on the same two open-drain
# lines.
#
# The bus: SCL and SDA, each pulled up and pulled low by whoever drives it (open
# drain: nobody drives a 1). A transfer begins with a start (SDA falls while SCL
# is high) and ends with a stop (SDA rises while SCL is high); in between, SDA
# changes only while SCL is low, and is taken while SCL is high. Each byte, most
# significant bit first, is followed by an acknowledge: the receiver pulls SDA
# low. A slave may hold SCL low to make the master wait (clock stretching).
#
#   python examples/i2c.py           # the testbench: the generator and the model
#   python examples/i2c.py --sim     # and the Verilog, the same clock for clock
#   python examples/i2c.py --syn     # in Quartus
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py2ver.protocol import (I2C, i2c_read_byte, i2c_start, i2c_stop,  # noqa: E402
                             i2c_write_byte, wait_until)

CLOCK = 50_000_000
Q = CLOCK // 100_000 // 4              # a quarter of a 100 kHz SCL period: 125 clocks
BUS = I2C(q=Q)                         # (its pins: scl, sda)


# ----------------------------- the master -----------------------------
# (the bus's steps, as its spec says: py2ver.protocol's i2c_start, i2c_stop,
# i2c_write_byte, i2c_read_byte, ...)

def i2c_master(port, bus=BUS):
    """port.start (a clock high): a register of the device at port.address (7 bits)
    written (port.value) or, with port.read, read (into port.got); port.nack if
    an acknowledge was missing, port.done high a clock when it's over. SCL and SDA
    are open drain."""
    port.scl = 1
    port.sda = 1
    while True:
        port.busy = 0
        port.done = 0
        yield from wait_until(port, "start")
        address = port.address
        register = port.register
        read = port.read
        value = port.value
        port.busy = 1
        yield from i2c_start(port, bus)
        n1 = yield from i2c_write_byte(port, bus, address << 1)  # the device, to write
        n2 = yield from i2c_write_byte(port, bus, register)      # the register's number
        if read:
            yield from i2c_start(port, bus)                      # a repeated start
            n3 = yield from i2c_write_byte(port, bus, (address << 1) | 1)   # the device, to read
            got = yield from i2c_read_byte(port, bus, 1)         # one byte, the last
            port.got = got
        else:
            n3 = yield from i2c_write_byte(port, bus, value)
        yield from i2c_stop(port, bus)
        port.nack = n1 | n2 | n3
        port.done = 1
        yield


ATTR = {"start": 1, "address": 7, "register": 8, "read": 1, "value": 8,
        "scl": {"open_drain": True}, "sda": {"open_drain": True}}


# ----------------------------- the testbench: an I2C memory -----------------------------
# (py2ver.devices.i2c_target's i2c_memory: a chip of registers on the bus, as a
# 24LC02 is, written as a generator too: it watches SCL and SDA a clock at a
# time, acknowledges what's its own, and may stretch the clock)

from py2ver.devices.i2c_target import i2c_memory  # noqa: E402


def host(ops):
    """A testbench driving the master: each op (address, register, read, value)
    started, its outcome (got, nack) when it's done."""
    def tb(dut):
        out = []
        for address, register, read, value in ops:
            yield from wait_until(dut, "busy", 0)
            dut.address, dut.register, dut.read, dut.value = address, register, read, value
            dut.start = 1
            yield
            dut.start = 0
            yield from wait_until(dut, "done")
            out.append((dut.got, dut.nack))
        return out
    return tb


def with_memory(ops, mem, transfers, stretch=0):
    """The host and the memory on one bus, at once."""
    from py2ver.protocol import together

    def tb(dut):
        done, log = yield from together(host(ops)(dut),
                                        i2c_memory(dut, mem, transfers, stretch=stretch))
        return done, log
    return tb


if __name__ == "__main__":
    import functools
    from py2ver import Py2ver

    ops = [(0x50, 0x12, 0, 0xAB),            # write 0xAB to register 0x12
           (0x50, 0x12, 1, 0),               # read register 0x12 back
           (0x51, 0x00, 0, 0x55)]            # no device at 0x51: no acknowledge
    q = 4 if "--fast" in sys.argv[1:] else Q
    p = Py2ver(functools.partial(i2c_master, bus=I2C(q=q)), ATTR)
    run = p.check(with_memory(ops, {}, 2, stretch=3 * q), verilog="--sim" in sys.argv[1:])
    done, log = run.value
    print(f"{len(run.inputs):,} clocks: the master got {[(hex(g), n) for g, n in done]} "
          f"(got, nack); the memory saw {log}")
    print("the same in the model" + (f" and the Verilog ({run.simulator})"
                                     if "--sim" in sys.argv[1:] else ""))
    if "--syn" in sys.argv[1:]:
        print(Py2ver(i2c_master, ATTR).SYN({"QUARTUS_DIR": os.environ.get(
            "QUARTUS_DIR", "/opt/intelFPGA/22.1std/quartus/bin")}))
