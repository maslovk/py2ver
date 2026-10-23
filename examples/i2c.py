# An I2C controller that writes and reads a register of a device, built from
# py2ver.protocol's I2C steps, and tested against a model of a memory chip on
# the same two lines.
#
# How I2C works: there are two lines, SCL (the clock) and SDA (the data). Both
# are open drain: anyone on the bus can pull a line low, and a pull-up makes it
# high when nobody does. A transfer begins with a start (SDA falls while SCL is
# high) and ends with a stop (SDA rises while SCL is high). In between, SDA only
# changes while SCL is low, and is read while SCL is high. After each byte (most
# significant bit first) the receiver acknowledges it by pulling SDA low. A slow
# device may hold SCL low to make the controller wait (clock stretching).
# See docs/processes.md for how processes work.
#
#   python examples/i2c.py           # test it against the generator
#   python examples/i2c.py --fast    # ... with a faster bus, for a quick run
#   python examples/i2c.py --sim     # also simulate the generated Verilog
#   python examples/i2c.py --syn     # also build it in Quartus
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py2ver.devices.i2c_target import i2c_memory  # noqa: E402
from py2ver.protocol import (I2C, i2c_read_byte, i2c_start, i2c_stop,  # noqa: E402
                             i2c_write_byte, together, wait_until)

CLOCK = 50_000_000
Q = CLOCK // 100_000 // 4              # a quarter of a 100 kHz SCL period: 125 clock cycles
BUS = I2C(q=Q)                         # on the pins scl and sda


# ----------------------------- the controller -----------------------------

def i2c_master(port, bus=BUS):
    """The controller: when port.start goes high, write or read one register.

    The device is at port.address (7 bits) and the register is port.register.
    With port.read low it writes port.value; with port.read high it reads the
    register into port.got. port.nack is 1 if the device didn't acknowledge,
    and port.done goes high for one cycle when it's over.
    """
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
        nack_address = yield from i2c_write_byte(port, bus, address << 1)   # address the device
        nack_register = yield from i2c_write_byte(port, bus, register)      # say which register
        if read:
            yield from i2c_start(port, bus)                                 # a repeated start
            nack_value = yield from i2c_write_byte(port, bus, (address << 1) | 1)  # now to read
            got = yield from i2c_read_byte(port, bus, 1)                    # one byte, the last
            port.got = got
        else:
            nack_value = yield from i2c_write_byte(port, bus, value)
        yield from i2c_stop(port, bus)
        port.nack = nack_address | nack_register | nack_value
        port.done = 1
        yield


# The inputs' widths; scl and sda are open drain.
ATTR = {"start": 1, "address": 7, "register": 8, "read": 1, "value": 8,
        "scl": {"open_drain": True}, "sda": {"open_drain": True}}


# ----------------------------- the testbench -----------------------------
# The memory chip on the other end is i2c_memory, from py2ver.devices.i2c_target.
# It's a generator too: it watches SCL and SDA one clock cycle at a time, answers
# its own address, and can stretch the clock.

def host(ops):
    """A testbench that drives the controller through a list of operations.

    Each operation is (address, register, read, value). Returns (got, nack)
    for each one, once it's done.
    """
    def tb(dut):
        results = []
        for address, register, read, value in ops:
            yield from wait_until(dut, "busy", 0)
            dut.address, dut.register, dut.read, dut.value = address, register, read, value
            dut.start = 1
            yield
            dut.start = 0
            yield from wait_until(dut, "done")
            results.append((dut.got, dut.nack))
        return results
    return tb


def with_memory(ops, mem, transfers, stretch=0):
    """A testbench: the host's operations and a memory chip, running at the same time."""
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
    results, log = run.value
    for (address, register, read, value), (got, nack) in zip(ops, results):
        what = f"read register {register:#04x}" if read else f"write {value:#04x} to {register:#04x}"
        outcome = "no answer (nack)" if nack else (f"got {got:#04x}" if read else "done")
        print(f"device {address:#04x}: {what}: {outcome}")
    print(f"The memory saw: {log}")
    print(f"({len(run.inputs):,} clock cycles; the hardware model"
          + (f" and the Verilog ({run.simulator})" if "--sim" in sys.argv[1:] else "")
          + " did the same)")
    if "--syn" in sys.argv[1:]:
        print(Py2ver(i2c_master, ATTR).SYN({"QUARTUS_DIR": os.environ.get(
            "QUARTUS_DIR", "/opt/intelFPGA/22.1std/quartus/bin")}))
