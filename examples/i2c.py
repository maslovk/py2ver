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
from py2ver.protocol import (i2c_read_byte, i2c_start, i2c_stop, i2c_write_byte,  # noqa: E402
                             wait_until)

CLOCK = 50_000_000
Q = CLOCK // 100_000 // 4              # a quarter of a 100 kHz SCL period: 125 clocks


# ----------------------------- the master -----------------------------
# (the bus's steps, as its spec says: py2ver.protocol's i2c_start, i2c_stop,
# i2c_write_byte, i2c_read_byte, ...)

def i2c_master(port, q=Q):
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
        yield from i2c_start(port, q)
        n1 = yield from i2c_write_byte(port, address << 1, q)    # the device, to write
        n2 = yield from i2c_write_byte(port, register, q)        # the register's number
        if read:
            yield from i2c_start(port, q)                        # a repeated start
            n3 = yield from i2c_write_byte(port, (address << 1) | 1, q)   # the device, to read
            got = yield from i2c_read_byte(port, 1, q)           # one byte, the last
            port.got = got
        else:
            n3 = yield from i2c_write_byte(port, value, q)
        yield from i2c_stop(port, q)
        port.nack = n1 | n2 | n3
        port.done = 1
        yield


ATTR = {"start": 1, "address": 7, "register": 8, "read": 1, "value": 8,
        "scl": {"open_drain": True}, "sda": {"open_drain": True}}


# ----------------------------- the testbench: an I2C memory -----------------------------

def wait_start(dut):
    """Until a start: SDA falling while SCL is high."""
    while True:
        yield from wait_until(dut, "sda", 1)
        yield from wait_until(dut, "sda", 0)
        if dut.scl:
            return


def take_bit(dut):
    """The next bit on the bus: SDA at SCL's next rising edge, once SCL falls; or
    'start' / 'stop' if SDA changed while SCL was high."""
    yield from wait_until(dut, "scl", 0)            # (after a start SCL is still high)
    yield from wait_until(dut, "scl", 1)
    first = dut.sda
    while dut.scl == 1:
        if dut.sda != first:
            return "start" if dut.sda == 0 else "stop"
        yield
    return first


def take_byte(dut):
    byte = 0
    for _ in range(8):
        b = yield from take_bit(dut)
        if b in ("start", "stop"):
            return b
        byte = (byte << 1) | b
    return byte


def acknowledge(dut, stretch):
    """SDA low through the next SCL pulse, SCL held low stretch clocks before it
    (clock stretching: the master waits)."""
    dut.sda = 0
    if stretch:
        dut.scl = 0
        yield stretch
        dut.scl = 1                                 # (the line high at once: the pulse)
    yield from wait_until(dut, "scl", 1)
    yield from wait_until(dut, "scl", 0)
    dut.sda = 1


def send_byte(dut, byte):
    """A byte out, a bit each SCL pulse; the master's acknowledge (0: one more)."""
    for i in range(8):
        dut.sda = (byte >> (7 - i)) & 1
        yield from wait_until(dut, "scl", 1)
        yield from wait_until(dut, "scl", 0)
    dut.sda = 1
    ack = yield from take_bit(dut)
    return ack


def i2c_memory(dut, mem, transfers, address=0x50, stretch=0):
    """A testbench playing an I2C memory at address, as a 24LC02 does: a write
    sends a register's number, then values; a read (after a write of the number)
    gets the values from it on. It acknowledges what's its own, holds SCL low
    stretch clocks after each byte it acknowledges, and returns after transfers
    bytes written or read: their log, (kind, register, value) each."""
    log, reg = [], 0
    dut.scl, dut.sda = 1, 1
    while len(log) < transfers:
        yield from wait_start(dut)
        while len(log) < transfers:                 # (after a start)
            b = yield from take_byte(dut)
            if b == "start":
                continue
            if b == "stop" or b >> 1 != address:    # (not ours: no acknowledge)
                break
            yield from acknowledge(dut, stretch)
            if b & 1:                               # read: values from reg on
                ack = 0
                while ack == 0:
                    log.append(("read", reg, mem[reg]))
                    ack = yield from send_byte(dut, mem[reg])
                    reg = (reg + 1) & 255
                break
            r = yield from take_byte(dut)           # write: the register's number
            if r in ("start", "stop"):
                break
            reg = r
            yield from acknowledge(dut, stretch)
            while True:                             # values, or a repeated start
                v = yield from take_byte(dut)
                if v in ("start", "stop"):
                    break
                mem[reg] = v
                log.append(("write", reg, v))
                reg = (reg + 1) & 255
                yield from acknowledge(dut, stretch)
            if v == "stop":
                break
    return log


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
    p = Py2ver(functools.partial(i2c_master, q=q), ATTR)
    run = p.check(with_memory(ops, {}, 2, stretch=3 * q), verilog="--sim" in sys.argv[1:])
    done, log = run.value
    print(f"{len(run.inputs):,} clocks: the master got {[(hex(g), n) for g, n in done]} "
          f"(got, nack); the memory saw {log}")
    print("the same in the model" + (f" and the Verilog ({run.simulator})"
                                     if "--sim" in sys.argv[1:] else ""))
    if "--syn" in sys.argv[1:]:
        print(Py2ver(i2c_master, ATTR).SYN({"QUARTUS_DIR": os.environ.get(
            "QUARTUS_DIR", "/opt/intelFPGA/22.1std/quartus/bin")}))
