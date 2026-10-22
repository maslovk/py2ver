# SPI (mode 0), from its spec to the FPGA: a master and a slave written as the
# protocol says, as Python generators in which a yield is a clock (see "Processes"
# in the README), connected as one design and checked by a testbench written the
# same way.
#
# Mode 0: the clock idle low; the chip select low for a transfer; each bit, most
# significant first, set while the clock is low and taken at its rising edge,
# both ways at once (mosi master to slave, miso slave to master).
#
#   python examples/spi.py           # the testbench: the generators, the machines' model
#   python examples/spi.py --sim     # and the Verilog, the same clock for clock
#   python examples/spi.py --syn     # the master and the slave in Quartus
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py2ver import connect  # noqa: E402
from py2ver.protocol import wait_until  # noqa: E402

HALF = 4                               # clocks a half SCLK period: 50 MHz / 8 = 6.25 MHz
BITS = 8


def spi_master(port, half=HALF, bits=BITS):
    """port.start (a clock high): send port.send, most significant bit first; the
    byte the slave sent back in port.received, port.done high a clock after."""
    port.cs_n = 1
    while True:
        port.done = 0
        port.busy = 0
        yield from wait_until(port, "start")
        out = port.send
        port.busy = 1
        port.cs_n = 0                              # select the slave
        got = 0
        for i in range(bits):
            port.mosi = (out >> (bits - 1 - i)) & 1   # set while the clock is low
            yield half
            port.sclk = 1                          # the rising edge: the slave takes it
            yield half
            got = got | (port.miso << (bits - 1 - i))  # (taken while the clock is high)
            port.sclk = 0
        yield half
        port.cs_n = 1
        port.received = got
        port.done = 1
        yield


def spi_slave(port, bits=BITS):
    """Selected (port.cs_n low): answer port.reply, take a byte into port.received,
    port.valid high a clock when it's in. The master's clock, much slower than
    this one, is followed by its edges."""
    while True:
        port.valid = 0
        yield from wait_until(port, "cs_n", 0)
        out = port.reply
        got = 0
        for i in range(bits):
            port.miso = (out >> (bits - 1 - i)) & 1
            yield from wait_until(port, "sclk", 1)     # the rising edge: take mosi
            got = got | (port.mosi << (bits - 1 - i))
            yield from wait_until(port, "sclk", 0)     # the falling edge: the next bit
        port.received = got
        port.valid = 1
        yield
        port.valid = 0
        yield from wait_until(port, "cs_n", 1)


def spi_pair():
    """The master and the slave on one SPI bus."""
    return connect(m=(spi_master, {"start": 1, "send": BITS}),
                   s=(spi_slave, {"reply": BITS}), name="spi_pair",
                   wires={"s.sclk": "m.sclk", "s.mosi": "m.mosi", "s.cs_n": "m.cs_n",
                          "m.miso": "s.miso"})


def exchange(pairs):
    """A testbench: for each (master's byte, slave's reply) a transfer; what each
    side received."""
    def tb(dut):
        got = []
        for send, reply in pairs:
            dut.reply = reply
            dut.send = send
            dut.start = 1
            yield
            dut.start = 0
            yield from wait_until(dut, "done")
            got.append((dut.m_received, dut.s_received))
            yield 3
        return got
    return tb


if __name__ == "__main__":
    import contextlib
    import io
    from py2ver import Py2ver

    pairs = [(0xA5, 0x3C), (0x01, 0x80), (0xFF, 0x00)]
    with contextlib.redirect_stdout(io.StringIO()):
        p = Py2ver(spi_pair())
    run = p.check(exchange(pairs), verilog="--sim" in sys.argv[1:])
    print(f"{len(run.inputs)} clocks: master and slave received "
          f"{[(hex(m), hex(s)) for m, s in run.value]} (sent {[(hex(a), hex(b)) for a, b in pairs]})")
    print("the same in the model" + (" and the Verilog" if "--sim" in sys.argv[1:] else ""))
    if "--syn" in sys.argv[1:]:
        q = {"QUARTUS_DIR": os.environ.get("QUARTUS_DIR", "/opt/intelFPGA/22.1std/quartus/bin")}
        with contextlib.redirect_stdout(io.StringIO()):
            m = Py2ver(spi_master, {"start": 1, "send": BITS, "miso": 1})
            s = Py2ver(spi_slave, {"reply": BITS, "sclk": 1, "mosi": 1, "cs_n": 1})
        for name, d in (("master", m), ("slave", s)):
            print(name, d.SYN(q))
