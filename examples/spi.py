# An SPI master and slave (mode 0), each written the way the protocol is described,
# connected into one design and tested together.
#
# SPI mode 0: the clock (sclk) is low when idle, and the chip select (cs_n) goes
# low for a transfer. Each bit is set while the clock is low and read on its
# rising edge, most significant bit first, in both directions at once: mosi from
# master to slave, miso from slave to master. See docs/processes.md for how
# processes work.
#
#   python examples/spi.py           # test it against the generators
#   python examples/spi.py --sim     # also simulate the generated Verilog
#   python examples/spi.py --syn     # also build the master and the slave in Quartus
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py2ver import connect  # noqa: E402
from py2ver.protocol import wait_until  # noqa: E402

HALF = 4                               # half an SCLK period, in clock cycles: 6.25 MHz at 50 MHz
BITS = 8


def spi_master(port, half=HALF, bits=BITS):
    """The master: when port.start goes high, send port.send to the slave.

    The byte the slave sends back at the same time ends up in port.received,
    and port.done goes high for one cycle when the transfer is over.
    """
    port.cs_n = 1
    while True:
        port.done = 0
        port.busy = 0
        yield from wait_until(port, "start")
        outgoing = port.send
        port.busy = 1
        port.cs_n = 0                              # select the slave
        incoming = 0
        for i in range(bits):
            port.mosi = (outgoing >> (bits - 1 - i)) & 1   # set the bit while the clock is low
            yield half
            port.sclk = 1                          # rising edge: the slave reads mosi
            yield half
            incoming = incoming | (port.miso << (bits - 1 - i))
            port.sclk = 0
        yield half
        port.cs_n = 1
        port.received = incoming
        port.done = 1
        yield


def spi_slave(port, bits=BITS):
    """The slave: while selected (port.cs_n low), receive a byte and answer with port.reply.

    The byte received ends up in port.received, with port.valid high for one
    cycle. The slave doesn't need the master's timing: it simply follows the
    edges of sclk.
    """
    while True:
        port.valid = 0
        yield from wait_until(port, "cs_n", 0)
        outgoing = port.reply
        incoming = 0
        for i in range(bits):
            port.miso = (outgoing >> (bits - 1 - i)) & 1
            yield from wait_until(port, "sclk", 1)     # rising edge: read mosi
            incoming = incoming | (port.mosi << (bits - 1 - i))
            yield from wait_until(port, "sclk", 0)     # falling edge: on to the next bit
        port.received = incoming
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
    """A testbench: one transfer for each (master's byte, slave's reply) pair.

    Returns what each side received, as (master, slave) pairs.
    """
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
    from py2ver import Py2ver

    pairs = [(0xA5, 0x3C), (0x01, 0x80), (0xFF, 0x00)]
    p = Py2ver(spi_pair())
    run = p.check(exchange(pairs), verilog="--sim" in sys.argv[1:])
    for (sent, reply), (master_got, slave_got) in zip(pairs, run.value):
        print(f"master sent {sent:#04x}, slave got {slave_got:#04x}; "
              f"slave replied {reply:#04x}, master got {master_got:#04x}")
    print(f"({len(run.inputs)} clock cycles; the hardware model"
          + (" and the Verilog" if "--sim" in sys.argv[1:] else "") + " did the same)")
    if "--syn" in sys.argv[1:]:
        q = {"QUARTUS_DIR": os.environ.get("QUARTUS_DIR", "/opt/intelFPGA/22.1std/quartus/bin")}
        m = Py2ver(spi_master, {"start": 1, "send": BITS, "miso": 1})
        s = Py2ver(spi_slave, {"reply": BITS, "sclk": 1, "mosi": 1, "cs_n": 1})
        for name, d in (("master", m), ("slave", s)):
            print(name, d.SYN(q))
