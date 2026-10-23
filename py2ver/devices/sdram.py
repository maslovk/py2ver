"""The DE0-Nano's SDRAM (IS42S16160B: 32 MB, 16 bits wide, 4 banks of 8192 rows
of 512 halfwords): a controller written as a process, and a model of the chip
for testbenches.

The controller serves a client 32-bit words, one access at a time, through a
request and a response:

    req, req_write, req_address, req_data, req_mask  -> the access (address in words,
                                                        the mask: the bytes to write)
    req_ready                                        <- 1 for the clock it is taken
    resp_valid, resp_data                            <- a read's word, for a clock

which a client process drives with its steps:

    yield from sdram_write(port, address, data)              # (mask=: some bytes only)
    data = yield from sdram_read(port, address)

A word is two halfwords next to each other in a row: the controller opens the
row, reads or writes the two (the second closing the row again), and every
7 us or so refreshes the chip. Its pins (pins=SDRAM_PINS) are the SDRAM's;
dq_out and dq_in share DRAM_DQ (dq_out drives it while dq_oe is 1), and
dram_clk is the clock itself, inverted, so that the chip samples a command half
a clock after it is set.

On the board each input from a pin passes two registers first (see pins.py),
so a read's data reaches the controller four clocks after the clock it gave
the READ: CAS latency 2, and those two. SdramModel gives a testbench the same.
"""
from typing import NamedTuple

from ..protocol import wait_until

WORDS = 1 << 23                          # 32 MB in 32-bit words


class SdramTiming(NamedTuple):
    """The chip's waits, in clocks at 50 MHz (a -7 part: 20 ns a clock)."""
    power_up: int = 10_000               # 200 us before the first command
    refresh: int = 300                   # between refreshes (8192 in 64 ms: 7.8 us at most)
    rcd: int = 2                         # ACTIVE to READ or WRITE
    rp: int = 2                          # PRECHARGE to ACTIVE
    rfc: int = 5                         # REFRESH to anything
    mrd: int = 2                         # LOAD MODE to anything
    after: int = 4                       # an access's last command to the next ACTIVE
    #                                      (write recovery and the auto precharge)
    read_delay: int = 4                  # READ to its data at the controller (CL 2, the pins' 2 registers)


TIMING = SdramTiming()
MODE = 0b000_1_00_010_0_000              # writes a word at a time, CAS latency 2, bursts of 1

# The pins on the DE0-Nano, for pins=
SDRAM_PINS = {"dram_clk": "DRAM_CLK", "cke": "DRAM_CKE", "cs_n": "DRAM_CS_N",
              "ras_n": "DRAM_RAS_N", "cas_n": "DRAM_CAS_N", "we_n": "DRAM_WE_N",
              "addr": "DRAM_ADDR", "ba": "DRAM_BA", "dqm": "DRAM_DQM",
              "dq_out": "DRAM_DQ", "dq_in": "DRAM_DQ"}

# The controller's attr: the client's inputs, and the pins' kinds
SDRAM_ATTR = {"req": 1, "req_write": 1, "req_address": 23, "req_data": 32, "req_mask": 4,
              "dq_in": 16, "dq_out": {"width": 16, "tristate": "dq_oe"},
              "dram_clk": {"width": 1, "clock": "inverted"}}


def sdram_command(port, ras_n, cas_n, we_n):
    """One command to the chip, for a clock (then NOP)."""
    port.cs_n = 0
    port.ras_n = ras_n
    port.cas_n = cas_n
    port.we_n = we_n
    yield
    port.cs_n = 1
    port.ras_n = 1
    port.cas_n = 1
    port.we_n = 1


def sdram_controller(port, timing=TIMING):
    """The controller: the chip set up, then the client's accesses and the refreshes."""
    port.dram_clk = 0                    # (the board drives it: the clock, inverted)
    port.cke = 1
    port.cs_n = 1
    port.ras_n = 1
    port.cas_n = 1
    port.we_n = 1
    port.addr = 0
    port.ba = 0
    port.dqm = 3
    port.dq_out = 0
    port.dq_oe = 0
    port.req_ready = 0
    port.resp_valid = 0
    port.resp_data = 0
    # ---- power-up: wait, precharge every bank, refresh twice, set the mode ----
    yield timing.power_up
    port.addr = 1 << 10                  # (A10: all banks)
    yield from sdram_command(port, 0, 1, 0)          # PRECHARGE
    yield timing.rp
    yield from sdram_command(port, 0, 0, 1)          # REFRESH
    yield timing.rfc
    yield from sdram_command(port, 0, 0, 1)
    yield timing.rfc
    port.addr = MODE
    port.ba = 0
    yield from sdram_command(port, 0, 0, 0)          # LOAD MODE
    yield timing.mrd
    port.dqm = 0
    since = 0                                        # clocks since the last refresh
    while True:
        if since >= timing.refresh:                  # ---- a refresh ----
            yield from sdram_command(port, 0, 0, 1)
            yield timing.rfc
            since = 0
        elif port.req:                               # ---- the client's access ----
            write = port.req_write
            word = port.req_address
            data = port.req_data
            mask = port.req_mask
            port.req_ready = 1
            port.ba = word >> 21                     # (bank, row, the first halfword's column)
            port.addr = (word >> 8) & 8191
            yield from sdram_command(port, 0, 1, 1)  # ACTIVE: the row open
            port.req_ready = 0
            yield timing.rcd - 1
            column = (word & 255) << 1
            if write:
                port.addr = column
                port.dq_out = data & 0xFFFF
                port.dqm = 3 ^ (mask & 3)            # (DQM high: that byte not written)
                port.dq_oe = 1
                yield from sdram_command(port, 1, 0, 0)          # WRITE the low half
                port.addr = (1 << 10) | (column + 1)             # (A10: then close the row)
                port.dq_out = data >> 16
                port.dqm = 3 ^ (mask >> 2)
                yield from sdram_command(port, 1, 0, 0)          # WRITE the high half
                port.dq_oe = 0
                port.dqm = 0
                yield timing.after
                since = since + timing.rcd + 2 + timing.after
            else:
                port.addr = column
                yield from sdram_command(port, 1, 0, 1)          # READ the low half
                port.addr = (1 << 10) | (column + 1)
                yield from sdram_command(port, 1, 0, 1)          # READ the high half
                yield timing.read_delay - 2
                low = port.dq_in
                yield
                port.resp_data = (port.dq_in << 16) | low
                port.resp_valid = 1
                yield
                port.resp_valid = 0
                since = since + timing.rcd + timing.read_delay + 2
        else:
            yield
            since = since + 1


# ----------------------------- a client's steps -----------------------------

def sdram_write(port, address, data, mask=15):
    """Write a word (the bytes that mask's bits choose) at a word address, and
    return once the controller has taken it."""
    port.req_address = address
    port.req_data = data
    port.req_mask = mask
    port.req_write = 1
    port.req = 1
    yield
    yield from wait_until(port, "req_ready")
    port.req = 0


def sdram_read(port, address):
    """Read the word at a word address."""
    port.req_address = address
    port.req_write = 0
    port.req = 1
    yield
    yield from wait_until(port, "req_ready")
    port.req = 0
    yield from wait_until(port, "resp_valid")
    return port.resp_data


# ----------------------------- a model, for testbenches -----------------------------

COMMANDS = {(0, 1, 1): "active", (1, 0, 1): "read", (1, 0, 0): "write",
            (0, 1, 0): "precharge", (0, 0, 1): "refresh", (0, 0, 0): "mode"}


class SdramModel:
    """The chip on the controller's pins, in a testbench.

    It opens and closes rows, writes halfwords (keeping the bytes DQM masks)
    and reads them: a read's data reaches dq_in `delay` clocks after the clock
    the controller gave the READ in (CAS latency 2, and the board's two input
    registers). It fails a testbench at a command the chip can't take then: a
    read or write with no row open, an ACTIVE to a bank whose row is still
    open, a refresh with a row open, a write not driving the bus.

    mem: halfword address -> value. commands: each command as it came,
    (clock, name, bank, address), for a test to check the timing of.
    """

    def __init__(self, delay=TIMING.read_delay):
        self.mem = {}
        self.delay = delay
        self.open = {}                           # bank -> its open row
        self.commands = []

    def run(self, dut):
        """A testbench step: forever (alongside the rest)."""
        pending = []                             # [clocks left, value] of reads on their way
        clock = 0
        while True:
            for p in pending:
                p[0] -= 1
            if pending and pending[0][0] == 0:
                dut.dq_in = pending.pop(0)[1]
            if dut.cs_n == 0:
                self.command(dut, clock, pending)
            clock += 1
            yield

    def command(self, dut, clock, pending):
        """The command on the pins this clock."""
        name = COMMANDS.get((dut.ras_n, dut.cas_n, dut.we_n))
        bank, a = dut.ba, dut.addr
        self.commands.append((clock, name, bank, a))
        where = f"SDRAM, clock {clock:,}"
        if name == "active":
            assert bank not in self.open, f"{where}: ACTIVE to bank {bank}, whose row is open"
            self.open[bank] = a
        elif name in ("read", "write"):
            assert bank in self.open, f"{where}: a {name} to bank {bank}, no row open"
            address = (bank << 22) | (self.open[bank] << 9) | (a & 511)
            if name == "write":
                assert dut.dq_oe, f"{where}: a write, the bus not driven"
                keep = (0xFF if dut.dqm & 1 else 0) | (0xFF00 if dut.dqm & 2 else 0)
                self.mem[address] = (self.mem.get(address, 0) & keep) | (dut.dq_out & ~keep & 0xFFFF)
            else:
                # (seen a clock after the controller gave it: on dq_in delay - 1 clocks on)
                pending.append([self.delay - 1, self.mem.get(address, 0)])
            if a & (1 << 10):                    # (auto precharge: the row closes)
                del self.open[bank]
        elif name == "precharge":
            if a & (1 << 10):
                self.open.clear()
            else:
                self.open.pop(bank, None)
        elif name == "refresh":
            assert not self.open, f"{where}: a refresh with rows open"

    def word(self, address):
        """The 32-bit word at a word address, as the controller stores it."""
        h = ((address >> 21) << 22) | (((address >> 8) & 8191) << 9) | ((address & 255) << 1)
        return self.mem.get(h, 0) | (self.mem.get(h + 1, 0) << 16)

