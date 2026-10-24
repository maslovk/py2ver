# A RISC-V computer on the FPGA: a CPU written in Python, running C programs,
# its console on the ILI9341 display and its keyboard the PC's.
#
# The CPU runs RV32IMA with machine, supervisor and user modes: the base
# integer instructions, multiply and divide (M), atomics (A), the CSRs (Zicsr),
# traps, delegated to supervisor mode as medeleg and mideleg say, and the
# timer's interrupt (user mode: for Linux's programs, their system calls an
# ecall from user mode; supervisor mode: for Linux with an MMU), and Sv32
# paging (a TLB of 32 pages for fetches, one for data). A clock to fetch an instruction, a clock to decode it and read its
# registers, a clock or more to carry it out (memory, multiplying, dividing:
# 32 clocks), a clock to write its result: about 12 million instructions a
# second. Its memory is mini-rv32ima's (the small RISC-V emulator that boots
# Linux), so that its Linux could run here later:
#
#   0x0000_0000  RAM, 16 KB: the program, loaded into it from the start (memory blocks)
#   0x1000_0000  a 16550-style UART: write a character to the screen (terminal.py's
#                terminal, and the PC's if it listens: the JTAG UART); read one typed
#                on the PC; +3, its line control (bit 7: the data register sets the
#                baud rate, ignored); +5, its status (bit 0: a character came in,
#                bit 5: room to write)
#   0x1000_1000  write: the 8 LEDs;  0x1000_1004  read: the keys (bits 0, 1; 0 pressed)
#                (KEY1 also sends the PC where the CPU is: pc, mcause, mepc, mtval)
#   0x1100_4000  the timer's mtimecmp (and +4, its upper half): the interrupt at mtime
#   0x1100_BFF8  mtime (and +4): microseconds since reset
#   0x1110_0000  write 0x5555: power off (the CPU stops); 0x7777: restart
#   0x8000_0000  the SDRAM, 32 MB (py2ver.devices.sdram's controller), instructions
#                and data through an 8 KB cache: a hit as fast as the RAM, a miss
#                the line of 8 words read in about 30 clocks; stores written
#                through (to the SDRAM, and to the cache if their line is there)
#
#   machine_timer --mtime--> cpu --characters--> terminal --color/valid/ready--> ili9341_display
#                             ^
#                             +--words--> sdram_controller --> the SDRAM's pins
#
# Programs are written in C or assembly (examples/riscv/) and built with the
# GNU RISC-V toolchain: riscv-none-elf-gcc (xPack's, unpacked in ~/opt/riscv,
# or on PATH, or named by RISCV_GCC).
#
#   python examples/riscv_computer.py          # the CPU's checks, then hello.c: test, print what they print
#   python examples/riscv_computer.py --sim    # also simulate the generated Verilog
#   python examples/riscv_computer.py --pins   # build hello.c's computer for the board
#   python examples/riscv_computer.py --pins memtest.c     # or another program's
#   python examples/riscv_computer.py --pins screen.c --framebuffer   # the display showing
#                                                # the framebuffer, not the console
import functools
import glob
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import terminal as T  # noqa: E402
from py2ver import connect  # noqa: E402
from py2ver.devices.sdram import (LINE_WORDS, SDRAM_ATTR, SDRAM_PINS, TIMING,  # noqa: E402
                                  WORDS as SDRAM_WORDS, sdram_controller, sdram_read_line,
                                  sdram_write)
from py2ver.protocol import (alongside, jtag_print, jtag_print_hex, stream_send,  # noqa: E402
                             wait_until)

HERE = os.path.dirname(os.path.abspath(__file__))
PROGRAMS = os.path.join(HERE, "riscv")

RAM_WORDS = 4096                         # 16 KB, from address 0
UART, UART_STATUS = 0x10000000, 0x10000005
LEDS, KEYS = 0x10001000, 0x10001004
MTIMECMP, MTIME = 0x11004000, 0x1100BFF8
SYSCON = 0x11100000
SDRAM = 0x80000000                       # (to 0x81FF_FFFF: an address there has 0x40 above bit 25)
# The SDRAM's cache: 8 KB, direct mapped, lines of LINE_WORDS words. A line's
# tag: the address's bits 13 to 24, and bit 12 set (1: the line holds them).
CACHE_WORDS = 2048
LINES = CACHE_WORDS // LINE_WORDS
CACHE_SHIFT = 13                         # (8 KB: the address's bits below 13 choose the word)
TAG_VALID = 4096
PC_PATIENCE = 100_000_000                # (2 s: the PC empties the JTAG UART every few ms, but
#   after a long write to the board, its next read can be a second away)
MASK = 0xFFFFFFFF                       # (the registers are 32 bits)
MISA = (1 << 30) | (1 << 20) | (1 << 18) | (1 << 12) | (1 << 8) | (1 << 0)   # 32 bits; U, S, M, I, A
SSTATUS = 0x000C0122                     # (sstatus: mstatus's SIE, SPIE, SPP, SUM, MXR)
MPRV = 1 << 17
LOAD, STORE, ATOMIC = 1, 2, 3            # (the data loop's accesses)
TLB_ENTRIES = 32                         # (each TLB's: a memory block's, 32 at least)

# ----------------------------- building programs -----------------------------


def toolchain():
    """The RISC-V gcc: RISCV_GCC, or xPack's in ~/opt/riscv, or one on PATH; None if none."""
    found = [os.environ.get("RISCV_GCC")]
    found += sorted(glob.glob(os.path.expanduser("~/opt/riscv/*/bin/riscv-none-elf-gcc")))
    found += [shutil.which("riscv-none-elf-gcc"), shutil.which("riscv64-unknown-elf-gcc")]
    return next((g for g in found if g and os.path.exists(g)), None)


def build(*sources, includes=(), defines=()):
    """A program's memory image (RAM_WORDS words, from 0): its sources (paths,
    or names in examples/riscv; C or assembly; a C program gets start.S,
    which calls its main) compiled for RV32IMA and linked to this computer.
    defines: the preprocessor's, such as "WORDS=24"."""
    gcc = toolchain()
    if gcc is None:
        raise RuntimeError("No RISC-V toolchain: unpack xPack's riscv-none-elf-gcc in ~/opt/riscv, "
                           "or set RISCV_GCC")
    paths = [s if os.path.isabs(s) else os.path.join(PROGRAMS, s) for s in sources]
    if any(s.endswith(".c") for s in sources):
        paths.insert(0, os.path.join(PROGRAMS, "start.S"))
    with tempfile.TemporaryDirectory() as work:
        elf, binary = os.path.join(work, "program.elf"), os.path.join(work, "program.bin")
        subprocess.run([gcc, "-march=rv32ima_zicsr_zifencei", "-mabi=ilp32", "-Os", "-nostdlib",
                        "-ffreestanding", "-Wl,--no-warn-rwx-segments", *[f"-I{i}" for i in includes],
                        *[f"-D{d}" for d in defines],
                        "-T", os.path.join(PROGRAMS, "link.ld"), "-o", elf, *paths, "-lgcc"],
                       check=True)
        objcopy = gcc[:-len("gcc")] + "objcopy"       # (riscv-none-elf-gcc: riscv-none-elf-objcopy)
        subprocess.run([objcopy, "-O", "binary", elf, binary], check=True)
        data = open(binary, "rb").read()
    if len(data) > 4 * RAM_WORDS:
        raise RuntimeError(f"The program is {len(data)} bytes: more than the RAM's {4 * RAM_WORDS}")
    data += bytes(4 * RAM_WORDS - len(data))
    return tuple(int.from_bytes(data[k:k + 4], "little") for k in range(0, len(data), 4))


# riscv-tests (github.com/riscv-software-src/riscv-tests), cloned next to the toolchain
RISCV_TESTS = os.path.expanduser("~/opt/riscv/riscv-tests")


def riscv_test(name, paging=False):
    """One of riscv-tests' programs (rv32ui/add, rv32um/div, rv32ua/amoadd_w...),
    built with this computer's environment (examples/riscv/env); paging: run
    in user mode with Sv32 paging on."""
    return build(os.path.join(RISCV_TESTS, "isa", name + ".S"),
                 defines=("PY2VER_PAGING",) if paging else (),
                 includes=(os.path.join(PROGRAMS, "env"), os.path.join(RISCV_TESTS, "env"),
                           os.path.join(RISCV_TESTS, "isa", "macros", "scalar")))


# ----------------------------- the CPU -----------------------------


def take_key(port):
    """A key from the PC, if one is waiting: (1, it), else (0, the input as
    it is). Takes a clock: kb_ready is high in it, so a key there is taken."""
    port.kb_ready = 1
    yield
    got = port.kb_valid
    key = port.kb
    port.kb_ready = 0
    return got, key


def tell_pc(port, text):
    """Text to the PC's console (the JTAG UART), waiting for room for it."""
    yield from jtag_print(port, text, data="jtag_tx", send="jtag_tx_send", room="jtag_tx_room")


def tell_pc_hex(port, value):
    """A 32-bit number to the PC's console, in hexadecimal."""
    yield from jtag_print_hex(port, value, 8, data="jtag_tx", send="jtag_tx_send", room="jtag_tx_room")


def rv32ima(port, image=(0x0000006F,) * RAM_WORDS, pc_patience=PC_PATIENCE):
    """The CPU, running from address 0 the memory image `image` (RAM_WORDS words).

    Outputs: the console's characters (tx, tx_valid; tx_ready: taken), and
    a copy for the PC (jtag_tx, jtag_tx_send; sent while jtag_tx_room); the
    PC keyboard's kb_ready, leds; the SDRAM controller's requests (req,
    req_write, req_line, req_address, req_data, req_mask). Inputs: tx_ready,
    jtag_tx_room; the keyboard's kb, kb_valid; keys; mtime_lo, mtime_hi (from
    machine_timer); the SDRAM controller's req_ready, resp_valid, resp_data.
    """
    port.tx_valid = 0
    port.kb_ready = 0
    port.leds = 0
    port.req = 0
    port.req_write = 0
    port.req_address = 0
    port.req_data = 0
    port.req_mask = 15
    port.req_line = 0
    port.jtag_tx = 0
    port.jtag_tx_send = 0
    ram = list(image)                            # the memory, the program in it
    cache = [0] * CACHE_WORDS                    # the SDRAM's cache: its lines' words,
    tags = [0] * LINES                           # and each line's tag (0: holds nothing)
    regs = [0] * 32                             # x0 to x31 (x0 never written: always 0)
    pc = 0
    # The privileged state: the mode, and the CSRs
    mode = 3                                     # 3: machine mode, 1: supervisor mode, 0: user mode
    interrupts = 0                               # mstatus.MIE: machine mode's interrupts on
    interrupts_before = 0                        # mstatus.MPIE: MIE before the trap
    mode_before = 3                              # mstatus.MPP: the mode before the trap
    s_interrupts = 0                             # mstatus.SIE: supervisor mode's interrupts on
    s_interrupts_before = 0                      # mstatus.SPIE: SIE before the trap
    s_mode_before = 0                            # mstatus.SPP: the mode before the trap (1 or 0)
    status_bits = 0                              # mstatus.MPRV, SUM, MXR, TVM, TW, TSR, in place (17 to 22)
    enabled = 0                                  # mie: the interrupts wanted
    pending = 0                                  # mip's bits that software sets: SSIP, STIP, SEIP
    exceptions_delegated = 0                     # medeleg: traps below machine mode, to supervisor mode
    interrupts_delegated = 0                     # mideleg: the same for interrupts
    trap_vector = 0                              # mtvec
    scratch = 0                                  # mscratch
    trap_pc = 0                                  # mepc
    cause = 0                                    # mcause
    trap_value = 0                               # mtval
    s_trap_vector = 0                            # stvec
    s_scratch = 0                                # sscratch
    s_trap_pc = 0                                # sepc
    s_cause = 0                                  # scause
    s_trap_value = 0                             # stval
    satp = 0                                     # satp (paging: its mode and root)
    # The TLBs, a fetch's and the data's: TLB_ENTRIES each, an entry for a 4 KB
    # page (a megapage's too, a page at a time), at its virtual page number's
    # lowest bits: that number (bits 28 up), the physical page's (8 to 27), the
    # page-table entry's flags (0 to 7, where they are in it); which hold one:
    # a bit each (sfence.vma, a write of satp: none)
    itlb = [0] * TLB_ENTRIES
    dtlb = [0] * TLB_ENTRIES
    itlb_valid = 0
    dtlb_valid = 0
    counters_on = 0                              # mcounteren: the counters supervisor mode may read
    s_counters_on = 0                            # scounteren: and user mode
    retired_lo = 0                               # instructions carried out (cycle, instret)
    retired_hi = 0
    compare_lo = MASK                            # mtimecmp: the timer's interrupt from then
    compare_hi = MASK
    reserved = 0                                 # lr.w's reservation: on, and where
    reservation = 0
    key_waiting = 0                              # a key taken from the PC, not yet read
    waiting_key = 0
    divisor_latch = 0                            # the UART's LCR bit 7: writes to its data set the baud rate
    patience = 0                                 # clocks to wait for room for the PC's copy (0: none listens)
    told = 0                                     # (1: KEY1 still held since the CPU's state was sent)
    while True:
        # ---- KEY1 pressed: where the CPU is, and its last trap, to the PC ----
        if port.keys & 2:
            told = 0
        elif not told:
            told = 1
            yield from tell_pc(port, b"\n[pc ")
            yield from tell_pc_hex(port, pc)
            yield from tell_pc(port, b" mcause ")
            yield from tell_pc_hex(port, cause)
            yield from tell_pc(port, b" mepc ")
            yield from tell_pc_hex(port, trap_pc)
            yield from tell_pc(port, b" mtval ")
            yield from tell_pc_hex(port, trap_value)
            yield from tell_pc(port, b"]\n")
        # (the timer's interrupt due: mip.MTIP)
        due = port.mtime_hi > compare_hi or (port.mtime_hi == compare_hi and port.mtime_lo >= compare_lo)
        trap = 0                                 # (1: a trap instead: its cause, its value)
        trap_cause = 0
        trap_with = 0
        # ---- fetch: the instruction at pc, through paging if it's on (below
        # machine mode, satp's mode Sv32): the page-table entries read first.
        # A word from the RAM, or from the SDRAM through the cache (on a miss,
        # the line read into it, and the cache looked at again) ----
        # (the TLB's entry for pc's page: read now, a clock on used if paging:
        # a hit that allows the fetch (X; U for user mode, not U for
        # supervisor mode) gives the physical address, else the walk)
        fetch_entry = itlb[(pc >> 12) & (TLB_ENTRIES - 1)]
        walk = 0                                 # (2: the root's entry next, 1: the leaf's, 0: the word)
        table = (satp & 0x3FFFFF) << 12
        physical = pc
        if satp >> 31 and mode < 3:
            yield
            if (itlb_valid >> ((pc >> 12) & (TLB_ENTRIES - 1))) & 1 and fetch_entry >> 28 == pc >> 12 \
                    and (fetch_entry >> 3) & 1 and (fetch_entry >> 4) & 1 == (1 if mode == 0 else 0):
                physical = (((fetch_entry >> 8) & 0xFFFFF) << 12) | (pc & 0xFFF)
            else:
                walk = 2
        while True:
            # (fetch_where, not the data loop's where: py2ver gives reads of a
            # memory at the same address one port, and the fetch's and the
            # data's must each have their own memory block)
            if walk == 2:
                fetch_where = (table + ((pc >> 22) << 2)) & MASK
            elif walk == 1:
                fetch_where = (table + (((pc >> 12) & 1023) << 2)) & MASK
            else:
                fetch_where = physical
            while True:
                from_ram = ram[(fetch_where >> 2) & (RAM_WORDS - 1)]
                from_cache = cache[(fetch_where >> 2) & (CACHE_WORDS - 1)]
                fetch_tag = tags[(fetch_where >> 5) & (LINES - 1)]
                yield
                if fetch_where >> 25 != SDRAM >> 25 or fetch_tag == ((fetch_where >> CACHE_SHIFT) & 4095) | TAG_VALID:
                    break
                yield from sdram_read_line(port, (fetch_where >> 2) & (SDRAM_WORDS - LINE_WORDS))
                base = (fetch_where >> 2) & (CACHE_WORDS - LINE_WORDS)
                k = 0
                while k < LINE_WORDS:
                    yield from wait_until(port, "resp_valid")
                    cache[base + k] = port.resp_data
                    k = k + 1
                    yield
                tags[(fetch_where >> 5) & (LINES - 1)] = ((fetch_where >> CACHE_SHIFT) & 4095) | TAG_VALID
                yield
            fetched = from_cache if fetch_where >> 25 == SDRAM >> 25 else from_ram
            if walk == 0:
                break
            # (a page-table entry: V R W X U G A D in bits 0 to 7, the page's
            # number above; a leaf (R or X) must allow the fetch: X, U for
            # user mode, not U for supervisor mode, A; a megapage aligned)
            leaf = (fetched >> 1) & 1 or (fetched >> 3) & 1
            user_page = (fetched >> 4) & 1
            if not fetched & 1 or ((fetched >> 2) & 1 and not (fetched >> 1) & 1) or (not leaf and walk == 1) \
                    or (leaf and (not (fetched >> 3) & 1 or user_page != (1 if mode == 0 else 0)
                                  or not (fetched >> 6) & 1 or (walk == 2 and (fetched >> 10) & 1023))):
                trap = 1                         # (a fetch's page fault)
                trap_cause = 12
                trap_with = pc
                break
            if leaf:                             # (the page: into the TLB)
                if walk == 2:                    # (a megapage: 4 MB)
                    physical = (((fetched >> 20) << 22) | (pc & 0x3FFFFF)) & MASK
                else:
                    physical = (((fetched >> 10) << 12) | (pc & 0xFFF)) & MASK
                walk = 0
                itlb[(pc >> 12) & (TLB_ENTRIES - 1)] = ((pc >> 12) << 28) | ((physical >> 12) << 8) | (fetched & 0xFF)
                itlb_valid = itlb_valid | (1 << ((pc >> 12) & (TLB_ENTRIES - 1)))
            else:
                table = ((fetched >> 10) << 12) & MASK
                walk = 1
            yield                                # (a clock before the next read: the entry used, the next asked for)
        instruction = fetched if not trap else 0x13  # (a fault: a nop, the trap pending)
        # ---- decode: the instruction's fields, the registers it reads ----
        op = instruction & 0x7F
        rd = (instruction >> 7) & 31
        f3 = (instruction >> 12) & 7
        f7 = instruction >> 25
        a = regs[(instruction >> 15) & 31]
        b = regs[(instruction >> 20) & 31]
        sign = instruction >> 31                 # (each immediate's sign: bit 31)
        imm_i = (instruction >> 20) - (sign << 12)
        imm_s = (((instruction >> 25) << 5) | ((instruction >> 7) & 31)) - (sign << 12)
        imm_b = ((sign << 12) | (((instruction >> 7) & 1) << 11) | (((instruction >> 25) & 63) << 5)
                 | (((instruction >> 8) & 15) << 1)) - (sign << 13)
        imm_u = instruction & 0xFFFFF000
        imm_j = ((sign << 20) | (((instruction >> 12) & 255) << 12) | (((instruction >> 20) & 1) << 11)
                 | (((instruction >> 21) & 1023) << 1)) - (sign << 21)
        yield
        # ---- execute ----
        result = 0
        write = 0                                # (1: rd gets result)
        access = 0                               # (a memory access to make: LOAD, STORE, ATOMIC)
        address = 0                              # (its address, its bytes, an atomic's kind)
        size = 4
        kind = 0
        next_pc = (pc + 4) & MASK
        sa = a - ((a >> 31) << 32)               # (a and b as signed numbers)
        sb = b - ((b >> 31) << 32)
        if op == 0x37:                           # lui
            result = imm_u
            write = 1
        elif op == 0x17:                         # auipc
            result = (pc + imm_u) & MASK
            write = 1
        elif op == 0x6F or op == 0x67:           # jal jalr (a target not a multiple of 4: a trap)
            result = next_pc
            target = (pc + imm_j) & MASK if op == 0x6F else (a + imm_i) & 0xFFFFFFFE
            if target & 2:
                trap = 1
                trap_cause = 0
                trap_with = target
            else:
                write = 1
                next_pc = target
        elif op == 0x63:                         # beq bne blt bge bltu bgeu
            taken = (f3 == 0 and a == b) or (f3 == 1 and a != b) or (f3 == 4 and sa < sb) \
                or (f3 == 5 and sa >= sb) or (f3 == 6 and a < b) or (f3 == 7 and a >= b)
            target = (pc + imm_b) & MASK
            if taken and target & 2:
                trap = 1
                trap_cause = 0
                trap_with = target
            elif taken:
                next_pc = target
        elif op == 0x33 and f7 == 1 and f3 < 4:  # mul mulh mulhsu mulhu
            left = sa if f3 == 1 or f3 == 2 else a
            right = sb if f3 == 1 else b
            yield
            product = left * right               # (a clock of its own: the multipliers)
            yield
            result = product & MASK if f3 == 0 else (product >> 32) & MASK
            write = 1
        elif op == 0x33 and f7 == 1:             # div divu rem remu: a bit a clock
            signed = f3 == 4 or f3 == 6
            dividend = (-sa if sa < 0 else sa) if signed else a
            divisor = (-sb if sb < 0 else sb) if signed else b
            quotient = 0
            remainder = 0
            if divisor == 0:                     # (by 0: all ones, the dividend left)
                quotient = MASK
                remainder = a
            else:
                # (long division, a bit a clock: the dividend's top bit into the
                # remainder, the divisor out of it if it goes, a 1 into the quotient)
                for i in range(32):
                    # (the remainder is under the divisor, so doubled it fits 33 bits)
                    remainder = ((remainder << 1) | (dividend >> 31)) & 0x1FFFFFFFF
                    dividend = (dividend << 1) & MASK
                    quotient = (quotient << 1) & MASK
                    if remainder >= divisor:
                        remainder = remainder - divisor
                        quotient = quotient | 1
                    yield
                if signed and (sa < 0) != (sb < 0):
                    quotient = -quotient & MASK
                if signed and sa < 0:
                    remainder = -remainder & MASK
            result = (quotient if f3 < 6 else remainder) & MASK
            write = 1
        elif op == 0x13 or op == 0x33:           # arithmetic: with an immediate, or two registers
            y = b if op == 0x33 else imm_i & MASK
            sy = y - ((y >> 31) << 32)
            shift = y & 31
            arithmetic = (instruction >> 30) & 1     # (sub, sra, srai)
            if f3 == 0:
                result = (a - y) & MASK if op == 0x33 and arithmetic else (a + y) & MASK
            elif f3 == 1:
                result = (a << shift) & MASK
            elif f3 == 2:
                result = 1 if sa < sy else 0
            elif f3 == 3:
                result = 1 if a < y else 0
            elif f3 == 4:
                result = a ^ y
            elif f3 == 5:
                result = (sa >> shift) & MASK if arithmetic else a >> shift
            elif f3 == 6:
                result = a | y
            else:
                result = a & y
            write = 1
        elif op == 0x03:                         # lb lh lw lbu lhu (the data loop's)
            access = LOAD
            address = (a + imm_i) & MASK
            size = 1 if f3 == 0 or f3 == 4 else (2 if f3 == 1 or f3 == 5 else 4)
        elif op == 0x23:                         # sb sh sw (the data loop's)
            access = STORE
            address = (a + imm_s) & MASK
            size = 1 if f3 == 0 else (2 if f3 == 1 else 4)
        elif op == 0x2F:                         # lr.w sc.w amo*.w (the data loop's)
            address = a
            size = 4
            kind = instruction >> 27
            if address & 3:
                trap = 1
                trap_cause = 6
                trap_with = address
            elif kind == 3 and not (reserved and reservation == address):
                result = 1                       # (sc.w: lr.w's reservation gone, nothing written)
                write = 1
                reserved = 0
            else:
                access = ATOMIC
        elif op == 0x73 and f3 == 0:             # ecall ebreak mret sret wfi sfence.vma
            function = instruction >> 20
            tvm = (status_bits >> 20) & 1
            if function == 0x302 and mode == 3:  # mret: back from machine mode's trap
                next_pc = trap_pc
                interrupts = interrupts_before
                interrupts_before = 1
                mode = mode_before
                mode_before = 0
                if mode != 3:
                    status_bits = status_bits & ~MPRV
            elif function == 0x102 and (mode == 3 or (mode == 1 and not (status_bits >> 22) & 1)):
                next_pc = s_trap_pc              # sret: back from supervisor mode's (TSR: not allowed)
                s_interrupts = s_interrupts_before
                s_interrupts_before = 1
                mode = s_mode_before
                s_mode_before = 0
                status_bits = status_bits & ~MPRV
            elif function == 0:
                trap = 1
                trap_cause = 8 + mode            # (ecall: from user mode 8, a system call; supervisor 9, machine 11)
            elif function == 1:
                trap = 1
                trap_cause = 3                   # (ebreak)
            elif function == 0x105 and not ((status_bits >> 21) & 1 and mode < 3):
                pass                             # (wfi: done at once; TW: not below machine mode)
            elif f7 == 0x09 and rd == 0 and (mode == 3 or (mode == 1 and not tvm)):
                itlb_valid = 0                   # (sfence.vma: the TLBs emptied; TVM: not in supervisor mode)
                dtlb_valid = 0
            else:
                trap = 1
                trap_cause = 2
                trap_with = instruction
        elif op == 0x73:                         # csrrw csrrs csrrc, and with an immediate
            csr = instruction >> 20
            source = (instruction >> 15) & 31 if f3 >= 5 else a
            # (csrrs and csrrc of x0 or 0: a read only)
            writes = f3 == 1 or f3 == 5 or (instruction >> 15) & 31 != 0
            mstatus = (s_interrupts << 1) | (interrupts << 3) | (s_interrupts_before << 5) \
                | (interrupts_before << 7) | (s_mode_before << 8) | (mode_before << 11) | status_bits
            mip = pending | (128 if due else 0)
            known = 1
            old = 0
            if csr == 0x300:
                old = mstatus
            elif csr == 0x100:
                old = mstatus & SSTATUS
            elif csr == 0x301:
                old = MISA
            elif csr == 0x304:
                old = enabled
            elif csr == 0x104:
                old = enabled & interrupts_delegated
            elif csr == 0x344:
                old = mip
            elif csr == 0x144:
                old = mip & interrupts_delegated
            elif csr == 0x305:
                old = trap_vector
            elif csr == 0x105:
                old = s_trap_vector
            elif csr == 0x340:
                old = scratch
            elif csr == 0x140:
                old = s_scratch
            elif csr == 0x341:
                old = trap_pc
            elif csr == 0x141:
                old = s_trap_pc
            elif csr == 0x342:
                old = cause
            elif csr == 0x142:
                old = s_cause
            elif csr == 0x343:
                old = trap_value
            elif csr == 0x143:
                old = s_trap_value
            elif csr == 0x302:
                old = exceptions_delegated
            elif csr == 0x303:
                old = interrupts_delegated
            elif csr == 0x306:
                old = counters_on
            elif csr == 0x106:
                old = s_counters_on
            elif csr == 0x180:
                old = satp
            elif csr == 0x310 or (csr >= 0xF11 and csr <= 0xF15):
                old = 0                          # (mstatush; mvendorid, marchid, mimpid, mhartid, mconfigptr)
            elif csr == 0xC00 or csr == 0xC02 or csr == 0xB00 or csr == 0xB02:
                old = retired_lo
            elif csr == 0xC80 or csr == 0xC82 or csr == 0xB80 or csr == 0xB82:
                old = retired_hi
            elif csr == 0xC01:
                old = port.mtime_lo
            elif csr == 0xC81:
                old = port.mtime_hi
            else:
                known = 0                        # (an unknown CSR: an illegal instruction)
            # (allowed: a CSR of this mode or below (bits 8, 9), not read-only
            # (bits 10, 11 both 1) if written; a counter as mcounteren and
            # scounteren allow; satp in supervisor mode unless TVM)
            counter = csr >> 8 == 0xC
            counter_allowed = not counter or ((mode == 3 or (counters_on >> (csr & 31)) & 1)
                                              and (mode != 0 or (s_counters_on >> (csr & 31)) & 1))
            if not known or (csr >> 8) & 3 > mode or (writes and csr >> 10 == 3) \
                    or not counter_allowed or (csr == 0x180 and mode == 1 and (status_bits >> 20) & 1):
                trap = 1
                trap_cause = 2
                trap_with = instruction
            else:
                new = source
                if f3 == 2 or f3 == 6:
                    new = old | source
                elif f3 == 3 or f3 == 7:
                    new = old & (MASK ^ source)
                if writes and (csr == 0x300 or csr == 0x100):
                    # (sstatus: its bits of mstatus, the rest kept)
                    status = new if csr == 0x300 else (mstatus & (MASK ^ SSTATUS)) | (new & SSTATUS)
                    s_interrupts = (status >> 1) & 1
                    interrupts = (status >> 3) & 1
                    s_interrupts_before = (status >> 5) & 1
                    interrupts_before = (status >> 7) & 1
                    s_mode_before = (status >> 8) & 1
                    mode_before = 0 if (status >> 11) & 3 == 2 else (status >> 11) & 3
                    status_bits = status & 0x7E0000
                elif writes and csr == 0x304:
                    enabled = new & 0xAAA
                elif writes and csr == 0x104:
                    enabled = (enabled & (MASK ^ interrupts_delegated)) | (new & interrupts_delegated)
                elif writes and csr == 0x344:
                    pending = new & 0x222
                elif writes and csr == 0x144:            # (sip: SSIP alone, if delegated)
                    pending = (pending & 0x220) | (new & interrupts_delegated & 2)
                elif writes and csr == 0x305:
                    trap_vector = new & 0xFFFFFFFD
                elif writes and csr == 0x105:
                    s_trap_vector = new & 0xFFFFFFFD
                elif writes and csr == 0x340:
                    scratch = new
                elif writes and csr == 0x140:
                    s_scratch = new
                elif writes and csr == 0x341:
                    trap_pc = new & 0xFFFFFFFC
                elif writes and csr == 0x141:
                    s_trap_pc = new & 0xFFFFFFFC
                elif writes and csr == 0x342:
                    cause = new
                elif writes and csr == 0x142:
                    s_cause = new
                elif writes and csr == 0x343:
                    trap_value = new
                elif writes and csr == 0x143:
                    s_trap_value = new
                elif writes and csr == 0x302:            # (ecall from machine mode: never delegated)
                    exceptions_delegated = new & 0xB3FF
                elif writes and csr == 0x303:
                    interrupts_delegated = new & 0x222
                elif writes and csr == 0x306:
                    counters_on = new
                elif writes and csr == 0x106:
                    s_counters_on = new
                elif writes and csr == 0x180:            # (Sv32's mode and root; no ASIDs)
                    satp = new & 0x803FFFFF
                    itlb_valid = 0
                    dtlb_valid = 0
                result = old
                write = 1
        elif op != 0x0F:                         # (fence, fence.i: nothing to do; else illegal)
            trap = 1
            trap_cause = 2
            trap_with = instruction
        # ---- the data: a load's, a store's or an atomic's words (two, across
        # a word's end: a word and a half at byte 3), each through paging if
        # it's on (below machine mode, or with MPRV MPP's; satp's mode Sv32),
        # its page-table entries read first; then the word read (from the RAM,
        # the SDRAM through the cache, a device's register) and, a write, its
        # bytes written into it (the SDRAM's line read into the cache first if
        # it isn't there; the SDRAM written through). An atomic: the word
        # read, then its new value written ----
        if access:
            effective = mode_before if status_bits & MPRV and mode == 3 else mode
            paging = satp >> 31 and effective < 3
            # (stores and atomics but lr.w: the page's W, D; the rest its R,
            # or X with MXR)
            storing = access == STORE or (access == ATOMIC and kind != 2)
            spread = (b & (255 if size == 1 else (65535 if size == 2 else MASK))) << ((address & 3) * 8)
            spread_mask = ((1 if size == 1 else (3 if size == 2 else 15)) << (address & 3)) & 127
            two = (address & 3) + size > 4
            data = spread & MASK
            byte_mask = spread_mask & 15
            writing = access == STORE or (access == ATOMIC and kind == 3)   # (sc.w: a write alone)
            at = address                         # (the word now: the first, then the second)
            second = 0
            low = 0
            # (the TLB's entry for the word's page, read now, used after the
            # turn's first clock: walk 3; a hit that allows the access gives
            # the physical address, else the walk)
            data_entry = dtlb[(address >> 12) & (TLB_ENTRIES - 1)]
            walk = 3 if paging else 0
            table = (satp & 0x3FFFFF) << 12
            physical = at
            while True:
                # (a clock first: each turn starts from one state, however the
                # last ended; else py2ver makes the turn's start once for each)
                yield
                done = 0
                if walk == 3:
                    allowed = (data_entry >> 2) & 1 and (data_entry >> 7) & 1 if storing \
                        else (data_entry >> 1) & 1 or (status_bits >> 19 & 1 and (data_entry >> 3) & 1)
                    user_page = (data_entry >> 4) & 1
                    user_allowed = user_page if effective == 0 else (not user_page or status_bits >> 18 & 1)
                    if (dtlb_valid >> ((at >> 12) & (TLB_ENTRIES - 1))) & 1 and data_entry >> 28 == at >> 12 \
                            and allowed and user_allowed:
                        physical = (((data_entry >> 8) & 0xFFFFF) << 12) | (at & 0xFFF)
                        walk = 0
                    else:
                        walk = 2
                if walk == 2:
                    where = (table + ((at >> 22) << 2)) & MASK
                elif walk == 1:
                    where = (table + (((at >> 12) & 1023) << 2)) & MASK
                else:
                    where = physical
                # ---- the word read, whatever comes next: a page-table entry,
                # the data, or the word a write goes into (the SDRAM's line in
                # the cache then; a device's register only for a load) ----
                value = 0
                if where < 4 * RAM_WORDS:
                    word = ram[where >> 2]
                    yield
                    value = word
                elif where >> 25 == SDRAM >> 25:
                    while True:
                        cached = cache[(where >> 2) & (CACHE_WORDS - 1)]
                        tag = tags[(where >> 5) & (LINES - 1)]
                        yield
                        if tag == ((where >> CACHE_SHIFT) & 4095) | TAG_VALID:
                            break
                        yield from sdram_read_line(port, (where >> 2) & (SDRAM_WORDS - LINE_WORDS))
                        base = (where >> 2) & (CACHE_WORDS - LINE_WORDS)
                        k = 0
                        while k < LINE_WORDS:
                            yield from wait_until(port, "resp_valid")
                            cache[base + k] = port.resp_data
                            k = k + 1
                            yield
                        tags[(where >> 5) & (LINES - 1)] = ((where >> CACHE_SHIFT) & 4095) | TAG_VALID
                        yield
                    value = cached
                elif walk == 0 and not writing and (where == UART or where == UART_STATUS):
                    if not key_waiting:          # (a key from the PC, kept until read)
                        got, key = yield from take_key(port)
                        key_waiting = got
                        waiting_key = key
                    if where == UART:
                        value = waiting_key if key_waiting else 0
                        key_waiting = 0
                    else:
                        value = (0x60 | key_waiting) << 8 * (UART_STATUS & 3)
                elif walk == 0 and not writing and where == KEYS:
                    value = port.keys
                elif walk == 0 and not writing and where == MTIMECMP:
                    value = compare_lo
                elif walk == 0 and not writing and where == MTIMECMP + 4:
                    value = compare_hi
                elif walk == 0 and not writing and where == MTIME:
                    value = port.mtime_lo
                elif walk == 0 and not writing and where == MTIME + 4:
                    value = port.mtime_hi
                # (what's done with the word: in one state, after a clock,
                # however the read ended; else py2ver makes it once for each)
                yield
                if walk:
                    # (a page-table entry: V R W X U G A D in bits 0 to 7, the
                    # page's number above; a leaf (R or X) must allow the
                    # access: W and D storing, else R or X with MXR; U for user
                    # mode, for supervisor mode not U unless SUM; A; a
                    # megapage aligned)
                    leaf = (value >> 1) & 1 or (value >> 3) & 1
                    allowed = (value >> 2) & 1 and (value >> 7) & 1 if storing \
                        else (value >> 1) & 1 or (status_bits >> 19 & 1 and (value >> 3) & 1)
                    user_page = (value >> 4) & 1
                    user_allowed = user_page if effective == 0 else (not user_page or status_bits >> 18 & 1)
                    if not value & 1 or ((value >> 2) & 1 and not (value >> 1) & 1) \
                            or (not leaf and walk == 1) \
                            or (leaf and (not allowed or not user_allowed or not (value >> 6) & 1
                                          or (walk == 2 and (value >> 10) & 1023))):
                        trap = 1                 # (a load's page fault, or a store's or an atomic's)
                        trap_cause = 15 if storing else 13
                        trap_with = at
                        done = 1
                    elif leaf:                   # (the page: into the TLB)
                        if walk == 2:            # (a megapage: 4 MB)
                            physical = (((value >> 20) << 22) | (at & 0x3FFFFF)) & MASK
                        else:
                            physical = (((value >> 10) << 12) | (at & 0xFFF)) & MASK
                        walk = 0
                        dtlb[(at >> 12) & (TLB_ENTRIES - 1)] = ((at >> 12) << 28) | ((physical >> 12) << 8) \
                            | (value & 0xFF)
                        dtlb_valid = dtlb_valid | (1 << ((at >> 12) & (TLB_ENTRIES - 1)))
                    else:
                        table = ((value >> 10) << 12) & MASK
                        walk = 1
                elif writing:
                    # ---- a word written: data's bytes that byte_mask chooses,
                    # into the word read (the RAM's, or the cache's; the SDRAM
                    # written through), or a device's register ----
                    bits = ((byte_mask & 1) * 0xFF) | (((byte_mask >> 1) & 1) * 0xFF00) \
                        | (((byte_mask >> 2) & 1) * 0xFF0000) | ((byte_mask >> 3) * 0xFF000000)
                    merged = (value & (MASK ^ bits)) | data
                    if where < 4 * RAM_WORDS:
                        ram[where >> 2] = merged
                    elif where >> 25 == SDRAM >> 25:
                        cache[(where >> 2) & (CACHE_WORDS - 1)] = merged
                        yield from sdram_write(port, (where >> 2) & (SDRAM_WORDS - 1), data, byte_mask)
                    elif where == UART and divisor_latch == 0:
                        # (a copy to the PC over the JTAG UART: while the PC reads it,
                        # waiting for room there, up to pc_patience clocks; the room
                        # not coming, the PC isn't reading: copies are dropped, not
                        # waited for, until there is room again)
                        waited = 0
                        while port.jtag_tx_room == 0 and waited < patience:
                            yield
                            waited = waited + 1
                        if port.jtag_tx_room:
                            port.jtag_tx = b & 255
                            port.jtag_tx_send = 1
                            yield
                            port.jtag_tx_send = 0
                            patience = pc_patience
                        else:
                            patience = 0
                        yield from stream_send(port, b & 255, data="tx", valid="tx_valid", ready="tx_ready")
                    elif where == UART + 3:      # (the line control register)
                        divisor_latch = (b >> 7) & 1
                    elif where == LEDS:
                        port.leds = b & 255
                    elif where == MTIMECMP:
                        compare_lo = b
                    elif where == MTIMECMP + 4:
                        compare_hi = b
                    elif where == SYSCON and b == 0x5555:
                        while True:              # (powered off)
                            yield
                    elif where == SYSCON and b == 0x7777:
                        next_pc = 0              # (restart)
                    yield                        # (the same, after a write)
                    if access == ATOMIC:
                        if kind == 3:            # (sc.w: done)
                            result = 0
                            reserved = 0
                        done = 1
                    elif two and not second:     # (a store's first word: then the second)
                        second = 1
                        at = (((address >> 2) + 1) << 2) & MASK
                        data = spread >> 32
                        byte_mask = spread_mask >> 4
                        data_entry = dtlb[(at >> 12) & (TLB_ENTRIES - 1)]
                        walk = 3 if paging else 0
                        table = (satp & 0x3FFFFF) << 12
                        physical = at
                    else:
                        done = 1
                elif access == ATOMIC:
                    result = value
                    if kind == 2:                # (lr.w: reserved)
                        reserved = 1
                        reservation = address
                        done = 1
                    else:                        # (then the new value written)
                        sw = value - ((value >> 31) << 32)
                        data = b                 # (amoswap)
                        if kind == 0:
                            data = (value + b) & MASK
                        elif kind == 4:
                            data = value ^ b
                        elif kind == 12:
                            data = value & b
                        elif kind == 8:
                            data = value | b
                        elif kind == 16:
                            data = value if sw < sb else b
                        elif kind == 20:
                            data = value if sw > sb else b
                        elif kind == 24:
                            data = value if value < b else b
                        elif kind == 28:
                            data = value if value > b else b
                        writing = 1
                elif two and not second:         # (a load's first word: then the second)
                    low = value
                    second = 1
                    at = (((address >> 2) + 1) << 2) & MASK
                    data_entry = dtlb[(at >> 12) & (TLB_ENTRIES - 1)]
                    walk = 3 if paging else 0
                    table = (satp & 0x3FFFFF) << 12
                    physical = at
                else:
                    done = 1
                if done:
                    break
            if access == LOAD:                   # (the bytes asked for, at the bottom)
                if two:
                    part = (((value << 32) | low) >> ((address & 3) * 8)) & MASK
                else:
                    part = value >> ((address & 3) * 8)
                if f3 == 0:
                    result = ((part & 255) - (((part >> 7) & 1) << 8)) & MASK
                elif f3 == 1:
                    result = ((part & 65535) - (((part >> 15) & 1) << 16)) & MASK
                elif f3 == 4:
                    result = part & 255
                elif f3 == 5:
                    result = part & 65535
                else:
                    result = part & MASK
            write = access != STORE
        yield
        # ---- the instruction's end, a clock of its own (one state for every
        # instruction: the trap's logic and the register file's write made once,
        # from registers, not again for each way through the instruction).
        # An interrupt waiting and wanted, if the instruction didn't trap, is
        # taken now, before the next. Machine mode's: wanted below machine mode,
        # or in it with MIE; supervisor mode's (delegated): in user mode, or in
        # supervisor mode with SIE; the first of MEI, MSI, MTI, SEI, SSI, STI ----
        waiting = (pending | (128 if due else 0)) & enabled
        for_machine = waiting & (MASK ^ interrupts_delegated) if mode < 3 or interrupts else 0
        for_supervisor = waiting & interrupts_delegated if mode == 0 or (mode == 1 and s_interrupts) else 0
        interrupt = for_machine if for_machine else for_supervisor
        number = 11 if interrupt & 0x800 else (3 if interrupt & 8 else (7 if interrupt & 128 else (
            9 if interrupt & 0x200 else (1 if interrupt & 2 else 5))))
        if trap:                                 # (an exception: the instruction not done)
            write = 0
            trap_at = pc
            to_supervisor = mode < 3 and (exceptions_delegated >> trap_cause) & 1
            vectored = 0
        elif interrupt:
            trap = 1
            trap_cause = 0x80000000 | number
            trap_with = 0
            trap_at = next_pc
            to_supervisor = for_machine == 0
            vectored = 1
        if trap:                                 # a trap: to mtvec, or delegated, to stvec
            if to_supervisor:
                s_trap_pc = trap_at
                s_cause = trap_cause
                s_trap_value = trap_with
                s_interrupts_before = s_interrupts
                s_interrupts = 0
                s_mode_before = 1 if mode == 1 else 0
                mode = 1
                next_pc = ((s_trap_vector & 0xFFFFFFFC)
                           + (4 * number if vectored and s_trap_vector & 1 else 0)) & MASK
            else:
                trap_pc = trap_at
                cause = trap_cause
                trap_value = trap_with
                interrupts_before = interrupts
                interrupts = 0
                mode_before = mode
                mode = 3
                next_pc = ((trap_vector & 0xFFFFFFFC) + (4 * number if vectored and trap_vector & 1 else 0)) & MASK
        pc = next_pc
        if write and rd != 0:
            regs[rd] = result
        if retired_lo == MASK:
            retired_lo = 0
            retired_hi = 0 if retired_hi == MASK else retired_hi + 1
        else:
            retired_lo = retired_lo + 1


CPU_ATTR = {"tx_ready": 1, "jtag_tx_room": 1, "kb": 8, "kb_valid": 1, "keys": 2,
            "mtime_lo": 32, "mtime_hi": 32,
            # (a product of two 33-bit signed numbers: past the 64 bits followed by itself)
            "product": {"width": 66, "signed": 1},
            # (the SDRAM controller's answers)
            "req_ready": 1, "resp_valid": 1, "resp_data": 32,
            # (the clocks waited for room for the PC's copy: up to PC_PATIENCE, 2 s)
            "waited": 27, "patience": 27}


def machine_timer(port, tick=50):
    """mtime: microseconds since reset (tick: clocks a microsecond), its lower
    and upper 32 bits."""
    port.mtime_lo = 0
    port.mtime_hi = 0
    while True:
        yield tick - 1
        if port.mtime_lo == MASK:
            port.mtime_lo = 0
            port.mtime_hi = 0 if port.mtime_hi == MASK else port.mtime_hi + 1
        else:
            port.mtime_lo = port.mtime_lo + 1


# ----------------------------- the computer -----------------------------

# On the board: the display's pins, the JTAG UART for the keyboard and a copy of
# the console, the LEDs and keys, the SDRAM's (the display and the SDRAM both
# have a cs_n: in the computer, display_cs_n and sdram_cs_n)
PINS = dict(T.PINS, kb="JTAG_RX_DATA", kb_valid="JTAG_RX_VALID", kb_ready="JTAG_RX_READY",
            jtag_tx="JTAG_TX_DATA", jtag_tx_send="JTAG_TX_VALID", jtag_tx_room="JTAG_TX_READY",
            leds="LED", keys="KEY")
for _name in ("rx", "rx_valid", "rx_ready"):    # (the terminal's bytes come from the CPU)
    PINS.pop(_name)
PINS["display_cs_n"] = PINS.pop("cs_n")
PINS.update({("sdram_cs_n" if _name == "cs_n" else _name): _pin for _name, _pin in SDRAM_PINS.items()})


# The CPU's requests to the SDRAM controller, and its answers
SDRAM_WIRES = {**{f"sdram.{n}": f"cpu.{n}" for n in ("req", "req_write", "req_address", "req_data",
                                                     "req_mask", "req_line")},
               **{f"cpu.{n}": f"sdram.{n}" for n in ("req_ready", "resp_valid", "resp_data")}}

# The screen's picture, for screen="framebuffer": 240 x 320 pixels, RGB565 (red in
# the top 5 bits), row after row from the top left, two pixels a word (the
# left one in the low half): 153,600 bytes in the SDRAM, below the device tree
FRAMEBUFFER = 0x81FD0000
SCREEN_WIDTH = 240


def framebuffer_scan(port, base=(FRAMEBUFFER >> 2) & (SDRAM_WORDS - 1), width=SCREEN_WIDTH,
                     height=T.SCREEN_HEIGHT):
    """The screen's picture from the SDRAM, frame after frame, for the display:
    a line of LINE_WORDS words (2 x LINE_WORDS pixels) at a time from the
    controller's second client (req2, req2_address; req2_ready, resp2_valid,
    resp_data), each pixel sent on (color, valid; ready). base: the picture's
    word address in the SDRAM."""
    port.req2 = 0
    port.req2_address = 0
    port.valid = 0
    port.color = 0
    words = [0] * LINE_WORDS                     # a line's words, as they came
    while True:
        address = base
        k = 0
        while k < width * height // (2 * LINE_WORDS):
            port.req2_address = address
            port.req2 = 1
            yield
            yield from wait_until(port, "req2_ready")
            port.req2 = 0
            j = 0
            while j < LINE_WORDS:
                yield from wait_until(port, "resp2_valid")
                words[j] = port.resp_data
                j = j + 1
                yield
            j = 0
            while j < LINE_WORDS:
                yield from stream_send(port, words[j] & 0xFFFF, data="color")
                yield from stream_send(port, words[j] >> 16, data="color")
                j = j + 1
            address = (address + LINE_WORDS) & (SDRAM_WORDS - 1)
            k = k + 1
        yield                                    # (a clock between frames)


def always_ready(port):
    """ready always 1: for screen="framebuffer", the console's characters
    taken at once (the PC has its copy)."""
    port.ready = 1
    while True:
        yield


def computer(image, lcd=T.LCD, columns=T.COLUMNS, rows=T.ROWS, height=T.SCREEN_HEIGHT,
             timing=TIMING, screen="terminal", width=SCREEN_WIDTH, framebuffer=FRAMEBUFFER):
    """The CPU running the memory image `image`, the SDRAM at 0x8000_0000.
    The display shows, screen="terminal": the console, as text; screen=
    "framebuffer": the picture at `framebuffer` in the SDRAM (width x height)."""
    cpu = functools.partial(rv32ima, image=image)
    controller = functools.partial(sdram_controller, timing=timing)
    wires = {"cpu.mtime_lo": "timer.mtime_lo", "cpu.mtime_hi": "timer.mtime_hi", **SDRAM_WIRES}
    if screen == "terminal":
        picture = functools.partial(T.terminal, columns=columns, rows=rows, height=height)
        width = columns * 8
        shows = dict(terminal=(picture, {"ready": 1}), name="riscv_computer")
        wires.update({"terminal.rx": "cpu.tx", "terminal.rx_valid": "cpu.tx_valid",
                      "cpu.tx_ready": "terminal.rx_ready", "display.color": "terminal.color",
                      "display.valid": "terminal.valid", "terminal.ready": "display.ready"})
    elif screen == "framebuffer":
        picture = functools.partial(framebuffer_scan, base=(framebuffer >> 2) & (SDRAM_WORDS - 1),
                                    width=width, height=height)
        shows = dict(picture=(picture, {"req2_ready": 1, "resp2_valid": 1, "resp_data": 32,
                                        "ready": 1, "k": 16}),       # (k: a frame's lines)
                     console=(always_ready, {}), name="riscv_framebuffer_computer")
        wires.update({"cpu.tx_ready": "console.ready", "display.color": "picture.color",
                      "display.valid": "picture.valid", "picture.ready": "display.ready",
                      "sdram.req2": "picture.req2", "sdram.req2_address": "picture.req2_address",
                      "picture.req2_ready": "sdram.req2_ready",
                      "picture.resp2_valid": "sdram.resp2_valid", "picture.resp_data": "sdram.resp_data"})
    else:
        raise ValueError(f"screen: 'terminal' or 'framebuffer', not {screen!r}")
    display = functools.partial(T.ili9341_display, lcd=lcd, width=width, height=height)
    return connect(timer=(machine_timer, {}), cpu=(cpu, CPU_ATTR),
                   display=(display, {"color": 16, "valid": 1}), sdram=(controller, SDRAM_ATTR),
                   **shows, wires=wires)


def cpu_with_sdram(image, timing=TIMING):
    """The CPU and the SDRAM controller alone (no timer, no display): for a
    testbench, with console(chip=) playing the SDRAM."""
    cpu = functools.partial(rv32ima, image=image)
    controller = functools.partial(sdram_controller, timing=timing)
    return connect(cpu=(cpu, CPU_ATTR), sdram=(controller, SDRAM_ATTR), name="cpu_with_sdram",
                   wires=SDRAM_WIRES)


def console(until, typed=b"", limit=2_000_000, chip=None):
    """A testbench for the CPU alone: it takes the console's characters (a
    clock after each, as the terminal would), types `typed` on the keyboard,
    and stops once the console has printed `until` (or after limit clocks).
    mtime stays 0. chip: an SdramModel, for cpu_with_sdram(). Returns what was
    printed."""
    def printing(dut):
        printed, k = bytearray(), 0
        dut.keys = 3
        for _ in range(limit):
            dut.tx_ready = 1 if dut.tx_valid else 0
            if dut.tx_valid:
                printed.append(dut.tx)
            dut.kb_valid = 0
            if k < len(typed) and dut.kb_ready:
                dut.kb, dut.kb_valid = typed[k], 1
                k += 1
            if printed.endswith(until):
                break
            yield
        return bytes(printed)

    def tb(dut):
        if chip is None:
            printed = yield from printing(dut)
        else:
            printed = yield from alongside(printing(dut), chip.run(dut))
        return printed
    return tb


if __name__ == "__main__":
    from py2ver import Py2ver

    args = sys.argv[1:]
    for sources, until, typed in [(("selftest.S",), b"\n", b""), (("traps.S",), b"\n", b""),
                                  (("hello.c",), b"I echo:\nhi", b"hi")]:
        p = Py2ver(functools.partial(rv32ima, image=build(*sources)), CPU_ATTR)
        run = p.check(console(until, typed), verilog="--sim" in args)
        print(f"{sources[0]} printed {run.value.decode()!r} in {len(run.inputs):,} clock cycles")
    if "--pins" in args:
        program = next((a for a in args if a.endswith((".c", ".S"))), "hello.c")
        screen = "framebuffer" if "--framebuffer" in args else "terminal"
        board = Py2ver(computer(build(program), screen=screen), pins=PINS)
        print(board.PINS({"QUARTUS_DIR": os.environ.get(
            "QUARTUS_DIR", "/opt/intelFPGA/22.1std/quartus/bin")}))
