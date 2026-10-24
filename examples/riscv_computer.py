# A RISC-V computer on the FPGA: a CPU written in Python, running C programs,
# its console on the ILI9341 display and its keyboard the PC's.
#
# The CPU runs RV32IMA with machine and user modes: the base integer
# instructions, multiply and divide (M), atomics (A), the CSRs (Zicsr), traps
# and the timer's interrupt (user mode: for Linux's programs, their system calls
# an ecall from user mode; no memory protection). A clock to fetch an instruction, a clock to decode it and read its
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
                                  WORDS as SDRAM_WORDS, sdram_controller, sdram_read,
                                  sdram_read_line, sdram_write)
from py2ver.protocol import alongside, stream_send, wait_until  # noqa: E402

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
MISA = (1 << 30) | (1 << 12) | (1 << 8) | (1 << 0)    # 32 bits; M, I, A

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


def riscv_test(name):
    """One of riscv-tests' programs (rv32ui/add, rv32um/div, rv32ua/amoadd_w...),
    built with this computer's environment (examples/riscv/env)."""
    return build(os.path.join(RISCV_TESTS, "isa", name + ".S"),
                 includes=(os.path.join(PROGRAMS, "env"),
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
    # Machine mode's state (the CSRs)
    interrupts = 0                               # mstatus.MIE: interrupts on
    interrupts_before = 0                        # mstatus.MPIE: MIE before the trap
    enabled = 0                                  # mie: the interrupts wanted (bit 7: the timer's)
    trap_vector = 0                              # mtvec
    scratch = 0                                  # mscratch
    trap_pc = 0                                  # mepc
    cause = 0                                    # mcause
    trap_value = 0                               # mtval
    retired_lo = 0                               # instructions carried out (cycle, instret)
    retired_hi = 0
    compare_lo = MASK                            # mtimecmp: the timer's interrupt from then
    compare_hi = MASK
    reserved = 0                                 # lr.w's reservation: on, and where
    reservation = 0
    key_waiting = 0                              # a key taken from the PC, not yet read
    waiting_key = 0
    user = 0                                     # 1: in user mode (Linux's programs), else machine mode
    user_before = 0                              # mstatus.MPP: the mode before the trap (1: user)
    divisor_latch = 0                            # the UART's LCR bit 7: writes to its data set the baud rate
    patience = 0                                 # clocks to wait for room for the PC's copy (0: none listens)
    while True:
        # ---- the timer's interrupt, if it is due and wanted: a trap now (in
        # user mode, whatever mstatus.MIE says) ----
        due = port.mtime_hi > compare_hi or (port.mtime_hi == compare_hi and port.mtime_lo >= compare_lo)
        if (interrupts or user) and (enabled >> 7) & 1 and due:
            trap_pc = pc
            cause = 0x80000007
            trap_value = 0
            interrupts_before = interrupts
            interrupts = 0
            user_before = user
            user = 0
            pc = ((trap_vector & 0xFFFFFFFC) + (28 if trap_vector & 1 else 0)) & MASK   # (vectored: + 4 x 7)
        # ---- fetch: from the RAM, or from the SDRAM through the cache (on a
        # miss, the line read into it, and the cache looked at again) ----
        while True:
            from_ram = ram[(pc >> 2) & (RAM_WORDS - 1)]
            from_cache = cache[(pc >> 2) & (CACHE_WORDS - 1)]
            fetch_tag = tags[(pc >> 5) & (LINES - 1)]
            yield
            if pc >> 25 != SDRAM >> 25 or fetch_tag == ((pc >> CACHE_SHIFT) & 4095) | TAG_VALID:
                break
            yield from sdram_read_line(port, (pc >> 2) & (SDRAM_WORDS - LINE_WORDS))
            base = (pc >> 2) & (CACHE_WORDS - LINE_WORDS)
            k = 0
            while k < LINE_WORDS:
                yield from wait_until(port, "resp_valid")
                cache[base + k] = port.resp_data
                k = k + 1
                yield
            tags[(pc >> 5) & (LINES - 1)] = ((pc >> CACHE_SHIFT) & 4095) | TAG_VALID
            yield
        instruction = from_cache if pc >> 25 == SDRAM >> 25 else from_ram
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
        trap = 0                                 # (1: a trap instead: its cause, its value)
        trap_cause = 0
        trap_with = 0
        next_pc = (pc + 4) & MASK
        sa = a - ((a >> 31) << 32)               # (a and b as signed numbers)
        sb = b - ((b >> 31) << 32)
        if op == 0x37:                           # lui
            result = imm_u
            write = 1
        elif op == 0x17:                         # auipc
            result = (pc + imm_u) & MASK
            write = 1
        elif op == 0x6F:                         # jal
            result = next_pc
            write = 1
            next_pc = (pc + imm_j) & MASK
        elif op == 0x67:                         # jalr
            result = next_pc
            write = 1
            next_pc = (a + imm_i) & 0xFFFFFFFE
        elif op == 0x63:                         # beq bne blt bge bltu bgeu
            taken = (f3 == 0 and a == b) or (f3 == 1 and a != b) or (f3 == 4 and sa < sb) \
                or (f3 == 5 and sa >= sb) or (f3 == 6 and a < b) or (f3 == 7 and a >= b)
            if taken:
                next_pc = (pc + imm_b) & MASK
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
        elif op == 0x03:                         # lb lh lw lbu lhu
            address = (a + imm_i) & MASK
            value = 0
            if (f3 == 2 and address & 3) or ((f3 == 1 or f3 == 5) and address & 1):
                trap = 1                         # (misaligned)
                trap_cause = 4
                trap_with = address
            elif address < 4 * RAM_WORDS:
                word = ram[address >> 2]
                yield
                value = word
            elif address >> 25 == SDRAM >> 25:      # (through the cache, as a fetch)
                while True:
                    cached = cache[(address >> 2) & (CACHE_WORDS - 1)]
                    tag = tags[(address >> 5) & (LINES - 1)]
                    yield
                    if tag == ((address >> CACHE_SHIFT) & 4095) | TAG_VALID:
                        break
                    yield from sdram_read_line(port, (address >> 2) & (SDRAM_WORDS - LINE_WORDS))
                    base = (address >> 2) & (CACHE_WORDS - LINE_WORDS)
                    k = 0
                    while k < LINE_WORDS:
                        yield from wait_until(port, "resp_valid")
                        cache[base + k] = port.resp_data
                        k = k + 1
                        yield
                    tags[(address >> 5) & (LINES - 1)] = ((address >> CACHE_SHIFT) & 4095) | TAG_VALID
                    yield
                value = cached
            elif address == UART or address == UART_STATUS:
                if not key_waiting:              # (a key from the PC, kept until read)
                    got, key = yield from take_key(port)
                    key_waiting = got
                    waiting_key = key
                if address == UART:
                    value = waiting_key if key_waiting else 0
                    key_waiting = 0
                else:
                    value = (0x60 | key_waiting) << 8 * (UART_STATUS & 3)
            elif address == KEYS:
                value = port.keys
            elif address == MTIMECMP:
                value = compare_lo
            elif address == MTIMECMP + 4:
                value = compare_hi
            elif address == MTIME:
                value = port.mtime_lo
            elif address == MTIME + 4:
                value = port.mtime_hi
            part = value >> ((address & 3) * 8)  # (the byte or half asked for, at the bottom)
            if f3 == 0:
                result = ((part & 255) - (((part >> 7) & 1) << 8)) & MASK
            elif f3 == 1:
                result = ((part & 65535) - (((part >> 15) & 1) << 16)) & MASK
            elif f3 == 4:
                result = part & 255
            elif f3 == 5:
                result = part & 65535
            else:
                result = value
            write = 1 - trap
        elif op == 0x23:                         # sb sh sw
            address = (a + imm_s) & MASK
            if (f3 == 2 and address & 3) or (f3 == 1 and address & 1):
                trap = 1                         # (misaligned)
                trap_cause = 6
                trap_with = address
            elif address < 4 * RAM_WORDS:
                k = address >> 2
                if f3 == 2:
                    ram[k] = b
                else:                            # (a byte or a half: into the word around it)
                    word = ram[k]
                    yield
                    at = (address & 3) * 8
                    keep = 255 if f3 == 0 else 65535
                    # (each & MASK: 32 bits, as the RAM's words; a half at byte 3 traps first)
                    ram[k] = (word & (MASK ^ ((keep << at) & MASK))) | (((b & keep) << at) & MASK)
            elif address >> 25 == SDRAM >> 25:
                # (written through: to the SDRAM, a byte or a half just those,
                # and into the cache if the line is there)
                at = (address & 3) * 8
                data = b
                byte_mask = 15
                bits = MASK
                if f3 == 0:
                    data = ((b & 255) << at) & MASK
                    byte_mask = 1 << (address & 3)
                    bits = (255 << at) & MASK
                elif f3 == 1:
                    data = ((b & 65535) << at) & MASK
                    byte_mask = (3 << (address & 3)) & 15
                    bits = (65535 << at) & MASK
                cached = cache[(address >> 2) & (CACHE_WORDS - 1)]
                tag = tags[(address >> 5) & (LINES - 1)]
                yield
                if tag == ((address >> CACHE_SHIFT) & 4095) | TAG_VALID:
                    cache[(address >> 2) & (CACHE_WORDS - 1)] = (cached & (MASK ^ bits)) | data
                yield from sdram_write(port, (address >> 2) & (SDRAM_WORDS - 1), data, byte_mask)
            elif address == UART and divisor_latch == 0:
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
            elif address == UART + 3:                # (the line control register)
                divisor_latch = (b >> 7) & 1
            elif address == LEDS:
                port.leds = b & 255
            elif address == MTIMECMP:
                compare_lo = b
            elif address == MTIMECMP + 4:
                compare_hi = b
            elif address == SYSCON and b == 0x5555:
                while True:                      # (powered off)
                    yield
            elif address == SYSCON and b == 0x7777:
                next_pc = 0                      # (restart)
        elif op == 0x2F:                         # lr.w sc.w amo*.w
            address = a
            k = (address >> 2) & (RAM_WORDS - 1)
            in_sdram = address >> 25 == SDRAM >> 25
            sdram_word = (address >> 2) & (SDRAM_WORDS - 1)
            kind = instruction >> 27
            if address & 3:
                trap = 1
                trap_cause = 6
                trap_with = address
            elif kind == 3:                      # sc.w: done if lr.w's reservation is still here
                if reserved and reservation == address:
                    if in_sdram:                 # (written through, as a store)
                        tag = tags[(address >> 5) & (LINES - 1)]
                        yield
                        if tag == ((address >> CACHE_SHIFT) & 4095) | TAG_VALID:
                            cache[(address >> 2) & (CACHE_WORDS - 1)] = b
                        yield from sdram_write(port, sdram_word, b)
                    else:
                        ram[k] = b
                    result = 0
                else:
                    result = 1
                reserved = 0
            else:
                if in_sdram:
                    # (from the SDRAM itself: every store goes there, so it
                    # holds what the cache does)
                    word = yield from sdram_read(port, sdram_word)
                else:
                    word = ram[k]
                    yield
                result = word
                if kind == 2:                    # lr.w
                    reserved = 1
                    reservation = address
                else:
                    sw = word - ((word >> 31) << 32)
                    new = b                      # (amoswap)
                    if kind == 0:
                        new = (word + b) & MASK
                    elif kind == 4:
                        new = word ^ b
                    elif kind == 12:
                        new = word & b
                    elif kind == 8:
                        new = word | b
                    elif kind == 16:
                        new = word if sw < sb else b
                    elif kind == 20:
                        new = word if sw > sb else b
                    elif kind == 24:
                        new = word if word < b else b
                    elif kind == 28:
                        new = word if word > b else b
                    if in_sdram:                 # (written through, as a store)
                        tag = tags[(address >> 5) & (LINES - 1)]
                        yield
                        if tag == ((address >> CACHE_SHIFT) & 4095) | TAG_VALID:
                            cache[(address >> 2) & (CACHE_WORDS - 1)] = new
                        yield from sdram_write(port, sdram_word, new)
                    else:
                        ram[k] = new
            write = 1 - trap
        elif op == 0x73 and f3 == 0:             # ecall ebreak mret wfi
            if (instruction >> 20) == 0x302:     # mret: back from the trap
                next_pc = trap_pc
                interrupts = interrupts_before
                interrupts_before = 1
                user = user_before               # (to the mode mstatus.MPP names, which is then user's)
                user_before = 1
            elif (instruction >> 20) == 0:
                trap = 1
                trap_cause = 8 if user else 11   # (ecall from user mode: a system call; from machine mode)
            elif (instruction >> 20) == 1:
                trap = 1
                trap_cause = 3                   # (ebreak)
        elif op == 0x73:                         # csrrw csrrs csrrc, and with an immediate
            csr = instruction >> 20
            source = (instruction >> 15) & 31 if f3 >= 5 else a
            mstatus = (interrupts << 3) | (interrupts_before << 7) | (0 if user_before else 3 << 11)
            old = 0                              # (unknown CSRs: 0, written to no effect)
            if csr == 0x300:
                old = mstatus
            elif csr == 0x301:
                old = MISA
            elif csr == 0x304:
                old = enabled
            elif csr == 0x305:
                old = trap_vector
            elif csr == 0x340:
                old = scratch
            elif csr == 0x341:
                old = trap_pc
            elif csr == 0x342:
                old = cause
            elif csr == 0x343:
                old = trap_value
            elif csr == 0x344:
                old = 128 if due else 0          # (mip: the timer's interrupt due)
            elif csr == 0xC00 or csr == 0xC02 or csr == 0xB00 or csr == 0xB02:
                old = retired_lo
            elif csr == 0xC80 or csr == 0xC82 or csr == 0xB80 or csr == 0xB82:
                old = retired_hi
            elif csr == 0xC01:
                old = port.mtime_lo
            elif csr == 0xC81:
                old = port.mtime_hi
            new = source
            if f3 == 2 or f3 == 6:
                new = old | source
            elif f3 == 3 or f3 == 7:
                new = old & (MASK ^ source)
            if csr == 0x300:
                interrupts = (new >> 3) & 1
                interrupts_before = (new >> 7) & 1
                user_before = 1 if (new >> 11) & 3 == 0 else 0  # (MPP: user, or else machine)
            elif csr == 0x304:
                enabled = new & 0x888
            elif csr == 0x305:
                trap_vector = new
            elif csr == 0x340:
                scratch = new
            elif csr == 0x341:
                trap_pc = new & 0xFFFFFFFC
            elif csr == 0x342:
                cause = new
            elif csr == 0x343:
                trap_value = new
            result = old
            write = 1
        elif op != 0x0F:                         # (fence, fence.i: nothing to do; else illegal)
            trap = 1
            trap_cause = 2
            trap_with = instruction
        if trap:                                 # a trap: to mtvec, as an interrupt goes
            trap_pc = pc
            cause = trap_cause
            trap_value = trap_with
            interrupts_before = interrupts
            interrupts = 0
            user_before = user
            user = 0
            next_pc = trap_vector & 0xFFFFFFFC
        pc = next_pc
        yield
        # ---- write back (a clock of its own: the register file takes the
        # result from a register, not from the whole of the clock's logic) ----
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
# the console, the LEDs and
# keys, the SDRAM's (the display and the SDRAM both have a cs_n: in the
# computer, display_cs_n and sdram_cs_n)
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


def computer(image, lcd=T.LCD, columns=T.COLUMNS, rows=T.ROWS, height=T.SCREEN_HEIGHT,
             timing=TIMING):
    """The CPU running the memory image `image`, its console the terminal on
    the display, the SDRAM at 0x8000_0000."""
    cpu = functools.partial(rv32ima, image=image)
    controller = functools.partial(sdram_controller, timing=timing)
    screen = functools.partial(T.terminal, columns=columns, rows=rows, height=height)
    width = columns * 8
    display = functools.partial(T.ili9341_display, lcd=lcd, width=width, height=height)
    return connect(timer=(machine_timer, {}), cpu=(cpu, CPU_ATTR), terminal=(screen, {"ready": 1}),
                   display=(display, {"color": 16, "valid": 1}), sdram=(controller, SDRAM_ATTR),
                   name="riscv_computer",
                   wires={"cpu.mtime_lo": "timer.mtime_lo", "cpu.mtime_hi": "timer.mtime_hi",
                          "terminal.rx": "cpu.tx", "terminal.rx_valid": "cpu.tx_valid",
                          "cpu.tx_ready": "terminal.rx_ready",
                          "display.color": "terminal.color", "display.valid": "terminal.valid",
                          "terminal.ready": "display.ready", **SDRAM_WIRES})


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
        board = Py2ver(computer(build(program)), pins=PINS)
        print(board.PINS({"QUARTUS_DIR": os.environ.get(
            "QUARTUS_DIR", "/opt/intelFPGA/22.1std/quartus/bin")}))
