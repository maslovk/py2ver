# A RISC-V computer on the FPGA: a CPU written in Python, running C programs,
# its console on the ILI9341 display and its keyboard the PC's.
#
# The CPU runs RV32I, RISC-V's base integer instructions, a clock to fetch an
# instruction, a clock to decode it and read its registers, a clock (or a few,
# for memory) to carry it out, a clock to write its result: about 12 million
# instructions a second. Its memory:
#
#   0x0000_0000  ROM, 16 KB: the program and its constants (built into the design)
#   0x0001_0000  RAM, 4 KB: its variables and stack (a list in a memory block)
#   0x1000_0000  write: a character on the screen (sent to terminal.py's terminal)
#   0x1000_0004  read: a key typed on the PC (the JTAG UART), or -1 if none
#   0x1000_0008  write: the 8 LEDs
#   0x1000_000C  read: the keys KEY0, KEY1 (bits 0, 1; 0 while pressed)
#
#   cpu --characters--> terminal --color/valid/ready--> ili9341_display
#
# Programs are written in C or assembly (examples/riscv/) and built with the
# GNU RISC-V toolchain: riscv-none-elf-gcc (xPack's, unpacked in ~/opt/riscv,
# or on PATH, or named by RISCV_GCC).
#
#   python examples/riscv_computer.py          # the CPU checks itself, then hello.c: test, print what they print
#   python examples/riscv_computer.py --sim    # also simulate the generated Verilog
#   python examples/riscv_computer.py --pins   # build hello.c's computer for the board
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
from py2ver.protocol import stream_send  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PROGRAMS = os.path.join(HERE, "riscv")

ROM_WORDS = 4096                         # 16 KB
RAM_BASE = 0x00010000
RAM_WORDS = 1024                         # 4 KB
IO_BASE = 0x10000000
CONSOLE, KEYBOARD, LEDS, KEYS = IO_BASE, IO_BASE + 4, IO_BASE + 8, IO_BASE + 12
MASK = 0xFFFFFFFF                        # (the registers are 32 bits)

# ----------------------------- building programs -----------------------------


def toolchain():
    """The RISC-V gcc: RISCV_GCC, or xPack's in ~/opt/riscv, or one on PATH; None if none."""
    found = [os.environ.get("RISCV_GCC")]
    found += sorted(glob.glob(os.path.expanduser("~/opt/riscv/*/bin/riscv-none-elf-gcc")))
    found += [shutil.which("riscv-none-elf-gcc"), shutil.which("riscv64-unknown-elf-gcc")]
    return next((g for g in found if g and os.path.exists(g)), None)


def build(*sources):
    """A program's words, from 0: its sources (in examples/riscv, C or
    assembly; a C program gets start.S, which calls its main) compiled for
    RV32I and linked to this computer's memory."""
    gcc = toolchain()
    if gcc is None:
        raise RuntimeError("No RISC-V toolchain: unpack xPack's riscv-none-elf-gcc in ~/opt/riscv, "
                           "or set RISCV_GCC")
    paths = [os.path.join(PROGRAMS, s) for s in sources]
    if any(s.endswith(".c") for s in sources):
        paths.insert(0, os.path.join(PROGRAMS, "start.S"))
    with tempfile.TemporaryDirectory() as work:
        elf, binary = os.path.join(work, "program.elf"), os.path.join(work, "program.bin")
        subprocess.run([gcc, "-march=rv32i", "-mabi=ilp32", "-Os", "-nostdlib", "-ffreestanding",
                        "-T", os.path.join(PROGRAMS, "link.ld"), "-o", elf, *paths, "-lgcc"],
                       check=True)
        objcopy = gcc[:-len("gcc")] + "objcopy"       # (riscv-none-elf-gcc: riscv-none-elf-objcopy)
        subprocess.run([objcopy, "-O", "binary", "-j", ".text", elf, binary], check=True)
        data = open(binary, "rb").read()
    data += bytes(-len(data) % 4)
    if len(data) > 4 * ROM_WORDS:
        raise RuntimeError(f"The program is {len(data)} bytes: more than the ROM's {4 * ROM_WORDS}")
    return tuple(int.from_bytes(data[k:k + 4], "little") for k in range(0, len(data), 4))


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


def rv32i(port, program=(0x0000006F,)):
    """The CPU, running `program` (its words, from address 0) from reset.

    Outputs: the console's characters (tx, tx_valid; tx_ready: taken), the
    PC keyboard's kb_ready, leds. Inputs: tx_ready; the keyboard's kb,
    kb_valid; keys.
    """
    port.tx_valid = 0
    port.kb_ready = 0
    port.leds = 0
    regs = [0] * 32                              # x0 to x31 (x0 never written: always 0)
    ram = [0] * RAM_WORDS
    pc = 0
    while True:
        # ---- fetch ----
        instruction = program[min(pc >> 2, len(program) - 1)]
        yield
        # ---- decode: the instruction's fields, the registers it reads ----
        op = instruction & 0x7F
        rd = (instruction >> 7) & 31
        f3 = (instruction >> 12) & 7
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
        next_pc = (pc + 4) & MASK
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
            sa = a - ((a >> 31) << 32)           # (as signed numbers)
            sb = b - ((b >> 31) << 32)
            taken = (f3 == 0 and a == b) or (f3 == 1 and a != b) or (f3 == 4 and sa < sb) \
                or (f3 == 5 and sa >= sb) or (f3 == 6 and a < b) or (f3 == 7 and a >= b)
            if taken:
                next_pc = (pc + imm_b) & MASK
        elif op == 0x13 or op == 0x33:           # arithmetic: with an immediate, or two registers
            y = b if op == 0x33 else imm_i & MASK
            shift = y & 31
            sa = a - ((a >> 31) << 32)
            sy = y - ((y >> 31) << 32)
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
            if address >= IO_BASE:
                if address == KEYBOARD:
                    got, key = yield from take_key(port)
                    value = key if got else MASK
                elif address == KEYS:
                    value = port.keys
            elif address >= RAM_BASE:
                ram_word = ram[((address - RAM_BASE) >> 2) & (RAM_WORDS - 1)]
                yield
                value = ram_word
            else:
                rom_word = program[min(address >> 2, len(program) - 1)]
                yield
                value = rom_word
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
            write = 1
        elif op == 0x23:                         # sb sh sw
            address = (a + imm_s) & MASK
            if address == CONSOLE:
                yield from stream_send(port, b & 255, data="tx", valid="tx_valid", ready="tx_ready")
            elif address == LEDS:
                port.leds = b & 255
            elif address >= RAM_BASE and address < IO_BASE:
                k = ((address - RAM_BASE) >> 2) & (RAM_WORDS - 1)
                if f3 == 2:
                    ram[k] = b
                else:                            # (a byte or a half: into the word around it)
                    ram_word = ram[k]
                    yield
                    at = (address & 3) * 8
                    keep = 255 if f3 == 0 else 65535
                    ram[k] = (ram_word & (MASK ^ (keep << at))) | ((b & keep) << at)
        # (fence, ecall, ebreak: nothing to do)
        pc = next_pc
        yield
        # ---- write back (a clock of its own: the register file takes the
        # result from a register, not from the whole of the clock's logic) ----
        if write and rd != 0:
            regs[rd] = result


CPU_ATTR = {"tx_ready": 1, "kb": 8, "kb_valid": 1, "keys": 2}

# ----------------------------- the computer -----------------------------

# On the board: the display's pins, the JTAG UART for the keyboard, the LEDs and keys
PINS = dict(T.PINS, kb="JTAG_RX_DATA", kb_valid="JTAG_RX_VALID", kb_ready="JTAG_RX_READY",
            leds="LED", keys="KEY")
for _name in ("rx", "rx_valid", "rx_ready"):    # (the terminal's bytes come from the CPU)
    PINS.pop(_name)


def computer(program, lcd=T.LCD, columns=T.COLUMNS, rows=T.ROWS, height=T.SCREEN_HEIGHT):
    """The CPU running `program`, its console the terminal on the display."""
    cpu = functools.partial(rv32i, program=program)
    screen = functools.partial(T.terminal, columns=columns, rows=rows, height=height)
    width = columns * 8
    display = functools.partial(T.ili9341_display, lcd=lcd, width=width, height=height)
    return connect(cpu=(cpu, CPU_ATTR), terminal=(screen, {"ready": 1}),
                   display=(display, {"color": 16, "valid": 1}), name="riscv_computer",
                   wires={"terminal.rx": "cpu.tx", "terminal.rx_valid": "cpu.tx_valid",
                          "cpu.tx_ready": "terminal.rx_ready",
                          "display.color": "terminal.color", "display.valid": "terminal.valid",
                          "terminal.ready": "display.ready"})


def console(until, typed=b"", limit=2_000_000):
    """A testbench for the CPU alone: it takes the console's characters (a
    clock after each, as the terminal would), types `typed` on the keyboard,
    and stops once the console has printed `until` (or after limit clocks).
    Returns what was printed."""
    def tb(dut):
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
    return tb


if __name__ == "__main__":
    from py2ver import Py2ver

    args = sys.argv[1:]
    for sources, until, typed in [(("selftest.S",), b"\n", b""),
                                  (("hello.c",), b"I echo:\nhi", b"hi")]:
        p = Py2ver(functools.partial(rv32i, program=build(*sources)), CPU_ATTR)
        run = p.check(console(until, typed), verilog="--sim" in args)
        print(f"{sources[0]} printed {run.value.decode()!r} in {len(run.inputs):,} clock cycles")
    if "--pins" in args:
        board = Py2ver(computer(build("hello.c")), pins=PINS)
        print(board.PINS({"QUARTUS_DIR": os.environ.get(
            "QUARTUS_DIR", "/opt/intelFPGA/22.1std/quartus/bin")}))
