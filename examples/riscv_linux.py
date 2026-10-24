# Linux on riscv_computer.py's RISC-V computer: mini-rv32ima's kernel (NOMMU,
# RV32IMA, machine mode; its programs BusyBox's, in an initramfs inside it)
# loaded into the SDRAM from the PC, and started.
#
# The board runs examples/riscv/boot.c from its RAM. Over the JTAG UART the PC
# sends it the kernel (to 0x8000_0000) and the device tree (examples/riscv/
# linux.dts, to the SDRAM's last 16 KB), checks each one's sum, and starts the
# kernel; then it is the console: what Linux prints comes here (and to the
# display), what is typed goes to it.
#
#   python examples/riscv_linux.py --pins      # build the computer with the boot loader for the board
#   python examples/riscv_linux.py --boot      # (the board programmed) load Linux, start it, its console
#   python examples/riscv_linux.py --boot --watch 60    # the console for 60 s, nothing typed
#   python examples/riscv_linux.py --console   # the console of a board already running Linux
#
# (Log in as root, no password. juart-terminal, in Quartus's bin directory, is a
# console too.)
#
# The kernel: linux-6.1.14-rv32nommu-cnl-1.zip of github.com/cnlohr/mini-rv32ima-images,
# unpacked to ~/opt/riscv/linux/image/Image (or LINUX_IMAGE names it).
import os
import select
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import riscv_computer as R  # noqa: E402

KERNEL = os.environ.get("LINUX_IMAGE", os.path.expanduser("~/opt/riscv/linux/image/Image"))
KERNEL_AT = 0x80000000
TREE_AT = 0x81FFC000                     # (after the memory linux.dts gives Linux)


def device_tree():
    """linux.dts compiled (dtc), its length made a multiple of 4."""
    with tempfile.TemporaryDirectory() as work:
        out = os.path.join(work, "linux.dtb")
        subprocess.run(["dtc", "-q", "-I", "dts", "-O", "dtb", "-o", out,
                        os.path.join(R.PROGRAMS, "linux.dts")], check=True)
        tree = open(out, "rb").read()
    return tree + bytes(-len(tree) % 4)


def word(n):
    return n.to_bytes(4, "little")


def load_record(address, data):
    """The boot loader's record storing data at address."""
    return b"L" + word(address) + word(len(data)) + data


def go_record(address, a1):
    """The boot loader's record starting the program at address (a0 0, a1)."""
    return b"G" + word(address) + word(a1)


def loaded_line(address, data):
    """What the boot loader answers once data is stored at address."""
    return f"loaded {len(data):08x} at {address:08x}: sum {sum(data) & 0xFFFFFFFF:08x}\n".encode()


def wait_for(link, text, timeout):
    """What the board sends, up to and including text (printed as it comes)."""
    got = b""
    end = time.time() + timeout
    while not got.endswith(text):
        if time.time() > end:
            raise TimeoutError(f"the board didn't say {text!r} (it said {got[-200:]!r})")
        piece = link.read(None, 0.2)
        got += piece
        sys.stdout.write(piece.decode(errors="replace"))
        sys.stdout.flush()
    return got


def boot(link, kernel=KERNEL):
    """Load the kernel and the device tree into the board, each checked, and
    start the kernel."""
    image = open(kernel, "rb").read()
    image += bytes(-len(image) % 4)
    tree = device_tree()
    for address, data in ((KERNEL_AT, image), (TREE_AT, tree)):
        print(f"sending {len(data):,} bytes to {address:#010x}", flush=True)
        start = time.time()
        link.write(load_record(address, data))
        wait_for(link, loaded_line(address, data), 60 + len(data) / 20_000)
        print(f"({time.time() - start:.1f} s)", flush=True)
    link.write(go_record(KERNEL_AT, TREE_AT))


def console(link, seconds=None):
    """Be the board's console: print what it sends, send what is typed (a line
    at a time, as the terminal gives it). Ctrl-C to stop. seconds: only print
    what it sends, for that long."""
    end = None if seconds is None else time.time() + seconds
    try:
        while end is None or time.time() < end:
            sys.stdout.write(link.read(None, 0.05).decode(errors="replace"))
            sys.stdout.flush()
            if end is None and select.select([sys.stdin], [], [], 0)[0]:
                line = sys.stdin.readline()
                if not line:
                    break
                link.write(line.encode())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--pins" in args:
        from py2ver import Py2ver
        board = Py2ver(R.computer(R.build("boot.c")), pins=R.PINS)
        print(board.PINS({"QUARTUS_DIR": os.environ.get(
            "QUARTUS_DIR", "/opt/intelFPGA/22.1std/quartus/bin")}))
    if "--boot" in args or "--console" in args:
        from py2ver.jtag_uart import JtagUart
        with JtagUart(os.environ.get("QUARTUS_DIR", "/opt/intelFPGA/22.1std/quartus/bin")) as link:
            if "--boot" in args:
                boot(link)
            watch = args[args.index("--watch") + 1] if "--watch" in args else None
            console(link, None if watch is None else float(watch))
