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
#   python examples/riscv_linux.py --pins --framebuffer   # its display the framebuffer (our kernel's)
#   python examples/riscv_linux.py --boot      # (the board programmed) load Linux, start it, its console
#   python examples/riscv_linux.py --boot --watch 60    # the console for 60 s, nothing typed
#   python examples/riscv_linux.py --console   # the console of a board already running Linux
#   python examples/riscv_linux.py --boot --kernel PATH # another kernel (our own, say)
#   python examples/riscv_linux.py --build-linux        # our own kernel and BusyBox (Buildroot)
#
# (Log in as root, no password. juart-terminal, in Quartus's bin directory, is a
# console too.)
#
# The kernel: linux-6.1.14-rv32nommu-cnl-1.zip of github.com/cnlohr/mini-rv32ima-images,
# unpacked to ~/opt/riscv/linux/image/Image (or LINUX_IMAGE names it).
#
# Our own (--build-linux): mini-rv32ima's Buildroot (git clone --recurse-submodules
# --depth 1 https://github.com/cnlohr/buildroot ~/opt/riscv/buildroot) with its
# configurations, the kernel's with KERNEL_OPTIONS (the framebuffer, Linux's
# console and logo on it, /dev/mem), BusyBox's with BUSYBOX_OPTIONS, the shell
# on the display too (SHELL_LINE); about an hour the first time. Its kernel:
# ~/opt/riscv/buildroot/output/images/Image (boot it with --kernel).
import hashlib
import os
import urllib.request
import select
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import riscv_computer as R  # noqa: E402

KERNEL = os.environ.get("LINUX_IMAGE", os.path.expanduser("~/opt/riscv/linux/image/Image"))
BUILDROOT = os.environ.get("BUILDROOT", os.path.expanduser("~/opt/riscv/buildroot"))
BUILT_KERNEL = os.path.join(BUILDROOT, "output", "images", "Image")
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


# ----------------------------- our own Linux -----------------------------

# mini-rv32ima's configurations (github.com/cnlohr/mini-rv32ima, configs/), each
# file's git hash, and where it goes in Buildroot's folder (the kernel's: with
# KERNEL_OPTIONS)
MINI_RV32IMA = "https://raw.githubusercontent.com/cnlohr/mini-rv32ima/master/configs/"
CONFIGS = {
    "buildroot_config": ("3d85234bed1f6a95651999d50395e5332624a800", ".config"),
    "busybox_config": ("2a2c0eff5daf6aefb1485fe158f2aa77c51990cb", "busybox_config"),
    "uclibc_config": ("bda6efc73b93bd4cda25b94c162850c2fba1419a", "uclibc_config"),
    "custom_kernel_config": ("ad1c3608add01c8bb38868b3318413c68e61c45d", "kernel_config"),
    "rootfsoverlay/etc/fstab": ("1ec4ecaca50f64db9bc9063b507a6744f445f8bc", "py2ver_overlay/etc/fstab"),
    "rootfsoverlay/etc/inittab": ("d20e6958e033594ae2ab694c14d74f2a65a049fc",
                                  "py2ver_overlay/etc/inittab"),
}

# The kernel's options beyond mini-rv32ima's: the framebuffer (linux.dts's
# simple-framebuffer, /dev/fb0), Linux's console on it with its logo, /dev/mem
KERNEL_OPTIONS = {"CONFIG_FB": "y", "CONFIG_FB_DEVICE": "y", "CONFIG_FB_SIMPLE": "y",
                  "CONFIG_FRAMEBUFFER_CONSOLE": "y", "CONFIG_LOGO": "y",
                  "CONFIG_LOGO_LINUX_CLUT224": "y", "CONFIG_DEVMEM": "y"}

# BusyBox's beyond mini-rv32ima's: script (the shell's words on the display too,
# see SHELL_LINE), fbset and fbsplash (pictures on /dev/fb0)
BUSYBOX_OPTIONS = {"CONFIG_SCRIPT": "y", "CONFIG_FBSET": "y", "CONFIG_FBSPLASH": "y"}

# The shell, on the console (the PC's: ttyS0) as in mini-rv32ima's inittab, all
# it prints also on the display (tty1, the framebuffer's console)
LOGIN_LINE = "console::respawn:/bin/login -f root"
SHELL_LINE = "console::respawn:/usr/bin/script -q -c '/bin/login -f root' /dev/tty1"


def git_hash(data):
    """A file's git hash (git hash-object's)."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def with_options(config, options):
    """A kernel configuration with options set (each one's old line, set or
    'is not set', taken out)."""
    lines = [line for line in config.splitlines()
             if not any(line.startswith(f"{o}=") or line == f"# {o} is not set" for o in options)]
    return "\n".join(lines + [f"{o}={v}" for o, v in options.items()]) + "\n"


def prepare_buildroot(buildroot=BUILDROOT):
    """mini-rv32ima's configurations into Buildroot's folder, each checked,
    the kernel's with KERNEL_OPTIONS, the root's files from the overlay."""
    if not os.path.isdir(buildroot):
        raise FileNotFoundError(f"no Buildroot in {buildroot}: git clone --recurse-submodules "
                                f"--depth 1 https://github.com/cnlohr/buildroot {buildroot}")
    for name, (sha, to) in CONFIGS.items():
        data = urllib.request.urlopen(MINI_RV32IMA + name, timeout=60).read()
        if git_hash(data) != sha:
            raise ValueError(f"{name}: not the file expected (its git hash {git_hash(data)})")
        if name == "custom_kernel_config":
            data = with_options(data.decode(), KERNEL_OPTIONS).encode()
        if name == "busybox_config":
            data = with_options(data.decode(), BUSYBOX_OPTIONS).encode()
        if name == "rootfsoverlay/etc/inittab":
            if LOGIN_LINE.encode() not in data:
                raise ValueError(f"{name}: no line {LOGIN_LINE!r} to change")
            data = data.replace(LOGIN_LINE.encode(), SHELL_LINE.encode())
        if name == "buildroot_config":       # (the overlay: in the same build)
            overlay = os.path.join(buildroot, "py2ver_overlay")
            data = data.replace(b'BR2_ROOTFS_OVERLAY=""', f'BR2_ROOTFS_OVERLAY="{overlay}"'.encode())
        path = os.path.join(buildroot, to)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "wb").write(data)
        if name == "uclibc_config":           # (its fragment too, as mini-rv32ima's Makefile)
            open(os.path.join(buildroot, "uclibc_config_extra"), "wb").write(data)


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
        screen = "framebuffer" if "--framebuffer" in args else "terminal"
        board = Py2ver(R.computer(R.build("boot.c"), screen=screen), pins=R.PINS)
        print(board.PINS({"QUARTUS_DIR": os.environ.get(
            "QUARTUS_DIR", "/opt/intelFPGA/22.1std/quartus/bin")}))
    if "--build-linux" in args:
        prepare_buildroot()
        # (BusyBox and the kernel built afresh: their configurations may be new)
        subprocess.run(["make", "-C", BUILDROOT, "busybox-dirclean", "linux-dirclean"], check=True)
        subprocess.run(["make", "-C", BUILDROOT], check=True)
        print(f"the kernel: {BUILT_KERNEL}")
    if "--boot" in args or "--console" in args:
        from py2ver.jtag_uart import JtagUart
        with JtagUart(os.environ.get("QUARTUS_DIR", "/opt/intelFPGA/22.1std/quartus/bin")) as link:
            if "--boot" in args:
                boot(link, args[args.index("--kernel") + 1] if "--kernel" in args else KERNEL)
            watch = args[args.index("--watch") + 1] if "--watch" in args else None
            console(link, None if watch is None else float(watch))
