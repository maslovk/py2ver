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
#   python examples/riscv_linux.py --build-linux-mmu    # Linux with the MMU, and OpenSBI
#   python examples/riscv_linux.py --boot --mmu         # (the board programmed) boot it
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
#
# With the MMU (--build-linux-mmu): the same Buildroot, into its output-mmu
# folder, with MMU_CONFIG: glibc, built for RV32IMA (no C, no floating point:
# the CPU's), Linux in supervisor mode with Sv32 paging (MMU_KERNEL_OPTIONS),
# OpenSBI in machine mode under it (its generic platform, fw_jump); BusyBox,
# the overlay, as before. --boot --mmu loads OpenSBI (to 0x8000_0000), the
# kernel (0x8040_0000) and linux_mmu.dts, and starts OpenSBI.
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
MMU_OUTPUT = os.path.join(BUILDROOT, "output-mmu")
MMU_KERNEL = os.path.join(MMU_OUTPUT, "images", "Image")
MMU_FIRMWARE = os.path.join(MMU_OUTPUT, "images", "fw_jump.bin")
KERNEL_AT = 0x80000000
FIRMWARE_AT = 0x80000000                 # (with the MMU: OpenSBI here,
MMU_KERNEL_AT = 0x80400000               # the kernel here, 4 MB on: Sv32's megapage)
TREE_AT = 0x81FFC000                     # (after the memory linux.dts gives Linux)


def device_tree(source="linux.dts", room=0):
    """A device tree (examples/riscv's source) compiled (dtc), room bytes spare
    in it (for what OpenSBI adds), its length made a multiple of 4."""
    with tempfile.TemporaryDirectory() as work:
        out = os.path.join(work, "tree.dtb")
        subprocess.run(["dtc", "-q", "-I", "dts", "-O", "dtb", "-p", str(room), "-o", out,
                        os.path.join(R.PROGRAMS, source)], check=True)
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


# ----------------------------- Linux with the MMU -----------------------------

# Buildroot's configuration (from qemu_riscv32_virt_defconfig): the CPU's ISA
# exactly (RV32IMA, ilp32: no C, no floating point), glibc, Linux 6.8-rc1 (its
# rv32 defconfig, with MMU_KERNEL_OPTIONS), OpenSBI's generic platform built
# for the same ISA (fw_jump: the kernel at 0x8040_0000, the device tree at
# TREE_AT: see OPENSBI_PATCH), the root an initramfs in the kernel; {overlay},
# {kernel_options}, {busybox_options}, {patches}: their files and folders
MMU_CONFIG = """\
BR2_riscv=y
BR2_riscv_custom=y
BR2_RISCV_ISA_RVI=y
BR2_RISCV_ISA_RVM=y
BR2_RISCV_ISA_RVA=y
# BR2_RISCV_ISA_RVF is not set
# BR2_RISCV_ISA_RVD is not set
# BR2_RISCV_ISA_RVC is not set
BR2_RISCV_32=y
BR2_RISCV_ABI_ILP32=y
BR2_TOOLCHAIN_BUILDROOT_GLIBC=y
BR2_KERNEL_HEADERS_AS_KERNEL=y
BR2_PACKAGE_HOST_LINUX_HEADERS_CUSTOM_6_5=y
BR2_TARGET_GENERIC_HOSTNAME="py2ver"
BR2_ROOTFS_OVERLAY="{overlay}"
BR2_LINUX_KERNEL=y
BR2_LINUX_KERNEL_CUSTOM_VERSION=y
BR2_LINUX_KERNEL_CUSTOM_VERSION_VALUE="6.8-rc1"
BR2_LINUX_KERNEL_DEFCONFIG="rv32"
BR2_LINUX_KERNEL_CONFIG_FRAGMENT_FILES="{kernel_options}"
BR2_LINUX_KERNEL_IMAGE=y
BR2_PACKAGE_BUSYBOX_CONFIG_FRAGMENT_FILES="{busybox_options}"
BR2_TARGET_ROOTFS_CPIO=y
BR2_TARGET_ROOTFS_INITRAMFS=y
# BR2_TARGET_ROOTFS_TAR is not set
BR2_TARGET_OPENSBI=y
BR2_TARGET_OPENSBI_PLAT="generic"
BR2_TARGET_OPENSBI_ADDITIONAL_VARIABLES="PLATFORM_RISCV_ISA=rv32ima_zicsr_zifencei PLATFORM_RISCV_ABI=ilp32 \
FW_TEXT_START=0x80000000 FW_JUMP_ADDR=0x80400000 FW_JUMP_FDT_ADDR=0x81FFC000"
BR2_GLOBAL_PATCH_DIR="{patches}"
"""

# OpenSBI's generic platform sets FW_JUMP_FDT_ADDR (0x8220_0000 on RV32: past
# our 32 MB) whatever Buildroot passes it (in the environment, which a
# makefile's assignment overrides): this patch lets ours stand
OPENSBI_PATCH = """\
--- a/platform/generic/objects.mk
+++ b/platform/generic/objects.mk
@@ -32,7 +32,7 @@
   # This needs to be 2MB aligned for 64-bit system
   FW_JUMP_ADDR=$(shell printf "0x%X" $$(($(FW_TEXT_START) + 0x200000)))
 endif
-FW_JUMP_FDT_ADDR=$(shell printf "0x%X" $$(($(FW_TEXT_START) + 0x2200000)))
+FW_JUMP_FDT_ADDR?=$(shell printf "0x%X" $$(($(FW_TEXT_START) + 0x2200000)))
 FW_PAYLOAD=y
 ifeq ($(PLATFORM_RISCV_XLEN), 32)
   # This needs to be 4MB aligned for 32-bit system
"""

# The kernel's options beyond rv32's defconfig: what the CPU hasn't (floating
# point, the C extension, vectors, Zbb, the cache operations, Svnapot, Svpbmt;
# EFI, whose image begins with a compressed instruction), what the board
# hasn't (PCI, USB, sound, a network, DRM, MMC, virtio; disks and their file
# systems: the root an initramfs; I2C, SPI, HID, sensors, a watchdog, a
# clock), what 32 MB can't spare (BPF, cgroups, namespaces, crypto, keys,
# security modules, profiling, perf, power management, debugfs; the sections
# 4 MB apart for their protection, STRICT_KERNEL_RWX: 30 MB of image), one
# hart; the framebuffer, its console and the logo, /dev/mem; the 8250 UART,
# polled.
# LIBCRC32C is turned off by name: the RISC-V defconfig's file-system modules
# select it as a module, and that value stays in .config after they are
# gone. Without modules a module becomes built in, and LIBCRC32C selects
# crypto again, and with it the random-number generator, whose jitter
# entropy source tests itself at boot for 80 s on this CPU (the watchdog's
# soft-lockup warnings).
MMU_KERNEL_OPTIONS = {
    "CONFIG_SMP": "n", "CONFIG_FPU": "n", "CONFIG_RISCV_ISA_C": "n", "CONFIG_RISCV_ISA_V": "n",
    "CONFIG_RISCV_ISA_ZBB": "n", "CONFIG_RISCV_ISA_ZICBOM": "n", "CONFIG_RISCV_ISA_ZICBOZ": "n",
    "CONFIG_RISCV_ISA_SVNAPOT": "n", "CONFIG_RISCV_ISA_SVPBMT": "n", "CONFIG_EFI": "n",
    "CONFIG_MODULES": "n", "CONFIG_PCI": "n", "CONFIG_USB_SUPPORT": "n", "CONFIG_SOUND": "n",
    "CONFIG_NET": "n", "CONFIG_DRM": "n", "CONFIG_MMC": "n", "CONFIG_VIRTIO_MENU": "n",
    "CONFIG_BLOCK": "n", "CONFIG_MTD": "n", "CONFIG_SCSI": "n", "CONFIG_ATA": "n", "CONFIG_MD": "n",
    "CONFIG_EXT4_FS": "n", "CONFIG_BTRFS_FS": "n", "CONFIG_FAT_FS": "n", "CONFIG_OVERLAY_FS": "n",
    "CONFIG_AUTOFS_FS": "n", "CONFIG_I2C": "n", "CONFIG_SPI": "n", "CONFIG_HID": "n",
    "CONFIG_HWMON": "n", "CONFIG_THERMAL": "n", "CONFIG_WATCHDOG": "n", "CONFIG_RTC_CLASS": "n",
    "CONFIG_BPF_SYSCALL": "n", "CONFIG_CGROUPS": "n", "CONFIG_NAMESPACES": "n", "CONFIG_CRYPTO": "n",
    "CONFIG_LIBCRC32C": "n", "CONFIG_KEYS": "n", "CONFIG_SECURITY": "n", "CONFIG_PROFILING": "n", "CONFIG_PERF_EVENTS": "n",
    "CONFIG_SUSPEND": "n", "CONFIG_CPU_IDLE": "n", "CONFIG_CPU_FREQ": "n", "CONFIG_DEBUG_FS": "n",
    "CONFIG_IKCONFIG": "n", "CONFIG_STRICT_KERNEL_RWX": "n",
    **KERNEL_OPTIONS,
    "CONFIG_SERIAL_8250": "y", "CONFIG_SERIAL_8250_CONSOLE": "y", "CONFIG_SERIAL_OF_PLATFORM": "y",
    "CONFIG_SERIAL_EARLYCON": "y", "CONFIG_BLK_DEV_INITRD": "y"}


# BusyBox's: BUSYBOX_OPTIONS, and no tc (BusyBox 1.36's tc wants the CBQ
# scheduler's definitions, which Linux 6.8's headers have dropped)
MMU_BUSYBOX_OPTIONS = {**BUSYBOX_OPTIONS, "CONFIG_TC": "n"}


def fragment(options):
    """A configuration fragment: each option set, or (n) not set."""
    return "".join(f"# {o} is not set\n" if v == "n" else f"{o}={v}\n" for o, v in options.items())


def prepare_mmu_buildroot(buildroot=BUILDROOT):
    """The MMU build's configuration (and the overlay, prepare_buildroot's)
    into Buildroot's folder: py2ver_mmu/defconfig, its two fragments, and
    the patches (OPENSBI_PATCH)."""
    prepare_buildroot(buildroot)
    folder = os.path.join(buildroot, "py2ver_mmu")
    os.makedirs(folder, exist_ok=True)
    files = {"kernel_options": os.path.join(folder, "kernel.fragment"),
             "busybox_options": os.path.join(folder, "busybox.fragment")}
    open(files["kernel_options"], "w").write(fragment(MMU_KERNEL_OPTIONS))
    open(files["busybox_options"], "w").write(fragment(MMU_BUSYBOX_OPTIONS))
    patches = os.path.join(folder, "patches")
    os.makedirs(os.path.join(patches, "opensbi"), exist_ok=True)
    open(os.path.join(patches, "opensbi", "0001-fw-jump-fdt-addr-from-outside.patch"), "w").write(OPENSBI_PATCH)
    config = os.path.join(folder, "defconfig")
    open(config, "w").write(MMU_CONFIG.format(overlay=os.path.join(buildroot, "py2ver_overlay"),
                                              patches=patches, **files))
    return config


def boot_mmu(link, firmware=MMU_FIRMWARE, kernel=MMU_KERNEL):
    """Load OpenSBI, the MMU kernel and linux_mmu.dts into the board, each
    checked, and start OpenSBI (a1: the device tree)."""
    images = [(FIRMWARE_AT, open(firmware, "rb").read()), (MMU_KERNEL_AT, open(kernel, "rb").read()),
              (TREE_AT, device_tree("linux_mmu.dts", room=4096))]
    for address, data in images:
        data += bytes(-len(data) % 4)
        print(f"sending {len(data):,} bytes to {address:#010x}", flush=True)
        start = time.time()
        link.write(load_record(address, data))
        wait_for(link, loaded_line(address, data), 60 + len(data) / 20_000)
        print(f"({time.time() - start:.1f} s)", flush=True)
    link.write(go_record(FIRMWARE_AT, TREE_AT))


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
    if "--build-linux-mmu" in args:
        config = prepare_mmu_buildroot()
        subprocess.run(["make", "-C", BUILDROOT, f"O={MMU_OUTPUT}", f"BR2_DEFCONFIG={config}", "defconfig"],
                       check=True)
        subprocess.run(["make", "-C", BUILDROOT, f"O={MMU_OUTPUT}", "busybox-dirclean", "linux-dirclean"],
                       check=True)
        subprocess.run(["make", "-C", BUILDROOT, f"O={MMU_OUTPUT}"], check=True)
        print(f"OpenSBI: {MMU_FIRMWARE}; the kernel: {MMU_KERNEL}")
    if "--boot" in args or "--console" in args:
        from py2ver.jtag_uart import JtagUart
        with JtagUart(os.environ.get("QUARTUS_DIR", "/opt/intelFPGA/22.1std/quartus/bin")) as link:
            if "--boot" in args and "--mmu" in args:
                boot_mmu(link)
            elif "--boot" in args:
                boot(link, args[args.index("--kernel") + 1] if "--kernel" in args else KERNEL)
            watch = args[args.index("--watch") + 1] if "--watch" in args else None
            console(link, None if watch is None else float(watch))
