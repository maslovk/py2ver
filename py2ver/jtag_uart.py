"""
The board's JTAG UART from Python: bytes to and from a design running on the
DE0-Nano (its JTAG UART pins, see pins.py), over the USB-Blaster:

    with p.jtag_uart() as link:          # (p: a Py2ver, its PINS() build on the board)
        link.write(b"hello")
        link.read(5)                     # -> b'hello'

Through Quartus's libjtag_atlantic (the library its terminals use: a round trip
in about 25 ms, some 200 KB/s), or where there's none, its juart-terminal (a
round trip in about 0.2 s, 1.3 KB/s to the board: it polls every 100 ms).
"""
from __future__ import annotations

import ctypes
import os
import selectors
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

from .defaults import DEFAULT_QUARTUS_DIR

TERMINAL = "juart-terminal"


def find_tool(name: str, quartus_dir: Optional[str] = None) -> str:
    """A Quartus tool's path: in quartus_dir, $QUARTUS_DIR, on PATH, or Quartus's
    usual install."""
    for d in (quartus_dir, os.environ.get("QUARTUS_DIR"), None, DEFAULT_QUARTUS_DIR):
        if d is None:
            found = shutil.which(name)
            if found:
                return found
            continue
        p = Path(d) / name
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"{name} not found (give QUARTUS_DIR: Quartus's bin directory)")


# libjtag_atlantic's open errors (jtagatlantic_get_error)
_OPEN_ERRORS = {
    -1: "can't reach the JTAG server (jtagd)",
    -2: "more than one cable: say which (JTAG_CABLE, as jtagconfig names it)",
    -3: "no such cable",
    -4: "the cable isn't plugged in",
    -5: "the cable isn't connected to the board, or the board is off",
    -6: "another program has the JTAG UART (a juart-terminal, or another connection)",
    -7: "more than one JTAG UART on the board",
    -8: "no JTAG UART on the board (is a build with the JTAG UART programmed?)",
    -9: "the board's JTAG UART doesn't suit this library",
}

_LIBRARY = {}                                    # (each Quartus's, loaded once)


def _library(quartus_dir: Optional[str]):
    """Quartus's libjtag_atlantic (its functions typed), or None if this Quartus
    has none."""
    try:
        bin_dir = Path(find_tool(TERMINAL, quartus_dir)).resolve().parent
    except FileNotFoundError:
        return None
    lib_dir = bin_dir.parent / "linux64"
    if not (lib_dir / "libjtag_atlantic.so").exists():
        return None
    if lib_dir not in _LIBRARY:
        # (its JTAG client first, for the library's own symbols)
        ctypes.CDLL(str(lib_dir / "libjtag_client.so"), mode=ctypes.RTLD_GLOBAL)
        lib = ctypes.CDLL(str(lib_dir / "libjtag_atlantic.so"))
        P, I, U, S = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint, ctypes.c_char_p

        def fn(name, mangled, res, *args):
            f = getattr(lib, mangled)
            f.restype, f.argtypes = res, list(args)
            setattr(lib, name, f)
        # (C++ names: open(cable, device, instance, application), get_error(&info),
        # write(link, data, n) / read(link, buffer, size): the bytes taken / read)
        fn("open", "_Z17jtagatlantic_openPKciiS0_", P, S, I, I, S)
        fn("get_error", "_Z22jtagatlantic_get_errorPPKc", I, ctypes.POINTER(S))
        fn("close", "_Z18jtagatlantic_closeP12JTAGATLANTIC", None, P)
        fn("write", "_Z18jtagatlantic_writeP12JTAGATLANTICPKcj", I, P, S, U)
        fn("read", "_Z17jtagatlantic_readP12JTAGATLANTICPcj", I, P, S, U)
        _LIBRARY[lib_dir] = lib
    return _LIBRARY[lib_dir]


class JtagUart:
    """A connection to the board's JTAG UART: write() bytes to the design, read()
    the bytes it sent. cable: the USB-Blaster as jtagconfig names it (default:
    the one there is); terminal: through juart-terminal even where Quartus has
    the library."""

    PIECE = 1024        # (the library: bytes written before what came back is read)

    def __init__(self, quartus_dir: Optional[str] = None, cable: Optional[str] = None,
                 connect_s: float = 1.0, terminal: bool = False) -> None:
        self._buf = b""
        self.proc = None
        self._link = None
        self._lib = None if terminal else _library(quartus_dir)
        if self._lib is not None:
            self._link = self._lib.open(cable.encode() if cable else None, -1, -1, b"py2ver")
            if not self._link:
                info = ctypes.c_char_p()
                code = self._lib.get_error(ctypes.byref(info))
                extra = f": {info.value.decode(errors='replace')}" if info.value else ""
                raise ConnectionError(f"JTAG UART: {_OPEN_ERRORS.get(code, f'error {code}')}"
                                      f"{extra}")
            self._chunk = ctypes.create_string_buffer(65536)
            return
        # (a byte 4 from the design is data: by default, ^D, the terminal would quit)
        args = ([find_tool(TERMINAL, quartus_dir), "-q", "--no-quit-on-ctrl-d"]
                + (["-c", cable] if cable else []))
        self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)
        self._sel = selectors.DefaultSelector()
        self._sel.register(self.proc.stdout, selectors.EVENT_READ)
        time.sleep(connect_s)                    # (the terminal finding the board)
        if self.proc.poll() is not None:
            err = self.proc.stderr.read().decode(errors="replace").strip()
            raise ConnectionError(f"{TERMINAL} stopped: {err or 'no JTAG UART found'} (is the "
                                  f"board on, and its PINS() build with the JTAG UART's pins "
                                  f"programmed?)")

    def alive(self) -> bool:
        """Still connected (not closed; the terminal still running)."""
        return self._link is not None if self.proc is None else self.proc.poll() is None

    def write(self, data: bytes) -> None:
        """Send bytes to the design."""
        data = bytes(data)
        if self._lib is not None:
            if self._link is None:
                raise ConnectionError("JTAG UART: closed")
            # (a piece at a time, what came back read between: the library's buffer
            # for it full, the board would stop taking ours)
            sent = 0
            while sent < len(data):
                piece = data[sent:sent + self.PIECE]
                k = self._lib.write(self._link, piece, len(piece))
                if k < 0:
                    raise ConnectionError(f"JTAG UART: the link lost (error {k})")
                sent += k
                if not self._pull() and k == 0:
                    time.sleep(0.001)
            return
        if self.proc.poll() is not None:
            raise ConnectionError(f"{TERMINAL} stopped (exit code {self.proc.returncode})")
        self.proc.stdin.write(data)
        self.proc.stdin.flush()

    def _pull(self) -> int:
        """The library: the bytes that have come, into the buffer; how many."""
        k = self._lib.read(self._link, self._chunk, len(self._chunk))
        if k < 0:
            raise ConnectionError(f"JTAG UART: the link lost (error {k})")
        self._buf += self._chunk.raw[:k]
        return k

    def read(self, n: Optional[int] = None, timeout: float = 2.0) -> bytes:
        """The bytes the design sent: n of them (or as many as come), waiting at
        most timeout seconds for them."""
        end = time.time() + timeout
        if self._lib is not None and self._link is not None:
            self._pull()                         # (what's come, even with no time to wait)
        while (n is None or len(self._buf) < n) and time.time() < end:
            if self._lib is not None:
                if self._link is None:
                    raise ConnectionError("JTAG UART: closed")
                if not self._pull():
                    time.sleep(0.0005)
                continue
            for key, _ in self._sel.select(timeout=min(0.1, max(0.0, end - time.time()))):
                chunk = os.read(key.fileobj.fileno(), 4096)
                if chunk:
                    self._buf += chunk
        k = len(self._buf) if n is None else min(n, len(self._buf))
        out, self._buf = self._buf[:k], self._buf[k:]
        return out

    def close(self) -> None:
        if self._lib is not None:
            if self._link is not None:
                self._lib.close(self._link)
                self._link = None
            return
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        for f in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            if f:
                try:
                    f.close()
                except OSError:                  # (the terminal gone: its pipe broken)
                    pass

    def __enter__(self) -> "JtagUart":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
