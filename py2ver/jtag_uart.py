"""
The board's JTAG UART from Python: bytes to and from a design running on the
DE0-Nano (its JTAG UART pins, see pins.py), over the USB-Blaster, through
Quartus's juart-terminal:

    with p.jtag_uart() as link:          # (p: a Py2ver, its PINS() build on the board)
        link.write(b"hello")
        link.read(5)                     # -> b'hello'
"""
from __future__ import annotations

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


class JtagUart:
    """A connection to the board's JTAG UART: write() bytes to the design, read()
    the bytes it sent. cable: the USB-Blaster as jtagconfig names it (default:
    the one there is)."""

    def __init__(self, quartus_dir: Optional[str] = None, cable: Optional[str] = None,
                 connect_s: float = 1.0) -> None:
        # (a byte 4 from the design is data: by default, ^D, the terminal would quit)
        args = ([find_tool(TERMINAL, quartus_dir), "-q", "--no-quit-on-ctrl-d"]
                + (["-c", cable] if cable else []))
        self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)
        self._sel = selectors.DefaultSelector()
        self._sel.register(self.proc.stdout, selectors.EVENT_READ)
        self._buf = b""
        time.sleep(connect_s)                    # (the terminal finding the board)
        if self.proc.poll() is not None:
            err = self.proc.stderr.read().decode(errors="replace").strip()
            raise ConnectionError(f"{TERMINAL} stopped: {err or 'no JTAG UART found'} (is the "
                                  f"board on, and its PINS() build with the JTAG UART's pins "
                                  f"programmed?)")

    def write(self, data: bytes) -> None:
        """Send bytes to the design."""
        if self.proc.poll() is not None:
            raise ConnectionError(f"{TERMINAL} stopped (exit code {self.proc.returncode})")
        self.proc.stdin.write(bytes(data))
        self.proc.stdin.flush()

    def read(self, n: Optional[int] = None, timeout: float = 2.0) -> bytes:
        """The bytes the design sent: n of them (or as many as come), waiting at
        most timeout seconds for them."""
        end = time.time() + timeout
        while (n is None or len(self._buf) < n) and time.time() < end:
            for key, _ in self._sel.select(timeout=min(0.1, max(0.0, end - time.time()))):
                chunk = os.read(key.fileobj.fileno(), 4096)
                if chunk:
                    self._buf += chunk
        k = len(self._buf) if n is None else min(n, len(self._buf))
        out, self._buf = self._buf[:k], self._buf[k:]
        return out

    def close(self) -> None:
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
