import contextlib
import io
import sys
from pathlib import Path

import pytest

from py2ver import Py2ver
from py2ver.devices.ili9341 import ILI9341, ILI9341Model

# examples/tunnel.py: flying down a checkered tube, each pixel's point on the
# wall from a table of a quarter of the screen, streamed to the display.

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
import tunnel as T  # noqa: E402


def _show(w, h, speed=12, twist=8):
    lcd = ILI9341(wake_clocks=50)
    with contextlib.redirect_stdout(io.StringIO()):
        return Py2ver(T.show_on_screen(lcd, T.tube(w, h), w, h, speed, twist)), lcd


@pytest.mark.parametrize("sim", [False, pytest.param(True, marks=pytest.mark.sim)])
def test_the_tunnel_s_frames_are_as_worked_out_directly(sim, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    w, h, count = 32, 40, 3
    p, lcd = _show(w, h)
    screen = ILI9341Model(lcd, width=w, height=h)
    shots = p.check(T.frames(screen, count, w, h), verilog=sim).value
    assert shots == [T.picture(k, T.tube(w, h), w, h, 12, 8) for k in range(count)]


def test_the_tunnel_s_tables_are_memory_blocks():
    p, _ = _show(32, 40)
    tables = [line for line in p.verilog.splitlines() if line.startswith("// Table")]
    assert any("80 x 18 bits" in t and "memory block" in t for t in tables)   # (the tube)
    assert any("256 x 16 bits" in t and "memory block" in t for t in tables)  # (the colors)
