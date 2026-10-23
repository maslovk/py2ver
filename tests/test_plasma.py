import contextlib
import io
import sys
from pathlib import Path

import pytest

from py2ver import Py2ver
from py2ver.devices.ili9341 import ILI9341, ILI9341Model

# examples/plasma.py: four sine waves added at each pixel pick a color, frame
# after frame, streamed to the display.

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
import plasma as P  # noqa: E402


@pytest.mark.parametrize("sim", [False, pytest.param(True, marks=pytest.mark.sim)])
def test_the_plasma_s_frames_are_as_worked_out_directly(sim, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    w, h, count, speed = 16, 12, 3, 40
    lcd = ILI9341(wake_clocks=50)
    screen = ILI9341Model(lcd, width=w, height=h)
    with contextlib.redirect_stdout(io.StringIO()):
        p = Py2ver(P.show_on_screen(lcd, w, h, speed))
    shots = p.check(P.frames(screen, count, w, h), verilog=sim).value
    assert shots == [P.picture(k * speed, w, h) for k in range(count)]
    assert len({color for shot in shots for row in shot for color in row}) > 20


def test_the_plasma_s_tables_are_memory_blocks():
    with contextlib.redirect_stdout(io.StringIO()):
        p = Py2ver(P.show_on_screen(ILI9341(wake_clocks=50), 16, 12))
    tables = [line for line in p.verilog.splitlines() if line.startswith("// Table")]
    assert sum("256 x 6 bits" in t and "memory block" in t for t in tables) == 6   # (the sines)
    assert any("256 x 16 bits" in t and "memory block" in t for t in tables)       # (the palette)
