import contextlib
import io
import sys
from pathlib import Path

import pytest

from py2ver import Py2ver
from py2ver.devices.ili9341 import ILI9341, ILI9341Model

# examples/fire.py: Doom's fire, its heats a list in a memory block, streamed
# to the display; KEY1 puts it out.

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
import fire as F  # noqa: E402


@pytest.mark.parametrize("sim", [False, pytest.param(True, marks=pytest.mark.sim)])
def test_the_fire_rises_and_goes_out(sim, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    w, h, count = 32, 32, 5
    lcd = ILI9341(wake_clocks=50)
    with contextlib.redirect_stdout(io.StringIO()):
        p = Py2ver(F.show_on_screen(lcd, w, h))
    assert "The list fire_heat in a memory block: 64 x 6 bits" in p.verilog
    screen = ILI9341Model(lcd, width=w, height=h)
    shots = p.check(F.frames(screen, count, w, h, press_at=3), verilog=sim).value
    white, black = F.FIRE[F.HOT], F.FIRE[0]

    def lit_rows(shot):                       # (rows of pixels with some heat)
        return sum(any(c != black for c in row) for row in shot)
    assert all(c == white for c in shots[0][-1])          # (the bottom row: hot)
    # (each frame's update before its picture: two rows of blocks in the first,
    # 4 pixels each, then higher each frame: a block, or two where the wind
    # wraps round the left edge into the row above, as in Doom's)
    lit = [lit_rows(s) for s in shots[:3]]
    assert lit[0] == 8 and lit[0] < lit[1] < lit[2]
    assert all(c == black for c in shots[4][-1])          # (KEY1 in frame 4: out)
