import contextlib
import io
import sys
from pathlib import Path

import pytest

from py2ver import Py2ver
from py2ver.devices.ili9341 import ILI9341, ILI9341Model
from py2ver.protocol import I2C, SPI

# examples/breakout.py: Breakout steered by the board's accelerometer, three
# processes joined (the sensor, the game, the display). A small field to keep
# the run short; even so a few frames are millions of clocks: slow.

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
import breakout as B  # noqa: E402

pytestmark = pytest.mark.slow
SMALL = B.field(96, 104, rows=1)         # (two bricks across, one row)


def _played(sim, count=4):
    lcd = ILI9341(spi=SPI(1, mode=0, miso=""), wake_clocks=50)
    with contextlib.redirect_stdout(io.StringIO()):
        p = Py2ver(B.show_on_screen(lcd, SMALL, bus=I2C(q=4), period=2_000, speed=12))
    screen = ILI9341Model(lcd, width=SMALL.width, height=SMALL.height)
    return p.check(B.frames(screen, count, SMALL, tilt=96), verilog=sim).value


def _paddle_left(shot):
    row = shot[SMALL.paddle_y]
    return row.index(B.PADDLE)


def _bricks(shot):
    return sum(c == B.ROW_COLORS[0] for row in shot for c in row)


@pytest.mark.parametrize("sim", [False, pytest.param(True, marks=pytest.mark.sim)])
def test_tilting_steers_and_the_ball_breaks_a_brick(sim, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    shots = _played(sim)
    # (tilted right, 96: the paddle 96 >> 4 = 6 pixels right a frame, from the
    # middle, once the sensor's first reading is in: after the first frame)
    assert [_paddle_left(s) for s in shots] == [24, 30, 36, 42]
    # (served in frame 2, the ball at 12 pixels a frame: a brick gone in frame 3)
    assert _bricks(shots[0]) == _bricks(shots[1]) == 2 * 30 * 14
    assert _bricks(shots[-1]) == 30 * 14
