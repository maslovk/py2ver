import contextlib
import io
import sys
from pathlib import Path

import pytest

from py2ver import Py2ver
from py2ver.devices.ili9341 import ILI9341, ILI9341Model
from py2ver.protocol import I2C, SPI

# examples/marble_maze.py: a marble rolled through a maze by tilting the board;
# four processes joined (the accelerometer, a stopwatch, the game, the
# display). A small maze to keep the run short; even so, slow.

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
import marble_maze as M  # noqa: E402

pytestmark = pytest.mark.slow
SMALL = M.mazes(80, 88, levels=1)        # (3 x 2 places)


def test_a_maze_is_one_path_between_any_two_places():
    blocks, across, down = M.random_maze(4, 3, seed=7)
    floor = {(c, r) for r in range(down) for c in range(across) if not blocks[r * across + c]}
    reached, todo = {(1, 1)}, [(1, 1)]
    while todo:
        c, r = todo.pop()
        for n in ((c + 1, r), (c - 1, r), (c, r + 1), (c, r - 1)):
            if n in floor and n not in reached:
                reached.add(n)
                todo.append(n)
    assert reached == floor                          # (every floor block reachable)
    assert all(blocks[k] for k in range(across))     # (walls all round)


@pytest.mark.parametrize("sim", [False, pytest.param(True, marks=pytest.mark.sim)])
def test_tilting_rolls_the_marble(sim, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    lcd = ILI9341(spi=SPI(1, mode=0, miso=""), wake_clocks=50)
    with contextlib.redirect_stdout(io.StringIO()):
        p = Py2ver(M.show_on_screen(lcd, SMALL, bus=I2C(q=4), period=2_000, tick=20_000, grip=3))
    screen = ILI9341Model(lcd, width=SMALL.width, height=SMALL.height)
    shots = p.check(M.frames(screen, 4, SMALL, tilt=(128, 0)), verilog=sim).value
    at = [M.marble_at(s) for s in shots]
    assert at[0] == (SMALL.start_x, SMALL.start_y)               # (where it starts)
    xs = [x for x, _ in at]
    assert all(y == SMALL.start_y for _, y in at)                 # (no tilt up or down)
    assert xs[1] - xs[0] < xs[3] - xs[2]                          # (rolling right, faster)
