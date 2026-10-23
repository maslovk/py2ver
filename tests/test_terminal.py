import contextlib
import io
import sys
from pathlib import Path

import pytest

from py2ver import Py2ver
from py2ver.devices import font as F
from py2ver.devices.ili9341 import ILI9341, ILI9341Model
from py2ver.protocol import SPI

# examples/terminal.py: the PC's bytes over the JTAG UART shown on the display,
# the screen's characters a list in a memory block, the font a table.

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
import terminal as T  # noqa: E402

COLUMNS, ROWS = 8, 8                     # (64 characters: the smallest list in a memory block)
WIDTH, HEIGHT = COLUMNS * F.WIDTH, ROWS * F.HEIGHT


def _shown(typed, count, sim):
    lcd = ILI9341(spi=SPI(1, mode=0, miso=""), wake_clocks=50)
    with contextlib.redirect_stdout(io.StringIO()):
        p = Py2ver(T.show_on_screen(lcd, COLUMNS, ROWS, HEIGHT))
    screen = ILI9341Model(lcd, width=WIDTH, height=HEIGHT)
    return p, p.check(T.typing(screen, typed, count, WIDTH, HEIGHT), verilog=sim).value


def test_a_font_glyph_is_its_character():
    a = F.glyph(ord("A"))
    assert len(F.FONT) == (F.LAST - F.FIRST + 1) * F.HEIGHT
    assert [f"{r:08b}" for r in a[2:5]] == ["00011000", "00100100", "01000010"]
    assert all(r == 0 for r in F.glyph(ord(" ")))


@pytest.mark.parametrize("sim", [False, pytest.param(True, marks=pytest.mark.sim)])
def test_what_is_typed_is_shown_and_rubbed_out(sim, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    p, shots = _shown(b"hix\x08\rok", 2, sim)
    assert "The list terminal_text in a memory block: 64 x 7 bits" in p.verilog
    assert any("Table in a memory block: 1235 x 8 bits" in line for line in p.verilog.splitlines())
    assert shots[-1] == T.picture(["py2ver o", "", "hi", "ok"], COLUMNS, ROWS, HEIGHT)


def test_the_text_scrolls_up_at_the_bottom(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _, shots = _shown(b"1\r2\r3\r4\r5\r6\r", 3, False)
    # (the greeting, a blank row, then 1 to 6: two rows too many, so up two)
    assert shots[-1] == T.picture(["", "1", "2", "3", "4", "5", "6"], COLUMNS, ROWS, HEIGHT)
