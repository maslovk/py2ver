import contextlib
import io
import sys
from pathlib import Path

import pytest

from py2ver import Py2ver

# The real board: a design's PINS() build programmed into a DE0-Nano plugged in,
# its answers over the JTAG UART checked against its generator's. Run with
# --hardware (they program the board; the board's flash is left as it is).

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
import jtag_echo as J  # noqa: E402


def _quiet(f, attr, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return Py2ver(f, attr, **kw)


def test_the_generator_says_what_the_board_must_answer():
    p = _quiet(J.jtag_echo, J.PINS)
    assert p.jtag_reference(b"hello") == b"hello"


@pytest.mark.hardware
def test_an_echo_on_the_board_answers_as_its_generator(quartus):
    p = _quiet(J.jtag_echo, J.PINS)
    p.PINS(quartus)
    p.PROGRAM(quartus)
    sent = bytes(range(32, 127)) + b"\x00\xff"
    expected = p.jtag_reference(sent)
    assert expected == sent
    with p.jtag_uart(quartus) as link:
        link.write(sent)
        assert link.read(len(expected), timeout=5) == expected
