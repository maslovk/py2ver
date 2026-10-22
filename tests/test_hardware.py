import contextlib
import io
import sys
from pathlib import Path

import pytest

from py2ver import Py2ver, signaltap

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


import accelerometer as A  # noqa: E402
import jtag_console as C  # noqa: E402


def test_an_accelerometer_read_on_a_bus():
    # (an ADXL345's registers played by examples/i2c.py's memory: X, Y, Z from 0x32)
    mem = {0x32: 0x03, 0x33: 0x00, 0x34: 0xF1, 0x35: 0xFF, 0x36: 0x1F, 0x37: 0x01}
    import functools
    p = _quiet(functools.partial(A.accelerometer, q=4, period=100), A.ATTR)
    text = p.check(A.with_a_memory(mem, A.ADXL345, transfers=8, until=b"z=0x011F\n"),
                   verilog=False).value
    assert text == b"adxl345: measuring\nx=0x0003 y=0xFFF1 z=0x011F\n"
    assert A.readings(text) == [(3, -15, 287)]
    assert mem[A.POWER_CTL] == 0x08                                # (measuring: switched on)


@pytest.mark.hardware
def test_the_board_s_accelerometer_feels_gravity(quartus):
    p = _quiet(A.accelerometer, A.PINS)
    p.PINS(quartus)
    p.PROGRAM(quartus)
    with p.jtag_uart(quartus) as link:
        text = link.read(timeout=3)
    assert text.startswith(b"adxl345: measuring\n")
    got = A.readings(text)
    assert len(got) >= 5
    for x, y, z in got:                                      # (1 g, whichever way it lies)
        assert 200 <= (x * x + y * y + z * z) ** 0.5 <= 330

FIELDS = [("byte_", 0, 8), ("valid", 8, 1), ("ready", 9, 1)]


def test_a_console_prints_what_it_does():
    text = _quiet(C.board_console(bounce=4), C.PINS).check(C.session(), verilog=False).value
    assert text == C.BANNER + b"presses: 0x01\npresses: 0x02\nswitches: 0xA\n"


@pytest.mark.hardware
def test_a_console_on_the_board(quartus):
    p = _quiet(C.board_console(), C.PINS)
    p.PINS(quartus)
    p.PROGRAM(quartus)
    with p.jtag_uart(quartus) as link:
        assert link.read(len(C.BANNER), timeout=5) == C.BANNER    # (printed at reset, kept)
        link.write(b"s")
        line = link.read(len(b"switches: 0x0\n"), timeout=5)
    assert line.startswith(b"switches: 0x") and line.endswith(b"\n")
    assert line[12:13] in b"0123456789ABCDEF"                  # (the board's SW[3:0])


def test_a_trigger_condition_bit_by_bit():
    assert signaltap.condition({"valid": 1}, FIELDS) == "'acq_trigger_in[8]' == high"
    both = signaltap.condition({"byte_": 0b101, "valid": "rising"}, FIELDS)
    assert both.startswith("'acq_trigger_in[0]' == high &amp;&amp; 'acq_trigger_in[1]' == low")
    assert both.endswith("'acq_trigger_in[8]' == rising edge")
    assert signaltap.condition(None, FIELDS) == ""
    with pytest.raises(ValueError, match="an edge is a bit's"):
        signaltap.condition({"byte_": "falling"}, FIELDS)
    with pytest.raises(ValueError, match="no signal 'txd'"):
        signaltap.condition({"txd": 1}, FIELDS)


def test_a_capture_read_back(tmp_path):
    (tmp_path / "signaltap.txt").write_text("signal\tbits\twidth\nbyte_\t7:0\t8\nvalid\t8\t1\n"
                                            "ready\t9\t1\n")
    header = "time unit: ns, " + ", ".join(f"acq_data_in[{i}]" for i in range(10)) + ", "

    def row(k, byte, valid, ready):
        bits = [(byte >> i) & 1 for i in range(8)] + [valid, ready]
        return f"{k}, " + ", ".join(map(str, bits)) + ", "
    lines = ["Groups:", "", "Data:", header, "0, " + "X, " * 10]
    lines += [row(k, 0, 0, 1) for k in range(1, 17)] + [row(17, 0x41, 1, 1), row(18, 0, 0, 0)]
    (tmp_path / "capture.csv").write_text("\n".join(lines) + "\n")
    (tmp_path / "capture.vcd").write_text("$comment Sample -16 (Start) $end\n")
    cap = signaltap.read_capture(tmp_path, signaltap.read_map(tmp_path / "signaltap.txt"),
                                 {"byte_": "byte"})
    assert len(cap.rows) == 18 and cap.trigger == 16
    assert cap.rows[16] == {"byte": 0x41, "valid": 1, "ready": 1}
    assert [k for k, _ in cap.changes()] == [-16, 0, 1]
    cap.write_vcd(str(tmp_path / "out.vcd"))
    assert "$comment trigger $end" in (tmp_path / "out.vcd").read_text()


@pytest.mark.hardware
def test_a_signaltap_capture_from_the_board(quartus):
    p = _quiet(J.jtag_echo, J.PINS, probes=True)
    p.PINS(quartus, signaltap=True)
    p.PROGRAM(quartus)
    with p.jtag_uart(quartus) as link:
        cap = p.CAPTURE(quartus, trigger={"valid": 1, "byte": ord("Q")},
                        during=lambda: link.write(b"Q"))
        assert link.read(1, timeout=3) == b"Q"
    at, after = cap.rows[cap.trigger], cap.rows[cap.trigger + 1]
    assert at["valid"] == 1 and at["byte"] == ord("Q")              # (the byte, taken)
    assert after["send"] == 1 and after["out"] == ord("Q") and after["ready"] == 0
    assert "jtag_echo.py:33" in cap.show()                         # (its state, by its line)


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
