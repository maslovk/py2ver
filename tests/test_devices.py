import contextlib
import io
import shutil

import pytest

from py2ver import Py2ver
from py2ver.devices import adxl345 as X
from py2ver.devices.adc128s022 import ADC128S022, adc128s022_model, adc128s022_read
from py2ver.devices.eeprom24 import eeprom_model, eeprom_read, eeprom_write
from py2ver.devices.ili9341 import (ILI9341, ILI9341Model, ili9341_fill, ili9341_init,
                                    ili9341_pixel, ili9341_window, rgb565)
from py2ver.protocol import I2C, SPI, alongside, together, wait_until

# Each chip's driver (py2ver.devices) against the chip's model on the same pins:
# the generator, the machine's model and (marked sim) the Verilog, clock for clock.

FAST = I2C(q=4)                          # (a quick bus: 16 clocks a bit)
I2C_ATTR = {"scl": {"open_drain": True}, "sda": {"open_drain": True}}


def _quiet(f, attr):
    with contextlib.redirect_stdout(io.StringIO()):
        return Py2ver(f, attr)


def until_done(*outs):
    """A testbench step: once the design's done, its outputs."""
    def watch(dut):
        yield from wait_until(dut, "done")
        return tuple(getattr(dut, o) for o in outs)
    return watch


def with_chip(watch, chip):
    def tb(dut):
        got, log = yield from together(watch(dut), chip(dut))
        return got, log
    return tb


def _check(p, tb, sim):
    """The Run of check() (with sim, the Verilog's too)."""
    if sim and not shutil.which("iverilog"):
        pytest.skip("needs Icarus")
    return p.check(tb, verilog=sim)


# ----------------------------------- ADXL345 -----------------------------------

def tilt(port, bus=FAST):
    port.scl = 1
    port.sda = 1
    port.done = 0
    yield 10
    n1 = yield from X.adxl345_start(port, bus)
    ident, n2 = yield from X.adxl345_id(port, bus)
    x, y, z, n3 = yield from X.adxl345_read_xyz(port, bus)
    port.ident = ident
    port.x = x
    port.y = y
    port.z = z
    port.nack = n1 | n2 | n3
    port.done = 1
    while True:
        yield


@pytest.mark.parametrize("sim", [False, pytest.param(True, marks=pytest.mark.sim)])
def test_an_adxl345_read_on_its_bus(sim, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    regs = X.registers(x=3, y=-15, z=287)
    p = _quiet(tilt, I2C_ATTR)
    (ident, x, y, z, nack), log = _check(p, with_chip(
        until_done("ident", "x", "y", "z", "nack"),
        lambda dut: X.adxl345_model(dut, regs, 8, bus=FAST)), sim).value
    assert (ident, nack) == (X.ID, 0)
    assert (X.signed16(x), X.signed16(y), X.signed16(z)) == (3, -15, 287)
    assert regs[X.POWER_CTL] == X.MEASURE and log[0] == ("write", X.POWER_CTL, X.MEASURE)


def test_no_adxl345_no_acknowledge(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    p = _quiet(tilt, I2C_ATTR)
    (nack,), _ = _check(p, with_chip(until_done("nack"), lambda dut: X.adxl345_model(
        dut, X.registers(), 0, address=0x53, bus=FAST)), False).value
    assert nack == 1


# ---------------------------------- ADC128S022 ----------------------------------

QUICK_ADC = ADC128S022(SPI(2, mode=3))


def two_inputs(port, adc=QUICK_ADC):
    port.cs_n = 1
    port.sclk = 1
    port.done = 0
    yield 4
    a = yield from adc128s022_read(port, adc, 5)
    b = yield from adc128s022_read(port, adc, 2)
    port.a = a
    port.b = b
    port.done = 1
    while True:
        yield


@pytest.mark.parametrize("sim", [False, pytest.param(True, marks=pytest.mark.sim)])
def test_an_adc128s022_read_input_by_input(sim, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    values = [0x111 * k for k in range(8)]
    p = _quiet(two_inputs, {"miso": 1, "cs_n": {"init": 1}, "sclk": {"init": 1}})
    (a, b), _ = _check(p, with_chip(until_done("a", "b"),
                                    adc128s022_model(values, 4, QUICK_ADC)), sim).value
    assert (a, b) == (values[5], values[2])                # (the frame's 4 zeros: 12 bits)


# ------------------------------------ ILI9341 ------------------------------------

QUICK_LCD = ILI9341(wake_clocks=20)
RED, GREEN, BLUE, GREY = rgb565(255, 0, 0), rgb565(0, 255, 0), rgb565(0, 0, 255), 0x8410


def draws(port, lcd=QUICK_LCD):
    port.done = 0
    yield from ili9341_init(port, lcd)
    yield from ili9341_fill(port, lcd, 1, 2, 3, 3, RED)        # a 3 x 2 rectangle
    yield from ili9341_window(port, lcd, 5, 0, 6, 1)            # a 2 x 2 one, pixel by pixel
    yield from ili9341_pixel(port, lcd, RED)
    yield from ili9341_pixel(port, lcd, GREEN)
    yield from ili9341_pixel(port, lcd, BLUE)
    yield from ili9341_pixel(port, lcd, GREY)
    port.done = 1
    while True:
        yield


@pytest.mark.parametrize("sim", [False, pytest.param(True, marks=pytest.mark.sim)])
def test_an_ili9341_drawn_on(sim, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    screen = ILI9341Model(QUICK_LCD, width=8, height=4)
    p = _quiet(draws, {"cs_n": {"init": 1}, "reset_n": {"init": 1}})

    def tb(dut):
        yield from alongside(wait_until(dut, "done"), screen.run(dut))
    _check(p, tb, sim)
    assert screen.awake and screen.on
    assert [c for c, _ in screen.commands[:4]] == [0x11, 0x3A, 0x36, 0x29]
    assert screen.commands[1][1] == [0x55]                      # 16 bits a pixel
    assert [row[1:4] for row in screen.pixels[2:4]] == [[RED] * 3] * 2
    assert [row[5:7] for row in screen.pixels[0:2]] == [[RED, GREEN], [BLUE, GREY]]
    assert sum(c != 0 for row in screen.pixels for c in row) == 10
    assert "miso" not in p.process.inputs                       # (it only listens)
    screen.save(str(tmp_path / "screen.png"), scale=4)
    assert (tmp_path / "screen.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_alongside_returns_with_its_main_step():
    def counter(n):
        for _ in range(n):
            yield
        return n

    def forever():
        while True:
            yield 3

    steps = alongside(counter(5), forever(), counter(2))
    clocks = 0
    try:
        while True:
            next(steps)
            clocks += 1
    except StopIteration as stop:
        assert (stop.value, clocks) == (5, 5)


# ------------------------------------ EEPROM ------------------------------------

def keeps(port, bus=FAST):
    port.scl = 1
    port.sda = 1
    port.done = 0
    yield 10
    n1 = yield from eeprom_write(port, bus, 0x10, 0x42)
    value, n2 = yield from eeprom_read(port, bus, 0x10)
    port.value = value
    port.nack = n1 | n2
    port.done = 1
    while True:
        yield


@pytest.mark.parametrize("sim", [False, pytest.param(True, marks=pytest.mark.sim)])
def test_an_eeprom_written_waited_for_and_read(sim, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    mem = {}
    p = _quiet(keeps, I2C_ATTR)
    run = _check(p, with_chip(until_done("value", "nack"), lambda dut: eeprom_model(
        dut, mem, 2, write_clocks=600, bus=FAST)), sim)
    (value, nack), log = run.value
    assert (value, nack) == (0x42, 0) and mem == {0x10: 0x42}
    assert log == [("write", 0x10, 0x42), ("read", 0x10, 0x42)]
    # (its write cycle waited out: polled, unanswered, till it was done)
    sda = [r["sda"] for r in run.outputs]
    starts = sum(1 for a, b in zip(sda, sda[1:]) if a == 1 and b == 0)
    assert starts >= 2 + 600 // (9 * 4 * FAST.q)
