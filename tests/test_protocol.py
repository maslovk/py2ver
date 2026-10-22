import contextlib
import io
import sys
from pathlib import Path

import pytest

from py2ver import Py2ver, process
from py2ver.protocol import shift_in, shift_out, together, uart_receive, uart_send, wait_until

# Steps of protocols (protocol.py), used by processes and by testbenches written as
# generators too (drive(), check()); SPI and WS2812 written from their specs.

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
import spi as S  # noqa: E402
import uart as U  # noqa: E402
import ws2812 as W  # noqa: E402


def _quiet(f, attr=None):
    with contextlib.redirect_stdout(io.StringIO()):
        return Py2ver(f, attr)


def host(text):
    """A testbench: each byte sent to the design's rx while what it sends back
    on tx is received; the bytes back."""
    def tb(dut):
        dut.rx = 1                                   # (the line idle)
        yield 10
        got = []
        for b in text:
            _, back = yield from together(uart_send(dut, "rx", b, U.BIT),
                                          uart_receive(dut, "tx", U.BIT))
            got.append(back)
        return bytes(got)
    return tb


def test_a_testbench_is_a_generator_too():
    run = _quiet(U.echo()).check(host(b"ok"), verilog=False)
    assert run.value == b"ok" and len(run.inputs) == len(run.outputs) > 2 * 10 * U.BIT
    assert run.outputs == process.simulate(U.echo(), run.inputs)    # (its clocks, replayed)


@pytest.mark.sim
def test_a_testbench_checks_the_verilog(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert _quiet(U.echo()).check(host(b"A")).value == b"A"


def test_a_check_says_where_it_differs(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    p = _quiet(U.echo())
    model = p.PY_process

    def wrong():                                     # (a model that's off at clock 1,000)
        def run(clocks):
            rows = model()(clocks)
            rows[1000] = dict(rows[1000], busy=1 - rows[1000]["busy"])
            return rows
        return run
    monkeypatch.setattr(p, "PY_process", wrong)
    with pytest.raises(process.CheckFailed, match=r"the model differs from the generator at "
                                                  r"clock 1,000: busy 0 / 1"):
        p.check(host(b"x"), verilog=False)
    assert (tmp_path / "uart_echo_generator.vcd").exists() and (tmp_path / "uart_echo_model.vcd").exists()


def test_a_testbench_writes_inputs_and_reads_outputs():
    def writes_an_output(dut):
        dut.tx = 1
        yield

    def reads_no_pin(dut):
        yield
        return dut.txd
    with pytest.raises(AttributeError, match="'tx' is an output of the design"):
        process.drive(U.echo(), writes_an_output)
    with pytest.raises(AttributeError, match="no pin 'txd'.*its outputs: valid, data, error, tx, busy"):
        process.drive(U.echo(), reads_no_pin)


def tx_with_steps(port, bit=U.BIT):
    """examples/uart.py's transmitter, its frame the library's step."""
    port.tx = 1
    while True:
        port.busy = 0
        yield from wait_until(port, "start")
        data = port.data
        port.busy = 1
        yield from uart_send(port, "tx", data, bit)


def test_a_library_step_is_built_in_the_process():
    clocks = U.frames(b"\x5a\x00")
    p = _quiet(tx_with_steps, U.TX)
    assert p.GEN_process()(clocks) == process.simulate(U.uart_tx, clocks, U.TX)
    assert p.PY_process()(clocks) == p.GEN_process()(clocks)
    # (the byte read as it is, no copy of it; shift_out's least significant first: folded)
    assert sorted(p.process.locals) == ["data_v", "shift_out_i"]


def msb_out(port):
    port.line = 0
    while True:
        yield from wait_until(port, "go")
        yield from shift_out(port, "line", port.value, 8, 3, msb_first=True)
        port.line = 0


def test_steps_most_significant_bit_first():
    def tb(dut):
        dut.value, dut.go = 0xB4, 1
        got = yield from shift_in(dut, "line", 8, 3, msb_first=True)   # (each bit's last clock)
        dut.go = 0
        return got
    p = _quiet(msb_out, {"go": 1, "value": 8})
    assert p.check(tb, verilog=False).value == 0xB4


@pytest.mark.parametrize("sim", [False, pytest.param(True, marks=pytest.mark.sim)])
def test_spi_master_and_slave_exchange_bytes(sim, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    pairs = [(0xA5, 0x3C), (0x01, 0x80), (0xFF, 0x00)]
    p = _quiet(S.spi_pair())
    run = p.check(S.exchange(pairs), verilog=sim)
    assert run.value == [(s, m) for m, s in pairs]          # (each the other's byte)
    # (the bus wired both ways, miso back to the master: none of it a design's input)
    assert {q.name for q in p.ir.inputs} == {"start", "send", "reply", "clk", "rst", "ce"}


@pytest.mark.parametrize("sim", [False, pytest.param(True, marks=pytest.mark.sim)])
def test_ws2812_leds_get_their_colors(sim, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    colors = [0xFF0000, 0x00FF00, 0x0000FF, 0x123456]
    run = _quiet(W.ws2812, W.ATTR).check(W.show(colors), verilog=sim)
    line = [r["dout"] for r in run.outputs]
    assert W.led_strip(line) == colors
    highs = {len(s) for s in "".join(map(str, line)).split("0") if s}
    assert highs == {W.T0H, W.T1H}                          # (0.4 us and 0.8 us)


def wait_both(port):
    while True:
        port.out = 0
        yield from wait_until(port, "a", 1)
        yield from wait_until(port, "a", 0)        # (a can't be 1 and 0 in one clock: waits)
        port.out = 1
        yield


def test_an_input_keeps_its_value_through_a_clock():
    p = _quiet(wait_both, {"a": 1})
    clocks = [{"a": a} for a in [0, 1, 1, 0, 0, 1, 0]]
    assert p.PY_process()(clocks) == p.GEN_process()(clocks)
    assert [r["out"] for r in p.GEN_process()(clocks)] == [0, 0, 0, 0, 1, 0, 0]
