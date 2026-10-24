import contextlib
import functools
import io
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from py2ver import Py2ver, process
from py2ver.protocol import (SPI, debounced, shift_in, shift_out, spi_transfer, together,
                             uart_receive, uart_send, wait_edge, wait_until)

# Steps of protocols (protocol.py), used by processes and by testbenches written as
# generators too (drive(), check()); SPI and WS2812 written from their specs.

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
import i2c as I2C  # noqa: E402
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


# ----------------------------- what the first users ran into -----------------------------

DEBOUNCE = 50_000


def knob(port, debounce=DEBOUNCE):
    """A rotary encoder: a debounced, a step each change of it, saturating."""
    a = 0
    ca = 0
    pos = 0
    port.position = 0
    while True:
        if port.a == a:
            ca = 0
        else:
            ca = ca + 1                      # (reset at debounce: 0..debounce)
            if ca == debounce:
                ca = 0
                a = port.a
                if a != port.b:
                    pos = min(pos + 1, 255)  # (saturating: 0..255)
                elif pos > 0:
                    pos = pos - 1
        port.position = pos
        yield


def pwm(port, n=2):
    port.pwm = 0
    while True:
        duty = port.duty
        for i in range(256):
            port.pwm = 1 if i < duty else 0
            yield n


def test_widths_follow_what_the_ifs_say():
    p = _quiet(knob, {"a": 1, "b": 1})
    w = {v: p.process.attr[v]["width"] for v in ["ca", "pos", "position"]}
    assert w == {"ca": DEBOUNCE.bit_length(), "pos": 8, "position": 8}
    small = _quiet(functools.partial(knob, debounce=3), {"a": 1, "b": 1})
    turn = [(1, 0), (1, 1), (0, 1), (0, 0)]                          # (clockwise: a step each a)
    clocks = [{"a": a, "b": b} for _ in range(140) for a, b in turn for _ in range(5)]
    gen = small.GEN_process()(clocks)
    assert small.PY_process()(clocks) == gen and max(r["position"] for r in gen) == 255


def test_min_and_max_inside_connect():
    p = _quiet(process.connect(enc=(functools.partial(knob, debounce=3), {"a": 1, "b": 1}),
                               drv=(pwm, {}), name="knob_pwm", wires={"drv.duty": "enc.position"}))
    clocks = [{"a": a, "b": b} for _ in range(20) for a, b in [(1, 0), (1, 1), (0, 1), (0, 0)]
              for _ in range(5)]
    assert p.PY_process()(clocks) == p.GEN_process()(clocks)


def test_a_pin_doesn_t_narrow_a_port_and_is_checked_at_once():
    with pytest.raises(ValueError, match=r"the port 'data' \(8 bits\): LED\[6:0\]: 7 pins for a "
                                         r"port of 8 bits \(give it 8 pins"):
        _quiet(U.uart_rx, {"rx": "GPIO_0_RX", "data": "LED[6:0]"})
    with pytest.raises(ValueError, match="the board has no GPIO_0"):
        _quiet(U.uart_rx, {"rx": "GPIO_0[0]"})                  # (when built, not at PINS())


def edge_with_timeout(port, limit):
    n = 0
    while port.sck == 0:                     # (left: sck is 1, it's a bit)
        if n == limit:
            return 0
        yield
        n += 1
    while port.sck == 1:
        if n == limit:
            return 0
        yield
        n += 1
    return 1


def bits_on_edges(port):
    while True:
        port.valid = 0
        data = 0
        ok = 1
        for i in range(10):
            ok = yield from edge_with_timeout(port, 1000)
            if not ok:
                break
            if i < 8:
                data = data | (port.dat << i)          # (i < 8 here: 8 bits)
        if ok:
            port.data = data
            port.valid = 1
            yield


def test_a_one_bit_input_not_0_is_1():
    p = _quiet(bits_on_edges, {"sck": 1, "dat": 1})
    assert p.process.attr["data"]["width"] == 8 and p.process.attr["edge_with_timeout_n"]["width"] == 10
    clocks = [{"sck": (k // 3) % 2, "dat": (k // 7) % 2} for k in range(400)]
    assert p.PY_process()(clocks) == p.GEN_process()(clocks)


def no_clock(port):
    while True:
        if port.a:
            yield
        port.x = 1


def two_counters(port):
    j = 0
    k = 0
    while True:
        j = j + port.a
        k = k + port.a
        port.x = (j + k) & 1
        yield


def test_errors_say_where_and_all_at_once():
    with pytest.raises(process.ProcessError, match=r"test_protocol\.py:\d+: no_clock: a loop that "
                                                   r"doesn't yield: in one clock it can go "
                                                   r"\S*test_protocol\.py:\d+ -> "):
        process.compile_process(no_clock, {"a": 1})
    with pytest.raises(process.ProcessError, match=r"'j' \(\S*test_protocol\.py:\d+\), 'k' "
                                                   r"\(\S*test_protocol\.py:\d+\) need: nothing"):
        process.compile_process(two_counters, {"a": 1})


def test_a_testbench_s_assert_says_its_clock():
    def tb(dut):
        dut.rx = 1
        yield 100
        assert dut.valid == 1, "nothing received"
    with pytest.raises(process.TestbenchFailed, match=r"the testbench at clock 100 \(tests/"
                                                      r"test_protocol\.py:\d+\): nothing received"):
        process.drive(U.echo(), tb)

    def forever(dut):
        yield from wait_until(dut, "valid")
    with pytest.raises(process.ProcessError, match=r"ran 50 clocks without returning, waiting at "
                                                   r".*\(forever\) -> py2ver/protocol\.py:\d+ "
                                                   r"\(wait_until\)"):
        process.drive(U.echo(), forever, limit=50)


def test_a_long_run_need_not_be_recorded():
    # (record=False: the same run, its value and clocks, none of its rows kept)
    def tb(dut):
        dut.rx = 1
        yield 250
        return dut.valid

    kept = process.drive(U.echo(), tb)
    light = process.drive(U.echo(), tb, record=False)
    assert light.value == kept.value and light.clocks == kept.clocks == len(kept.outputs) == 250
    assert light.inputs == [] and light.outputs == []


def test_a_testbench_stuck_in_together_says_which_step():
    def never(dut):
        yield from wait_until(dut, "error")            # (the echo's error: never)

    def tb(dut):
        dut.rx = 1
        yield from together(uart_send(dut, "rx", 0x41, U.BIT), never(dut))
    with pytest.raises(process.ProcessError, match=r"\(together\) \[\S*test_protocol\.py:\d+ "
                                                   r"\(never\) -> py2ver/protocol\.py:\d+ "
                                                   r"\(wait_until\)\]"):
        process.drive(U.echo(), tb, limit=12 * U.BIT)


@pytest.mark.sim
def test_a_check_says_which_simulator(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PY2VER_SIM", "icarus")
    assert _quiet(U.echo()).check(host(b"Z")).simulator == "icarus"


# ----------------------------- open-drain pins: I2C -----------------------------

def pulls(port):
    """An open-drain line pulled low two clocks of four; port.seen: the line as
    read (low when the other side pulls it too)."""
    while True:
        port.sda = 0
        port.seen = port.sda
        yield 2
        port.sda = 1                           # (let go: high, unless someone pulls it)
        port.seen = port.sda
        yield 2


def test_an_open_drain_pin_reads_its_line():
    p = _quiet(pulls, {"sda": {"open_drain": True}})
    assert [q.name for q in p.ir.inputs][:1] == ["sda_in"]
    assert {q.name for q in p.ir.outputs} == {"sda", "seen"}

    def tb(dut):
        lines = []
        for k in range(16):
            dut.sda = 0 if 8 <= k < 12 else 1  # (the testbench pulls it low a while)
            lines.append((dut.sda, dut.seen))
            yield
        return lines
    run = p.check(tb, verilog=False)
    # (the line: low when either side pulls it; the process reads it so)
    assert [line for line, _ in run.value] == [1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1]
    assert run.inputs[9]["sda_in"] == 0 and run.inputs[3]["sda_in"] == 1


@pytest.mark.parametrize("sim", [False, pytest.param(True, marks=pytest.mark.sim)])
def test_an_i2c_master_writes_and_reads_a_memory(sim, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    ops = [(0x50, 0x12, 0, 0xAB), (0x50, 0x12, 1, 0), (0x51, 0x00, 0, 0x55)]
    p = _quiet(functools.partial(I2C.i2c_master, bus=I2C.I2C(q=4)), I2C.ATTR)
    run = p.check(I2C.with_memory(ops, {}, 2, stretch=12), verilog=sim)
    done, log = run.value
    assert log == [("write", 0x12, 0xAB), ("read", 0x12, 0xAB)]
    assert [nack for _, nack in done] == [0, 0, 1] and done[1][0] == 0xAB   # (no 0x51: no ack)


def test_open_drain_pins_on_the_board():
    attr = dict(I2C.ATTR, scl=dict(I2C.ATTR["scl"], pin="GPIO_0[2]"),
                sda=dict(I2C.ATTR["sda"], pin="I2C_SDAT"))
    top = _quiet(I2C.i2c_master, attr).board_top()
    assert "assign GPIO_0[2] = scl_w ? 1'bz : 1'b0;" in top and "scl_in_s1 <= GPIO_0[2]" in top
    assert "inout I2C_SDAT" in top and "assign I2C_SDAT = sda_w ? 1'bz : 1'b0;" in top
    assert "reg sda_in_s1 = 1, sda_in_s2 = 1;" in top               # (a line: high, let go)
    with pytest.raises(ValueError, match="can't float"):
        _quiet(I2C.i2c_master, dict(I2C.ATTR, sda=dict(I2C.ATTR["sda"], pin="LED[0]")))
    with pytest.raises(process.ProcessError, match="open-drain pin: a bus of them isn't wired"):
        process.connect(a=(pulls, {"sda": {"open_drain": True}}), b=(U.uart_rx, U.RX),
                        wires={"b.rx": "a.sda"})


_OD_TB = """
`timescale 1ns/1ps
module tb;
    reg clk = 0;
    always #10 clk = !clk;
    wire [33:2] gpio;
    pullup(gpio[2]);                               // (the bus's pull-up)
    reg pull = 0;
    assign gpio[2] = pull ? 1'b0 : 1'bz;           // (another device on the line)
    wire [7:0] led;
    DE0_Nano board(.CLOCK_50(clk), .GPIO_0(gpio), .KEY(2'b11), .LED(led));
    integer k;
    initial begin
        repeat (40) @(posedge clk);
        for (k = 0; k < 16; k = k + 1) begin
            @(posedge clk); #1 $display("LINE %0d", gpio[2]);
        end
        pull = 1;
        repeat (8) @(posedge clk);
        #1 $display("PULLED %0d SEEN %0d", gpio[2], led[0]);
        $finish;
    end
endmodule
"""


@pytest.mark.sim
@pytest.mark.skipif(not shutil.which("iverilog"), reason="needs Icarus")
def test_an_open_drain_pin_floats_or_pulls_low_on_the_board(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    d = _quiet(pulls, {"sda": {"open_drain": True, "pin": "GPIO_0[2]"},
                       "seen": "LED[0]"}).gen_pins_project()
    (d / "tb.v").write_text(_OD_TB, encoding="utf-8")
    subprocess.run(["iverilog", "-o", "tb.vvp", "tb.v", "DE0_Nano.v", "pulls.v"], cwd=d, check=True)
    out = subprocess.run(["vvp", "-n", "tb.vvp"], cwd=d, check=True, capture_output=True,
                         text=True).stdout.splitlines()
    lines = [int(l.split()[1]) for l in out if l.startswith("LINE")]
    assert set(lines) == {0, 1} and lines.count(0) == lines.count(1)   # (low two of four clocks)
    assert "PULLED 0 SEEN 0" in out                       # (pulled by the other: seen low)


def spi_loop(port, mode=0):
    """A word out and in with spi_transfer, in mode, its result kept."""
    port.sclk = mode >> 1                  # (its idle level)
    port.cs_n = 1
    while True:
        yield from wait_until(port, "go")
        word = port.word
        port.cs_n = 0
        yield 3
        got = yield from spi_transfer(port, SPI(3, mode), word, 12)   # (a record made here)
        port.cs_n = 1
        port.got = got
        port.done = 1
        yield
        port.done = 0


@pytest.mark.parametrize("mode", [0, 1, 2, 3])
def test_spi_in_each_mode(mode):
    def tb(dut):                           # (MISO: MOSI, looped back; SCLK's edges counted)
        yield                              # (its outputs as it set them: a clock on)
        dut.word, dut.go = 0xA5C, 1
        edges, last, idle = 0, dut.sclk, dut.sclk
        while True:
            dut.miso = dut.mosi
            if dut.sclk != last:
                edges += 1
                last = dut.sclk
            if dut.done:
                break
            yield
            dut.go = 0
        return dut.got, edges, idle, dut.sclk
    p = _quiet(functools.partial(spi_loop, mode=mode), {"go": 1, "word": 12, "miso": 1})
    got, edges, idle, end = p.check(tb, verilog=False).value
    assert got == 0xA5C and edges == 24 and idle == end == mode >> 1   # (CPOL: idle level)


def bouncing_switch(port):
    """A switch's count of presses, read through a debouncer (connect())."""
    count = 0
    port.count = 0
    while True:
        yield from wait_edge(port, "key")
        count = (count + 1) & 15
        port.count = count


def test_edges_and_a_debouncer():
    design = process.connect(db=(functools.partial(debounced, clocks=4), {"raw": 1}),
                             sw=(bouncing_switch, {}), name="presses",
                             wires={"sw.key": "db.clean"})
    p = _quiet(design)

    def tb(dut):
        for _ in range(3):                             # (three presses, each bouncing)
            for v in [1, 0, 1, 0, 0, 1]:
                dut.raw = v
                yield
            yield 10
            for v in [0, 1, 0, 1, 1, 0]:
                dut.raw = v
                yield
            yield 10
        return dut.count
    assert p.check(tb, verilog=False).value == 3
