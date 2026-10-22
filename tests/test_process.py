import contextlib
import functools
import io
import random
import shutil
import sys
from pathlib import Path

import pytest

from py2ver import Py2ver, process

# Processes (process.py): a protocol written as a Python generator, a yield a
# clock, built as a state machine that does what the generator does, clock for
# clock.

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
import uart as U  # noqa: E402


def _quiet(f, attr):
    with contextlib.redirect_stdout(io.StringIO()):
        return Py2ver(f, attr)


def _line(data, bit=U.BIT, idle=5):
    """The line of a sender bit clocks a bit: a frame (start, 8 bits, stop) a byte."""
    out = [1] * idle
    for d in data:
        for b in [0] + [(d >> i) & 1 for i in range(8)] + [1]:
            out += [b] * bit
        out += [1] * idle
    return [{"rx": b} for b in out]


def _received(rows):
    return bytes(r["data"] for r in rows if r["valid"])


def test_a_uart_s_frames_go_through_as_the_generators_say():
    message = b"\x00\xffpy2ver\x55"
    line = [{"rx": r["tx"]} for r in process.simulate(U.uart_tx, U.frames(message), U.TX)]
    assert _received(process.simulate(U.uart_rx, line, U.RX)) == message


def test_a_process_s_ports_are_its_pins():
    # (its state's registers, their next values: no ports)
    tx, rx = _quiet(U.uart_tx, U.TX), _quiet(U.uart_rx, U.RX)
    assert [q.name for q in tx.ir.inputs] == ["data", "start", "clk", "rst", "ce"]
    assert [q.name for q in tx.ir.outputs] == ["tx", "busy"]
    assert [q.name for q in rx.ir.outputs] == ["valid", "data", "error"]


def test_the_machine_is_the_generator_clock_for_clock():
    tx = _quiet(U.uart_tx, U.TX)
    clocks = U.frames(b"\xa5\x0f")
    # (and a start while a frame goes out: not taken)
    clocks[2000] = {"start": 1, "data": 0x33}
    assert tx.PY_process()(clocks) == tx.GEN_process()(clocks)
    rx = _quiet(U.uart_rx, U.RX)
    r = random.Random(1)
    line = _line(b"ok") + [{"rx": 0}] * 40 + [{"rx": 1}] * 400       # (a glitch: no frame)
    line += [{"rx": r.randint(0, 1)} for _ in range(3000)]           # (noise)
    assert rx.PY_process()(line) == rx.GEN_process()(line)
    assert _received(rx.PY_process()(line))[:2] == b"ok"


@pytest.mark.parametrize("off", [-0.03, 0.03])
def test_a_receiver_takes_a_sender_three_percent_off(off):
    rx = _quiet(U.uart_rx, U.RX)
    message = bytes(range(0, 256, 37))
    rows = rx.PY_process()(_line(message, bit=round(U.BIT * (1 + off))))
    assert _received(rows) == message and not any(r["error"] for r in rows if r["valid"])


@pytest.mark.sim
@pytest.mark.skipif(not shutil.which("iverilog"), reason="needs Icarus")
def test_a_uart_in_hardware(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PY2VER_SIM", "icarus")
    tx = _quiet(U.uart_tx, U.TX)
    clocks = U.frames(b"\x5aA")
    sent = tx.TB_process()(clocks)
    assert sent == tx.PY_process()(clocks)
    rx = _quiet(U.uart_rx, U.RX)
    line = [{"rx": r["tx"]} for r in sent]
    got = rx.TB_process()(line)
    assert got == rx.PY_process()(line) and _received(got) == b"\x5aA"


# ----------------------------- what else a process may say -----------------------------

def pulses(port):
    """Pulses as long as port.n says, a gap after each; port.stop ends a train."""
    while True:
        port.out = 0
        while port.n == 0:
            yield
        n = port.n
        for k in range(3, 0, -1):            # (a range counting down)
            if port.stop:                    # (a break out of a loop that yields)
                break
            port.out = 1
            yield n                          # (a wait as long as a value)
            port.out = 0
            if k == 2:                       # (an if with a yield)
                yield 2 * n
            else:
                yield
        port.done = 1
        yield
        port.done = 0


PULSES = {"n": {"width": 4}, "stop": {"width": 1}, "out": {"width": 1}, "done": {"width": 1}}


def test_a_process_s_waits_breaks_and_ranges():
    r = random.Random(3)
    clocks = []
    for _ in range(40):
        clocks += [{"n": r.choice([0, 0, 1, 3, 7]), "stop": int(r.random() < 0.05)}] * r.randint(1, 9)
    p = _quiet(pulses, PULSES)
    gen = p.GEN_process()(clocks)
    assert p.PY_process()(clocks) == gen
    assert sum(r["out"] for r in gen) > 20 and sum(r["done"] for r in gen) > 2


@pytest.mark.sim
def test_a_process_s_waits_in_hardware(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    r = random.Random(4)
    clocks = [{"n": r.choice([0, 2, 5]), "stop": int(r.random() < 0.02)} for _ in range(300)]
    p = _quiet(pulses, PULSES)
    assert p.TB_process()(clocks) == p.PY_process()(clocks)


def spins(port):
    while True:
        while port.a:            # (waits without a yield: no clock passes)
            port.x = 1
        yield


def counts(port):
    k = 0
    while True:
        k = k + port.a           # (grows without end)
        port.x = k & 1
        yield


# ----------------------------- steps, parameters, processes together -----------------------------

def first_high(port, limit):
    """Clocks until port.a, limit at most: how many."""
    for k in range(limit):
        if port.a:
            return k                         # (a return out of a loop: the step's value)
        yield
    return limit


def pulse(port, n):
    port.out = 1
    yield n
    port.out = 0


def measures(port):
    while True:
        k = yield from first_high(port, 6)
        port.count = k
        yield from pulse(port, k + 1)        # (an expression: the step's local, set here)
        yield from pulse(port, 2)            # (one step's locals for both)
        if port.stop:                        # (a return without a yield: the end)
            return
        yield


MEASURES = {"a": {"width": 1}, "stop": {"width": 1}, "out": {"width": 1}, "count": {"width": 3}}


def _measured():
    r = random.Random(5)
    return [{"a": int(r.random() < 0.2), "stop": int(k > 250 and r.random() < 0.1)}
            for k in range(400)]


def test_a_step_is_a_generator_of_its_own():
    p = _quiet(measures, MEASURES)
    clocks = _measured()
    gen = p.GEN_process()(clocks)
    assert p.PY_process()(clocks) == gen
    assert {r["count"] for r in gen} >= {0, 1, 6} and gen[-1] == gen[-2]   # (ended)
    assert sorted(p.process.locals) == ["first_high_k", "k", "pulse_n"]


@pytest.mark.sim
def test_a_step_in_hardware(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    p = _quiet(measures, MEASURES)
    clocks = _measured()
    assert p.TB_process()(clocks) == p.GEN_process()(clocks)


def test_a_parameter_is_a_constant_a_partial_its_value():
    half = U.BIT // 2
    tx = _quiet(functools.partial(U.uart_tx, bit=half), U.TX)
    clocks = U.frames(b"ok")
    sent = tx.PY_process()(clocks)
    assert sent == tx.GEN_process()(clocks)
    line = [r["tx"] for r in sent]
    start = line.index(0)
    assert line[start:start + half + 1] == [0] * half + [1]       # ('o': its bit 0 high)
    rx = _quiet(functools.partial(U.uart_rx, bit=half), U.RX)
    assert _received(rx.PY_process()([{"rx": b} for b in line])) == b"ok"


def test_processes_together_echo_what_they_hear():
    line = _line(b"echo") + [{"rx": 1}] * (12 * U.BIT)
    gen = process.simulate(U.echo(), line)
    assert _received(process.simulate(U.uart_rx, [{"rx": r["tx"]} for r in gen], U.RX)) == b"echo"
    p = _quiet(U.echo(), None)                 # (each process's attr its own: none here)
    assert [q.name for q in p.ir.inputs] == ["rx", "clk", "rst", "ce"]
    assert {q.name: q.width for q in p.ir.outputs} == {"valid": 1, "data": 8, "error": 1,
                                                        "tx": 1, "busy": 1}
    assert p.PY_process()(line) == gen


@pytest.mark.sim
def test_processes_together_in_hardware(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    line = _line(b"A") + [{"rx": 1}] * (11 * U.BIT)
    p = _quiet(U.echo(), None)
    assert p.TB_process()(line) == process.simulate(U.echo(), line)


def selfish(port):
    port.x = 1
    yield from selfish(port)


def needs(port, n):
    port.x = 1
    yield n


def ranges(port):
    port.x = 1
    yield from range(3)


def test_what_steps_and_wires_can_t_be_says_so():
    one = {"x": {"width": 1}}
    with pytest.raises(process.ProcessError, match="uses itself"):
        process.compile_process(selfish, one)
    with pytest.raises(process.ProcessError, match="'n' has no value"):
        process.compile_process(needs, one)
    with pytest.raises(process.ProcessError, match="a generator function's call"):
        process.compile_process(ranges, one)
    with pytest.raises(process.ProcessError, match="tx.date isn't an input of tx"):
        process.connect(rx=U.uart_rx, tx=U.uart_tx, wires={"tx.date": "rx.data"})
    with pytest.raises(process.ProcessError, match="rx.rx isn't an output of rx"):
        process.connect(rx=U.uart_rx, tx=U.uart_tx, wires={"tx.data": "rx.rx"})


# ----------------------------- what was built, and how it ran -----------------------------

def test_a_process_says_what_it_built():
    text = _quiet(U.uart_rx, U.RX).explain()
    assert text.startswith("uart_rx: 6 states")                       # (no end: it never returns)
    assert "(the end" not in text
    assert "yield bit // 2" in text and "uart.py:" in text            # (each state: its yield)
    # (a step's: which, and where it's used)
    assert "(in wait_until from examples/uart.py:" in text and "(in read_byte from " in text
    assert "outputs  valid 1, data 8, error 1" in text                # (followed: no attr)
    assert "data_v: the local data (named apart from the pin port.data)" in text
    echo = _quiet(U.echo(), None).explain()
    assert "rx: uart_rx: 6 states" in echo and "tx: uart_tx: 5 states" in echo


def test_waveforms_and_where_two_runs_differ(tmp_path):
    p = _quiet(U.uart_tx, U.TX)
    clocks = U.frames(b"\x01")
    rows = p.GEN_process(vcd=str(tmp_path / "tx.vcd"))(clocks)
    vcd = (tmp_path / "tx.vcd").read_text()
    assert "$var wire 8 " in vcd and " data $end" in vcd and " clk $end" in vcd
    assert f"#{20 * len(clocks) - 10}" in vcd                     # (every clock's)
    assert process.difference(rows, p.PY_process()(clocks)) is None
    other = [dict(r) for r in rows]
    other[500]["tx"] ^= 1
    assert process.difference(rows, other) == f"clock 500: tx {rows[500]['tx']} / {other[500]['tx']}"


def test_an_error_says_where_in_the_source():
    with pytest.raises(process.ProcessError, match=r"test_process\.py:\d+: a while loop that"):
        process.compile_process(spins, {"a": 1})
    with pytest.raises(process.ProcessError, match=r"test_process\.py:\d+: 'k': its values grow"):
        process.compile_process(counts, {"a": 1})
    with pytest.raises(process.ProcessError, match=r"test_process\.py:\d+: selfish uses itself"):
        process.compile_process(selfish, {})


def test_what_a_process_can_t_be_says_so():
    with pytest.raises(process.ProcessError, match="width"):
        process.compile_process(U.uart_tx, {"tx": {"width": 1}})
    with pytest.raises(process.ProcessError, match="doesn't yield"):
        process.compile_process(spins, {"a": {"width": 1}, "x": {"width": 1}})
    with pytest.raises(process.ProcessError, match="give its width"):
        process.compile_process(counts, {"a": {"width": 1}, "x": {"width": 1}})
    # (its width given: built)
    process.compile_process(counts, {"a": {"width": 1}, "x": {"width": 1}, "k": {"width": 8}})
