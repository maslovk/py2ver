import contextlib
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


def test_what_a_process_can_t_be_says_so():
    with pytest.raises(process.ProcessError, match="width"):
        process.compile_process(U.uart_tx, {"tx": {"width": 1}})
    with pytest.raises(process.ProcessError, match="doesn't yield"):
        process.compile_process(spins, {"a": {"width": 1}, "x": {"width": 1}})
    with pytest.raises(process.ProcessError, match="give its width"):
        process.compile_process(counts, {"a": {"width": 1}, "x": {"width": 1}})
    # (its width given: built)
    process.compile_process(counts, {"a": {"width": 1}, "x": {"width": 1}, "k": {"width": 8}})
