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


def divides(port, n, d):
    """n // d and n % d, by subtraction a clock at a time."""
    q = 0
    while n >= d:
        n = n - d
        q = q + 1
        yield
    return q, n


def swapped(port, x, y):
    yield
    return y, x


def early(port, x):
    if x > 100:
        return 0, 1                          # (before its yield)
    yield
    return x, x + 1


def divider(port):
    a, b = 1, 2
    while True:
        q, r = yield from divides(port, port.n, port.d)
        port.q = q
        port.r = r
        a, b = yield from swapped(port, a, b)   # (its names the caller's own: at once, as Python)
        port.a = a
        lo, hi = yield from early(port, port.n)
        port.lo = lo
        port.hi = hi
        yield


DIVIDER = {"n": 8, "d": 8, "q": 8, "r": 8, "a": 2, "b": 2, "lo": 8, "hi": 9, "divides_q": 8}


SEGMENT = 100       # (clocks each n, d is held: a pass, 35 at most with d >= 8, done in it)


def _divisions():
    r = random.Random(3)
    pairs = [(r.randrange(256), r.randrange(8, 40)) for _ in range(12)]
    return pairs, [{"n": n, "d": d} for n, d in pairs for _ in range(SEGMENT)]


def test_a_step_returns_several_values():
    p = _quiet(divider, DIVIDER)
    pairs, clocks = _divisions()
    gen = p.GEN_process()(clocks)
    assert p.PY_process()(clocks) == gen
    for k, (n, d) in enumerate(pairs):
        g = gen[(k + 1) * SEGMENT - 1]
        assert (g["q"], g["r"]) == divmod(n, d)
        assert (g["lo"], g["hi"]) == ((0, 1) if n > 100 else (n, n + 1))
    assert {g["a"] for g in gen[SEGMENT:]} == {1, 2}               # (swapped: each time)


@pytest.mark.sim
def test_a_step_s_values_in_hardware(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    p = _quiet(divider, DIVIDER)
    clocks = _divisions()[1]
    assert p.TB_process()(clocks) == p.GEN_process()(clocks)


def one_value(port):
    yield
    return 5


def wants_two(port):
    while True:
        a, b = yield from one_value(port)
        port.x = a
        yield


def wants_one(port):
    while True:
        x = yield from swapped(port, 1, 0)
        port.x = x
        yield


def test_values_a_step_returns_are_as_many_as_assigned():
    one = {"x": {"width": 1}}
    with pytest.raises(process.ProcessError, match="one_value returns one value .*, 2 values "
                                                   "assigned here"):
        process.compile_process(wants_two, one)
    with pytest.raises(process.ProcessError, match="swapped returns 2 values .*, one value "
                                                   "assigned here"):
        process.compile_process(wants_one, one)


from dataclasses import dataclass  # noqa: E402
from typing import NamedTuple  # noqa: E402


class Line(NamedTuple):
    pin: str = "out"
    clocks: int = 3


@dataclass(frozen=True)
class Board:
    line: Line
    level: int = 1


BOARD = Board(Line("out", 2))


def pulse_on(port, line, level):
    setattr(port, line.pin, level)
    yield line.clocks
    setattr(port, line.pin, 1 - level)
    yield line.clocks


def board_pulses(port, board=BOARD):
    while True:
        yield from pulse_on(port, board.line, board.level)     # (a record's field, a record)
        yield from pulse_on(port, BOARD.line, 0)               # (the module's)
        yield from pulse_on(port, Line("out", board.line.clocks + 1), 1)   # (made here)
        yield


def test_a_record_s_fields_are_constants_where_it_s_used():
    for p, first in [(_quiet(board_pulses, {"out": 1}), 2),
                     (_quiet(functools.partial(board_pulses, board=Board(Line("out", 4))),
                             {"out": 1}), 4)]:
        clocks = [{}] * 60
        gen = p.GEN_process()(clocks)
        assert p.PY_process()(clocks) == gen
        line = [g["out"] for g in gen]
        assert line[1:1 + first + 1] == [1] * first + [0]          # (its clocks: the record's)
    assert "board" not in p.process.locals and "line" not in " ".join(p.process.locals)


def assigns_a_field(port, line=Line()):
    line.clocks = 4
    yield


def keeps_a_record(port, line=Line()):
    x = line
    port.out = 1
    yield x


def wrong_field(port, line=Line()):
    yield line.clock


def not_constants(port):
    n = port.n
    yield from pulse_on(port, Line("out", n), 1)


def test_a_record_is_constants_or_says_so():
    one = {"out": {"width": 1}}
    with pytest.raises(process.ProcessError, match="Line's fields are constants: 'clocks' can't"):
        process.compile_process(assigns_a_field, one)
    with pytest.raises(process.ProcessError, match="a Line is a record of constants: read its "
                                                   "fields"):
        process.compile_process(keeps_a_record, one)
    with pytest.raises(process.ProcessError, match="Line has no field 'clock'"):
        process.compile_process(wrong_field, one)
    with pytest.raises(process.ProcessError, match=r"Line\(\.\.\.\) of constants"):
        process.compile_process(not_constants, {"out": 1, "n": 4})
    with pytest.raises(process.ProcessError, match="a record of them"):
        process.compile_process(functools.partial(board_pulses, board=1.5), one)


import py2ver.protocol as P  # noqa: E402
from py2ver.devices import adxl345 as ADXL  # noqa: E402


def by_module(port):
    while True:
        yield from P.wait_until(port, "go")                 # (a module's step)
        port.out = ADXL.MEASURE                             # (a module's constant)
        yield P.I2C(q=3).q                                  # (a module's record, made here)
        port.out = 0
        yield


def test_steps_constants_and_records_by_their_module():
    p = _quiet(by_module, {"go": 1, "out": 4})
    clocks = [{"go": 0}] * 3 + [{"go": 1}] + [{"go": 0}] * 8
    gen = p.GEN_process()(clocks)
    assert p.PY_process()(clocks) == gen
    assert [g["out"] for g in gen] == [0] * 4 + [8] * 3 + [0] * 5


def sets_then_waits(port):
    port.a = 1                   # a constant before the first yield: a's value from reset
    port.b = 5
    port.c = port.x              # not a constant: c starts at 0
    if port.x:
        port.d = 1               # after an if: d starts at 0
    yield
    while True:
        port.a = 0
        port.b = port.x
        yield


def _starts(p):
    clocks = [{"x": 2}] * 4
    gen = p.GEN_process()(clocks)
    assert p.PY_process()(clocks) == gen
    return gen


def test_an_output_starts_at_what_is_set_before_the_first_yield():
    first = _starts(_quiet(sets_then_waits, {"x": 3}))[0]
    assert (first["a"], first["b"], first["c"], first["d"]) == (1, 5, 0, 0)
    overridden = _starts(_quiet(sets_then_waits, {"x": 3, "a": {"init": 0}}))[0]
    assert overridden["a"] == 0                                 # (attr's 'init' wins)


@pytest.mark.sim
def test_an_output_s_value_from_reset_in_hardware(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    p = _quiet(sets_then_waits, {"x": 3})
    clocks = [{"x": 2}] * 4
    assert p.TB_process()(clocks) == p.GEN_process()(clocks)


def walks(port):
    k = 0
    while True:
        port.walk = 1 << k
        yield 3
        k = 0 if k == 4 else k + 1          # (a conditional expression: its test bounds k)


def test_a_conditional_expression_s_test_bounds_its_branches():
    p = _quiet(walks, None)
    assert p.process.attr["k"]["width"] == 3 and p.process.attr["walk"]["width"] == 5
    clocks = [{}] * 40
    gen = p.GEN_process()(clocks)
    assert p.PY_process()(clocks) == gen
    assert {g["walk"] for g in gen} == {0, 1, 2, 4, 8, 16}


def halve_until_small(port, x):
    while x > 3:
        x = x >> 1                           # (a parameter the step changes: a local too)
        yield
    return x


def halves(port):
    while True:
        if port.pick:
            a = yield from halve_until_small(port, port.v)
        else:
            a = yield from halve_until_small(port, 0)
        port.out = a
        yield


def test_a_step_s_changed_parameter_is_one_local():
    # (it was named twice, halve_until_small2_x, which made {'x': ...} in attr
    # look like a local of two steps)
    p = _quiet(halves, {"v": 8, "pick": 1, "x": 9})
    assert p.process.locals == ["halve_until_small_x", "a"]     # (both calls: one register)
    assert p.process.attr["halve_until_small_x"]["width"] == 9  # ('x' in attr: its own name)
    clocks = [{"v": v, "pick": 1} for v in (200, 7, 64, 3)] * 5
    assert p.PY_process()(clocks) == p.GEN_process()(clocks)


PATH = tuple((k * 7919) % 2001 - 1000 for k in range(100))      # (signed entries)


def follows_path(port):
    k = 0
    x = 0
    while True:
        if port.hold:
            x = 0                            # (a constant, or the table's entry)
        else:
            x = PATH[k]
        yield                                # (x used only after a yield: a memory block)
        port.out = x * 3
        k = 0 if k == len(PATH) - 1 else k + 1


def reads_path_at_once(port):
    k = 0
    while True:
        port.out = PATH[k] + 1               # (used in the clock it's read: logic)
        yield 2
        k = 0 if k == len(PATH) - 1 else k + 1


_PATH_CLOCKS = [{"hold": int(i % 37 < 5)} for i in range(250)]


def test_a_table_whose_entry_waits_for_a_clock_is_a_memory_block():
    p = _quiet(follows_path, {"hold": 1})
    assert "x__q <= x__rom[x__q__address];" in p.verilog
    gen = p.GEN_process()(_PATH_CLOCKS)
    assert p.PY_process()(_PATH_CLOCKS) == gen
    assert {g["out"] for g in gen} >= {0, 3 * PATH[1], 3 * min(PATH)}
    at_once = _quiet(reads_path_at_once, None)
    assert "// Table: 100 x 11 bits" in at_once.verilog
    assert "memory block" not in at_once.verilog


def reads_from_the_end(port):
    k = 0
    while True:
        x = PATH[k - 50]                     # (negative for k < 50: from the end, as in Python)
        yield
        port.out = x
        k = 0 if k == len(PATH) - 1 else k + 1


def test_a_table_s_negative_index_counts_from_the_end():
    p = _quiet(reads_from_the_end, None)
    assert "Table in a memory block: 100 x 11 bits" in p.verilog
    clocks = [{}] * 250
    gen = p.GEN_process()(clocks)
    assert p.PY_process()(clocks) == gen
    assert PATH[-50] in {g["out"] for g in gen}


def keeps_values(port):
    values = [0] * 8                         # (a list local: a register per element)
    while True:
        for i in range(8):
            values[i] = port.x + i           # (a variable index: the element it is)
            yield
        values[-1] = values[0] + 100         # (a constant one, from the end as in Python)
        total = 0
        for i in range(len(values)):
            v = values[i]
            yield
            total = total + v
        port.out = total
        yield


def test_a_list_local_is_a_register_per_element():
    p = _quiet(keeps_values, {"x": 8, "total": 12})
    assert [n for n in p.process.locals if n.startswith("values")] == [f"values_{k}" for k in range(8)]
    assert p.process.attr["values_0"]["width"] == 8                 # (x + 0: each its own width)
    assert p.process.attr["values_7"]["width"] == 9
    clocks = [{"x": k} for k in range(60)]
    assert p.PY_process()(clocks) == p.GEN_process()(clocks)
    steady = p.GEN_process()([{"x": 4}] * 40)
    assert steady[-1]["out"] == sum(range(4, 11)) + 104             # (4..10, and 4 + 100 last)


def whole_list(port):
    values = [0] * 4
    port.out = values
    yield


def two_lengths(port):
    values = [0] * 4
    yield
    values = [0] * 5
    yield


def test_what_a_list_local_can_t_do_says_so():
    with pytest.raises(process.ProcessError, match="used by its elements"):
        _quiet(whole_list, None)
    with pytest.raises(process.ProcessError, match="different lengths"):
        _quiet(two_lengths, None)


@pytest.mark.sim
def test_a_table_in_a_memory_block_in_hardware(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    p = _quiet(follows_path, {"hold": 1})
    assert p.TB_process()(_PATH_CLOCKS) == p.GEN_process()(_PATH_CLOCKS)
    q = _quiet(reads_path_at_once, None)
    assert q.TB_process()([{}] * 250) == q.GEN_process()([{}] * 250)
    end = _quiet(reads_from_the_end, None)
    assert end.TB_process()([{}] * 250) == end.GEN_process()([{}] * 250)
    kept = _quiet(keeps_values, {"x": 8, "total": 12})
    clocks = [{"x": k} for k in range(60)]
    assert kept.TB_process()(clocks) == kept.GEN_process()(clocks)


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
    assert "where it ends" not in text
    assert "yield bit // 2" in text and "uart.py:" in text            # (each state: its yield)
    # (a step's: which, and where it's used)
    assert "(in wait_until, called from examples/uart.py:" in text
    assert "(in read_byte, called from " in text
    assert "outputs  valid 1, data 8, error 1" in text                # (followed: no attr)
    assert ("Note: data_v is the local variable data, renamed so that it doesn't clash with "
            "the pin port.data.") in text
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
    with pytest.raises(process.ProcessError,
                       match=r"test_process\.py:\d+: py2ver can't tell how many bits 'k' needs"):
        process.compile_process(counts, {"a": 1})
    with pytest.raises(process.ProcessError, match=r"test_process\.py:\d+: selfish uses itself"):
        process.compile_process(selfish, {})


def test_what_a_process_can_t_be_says_so():
    with pytest.raises(process.ProcessError, match="width"):
        process.compile_process(U.uart_tx, {"tx": {"width": 1}})
    with pytest.raises(process.ProcessError, match="doesn't yield"):
        process.compile_process(spins, {"a": {"width": 1}, "x": {"width": 1}})
    with pytest.raises(process.ProcessError, match="can't tell how many bits .* Give it a width"):
        process.compile_process(counts, {"a": {"width": 1}, "x": {"width": 1}})
    # (its width given: built)
    process.compile_process(counts, {"a": {"width": 1}, "x": {"width": 1}, "k": {"width": 8}})
