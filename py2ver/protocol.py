"""
Steps of protocols, each a generator: used with `yield from` in a process (built
in it, see process.py) or in a testbench (drive(), check(): run as Python).
A step's pin is given by its name, a constant:

    yield from uart_send(port, "tx", data, BIT)       # a process sending a byte
    yield from uart_send(dut, "rx", 0x41, BIT)         # a testbench, to the design

    byte = yield from uart_receive(port, "rx", BIT)

debounced() is a process of its own, to connect() before a switch or a slow
line: connect(db=(functools.partial(debounced, clocks=50_000), {'raw': 1}), ...).
"""


def wait_until(port, pin, value=1):
    """Clocks pass until the pin is value (none, if it already is)."""
    while getattr(port, pin) != value:
        yield


def wait_edge(port, pin, rising=True):
    """Clocks pass until the pin's next edge: rising (low, then high) or, with
    rising=False, falling. A clock-following receiver's step (PS/2, SPI)."""
    yield from wait_until(port, pin, 0 if rising else 1)
    yield from wait_until(port, pin, 1 if rising else 0)


def debounced(port, clocks):
    """A process (connect() it before what reads the pin): port.clean follows
    port.raw once raw has kept its new value clocks clocks (a switch's bounces,
    a slow edge's: none get through)."""
    level = 0
    n = 0
    port.clean = 0
    while True:
        if port.raw == level:
            n = 0
        else:
            n = n + 1
            if n == clocks:
                level = port.raw
                n = 0
                port.clean = level
        yield


def wait_cycles(n):
    """n clocks pass."""
    yield n


def shift_out(port, pin, value, bits, clocks, msb_first=False):
    """value's bits on the pin, each for clocks clocks: the least significant
    first, or (msb_first) the most."""
    for i in range(bits):
        if msb_first:
            setattr(port, pin, (value >> (bits - 1 - i)) & 1)
        else:
            setattr(port, pin, (value >> i) & 1)
        yield clocks


def shift_in(port, pin, bits, clocks, msb_first=False):
    """bits bits from the pin, each taken clocks clocks after the one before (the
    first: clocks from now); the value, its first bit the least significant or
    (msb_first) the most."""
    value = 0
    for i in range(bits):
        yield clocks
        if msb_first:
            value = value | (getattr(port, pin) << (bits - 1 - i))
        else:
            value = value | (getattr(port, pin) << i)
    return value


def together(*steps):
    """Steps at once, a clock at a time, in a testbench (a process is one sequence:
    there, connect() processes); their values, when all have returned:

        _, got = yield from together(uart_send(dut, "rx", b, BIT),
                                     uart_receive(dut, "tx", BIT))
    """
    waits = [0] * len(steps)
    values = [None] * len(steps)
    done = [False] * len(steps)
    while True:
        for k, step in enumerate(steps):
            if not done[k] and waits[k] == 0:
                try:
                    n = next(step)
                    waits[k] = 1 if n is None else int(n)
                except StopIteration as e:
                    done[k], values[k] = True, e.value
        if all(done):
            return values
        yield
        waits = [w - 1 for w in waits]


def uart_send(port, pin, byte, bit):
    """A UART frame (8N1) of byte on the pin, bit clocks a bit: the start bit
    low, eight bits least significant first, the stop bit high."""
    setattr(port, pin, 0)
    yield bit
    yield from shift_out(port, pin, byte, 8, bit)
    setattr(port, pin, 1)
    yield bit


def uart_receive(port, pin, bit):
    """A UART frame's byte from the pin: from its start bit's edge, each bit taken
    at its middle; it returns at the stop bit's middle."""
    yield from wait_until(port, pin, 0)
    yield bit // 2                                   # (the start bit's middle)
    byte = yield from shift_in(port, pin, 8, bit)    # (each bit's middle)
    yield bit                                        # (the stop bit's middle)
    return byte
