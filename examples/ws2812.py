# WS2812 (NeoPixel) LEDs, from their spec to the FPGA: the driver written as the
# datasheet says, a Python generator in which a yield is a clock (see "Processes"
# in the README), checked by a testbench that reads its line back as an LED does.
#
# The line: each bit a high pulse then low, 1.25 us in all: a 0 high 0.4 us, a 1
# high 0.8 us (each +-0.15 us); 24 bits an LED, green, red, blue, most significant
# first; the line low 50 us or more (280 us for newer LEDs): the LEDs show what
# they got.
#
#   python examples/ws2812.py           # the testbench: the generator, the model
#   python examples/ws2812.py --sim     # and the Verilog, the same clock for clock
#   python examples/ws2812.py --syn     # in Quartus
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py2ver.protocol import wait_until  # noqa: E402

CLOCK = 50_000_000
NS = CLOCK / 1e9                          # clocks a nanosecond
T0H, T0L = round(400 * NS), round(850 * NS)      # a 0: 20 clocks high, 42 low
T1H, T1L = round(800 * NS), round(450 * NS)      # a 1: 40 high, 22 low
RESET = round(300_000 * NS)               # 300 us low: 15,000 clocks


def ws2812(port):
    """port.start (a clock high, port.ready high): port.color (24 bits, GRB) to the
    next LED on port.dout; port.last too: then the reset, the LEDs show it."""
    port.dout = 0
    while True:
        port.ready = 1
        yield from wait_until(port, "start")
        grb = port.color
        last = port.last
        port.ready = 0
        for i in range(24):
            port.dout = 1
            if (grb >> (23 - i)) & 1:
                yield T1H
                port.dout = 0
                yield T1L
            else:
                yield T0H
                port.dout = 0
                yield T0L
        if last:
            yield RESET


ATTR = {"start": 1, "color": 24, "last": 1}


def show(colors):
    """A testbench: the colors (0xGGRRBB each) sent, the last one last."""
    def tb(dut):
        for k, c in enumerate(colors):
            yield from wait_until(dut, "ready")
            dut.color, dut.last, dut.start = c, int(k == len(colors) - 1), 1
            yield
            dut.start = 0
        yield
        yield from wait_until(dut, "ready")
    return tb


def led_strip(line):
    """What LEDs make of the line (dout a clock each): each high pulse a bit, a 1
    if longer than 0.6 us; the colors, 24 bits each, up to a reset."""
    bits, colors, high, low = [], [], 0, 0
    for v in line + [0]:
        if v:
            if low and high:                  # (a pulse and its low: a bit)
                bits.append(int(high > 600 * NS))
                high = 0
            low = 0
            high += 1
        else:
            if high and not low:
                bits.append(int(high > 600 * NS))
                high = 0
            low += 1
            if low == round(50_000 * NS):         # (a reset)
                break
    for k in range(0, len(bits) - len(bits) % 24, 24):
        colors.append(int("".join(map(str, bits[k:k + 24])), 2))
    return colors


if __name__ == "__main__":
    import contextlib
    import io
    from py2ver import Py2ver

    colors = [0xFF0000, 0x00FF00, 0x0000FF, 0x123456]
    with contextlib.redirect_stdout(io.StringIO()):
        p = Py2ver(ws2812, ATTR)
    run = p.check(show(colors), verilog="--sim" in sys.argv[1:])
    got = led_strip([r["dout"] for r in run.outputs])
    print(f"{len(run.inputs):,} clocks; the LEDs got {[hex(c) for c in got]} "
          f"(sent {[hex(c) for c in colors]})")
    print("the same in the model" + (" and the Verilog" if "--sim" in sys.argv[1:] else ""))
    if "--syn" in sys.argv[1:]:
        print(p.SYN({"QUARTUS_DIR": os.environ.get("QUARTUS_DIR",
                                                   "/opt/intelFPGA/22.1std/quartus/bin")}))
