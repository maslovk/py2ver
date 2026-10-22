import contextlib
import io
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from py2ver import Py2ver
from py2ver.pins import board_pins, parse_pin

# A design on the DE0-Nano's own pins (pins.py): each port's attr 'pin', the
# board's top (DE0_Nano) around the design.

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
import uart as U  # noqa: E402


def _quiet(f, attr):
    with contextlib.redirect_stdout(io.StringIO()):
        return Py2ver(f, attr)


def _receiver():
    """The UART's receiver: rx on GPIO_0[0], the byte on the LEDs."""
    return _quiet(U.uart_rx, dict(U.RX, rx=dict(U.RX["rx"], pin="GPIO_0[0]"),
                                  data=dict(U.RX["data"], pin="LED[7:0]")))


def test_a_pin_is_the_board_s():
    pins = board_pins()
    assert pins["LED"] == 8 and pins["GPIO_0"] == 34 and pins["KEY"] == 2
    assert parse_pin("LED[3:0]", 4, pins) == ["LED[0]", "LED[1]", "LED[2]", "LED[3]"]
    assert parse_pin("LED", 2, pins) == ["LED[0]", "LED[1]"]
    assert parse_pin("GPIO_0[5]", 1, pins) == ["GPIO_0[5]"]
    assert parse_pin("GPIO_0_IN[1]", 1, pins) == ["GPIO_0_IN[1]"]
    for spec, width, says in [("LEDS[0]", 1, "No DE0-Nano pin"), ("LED[3:0]", 8, "4 pins"),
                              ("LED[0:3]", 4, "high:low"), ("LED[9:8]", 2, "has 8 pins")]:
        with pytest.raises(ValueError, match=says):
            parse_pin(spec, width, pins)


def test_a_pin_goes_to_one_port_the_way_it_can():
    attr = dict(U.RX, rx=dict(U.RX["rx"], pin="LED[0]"))
    with pytest.raises(ValueError, match="an output pin"):
        _quiet(U.uart_rx, attr).board_top()
    attr = dict(U.RX, rx=dict(U.RX["rx"], pin="GPIO_0[0]"), data=dict(U.RX["data"], pin="GPIO_0[7:0]"))
    with pytest.raises(ValueError, match="both"):
        _quiet(U.uart_rx, attr).board_top()
    attr = dict(U.RX, error=dict(U.RX["error"], pin="SW[0]"))
    with pytest.raises(ValueError, match="an input pin"):
        _quiet(U.uart_rx, attr).board_top()


def test_the_board_s_top_takes_the_design_between_its_pins():
    top = _receiver().board_top()
    assert "module DE0_Nano(" in top and "inout [33:0] GPIO_0" in top and "output [7:0] LED" in top
    assert "rx_s1 <= GPIO_0[0]" in top and ".rx(rx_s2)" in top     # (through two registers)
    assert "assign LED[7] = data_w[7];" in top
    assert "|| !KEY[0]" in top                                       # (KEY[0] resets)


def test_processes_together_take_the_board_s_pins():
    top = _quiet(U.echo(), U.ECHO_PINS).board_top()
    assert "rx_s1 <= GPIO_0[0]" in top and "assign GPIO_0[1] = tx_w;" in top
    assert "assign GPIO_0[0] = 1'bz;" in top and "uart_echo u_design(" in top


_TB = """
`timescale 1ns/1ps
module tb;
    reg clk = 0;
    always #10 clk = !clk;
    reg line = 1;
    wire [33:0] gpio;
    assign gpio[0] = line;
    wire [7:0] led;
    DE0_Nano board(.CLOCK_50(clk), .GPIO_0(gpio), .KEY(2'b11), .LED(led));
    task send(input [7:0] b);
        integer i;
        begin
            line = 0; repeat (BIT) @(posedge clk);
            for (i = 0; i < 8; i = i + 1) begin line = b[i]; repeat (BIT) @(posedge clk); end
            line = 1; repeat (BIT) @(posedge clk);
        end
    endtask
    initial begin
        repeat (50) @(posedge clk);
        send(8'ha5); repeat (20) @(posedge clk); $display("LED %h", led);
        send(8'h3c); repeat (20) @(posedge clk); $display("LED %h", led);
        $finish;
    end
endmodule
"""


@pytest.mark.sim
@pytest.mark.skipif(not shutil.which("iverilog"), reason="needs Icarus")
def test_a_receiver_on_the_board_s_pins_lights_the_byte(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    d = _receiver().gen_pins_project()
    (d / "tb.v").write_text(_TB.replace("BIT", str(U.BIT)), encoding="utf-8")
    subprocess.run(["iverilog", "-o", "tb.vvp", "tb.v", "DE0_Nano.v", "uart_rx.v"],
                   cwd=d, check=True)
    out = subprocess.run(["vvp", "-n", "tb.vvp"], cwd=d, check=True, capture_output=True,
                         text=True).stdout
    assert [l for l in out.splitlines() if l.startswith("LED")] == ["LED a5", "LED 3c"]


def test_a_receiver_on_the_board_s_pins_meets_50_mhz(quartus):
    r = _receiver().PINS(quartus)
    assert r.device == "EP4CE22F17C6" and r.fmax_mhz > 50 and r.registers > 20
