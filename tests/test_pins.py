import contextlib
import io
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from py2ver import Py2ver
from py2ver.pins import board_pins, parse_pin, pin_width

# A design on the DE0-Nano's own pins (pins.py): each port's attr 'pin', the
# board's top (DE0_Nano) around the design.

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
import uart as U  # noqa: E402


def _quiet(f, attr):
    with contextlib.redirect_stdout(io.StringIO()):
        return Py2ver(f, attr)


def _receiver():
    """The UART's receiver: rx on the board's serial pin, the byte on the LEDs."""
    return _quiet(U.uart_rx, {"rx": "GPIO_0_RX", "data": "LED[7:0]"})


def test_a_pin_is_the_board_s():
    pins = board_pins()
    assert pins["LED"] == list(range(8)) and pins["KEY"] == [0, 1]
    # (GPIO_0's pins 0 and 1: the board's serial pins, by their own names)
    assert pins["GPIO_0"] == list(range(2, 34)) and pins["GPIO_0_RX"] == pins["GPIO_0_TX"] == []
    assert parse_pin("LED[3:0]", 4, pins) == ["LED[0]", "LED[1]", "LED[2]", "LED[3]"]
    assert parse_pin("LED", 2, pins) == ["LED[0]", "LED[1]"]
    assert parse_pin("GPIO_0", 2, pins) == ["GPIO_0[2]", "GPIO_0[3]"]
    assert parse_pin("GPIO_0[5]", 1, pins) == ["GPIO_0[5]"]
    assert parse_pin("GPIO_0_IN[1]", 1, pins) == ["GPIO_0_IN[1]"]
    for spec, width, says in [("LEDS[0]", 1, "No DE0-Nano pin"), ("LED[3:0]", 8, "4 pins"),
                              ("LED[0:3]", 4, "high:low"), ("LED[9:8]", 2, "no LED\\[8\\]"),
                              ("GPIO_0[0]", 1, "no GPIO_0\\[0\\].*GPIO_0_RX, GPIO_0_TX")]:
        with pytest.raises(ValueError, match=says):
            parse_pin(spec, width, pins)


def test_a_pin_goes_to_one_port_the_way_it_can():
    with pytest.raises(ValueError, match="an output pin"):
        _quiet(U.uart_rx, {"rx": "LED[0]"}).board_top()
    with pytest.raises(ValueError, match="both"):
        _quiet(U.uart_rx, {"rx": "GPIO_0[2]", "data": "GPIO_0[9:2]"}).board_top()
    with pytest.raises(ValueError, match="an input pin"):
        _quiet(U.uart_rx, {"rx": 1, "error": "SW[0]"}).board_top()


def test_a_pin_gives_its_port_its_width():
    assert [pin_width(s) for s in ["LED[7:0]", "LED[2]", "LED", "GPIO_0", "GPIO_0_TX"]] \
        == [8, 1, 8, 32, 1]
    p = _receiver()                         # (data: 8 bits from LED[7:0])
    assert {q.name: q.width for q in p.ir.outputs} == {"valid": 1, "data": 8, "error": 1}


def test_the_board_s_top_takes_the_design_between_its_pins():
    top = _quiet(U.uart_rx, {"rx": "GPIO_0[2]", "data": "LED"}).board_top()
    assert "module DE0_Nano(" in top and "inout [33:2] GPIO_0" in top and "output [7:0] LED" in top
    assert "rx_s1 <= GPIO_0[2]" in top and ".rx(rx_s2)" in top     # (through two registers)
    assert "assign GPIO_0 = 32'bz;" in top              # (no output drives GPIO_0: read only)
    assert "assign LED[7] = data_w[7];" in top
    assert "|| !KEY[0]" in top                                       # (KEY[0] resets)


def test_processes_together_take_the_board_s_pins():
    top = _quiet(U.echo(), U.ECHO_PINS).board_top()
    assert "input GPIO_0_RX" in top and "output GPIO_0_TX" in top
    assert "rx_s1 <= GPIO_0_RX" in top and "assign GPIO_0_TX = tx_w;" in top
    assert "uart_echo u_design(" in top


_TB = """
`timescale 1ns/1ps
module tb;
    reg clk = 0;
    always #10 clk = !clk;
    reg line = 1;
    wire tx;
    wire [7:0] led;
    DE0_Nano board(.CLOCK_50(clk), .GPIO_0_RX(line), .GPIO_0_TX(tx), .KEY(2'b11), .LED(led));
    task send(input [7:0] b);
        integer i;
        begin
            line = 0; repeat (BIT) @(posedge clk);
            for (i = 0; i < 8; i = i + 1) begin line = b[i]; repeat (BIT) @(posedge clk); end
            line = 1; repeat (BIT) @(posedge clk);
        end
    endtask
    reg [7:0] got;
    task receive;
        integer i;
        begin
            @(negedge tx); repeat (BIT / 2) @(posedge clk);
            for (i = 0; i < 8; i = i + 1) begin repeat (BIT) @(posedge clk); got[i] = tx; end
            repeat (BIT) @(posedge clk);
        end
    endtask
    initial begin
        repeat (50) @(posedge clk);
        fork send(8'ha5); receive; join
        $display("LED %h TX %h", led, got);
        fork send(8'h3c); receive; join
        $display("LED %h TX %h", led, got);
        $finish;
    end
endmodule
"""


@pytest.mark.sim
@pytest.mark.skipif(not shutil.which("iverilog"), reason="needs Icarus")
def test_the_echo_on_the_board_s_pins_sends_back_what_it_hears(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    d = _quiet(U.echo(), U.ECHO_PINS).gen_pins_project()
    (d / "tb.v").write_text(_TB.replace("BIT", str(U.BIT)), encoding="utf-8")
    subprocess.run(["iverilog", "-o", "tb.vvp", "tb.v", "DE0_Nano.v", "uart_echo.v"],
                   cwd=d, check=True)
    out = subprocess.run(["vvp", "-n", "tb.vvp"], cwd=d, check=True, capture_output=True,
                         text=True).stdout
    assert [l for l in out.splitlines() if l.startswith("LED")] == ["LED a5 TX a5", "LED 3c TX 3c"]


def test_the_echo_on_the_board_s_pins_meets_50_mhz(quartus):
    p = _quiet(U.echo(), U.ECHO_PINS)
    r = p.PINS(quartus)
    assert r.device == "EP4CE22F17C6" and r.fmax_mhz > 50 and r.registers > 40
    # (each pin where the board's assignments put it: the serial pins on D3 and C3)
    placed = (p._pins_dir / "DE0_Nano.pin").read_text(encoding="utf-8")
    assert any(l.split(":")[0].strip() == "GPIO_0_RX" and l.split(":")[1].strip() == "D3"
               and l.split(":")[-1].strip() == "Y" for l in placed.splitlines() if ":" in l)
    assert any(l.split(":")[0].strip() == "GPIO_0_TX" and l.split(":")[1].strip() == "C3"
               for l in placed.splitlines() if ":" in l)
