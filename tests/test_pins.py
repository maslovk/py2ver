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


def _quiet(f, attr, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return Py2ver(f, attr, **kw)


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


def test_unused_leds_are_off_and_pins_count_bits():
    top = _quiet(U.uart_tx, {"start": 1, "data": 8, "tx": "GPIO_0_TX", "busy": "LED[0]"}).board_top()
    assert "assign LED[0] = busy_w;" in top and "assign LED[7] = 1'b0;" in top
    with pytest.raises(ValueError, match=r"the port 'data' \(8 bits\): SW: 4 pins for a port of 8"):
        _quiet(U.uart_tx, {"start": 1, "data": {"width": 8, "pin": "SW"}})
    with pytest.raises(ValueError, match=r"the port 'tx' \(1 bit\): GPIO_0\[1\]"):
        _quiet(U.uart_tx, {"start": 1, "data": 8, "tx": "GPIO_0[1]"})


def test_signaltap_records_the_pins_and_the_state():
    p = _quiet(U.echo(), dict(U.ECHO_PINS), probes=True)
    probes = [q.name for q in p.ir.outputs][5:]
    assert probes[:2] == ["rx___pc_out", "rx___wait_out"] and "tx_i_out" in probes
    top = p.board_top(signaltap=True)
    assert "sld_signaltap #(" in top and ".sld_node_info(806383104)" in top
    assert ".sld_data_bits(61)" in top and ".sld_sample_depth(4096)" in top    # (61 bits: 4K)
    assert ".acq_data_in({tx_i_out_w, " in top and ", valid_w, rx_s2})" in top  # (bit 0: rx)
    mapping = p._signaltap(True, p._pin_map()[0])[1]
    assert mapping[1:3] == ["rx\t0\t1", "valid\t1\t1"] and "data\t9:2\t8" in mapping
    small = _quiet(U.echo(), dict(U.ECHO_PINS)).board_top(
        signaltap={"signals": ["rx", "tx"], "depth": 8192})
    assert ".sld_data_bits(2)" in small and ".acq_data_in({tx_w, rx_s2})" in small
    assert "sld_signaltap" not in _quiet(U.echo(), dict(U.ECHO_PINS)).board_top()
    with pytest.raises(ValueError, match="no port \\['rx___pc_out'\\]"):
        _quiet(U.echo(), dict(U.ECHO_PINS)).board_top(signaltap={"signals": ["rx___pc_out"]})
    with pytest.raises(ValueError, match="a depth of 1000"):
        p.board_top(signaltap={"depth": 1000})
    with pytest.raises(ValueError, match="probes=True: a process's state"):
        Py2ver(lambda x: x + 1, {"x": 8}, probes=True)


def test_signaltap_on_the_board_in_quartus(quartus):
    p = _quiet(U.echo(), dict(U.ECHO_PINS), probes=True)
    r = p.PINS(quartus, signaltap={"depth": 1024})
    assert r.memory_bits == 61 * 1024                    # (its buffer)
    assert r.fmax_mhz > 100 and r.meets_timing           # (the design's clock, not the JTAG's)
    assert "rx___pc_out\t15:13\t3" in (p._pins_dir / "signaltap.txt").read_text()


import jtag_echo as J  # noqa: E402

# (the PC's side of the JTAG UART, as the JTAG UART IP's handshake has it: a byte on
# t_dat while t_ena, given only while t_dav; the FPGA's back on r_dat with r_val the
# clock after r_ena; Intel's alt_jtag_atlantic has no simulation model)
_JTAG_PC = """
module alt_jtag_atlantic #(parameter INSTANCE_ID = 0, LOG2_RXFIFO_DEPTH = 6,
                           LOG2_TXFIFO_DEPTH = 6, SLD_AUTO_INSTANCE_INDEX = "YES") (
    input clk, input rst_n,
    input [7:0] r_dat, input r_val, output reg r_ena = 1'b0,
    output reg [7:0] t_dat = 8'd0, input t_dav, output reg t_ena = 1'b0, output t_pause);
    assign t_pause = 1'b0;
    reg [7:0] text [0:4];
    integer k = 0, n = 0;
    initial begin text[0] = "h"; text[1] = "e"; text[2] = "l"; text[3] = "l"; text[4] = "o"; end
    always @(posedge clk) begin
        t_ena <= 1'b0;
        r_ena <= (n % 3) != 0;                      // (the PC not always ready)
        n = n + 1;
        if (rst_n && t_dav && k < 5) begin t_dat <= text[k]; t_ena <= 1'b1; k = k + 1; end
        if (r_val) $display("PC %c", r_dat);
    end
endmodule
"""

_JTAG_TB = """
`timescale 1ns/1ps
module tb;
    reg clk = 0;
    always #10 clk = !clk;
    wire [7:0] led;
    DE0_Nano board(.CLOCK_50(clk), .KEY(2'b11), .LED(led));
    initial begin repeat (400) @(posedge clk); $display("LED %c", led); $finish; end
endmodule
"""


def test_the_jtag_uart_s_pins():
    top = _quiet(J.jtag_echo, J.PINS).board_top()
    assert "alt_jtag_atlantic #(" in top and ".byte_(jtag_rx_data)" in top
    assert "assign jtag_tx_data = out_w;" in top and "assign jtag_rx_ready = ready_w;" in top
    assert "jtag_rx_data_s1" not in top                    # (its clock: no two registers)
    with pytest.raises(ValueError, match="JTAG_TX_DATA is for an output of the design's"):
        _quiet(J.jtag_echo, dict(J.PINS, byte="JTAG_TX_DATA", out="GPIO_0[9:2]"))
    # (from the PC: its 8 bits exactly; to it, a narrower port is fine: text's 7)
    with pytest.raises(ValueError, match=r"the port 'valid' \(1 bit\): JTAG_RX_DATA: 8 bits "
                                         r"for a port of 1"):
        _quiet(J.jtag_echo, dict(J.PINS, valid={"width": 1, "pin": "JTAG_RX_DATA"}, byte="SW"))
    assert pin_width("JTAG_RX_DATA") == 8 and pin_width("JTAG_TX_VALID") == 1


def test_pins_given_apart_from_attr():
    # pins=: the same board as each port's 'pin' in attr, merged with the rest of attr
    assert _quiet(J.jtag_echo, None, pins=J.PINS).board_top() == \
        _quiet(J.jtag_echo, J.PINS).board_top()
    p = _quiet(U.uart_rx, {"rx": 1}, pins={"rx": "GPIO_0_RX", "data": "LED[7:0]"})
    assert p.board_top() == _receiver().board_top()
    with pytest.raises(ValueError, match="'rx' is on GPIO_0_RX here, but on KEY\\[0\\] in attr"):
        _quiet(U.uart_rx, {"rx": "KEY[0]"}, pins={"rx": "GPIO_0_RX"})
    with pytest.raises(ValueError, match="needs a pin's name, such as 'LED\\[0\\]'"):
        _quiet(U.uart_rx, {"rx": 1}, pins={"rx": 3})
    with pytest.raises(ValueError, match="A pin for no port: \\['led'\\]"):
        _quiet(U.uart_rx, {"rx": 1}, pins={"rx": "GPIO_0_RX", "led": "LED[0]"})


@pytest.mark.sim
@pytest.mark.skipif(not shutil.which("iverilog"), reason="needs Icarus")
def test_an_echo_over_the_jtag_uart_on_the_board(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    d = _quiet(J.jtag_echo, J.PINS).gen_pins_project()
    (d / "tb.v").write_text(_JTAG_TB, encoding="utf-8")
    (d / "pc.v").write_text(_JTAG_PC, encoding="utf-8")
    subprocess.run(["iverilog", "-o", "tb.vvp", "tb.v", "pc.v", "DE0_Nano.v", "jtag_echo.v"],
                   cwd=d, check=True)
    out = subprocess.run(["vvp", "-n", "tb.vvp"], cwd=d, check=True, capture_output=True,
                         text=True).stdout.splitlines()
    assert "".join(l[-1] for l in out if l.startswith("PC ")) == "hello"   # (all back, in order)
    assert "LED o" in out


def test_the_jtag_uart_in_quartus(quartus):
    p = _quiet(J.jtag_echo, J.PINS)
    r = p.PINS(quartus)
    assert r.meets_timing and r.fmax_mhz > 50
    # (Intel's JTAG UART core in it, on the JTAG hub)
    built = (p._pins_dir / "DE0_Nano.map.rpt").read_text(encoding="utf-8", errors="replace")
    assert "alt_jtag_atlantic:u_jtag_uart" in built and "sld_hub:auto_hub" in built


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
