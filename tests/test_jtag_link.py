import random
import shutil
import stat
import subprocess

import pytest

from py2ver import Py2ver
from py2ver.defaults import DEFAULT_TEMPLATE_DIR
from test_state_stream import LPF_ATTR, lpf_step
from test_syn_report import FIXTURES
from test_virtual_board import _fake_quartus_with_reports, board_add_sub

# HW() over the JTAG UART (LINK='jtag'): the board's words over the USB-Blaster
# (jtag_transceiver, jtag_bytes) instead of a USB-UART adapter.

ADD_ATTR = {"a": {"signed": 0, "width": 8}, "b": {"signed": 0, "width": 8},
            "s": {"signed": 0, "width": 9}, "d": {"signed": 1, "width": 9}}


def test_the_board_s_project_over_the_jtag_uart(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    tool = _fake_quartus_with_reports(tmp_path, FIXTURES / "lpf_step" / "output_files")
    p = Py2ver(lpf_step, LPF_ATTR)
    p.HW({"QUARTUS_DIR": tool, "LINK": "jtag"})
    project = tmp_path / "hw" / "lpf_step"
    top = (project / "DE0_Nano.v").read_text(encoding="utf-8")
    assert "jtag_transceiver #(" in top and "uart_transceiver #(" not in top
    qsf = (project / "lpf_step.qsf").read_text(encoding="utf-8")
    assert "uart/jtag_bytes.sv" in qsf and "uart/jtag_transceiver.sv" in qsf
    assert "alt_jtag_atlantic" in (project / "uart" / "jtag_bytes.sv").read_text(encoding="utf-8")


def test_the_default_link_is_the_uart_and_another_is_refused(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    tool = _fake_quartus_with_reports(tmp_path, FIXTURES / "lpf_step" / "output_files")
    p = Py2ver(lpf_step, LPF_ATTR)
    p.HW({"QUARTUS_DIR": tool})
    top = (tmp_path / "hw" / "lpf_step" / "DE0_Nano.v").read_text(encoding="utf-8")
    assert "uart_transceiver #(" in top and "jtag_transceiver" not in top
    with pytest.raises(ValueError, match="LINK: 'usb'"):
        p.HW({"QUARTUS_DIR": tool, "LINK": "usb"})


def _sdram_design(n=64, w=20):
    import contextlib
    import io
    import test_sdram as S
    with contextlib.redirect_stdout(io.StringIO()):
        return Py2ver(S._wide_ports, {"x": {"signed": 1, "width": w, "length": n}},
                      sequential=True, sdram=0)


def test_the_sdram_board_s_commands_over_the_jtag_uart(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    project = _sdram_design().gen_hw_project(link="jtag")
    top = (project / "DE0_Nano.v").read_text(encoding="utf-8")
    assert "jtag_bytes u_jtag(" in top and "uart_rx_8n1" not in top
    assert "assign rx_accept = state == S_CMD" in top           # (bytes taken only where read)


def test_no_terminal_is_no_board(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    tool = _fake_quartus_with_reports(tmp_path, FIXTURES / "lpf_step" / "output_files")
    term = tmp_path / "quartus_bin" / "juart-terminal"     # (a terminal that finds no cable)
    term.write_text("#!/bin/sh\necho 'No JTAG hardware available' >&2\nexit 1\n", encoding="utf-8")
    term.chmod(term.stat().st_mode | stat.S_IEXEC)
    p = Py2ver(lpf_step, LPF_ATTR)
    hw = p.HW({"QUARTUS_DIR": tool, "LINK": "jtag"})
    with pytest.raises(RuntimeError, match="JTAG communication failed.*No JTAG hardware"):
        hw(256)
    with pytest.raises(RuntimeError, match="JTAG communication failed"):
        p.HW_batch()([(256,), (0,)])


# --------------------- the transceiver, against a stand-in PC ---------------------

# alt_jtag_atlantic as the PC drives it: its bytes offered every clock there's
# room (t_dav), the board's taken only on 4 clocks in 7 (r_ena); each byte from
# the board printed
_ATLANTIC = """
module alt_jtag_atlantic #(parameter INSTANCE_ID = 0, LOG2_RXFIFO_DEPTH = 6,
                           LOG2_TXFIFO_DEPTH = 6, SLD_AUTO_INSTANCE_INDEX = "YES") (
    input clk, input rst_n, input [7:0] r_dat, input r_val, output reg r_ena = 1'b0,
    output reg [7:0] t_dat = 8'd0, input t_dav, output reg t_ena = 1'b0, output t_pause);
    reg [7:0] to_board [0:{n_in}-1];
    initial $readmemh("to_board.hex", to_board);
    integer sent = 0, got = 0, tick = 0;
    assign t_pause = 1'b0;
    always @(posedge clk) begin
        tick <= tick + 1;
        t_ena <= 1'b0;
        if (rst_n && t_dav && sent < {n_in}) begin
            t_dat <= to_board[sent]; t_ena <= 1'b1; sent <= sent + 1;
        end
        r_ena <= rst_n && (tick % 7) < 4;
        if (r_val) begin
            $display("B %02x", r_dat);
            got = got + 1;
            if (got == {n_out}) begin $display("DONE"); $finish; end
        end
    end
endmodule
"""

# The board: a core answering each word 5 clocks after it ({count, ~word}); a
# word while it holds one, an error
_BENCH = """
module bench;
    reg clk = 1'b0;
    always #10 clk = ~clk;
    reg rst_n = 1'b0;
    initial begin #95 rst_n = 1'b1; #20000000 $display("TIMEOUT"); $finish; end
    wire [15:0] rx_data;
    wire rx_valid, tx_busy;
    reg [23:0] tx_data = 24'd0;
    reg tx_valid = 1'b0;
    jtag_transceiver #(.TX_WIDTH(24), .RX_WIDTH(16)) dut(.clk(clk), .rst_n(rst_n),
        .tx_data(tx_data), .tx_valid(tx_valid), .tx_busy(tx_busy),
        .rx_data(rx_data), .rx_valid(rx_valid));
    reg busy = 1'b0;
    reg [15:0] held = 16'd0;
    integer wait_n = 0, words = 0;
    always @(posedge clk) begin
        tx_valid <= 1'b0;
        if (rx_valid) begin
            if (busy) $display("ERROR a word while one is held");
            busy <= 1'b1; held <= rx_data; wait_n <= 5;
        end else if (busy) begin
            if (wait_n == 0) begin
                tx_data <= {{words[7:0], ~held}}; tx_valid <= 1'b1; busy <= 1'b0; words <= words + 1;
            end else wait_n <= wait_n - 1;
        end
    end
endmodule
"""


@pytest.mark.sim
def test_words_sent_together_answered_in_turn(tmp_path):
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("Icarus Verilog tools are not installed")
    rng = random.Random(7)
    words = [rng.randrange(1 << 16) for _ in range(40)] + [0x0004, 0xFFFF, 0x0000]
    sent = b"".join(w.to_bytes(2, "little") for w in words)
    (tmp_path / "to_board.hex").write_text("\n".join(f"{b:02x}" for b in sent) + "\n")
    sources = []
    for name, text in [("atlantic.v", _ATLANTIC.format(n_in=len(sent), n_out=3 * len(words))),
                       ("bench.v", _BENCH.format())]:
        (tmp_path / name).write_text(text, encoding="utf-8")
        sources.append(name)
    for name in ("jtag_bytes", "jtag_transceiver"):
        shutil.copy(DEFAULT_TEMPLATE_DIR / "hw" / "uart" / f"{name}.txt", tmp_path / f"{name}.sv")
        sources.append(f"{name}.sv")
    subprocess.run(["iverilog", "-g2012", "-s", "bench", "-o", "bench.vvp", *sources],
                   cwd=tmp_path, check=True, capture_output=True, text=True)
    out = subprocess.run(["vvp", "-n", "bench.vvp"], cwd=tmp_path, check=True,
                         capture_output=True, text=True).stdout
    assert "ERROR" not in out and "DONE" in out, out[-2000:]
    got = bytes(int(line.split()[1], 16) for line in out.splitlines() if line.startswith("B "))
    want = b"".join(((~w) & 0xFFFF).to_bytes(2, "little") + bytes([i]) for i, w in enumerate(words))
    assert got == want


# ------------------------------------ the board ------------------------------------

@pytest.mark.hardware
def test_an_adder_on_the_board_over_the_jtag_uart(quartus):
    p = Py2ver(board_add_sub, ADD_ATTR)
    hw = p.HW(dict(quartus, LINK="jtag"))
    assert hw(7, 3) == (10, 4) and hw(0, 4) == (4, -4)            # (a 4 back: data, not ^D)
    vectors = [(a, b) for a in range(256) for b in (0, 255 - a, a * 7 % 256)]
    assert p.HW_batch()(vectors) == [board_add_sub(*v) for v in vectors]


@pytest.mark.hardware
def test_a_filter_s_state_on_the_board_over_the_jtag_uart(quartus):
    p = Py2ver(lpf_step, LPF_ATTR)
    p.HW(dict(quartus, LINK="jtag"))
    xs = [256, 256, 512, -2048, 100, 0] * 50
    assert p.HW_batch()([(x,) for x in xs]) == p.PY_stream()(xs)


@pytest.mark.hardware
def test_lists_in_the_board_s_sdram_over_the_jtag_uart(quartus):
    # (the tables, the input lists written into the chip, the returned ones read back)
    p = _sdram_design()
    hw = p.HW(dict(quartus, LINK="jtag"))
    r = random.Random(2)
    vectors = [([r.randint(-2 ** 19, 2 ** 19 - 1) for _ in range(64)],) for _ in range(3)]
    assert [hw(*v) for v in vectors] == p.PY_batch()(vectors)
