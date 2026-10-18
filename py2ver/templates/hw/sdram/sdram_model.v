// sdram_model.v
// A behavioral model of the DE0-Nano's SDRAM (IS42S16160B) for simulation:
// burst length 1, the CAS latency of its mode register, banks with open rows.
// It checks the timing a controller must keep (ACTIVE to READ / WRITE, PRECHARGE
// to ACTIVE, ACTIVE to PRECHARGE, REFRESH to anything, the refresh interval,
// accesses to a closed bank) and counts the violations in `errors` ($display'ing
// the first few). Contents start from files: $readmemh at `base` (+define via the
// testbench: INIT_FILE / INIT_BASE pairs are loaded by py2ver's testbench).
`timescale 1ns/1ps
module sdram_model #(
    parameter ADDR_BITS = 24,    // {bank 2, row 13, col 9}
    parameter T_RCD   = 1,       // (the chip's own minimums, in its clocks at 50 MHz)
    parameter T_RP    = 1,
    parameter T_RAS   = 3,
    parameter T_RFC   = 4,
    parameter T_REF_MAX = 400    // most clocks between refreshes (7.8 us at 50 MHz)
) (
    input             clk,       // DRAM_CLK
    input             cke,
    input             cs_n,
    input             ras_n,
    input             cas_n,
    input             we_n,
    input      [1:0]  ba,
    input      [12:0] sa,
    input      [1:0]  dqm,
    inout      [15:0] dq
);
    reg [15:0] mem [0:(1 << ADDR_BITS) - 1];
    reg [3:0]  open;
    reg [12:0] row [0:3];
    integer    since_act [0:3];
    integer    since_pre [0:3];
    integer    since_ref, since_refresh;
    integer    cl;
    integer    errors;
    reg [15:0] out_data [0:7];
    reg [7:0]  out_valid;
    reg        driving;
    reg [15:0] driven;
    assign #1 dq = driving ? driven : 16'bz;

    task error(input [8*64-1:0] what);
        begin
            errors = errors + 1;
            if (errors <= 10) $display("sdram_model: %0s at %0t", what, $time);
        end
    endtask

    integer i;
    initial begin
        open = 0;
        cl = 2;
        errors = 0;
        out_valid = 0;
        driving = 0;
        since_ref = 1000000;
        since_refresh = 0;
        for (i = 0; i < 4; i = i + 1) begin
            since_act[i] = 1000000;
            since_pre[i] = 1000000;
            row[i] = 0;
        end
    end

    wire [3:0] cmd = {cs_n, ras_n, cas_n, we_n};
    always @(posedge clk) begin
        // (data out: the read issued cl clocks ago)
        driving <= out_valid[cl - 1];
        driven <= out_data[cl - 1];
        out_valid <= {out_valid[6:0], 1'b0};
        for (i = 7; i > 0; i = i - 1) out_data[i] <= out_data[i - 1];
        for (i = 0; i < 4; i = i + 1) begin
            since_act[i] = since_act[i] + 1;
            since_pre[i] = since_pre[i] + 1;
        end
        since_ref = since_ref + 1;
        since_refresh = since_refresh + 1;
        if (since_refresh > T_REF_MAX && open !== 4'bx) begin
            error("refresh interval exceeded");
            since_refresh = 0;
        end
        if (cke && !cs_n) begin
            if (since_ref < T_RFC && cmd != 4'b0111) error("command during refresh");
            case (cmd)
            4'b0011: begin                                     // ACTIVE
                if (open[ba]) error("ACTIVE to an open bank");
                if (since_pre[ba] < T_RP) error("ACTIVE too soon after PRECHARGE");
                open[ba] = 1'b1;
                row[ba] = sa;
                since_act[ba] = 0;
            end
            4'b0101: begin                                     // READ
                if (!open[ba]) error("READ of a closed bank");
                if (since_act[ba] < T_RCD) error("READ too soon after ACTIVE");
                out_valid[0] <= 1'b1;
                out_data[0] <= mem[{ba, row[ba], sa[8:0]}];
            end
            4'b0100: begin                                     // WRITE
                if (!open[ba]) error("WRITE to a closed bank");
                if (since_act[ba] < T_RCD) error("WRITE too soon after ACTIVE");
                mem[{ba, row[ba], sa[8:0]}] = dq;
            end
            4'b0010: begin                                     // PRECHARGE (A10: all)
                for (i = 0; i < 4; i = i + 1)
                    if (sa[10] || ba == i) begin
                        if (open[i] && since_act[i] < T_RAS) error("PRECHARGE too soon after ACTIVE");
                        open[i] = 1'b0;
                        since_pre[i] = 0;
                    end
            end
            4'b0001: begin                                     // AUTO REFRESH
                if (open != 0) error("REFRESH with a bank open");
                since_ref = 0;
                since_refresh = 0;
            end
            4'b0000: cl = sa[6:4];                             // LOAD MODE
            default: ;
            endcase
        end
    end
endmodule
