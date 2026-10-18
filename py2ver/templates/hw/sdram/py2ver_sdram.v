// py2ver_sdram.v
// A controller for the DE0-Nano's SDRAM (IS42S16160B: 32 MB, 16-bit, 4 banks of
// 8192 rows of 512 words), for a design that reads and writes it a word at a
// time. The SDRAM's clock is the inverse of clk (DRAM_CLK = ~clk): a command
// registered at a rising edge of clk is sampled half a clock later, and a read's
// data (CAS latency 2, burst length 1) is captured CL + 1 = 3 clocks after its
// READ command.
//
// The request side takes one access a clock: req with we, addr = {bank, row,
// col} and wdata. ready says the access is issued at this edge; while it is low
// (a row being opened, the bank precharged, a refresh) the request must be held,
// which a design does by stopping its clock enable on !ready. Each bank keeps its
// row open, so accesses within a row run one a clock. Read data come back in
// order, on rvalid, READ_LATENCY clocks after the edge that issued them.
module py2ver_sdram #(
    parameter T_INIT  = 10000,   // clocks of 200 us before the first command (50 MHz)
    parameter T_REF   = 380,     // clocks between refreshes (64 ms / 8192 rows = 7.8 us)
    parameter T_RCD   = 2,       // ACTIVE to READ / WRITE
    parameter T_RP    = 2,       // PRECHARGE to ACTIVE
    parameter T_RAS   = 3,       // ACTIVE to PRECHARGE
    parameter T_RFC   = 5,       // REFRESH to anything
    parameter T_WR    = 2,       // WRITE to PRECHARGE
    parameter T_MRD   = 2        // LOAD MODE to anything
) (
    input             clk,
    input             rst,
    input             req,
    input             we,
    input      [23:0] addr,
    input      [15:0] wdata,
    output            ready,
    output reg        rvalid,
    output reg [15:0] rdata,
    output            init_done,
    // the SDRAM
    output reg [12:0] sa,
    output reg  [1:0] ba,
    output reg        cs_n,
    output reg        ras_n,
    output reg        cas_n,
    output reg        we_n,
    output            cke,
    output      [1:0] dqm,
    inout      [15:0] dq
);
    localparam CL = 2;
    localparam READ_LATENCY = CL + 1;

    // Commands {cs_n, ras_n, cas_n, we_n}
    localparam C_NOP = 4'b0111, C_ACT = 4'b0011, C_READ = 4'b0101, C_WRITE = 4'b0100,
               C_PRE = 4'b0010, C_REF = 4'b0001, C_MRS = 4'b0000;

    localparam S_WAIT = 3'd0, S_PREALL = 3'd1, S_REF1 = 3'd2, S_REF2 = 3'd3, S_MRS = 3'd4,
               S_RUN = 3'd5, S_REFRESH = 3'd6;

    reg  [2:0]  state;
    reg  [15:0] wait_n;          // clocks left before the next command
    reg  [15:0] ref_n;           // clocks to the next refresh
    reg  [3:0]  open;            // a bank's row is open
    reg  [12:0] row [0:3];       // which
    reg  [3:0]  act_n [0:3];     // clocks since ACTIVE (saturating)
    reg  [3:0]  wr_n [0:3];      // clocks since a WRITE (saturating)
    reg  [3:0]  rd_busy;         // clocks the data bus is still taken by a read
    reg  [READ_LATENCY-1:0] inflight;
    reg  [15:0] dq_out;
    reg         dq_oe;

    assign cke = 1'b1;
    assign dqm = 2'b00;
    assign dq = dq_oe ? dq_out : 16'bz;
    assign init_done = (state == S_RUN) || (state == S_REFRESH);

    wire [1:0]  rb = addr[23:22];
    wire [12:0] rr = addr[21:9];
    wire [8:0]  rc = addr[8:0];
    wire hit = open[rb] && row[rb] == rr;
    wire refresh_due = ref_n == 0;
    // Every open bank may be precharged (ACTIVE and a write far enough back)
    wire pre_ok = (!open[0] || (act_n[0] >= T_RAS && wr_n[0] >= T_WR))
               && (!open[1] || (act_n[1] >= T_RAS && wr_n[1] >= T_WR))
               && (!open[2] || (act_n[2] >= T_RAS && wr_n[2] >= T_WR))
               && (!open[3] || (act_n[3] >= T_RAS && wr_n[3] >= T_WR));
    // An access issued now: its row open (and ACTIVE long enough ago), no refresh
    // waiting, and a write not on top of a read's data on the bus
    assign ready = state == S_RUN && req && !refresh_due && hit && act_n[rb] >= T_RCD
                   && !(we && rd_busy != 0);

    integer i;
    always @(posedge clk) begin
        {cs_n, ras_n, cas_n, we_n} <= C_NOP;
        dq_oe <= 1'b0;
        inflight <= {inflight[READ_LATENCY-2:0], 1'b0};
        rvalid <= inflight[READ_LATENCY-1];
        if (inflight[READ_LATENCY-1]) rdata <= dq;
        if (rd_busy != 0) rd_busy <= rd_busy - 1'b1;
        for (i = 0; i < 4; i = i + 1) begin
            if (act_n[i] != 4'hf) act_n[i] <= act_n[i] + 1'b1;
            if (wr_n[i] != 4'hf) wr_n[i] <= wr_n[i] + 1'b1;
        end
        if (ref_n != 0 && init_done) ref_n <= ref_n - 1'b1;
        if (rst) begin
            state <= S_WAIT;
            wait_n <= T_INIT;
            open <= 4'b0;
            inflight <= 0;
            rvalid <= 1'b0;
            rd_busy <= 0;
            ref_n <= T_REF;
        end else if (wait_n != 0) begin
            wait_n <= wait_n - 1'b1;
        end else begin
            case (state)
            S_WAIT: begin                                  // precharge all
                {cs_n, ras_n, cas_n, we_n} <= C_PRE;
                sa[10] <= 1'b1;
                wait_n <= T_RP;
                state <= S_PREALL;
            end
            S_PREALL: begin
                {cs_n, ras_n, cas_n, we_n} <= C_REF;
                wait_n <= T_RFC;
                state <= S_REF1;
            end
            S_REF1: begin
                {cs_n, ras_n, cas_n, we_n} <= C_REF;
                wait_n <= T_RFC;
                state <= S_REF2;
            end
            S_REF2: begin                                  // mode: CL 2, burst of 1
                {cs_n, ras_n, cas_n, we_n} <= C_MRS;
                ba <= 2'b00;
                sa <= 13'b000_1_00_010_0_000;              // write burst single, CL 2, BL 1
                wait_n <= T_MRD;
                state <= S_MRS;
            end
            S_MRS: begin
                state <= S_RUN;
                ref_n <= T_REF;
            end
            S_RUN: begin
                if (refresh_due) begin
                    if (rd_busy != 0 || !pre_ok) begin
                        // (wait: a read's data on the bus, a bank opened just now)
                    end else if (open != 0) begin          // (all rows closed first)
                        {cs_n, ras_n, cas_n, we_n} <= C_PRE;
                        sa[10] <= 1'b1;
                        open <= 4'b0;
                        wait_n <= T_RP;
                    end else begin
                        {cs_n, ras_n, cas_n, we_n} <= C_REF;
                        wait_n <= T_RFC;
                        ref_n <= T_REF;
                    end
                end else if (ready) begin                  // the access itself
                    ba <= rb;
                    sa <= {4'b0, rc};                      // (A10 low: no auto precharge)
                    if (we) begin
                        {cs_n, ras_n, cas_n, we_n} <= C_WRITE;
                        dq_out <= wdata;
                        dq_oe <= 1'b1;
                        wr_n[rb] <= 0;
                    end else begin
                        {cs_n, ras_n, cas_n, we_n} <= C_READ;
                        inflight[0] <= 1'b1;
                        rd_busy <= READ_LATENCY;
                    end
                end else if (req && !refresh_due) begin
                    if (open[rb] && !hit) begin            // another row: close this one
                        if (act_n[rb] >= T_RAS && wr_n[rb] >= T_WR) begin
                            {cs_n, ras_n, cas_n, we_n} <= C_PRE;
                            ba <= rb;
                            sa[10] <= 1'b0;
                            open[rb] <= 1'b0;
                            wait_n <= T_RP - 1;
                        end
                    end else if (!open[rb]) begin          // open the row
                        {cs_n, ras_n, cas_n, we_n} <= C_ACT;
                        ba <= rb;
                        sa <= rr;
                        open[rb] <= 1'b1;
                        row[rb] <= rr;
                        act_n[rb] <= 0;
                    end
                end
            end
            default: state <= S_RUN;
            endcase
        end
    end

    initial begin
        for (i = 0; i < 4; i = i + 1) begin
            act_n[i] = 4'hf;
            wr_n[i] = 4'hf;
            row[i] = 0;
        end
        state = S_WAIT;
        wait_n = T_INIT;
        open = 0;
        inflight = 0;
        rd_busy = 0;
        rvalid = 0;
        ref_n = T_REF;
        {cs_n, ras_n, cas_n, we_n} = C_NOP;
        sa = 0;
        ba = 0;
        dq_oe = 0;
        dq_out = 0;
        rdata = 0;
    end
endmodule
