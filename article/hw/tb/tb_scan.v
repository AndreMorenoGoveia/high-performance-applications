// scan_lane: NS sequences of T tokens, state cleared between sequences.
// Token word = {x[7:0], b[N*8], c[N*8], m, k[3:0], j[3:0], a[15:0]} (LSB = a).
`timescale 1ns / 1ps
module tb_scan;
    parameter N = 16, STATE_W = 32, GUARD = 10, ROUND = 1, DECAY = 0, Y_W = 24;
    parameter NS = 4, T = 256;
    parameter TF = "vectors/scan_tok.hex";
    parameter YF = "vectors/scan_y.hex";
    localparam TW = 8 + 16*N + 1 + 4 + 4 + 16;
    reg clk = 0, rst_n = 0, clear = 0, vin = 0;
    reg [TW-1:0] tok;
    wire vout;
    wire signed [Y_W-1:0] y;
    reg [TW-1:0]  toks [0:NS*T-1];
    reg [Y_W-1:0] ys   [0:NS*T-1];
    integer s, t, got, err;
    always #2.5 clk = ~clk;
    scan_lane #(.N(N), .STATE_W(STATE_W), .GUARD(GUARD), .ROUND(ROUND), .DECAY(DECAY), .Y_W(Y_W)) dut (
        .clk(clk), .rst_n(rst_n), .clear(clear), .valid_in(vin),
        .x_in(tok[TW-1 -: 8]), .b_flat(tok[TW-9 -: 8*N]), .c_flat(tok[TW-9-8*N -: 8*N]),
        .m_in(tok[24]), .k_in(tok[23:20]), .j_in(tok[19:16]), .a_q15_in(tok[15:0]),
        .valid_out(vout), .y_out(y));
    always @(posedge clk) if (vout) begin
        if (y !== ys[got]) err = err + 1;
        got = got + 1;
    end
    initial begin
        $readmemh(TF, toks); $readmemh(YF, ys);
        got = 0; err = 0;
        #12 rst_n = 1;
        for (s = 0; s < NS; s = s + 1) begin
            @(negedge clk); vin = 0; clear = 1;
            @(negedge clk); clear = 0;
            for (t = 0; t < T; t = t + 1) begin
                vin = 1; tok = toks[s*T + t];
                @(negedge clk);
            end
            vin = 0;
            repeat (12) @(negedge clk);   // drain before next clear
        end
        $display("RESULT scan STATE_W=%0d GUARD=%0d ROUND=%0d DECAY=%0d checked=%0d mismatches=%0d",
                 STATE_W, GUARD, ROUND, DECAY, got, err + (NS*T - got));
        $finish;
    end
endmodule
