// bitlinear_engine: load weights via the w_* port, then NX activation vectors.
`timescale 1ns / 1ps
module tb_engine;
    parameter P = 16, K = 64, M = 32, ENC = 2, NX = 4;
    parameter WF = "vectors/eng_w.hex";
    localparam WW = (ENC == 1) ? 2 * P : 8 * P;
    localparam DEPTH = (ENC == 2) ? ((M * (K / P)) + 4) / 5 : M * (K / P);
    parameter XF = "vectors/eng_x.hex";
    parameter YF = "vectors/eng_y.hex";
    reg clk = 0, rst_n = 0, x_we = 0, start = 0, w_we = 0;
    reg [$clog2(DEPTH)-1:0] w_addr;
    reg [WW-1:0] w_data;
    reg [WW-1:0] ws [0:DEPTH-1];
    reg [$clog2(K)-1:0] x_addr;
    reg [7:0] x_data;
    wire y_valid, busy;
    wire signed [23:0] y_data;
    reg [7:0]  xs [0:NX*K-1];
    reg [23:0] ys [0:NX*M-1];
    integer v, i, got, err, t_start, cycles;
    always #2.5 clk = ~clk;
    bitlinear_engine #(.P(P), .K(K), .M(M), .ENC(ENC)) dut (
        .clk(clk), .rst_n(rst_n), .w_we(w_we), .w_addr(w_addr), .w_data(w_data), .x_we(x_we), .x_addr(x_addr), .x_data(x_data),
        .start(start), .y_valid(y_valid), .y_data(y_data), .busy(busy));
    always @(posedge clk) if (y_valid) begin
        if (y_data !== ys[got]) err = err + 1;
        got = got + 1;
        cycles = ($time - t_start) / 5;     // start edge -> last output (5 ns clock)
    end
    initial begin
        $readmemh(XF, xs); $readmemh(YF, ys); $readmemh(WF, ws);
        got = 0; err = 0;
        #12 rst_n = 1;
        for (i = 0; i < DEPTH; i = i + 1) begin
            @(negedge clk); w_we = 1; w_addr = i; w_data = ws[i];
        end
        @(negedge clk); w_we = 0;
        for (v = 0; v < NX; v = v + 1) begin
            for (i = 0; i < K; i = i + 1) begin
                @(negedge clk); x_we = 1; x_addr = i; x_data = xs[v*K + i];
            end
            @(negedge clk); x_we = 0; start = 1; t_start = $time + 2.5;
            @(negedge clk); start = 0;
            @(negedge clk); while (busy) @(negedge clk);
        end
        repeat (4) @(posedge clk);
        $display("RESULT engine P=%0d K=%0d M=%0d ENC=%0d checked=%0d mismatches=%0d cycles=%0d",
                 P, K, M, ENC, got, err + (NX*M - got), cycles);
        $finish;
    end
endmodule
