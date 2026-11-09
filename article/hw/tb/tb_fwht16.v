// Streams NV random int8 vectors through fwht16 (1 per cycle) and compares to golden.
`timescale 1ns / 1ps
module tb_fwht16;
    parameter NV = 1000;
    parameter XF = "vectors/fwht_x.hex";
    parameter YF = "vectors/fwht_y.hex";
    reg clk = 0, rst_n = 0, vin = 0;
    reg  [127:0] x;
    wire [191:0] y;
    wire vout;
    reg [127:0] xs [0:NV-1];
    reg [191:0] ys [0:NV-1];
    integer i, got, err;
    always #2.5 clk = ~clk;
    fwht16 dut (.clk(clk), .rst_n(rst_n), .valid_in(vin), .x_flat(x), .valid_out(vout), .y_flat(y));
    always @(posedge clk) if (vout) begin
        if (y !== ys[got]) err = err + 1;
        got = got + 1;
    end
    initial begin
        $readmemh(XF, xs); $readmemh(YF, ys);
        got = 0; err = 0;
        #12 rst_n = 1;
        for (i = 0; i < NV; i = i + 1) begin
            @(negedge clk); vin = 1; x = xs[i];
        end
        @(negedge clk); vin = 0;
        repeat (10) @(posedge clk);
        $display("RESULT fwht16 checked=%0d mismatches=%0d", got, err + (NV - got));
        $finish;
    end
endmodule
