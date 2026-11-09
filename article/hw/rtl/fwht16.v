// Exact 16-point Fast Walsh-Hadamard Transform, 4 registered butterfly stages,
// 0 multipliers. INT8 in -> 12-bit out (|y| <= 2048), latency 4 cycles.
// The 1/sqrt(16) normalisation is folded into the following requantisation scale
// (golden: fwht16_int).
`timescale 1ns / 1ps

module fwht16 (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          valid_in,
    input  wire [127:0]  x_flat,     // 16 x int8
    output wire          valid_out,
    output wire [191:0]  y_flat      // 16 x int12
);
    wire [191:0] st0;
    reg  [4*192-1:0] st;             // outputs of stages 1..4
    reg  [3:0] v;

    genvar s, e;
    generate
        for (e = 0; e < 16; e = e + 1) begin : g_in
            assign st0[12*e +: 12] = {{4{x_flat[8*e+7]}}, x_flat[8*e +: 8]};
        end
        for (s = 0; s < 4; s = s + 1) begin : g_stage
            wire [191:0] src;
            if (s == 0) begin : g_first
                assign src = st0;
            end else begin : g_next
                assign src = st[(s-1)*192 +: 192];
            end
            for (e = 0; e < 16; e = e + 1) begin : g_elem
                localparam integer STR = 1 << s;
                localparam integer LO  = (e & STR) ? e - STR : e;   // pair (LO, LO+STR)
                wire signed [11:0] a = src[12*LO +: 12];
                wire signed [11:0] b = src[12*(LO+STR) +: 12];
                always @(posedge clk) st[s*192 + 12*e +: 12] <= (e & STR) ? a - b : a + b;
            end
        end
    endgenerate

    always @(posedge clk or negedge rst_n)
        if (!rst_n) v <= 4'd0;
        else        v <= {v[2:0], valid_in};

    assign valid_out = v[3];
    assign y_flat    = st[3*192 +: 192];
endmodule
