// Pipelined balanced binary adder tree: P signed inputs -> 1 sum.
// One register per tree level, latency = log2(P) cycles. P must be a power of 2.
`timescale 1ns / 1ps

module adder_tree #(
    parameter integer P     = 16,
    parameter integer IN_W  = 9,
    parameter integer OUT_W = 24
) (
    input  wire                    clk,
    input  wire [P*IN_W-1:0]       in_flat,
    output wire signed [OUT_W-1:0] sum
);
    // Heap layout: node n (1..P-1) = node 2n + node 2n+1; leaves are inputs P..2P-1.
    reg [P*OUT_W-1:0] nodes;

    genvar n;
    generate
        for (n = 1; n < P; n = n + 1) begin : g_node
            wire signed [OUT_W-1:0] a, b;
            if (2 * n >= P) begin : g_leaf
                assign a = {{(OUT_W-IN_W){in_flat[(2*n-P)*IN_W+IN_W-1]}},   in_flat[(2*n-P)*IN_W   +: IN_W]};
                assign b = {{(OUT_W-IN_W){in_flat[(2*n+1-P)*IN_W+IN_W-1]}}, in_flat[(2*n+1-P)*IN_W +: IN_W]};
            end else begin : g_inner
                assign a = nodes[(2*n)*OUT_W   +: OUT_W];
                assign b = nodes[(2*n+1)*OUT_W +: OUT_W];
            end
            always @(posedge clk) nodes[n*OUT_W +: OUT_W] <= a + b;
        end
    endgenerate

    assign sum = nodes[1*OUT_W +: OUT_W];
endmodule
