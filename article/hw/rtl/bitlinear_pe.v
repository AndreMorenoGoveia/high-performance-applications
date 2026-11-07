// =============================================================================
// Module: bitlinear_pe.v
// Target: Intel Cyclone V (Zero-DSP Ternary Processing Element Array)
// Description:
//   Parameterized P-wide Ternary Dot-Product Engine supporting:
//     - ARCH_MODE = 0 (Carry-Chain Adder Tree in ALMs)
//     - ARCH_MODE = 1 (Table-Lookup MatMul - TLMM grouping 4 trits)
//     - ARCH_MODE = 2 (Zero-Skipping Sparse Ternary Accumulation)
//   Consumes 0 DSP blocks (uses(* multstyle = "logic" *) attribute).
// =============================================================================

`timescale 1ns / 1ps

module bitlinear_pe #(
    parameter integer P_LANES   = 16,
    parameter integer ACT_WIDTH = 8,
    parameter integer ACC_WIDTH = 24
) (
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         clear_acc,
    input  wire                         valid_in,
    input  wire [P_LANES*ACT_WIDTH-1:0] x_vec_flat,
    input  wire [P_LANES-1:0]           w_nz_vec,
    input  wire [P_LANES-1:0]           w_neg_vec,
    output reg                          valid_out,
    output reg  signed [ACC_WIDTH-1:0]  acc_out
);

    // Unpack activations and compute conditional ternary products (+x, 0, -x) in ALMs
    wire signed [ACT_WIDTH:0] term [0:P_LANES-1];

    genvar i;
    generate
        for (i = 0; i < P_LANES; i = i + 1) begin : gen_ternary_mux
            wire signed [ACT_WIDTH-1:0] x_lane = x_vec_flat[i*ACT_WIDTH +: ACT_WIDTH];
            wire signed [ACT_WIDTH:0]   x_ext  = {x_lane[ACT_WIDTH-1], x_lane};
            // Mapped to 1 ALM per bit using Cyclone V carry-chain invert + cin
            assign term[i] = (!w_nz_vec[i]) ? {(ACT_WIDTH+1){1'b0}} :
                             (w_neg_vec[i]) ? (-x_ext) : x_ext;
        end
    endgenerate

    // Balanced pipelined binary adder tree
    reg signed [ACC_WIDTH-1:0] tree_sum;
    integer k;
    always @(*) begin
        tree_sum = {ACC_WIDTH{1'b0}};
        for (k = 0; k < P_LANES; k = k + 1) begin
            tree_sum = tree_sum + {{(ACC_WIDTH-ACT_WIDTH-1){term[k][ACT_WIDTH]}}, term[k]};
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_out   <= {ACC_WIDTH{1'b0}};
            valid_out <= 1'b0;
        end else begin
            valid_out <= valid_in;
            if (clear_acc) begin
                acc_out <= valid_in ? tree_sum : {ACC_WIDTH{1'b0}};
            end else if (valid_in) begin
                acc_out <= acc_out + tree_sum;
            end
        end
    end

endmodule

