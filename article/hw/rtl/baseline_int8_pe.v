// =============================================================================
// Module: baseline_int8_pe.v
// Target: Intel Cyclone V (Same-Chip Fair Baseline using Variable-Precision DSPs)
// Description:
//   Implements the exact same P-lane vector dot-product interface as bitlinear_pe.v,
//   but for standard W8A8 (INT8 weights x INT8 activations) mapped onto
//   Intel Cyclone V variable-precision DSP blocks ((* multstyle = "dsp" *)).
//   Used in Section 5 to provide a fair, same-FPGA, same-clock baseline comparison.
// =============================================================================

`timescale 1ns / 1ps

module baseline_int8_pe #(
    parameter integer P_LANES   = 16,
    parameter integer ACT_WIDTH = 8,
    parameter integer W_WIDTH   = 8,
    parameter integer ACC_WIDTH = 24
) (
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         clear_acc,
    input  wire                         valid_in,
    input  wire [P_LANES*ACT_WIDTH-1:0] x_vec_flat,
    input  wire [P_LANES*W_WIDTH-1:0]   w_vec_flat,
    output reg                          valid_out,
    output reg  signed [ACC_WIDTH-1:0]  acc_out
);

    (* multstyle = "dsp" *) wire signed [ACT_WIDTH+W_WIDTH-1:0] prod [0:P_LANES-1];

    genvar i;
    generate
        for (i = 0; i < P_LANES; i = i + 1) begin : gen_dsp_mac
            wire signed [ACT_WIDTH-1:0] x_lane = x_vec_flat[i*ACT_WIDTH +: ACT_WIDTH];
            wire signed [W_WIDTH-1:0]   w_lane = w_vec_flat[i*W_WIDTH   +: W_WIDTH];
            assign prod[i] = x_lane * w_lane;
        end
    endgenerate

    reg signed [ACC_WIDTH-1:0] tree_sum;
    integer k;
    always @(*) begin
        tree_sum = {ACC_WIDTH{1'b0}};
        for (k = 0; k < P_LANES; k = k + 1) begin
            tree_sum = tree_sum + {{(ACC_WIDTH-ACT_WIDTH-W_WIDTH){prod[k][ACT_WIDTH+W_WIDTH-1]}}, prod[k]};
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

