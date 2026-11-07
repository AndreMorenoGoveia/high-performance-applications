// =============================================================================
// Module: cpot_selective_scan.v
// Target: Intel Cyclone V (ALMs for CPoT Decay + Variable-Precision DSPs for B*x, C*h)
// Description:
//   Implements the Mamba-2 (SSD) Recurrent Selective Scan step with:
//     1. Bimodal Complementary Power-of-Two (CPoT) Discretization for A_bar_t:
//          mode_bit == 0: h_decay = (h_prev >>> shift_k)
//          mode_bit == 1: h_decay = h_prev - (h_prev >>> shift_k)
//        Preserves long-memory channels (A_bar in [0.90, 0.999]) with 0 multipliers!
//     2. Dedicated Cyclone V DSP multipliers for dynamic input injection (B_t * x_t),
//        state readout (C_t * h_t), and 16-segment PWL SiLU gating.
//     3. On-chip O(1) stationary state register file / M10K BRAM.
// =============================================================================

`timescale 1ns / 1ps

module cpot_selective_scan #(
    parameter integer D_STATE = 16
) (
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     valid_in,
    input  wire                     cpot_mode_bit, // 0: 2^-k, 1: 1 - 2^-k (long-memory)
    input  wire [3:0]               cpot_shift_k,  // k in 1..12
    input  wire signed [7:0]        x_in,          // INT8 activation after Conv1D + PWL SiLU
    input  wire signed [15:0]       z_gate_q88,    // Q8.8 gate activation after PWL SiLU
    input  wire [D_STATE*8-1:0]     b_vec_flat,    // D_STATE x INT8 B_t vector
    input  wire [D_STATE*8-1:0]     c_vec_flat,    // D_STATE x INT8 C_t vector
    output reg                      valid_out,
    output reg  signed [15:0]       y_gated_q88    // Q8.8 gated SSM output
);

    // O(1) Stationary State Registers (Q8.8 fixed-point, 16-bit per state dimension)
    reg signed [15:0] h_state [0:D_STATE-1];

    wire signed [7:0] b_elem [0:D_STATE-1];
    wire signed [7:0] c_elem [0:D_STATE-1];

    genvar gi;
    generate
        for (gi = 0; gi < D_STATE; gi = gi + 1) begin : gen_unpack_bc
            assign b_elem[gi] = b_vec_flat[gi*8 +: 8];
            assign c_elem[gi] = c_vec_flat[gi*8 +: 8];
        end
    endgenerate

    // Combinational CPoT decay + B_t * x_t update
    wire signed [15:0] h_shifted [0:D_STATE-1];
    wire signed [15:0] h_decay   [0:D_STATE-1];
    (* multstyle = "dsp" *) wire signed [15:0] bx_prod [0:D_STATE-1];
    wire signed [16:0] h_next_full [0:D_STATE-1];
    wire signed [15:0] h_next    [0:D_STATE-1];
    (* multstyle = "dsp" *) wire signed [23:0] ch_prod [0:D_STATE-1];

    generate
        for (gi = 0; gi < D_STATE; gi = gi + 1) begin : gen_ssm_lanes
            assign h_shifted[gi]   = h_state[gi] >>> cpot_shift_k;
            assign h_decay[gi]     = cpot_mode_bit ? (h_state[gi] - h_shifted[gi]) : h_shifted[gi];
            assign bx_prod[gi]     = (b_elem[gi] * x_in) >>> 4;
            assign h_next_full[gi] = $signed(h_decay[gi]) + $signed(bx_prod[gi]);

            // Saturate to 16-bit signed Q8.8
            assign h_next[gi] = (h_next_full[gi] > 17'sd32767)  ? 16'sd32767 :
                                (h_next_full[gi] < -17'sd32768) ? -16'sd32768 :
                                h_next_full[gi][15:0];

            assign ch_prod[gi] = h_next[gi] * c_elem[gi];
        end
    endgenerate

    reg signed [23:0] y_sum;
    integer k;
    always @(*) begin
        y_sum = 24'sd0;
        for (k = 0; k < D_STATE; k = k + 1) begin
            y_sum = y_sum + (ch_prod[k] >>> 8);
        end
    end

    (* multstyle = "dsp" *) wire signed [31:0] gated_full = y_sum[15:0] * z_gate_q88;

    integer j;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out   <= 1'b0;
            y_gated_q88 <= 16'sd0;
            for (j = 0; j < D_STATE; j = j + 1) begin
                h_state[j] <= 16'sd0;
            end
        end else begin
            valid_out <= valid_in;
            if (valid_in) begin
                for (j = 0; j < D_STATE; j = j + 1) begin
                    h_state[j] <= h_next[j];
                end
                y_gated_q88 <= gated_full[23:8];
            end
        end
    end

endmodule

