// =============================================================================
// Module: fwht_16_stream.v
// Target: Intel Cyclone V (Zero-DSP Multiplier-Free Butterfly Pipeline)
// Description:
//   Computes a 16-point Fast Walsh-Hadamard Transform (FWHT) over 8-bit signed
//   activations in 4 pipelined butterfly stages of pure additions/subtractions,
//   followed by an exact 2-bit arithmetic right shift (1/sqrt(16) = 1/4 = 2^-2).
//   Uses 0 DSP blocks (100% mapped to Cyclone V ALM carry-chains).
// =============================================================================

`timescale 1ns / 1ps

module fwht_16_stream (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         valid_in,
    input  wire [127:0] x_in_flat,   // 16 x 8-bit signed activations
    output reg          valid_out,
    output reg  [127:0] y_out_flat   // 16 x 8-bit signed rotated activations
);

    // Unpack input vector
    wire signed [7:0] x_in [0:15];
    genvar gi;
    generate
        for (gi = 0; gi < 16; gi = gi + 1) begin : gen_unpack
            assign x_in[gi] = x_in_flat[gi*8 +: 8];
        end
    endgenerate

    // Pipeline registers for 4 butterfly stages (12-bit internal width to prevent overflow)
    reg signed [11:0] s1 [0:15];
    reg signed [11:0] s2 [0:15];
    reg signed [11:0] s3 [0:15];
    reg signed [11:0] s4 [0:15];
    reg [3:0]         v_pipe;

    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v_pipe    <= 4'b0000;
            valid_out <= 1'b0;
            y_out_flat <= 128'd0;
            for (i = 0; i < 16; i = i + 1) begin
                s1[i] <= 12'sd0;
                s2[i] <= 12'sd0;
                s3[i] <= 12'sd0;
                s4[i] <= 12'sd0;
            end
        end else begin
            v_pipe <= {v_pipe[2:0], valid_in};

            // Stage 1: Stride 1
            for (i = 0; i < 16; i = i + 2) begin
                s1[i]   <= $signed(x_in[i]) + $signed(x_in[i+1]);
                s1[i+1] <= $signed(x_in[i]) - $signed(x_in[i+1]);
            end

            // Stage 2: Stride 2
            for (i = 0; i < 16; i = i + 4) begin
                s2[i]   <= s1[i]   + s1[i+2];
                s2[i+1] <= s1[i+1] + s1[i+3];
                s2[i+2] <= s1[i]   - s1[i+2];
                s2[i+3] <= s1[i+1] - s1[i+3];
            end

            // Stage 3: Stride 4
            for (i = 0; i < 16; i = i + 8) begin
                s3[i+0] <= s2[i+0] + s2[i+4];
                s3[i+1] <= s2[i+1] + s2[i+5];
                s3[i+2] <= s2[i+2] + s2[i+6];
                s3[i+3] <= s2[i+3] + s2[i+7];
                s3[i+4] <= s2[i+0] - s2[i+4];
                s3[i+5] <= s2[i+1] - s2[i+5];
                s3[i+6] <= s2[i+2] - s2[i+6];
                s3[i+7] <= s2[i+3] - s2[i+7];
            end

            // Stage 4: Stride 8
            for (i = 0; i < 8; i = i + 1) begin
                s4[i]   <= s3[i] + s3[i+8];
                s4[i+8] <= s3[i] - s3[i+8];
            end

            // Output stage: Exact 1/sqrt(16) = 1/4 scaling via >>> 2 and saturation to [-127, 127]
            valid_out <= v_pipe[3];
            for (i = 0; i < 16; i = i + 1) begin
                if ((s4[i] >>> 2) > 12'sd127)
                    y_out_flat[i*8 +: 8] <= 8'sd127;
                else if ((s4[i] >>> 2) < -12'sd127)
                    y_out_flat[i*8 +: 8] <= -8'sd127;
                else
                    y_out_flat[i*8 +: 8] <= s4[i][9:2];
            end
        end
    end

endmodule

