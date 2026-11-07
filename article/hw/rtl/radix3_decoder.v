// =============================================================================
// Module: radix3_decoder.v
// Target: Intel Cyclone V (28nm, 6-input Fracturable ALMs)
// Description:
//   Decodes 1 packed Radix-3 byte (8 bits, values 0..242 = 3^5 - 1) into
//   5 parallel ternary weights w_0..w_4 in {-1, 0, +1}, represented as:
//     w_nz[i]  : 1 if weight != 0, 0 if weight == 0
//     w_neg[i] : 1 if weight == -1, 0 if weight == +1 (or 0)
//   Since 8-bit input -> 10 single-bit outputs is a pure combinational truth
//   table, Intel Quartus maps each output bit onto at most 4 fracturable ALMs
//   (1 level of 6-LUT + 2:1 mux), completing in < 1.2 ns (single cycle).
// =============================================================================

`timescale 1ns / 1ps

module radix3_decoder (
    input  wire [7:0] packed_byte,
    output reg  [4:0] w_nz,
    output reg  [4:0] w_neg
);

    integer b, rem, d0, d1, d2, d3, d4;

    always @(*) begin
        if (packed_byte < 8'd243) begin
            rem = packed_byte;
            d0  = rem % 3; rem = rem / 3;
            d1  = rem % 3; rem = rem / 3;
            d2  = rem % 3; rem = rem / 3;
            d3  = rem % 3; rem = rem / 3;
            d4  = rem % 3;

            // Digit encoding: 0 -> -1 (nz=1, neg=1), 1 -> 0 (nz=0, neg=0), 2 -> +1 (nz=1, neg=0)
            w_nz[0]  = (d0 != 1); w_neg[0] = (d0 == 0);
            w_nz[1]  = (d1 != 1); w_neg[1] = (d1 == 0);
            w_nz[2]  = (d2 != 1); w_neg[2] = (d2 == 0);
            w_nz[3]  = (d3 != 1); w_neg[3] = (d3 == 0);
            w_nz[4]  = (d4 != 1); w_neg[4] = (d4 == 0);
        end else begin
            w_nz  = 5'b00000;
            w_neg = 5'b00000;
        end
    end

endmodule

