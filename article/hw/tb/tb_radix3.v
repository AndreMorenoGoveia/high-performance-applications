// Exhaustive check of radix3_decoder (256 codes) against golden RADIX3_LUT.
`timescale 1ns / 1ps
module tb_radix3;
    parameter VEC = "vectors/radix3_expected.hex";
    reg  [7:0] code;
    wire [9:0] trits;
    reg  [9:0] exp_mem [0:255];
    integer i, err;
    radix3_decoder dut (.packed_byte(code), .trits(trits));
    initial begin
        $readmemh(VEC, exp_mem);
        err = 0;
        for (i = 0; i < 256; i = i + 1) begin
            code = i; #1;
            if (trits !== exp_mem[i]) err = err + 1;
        end
        $display("RESULT radix3 checked=256 mismatches=%0d", err);
        $finish;
    end
endmodule
