// =============================================================================
// Module: bitmamba_axi_top.v
// Target: Intel Cyclone V SoC (5CSEBA6U23I7 DE10-Nano / 5CSEMA5F31C6 DE1-SoC)
// Description:
//   Top-Level BitMamba-CV Accelerator Core integrating:
//     1. On-Chip M10K Dual-Port Weight BRAM (Radix-3 Packed: 5 trits/byte)
//     2. 4x Parallel Radix-3 Combinational LUT Decoders (20 trits/cycle -> 16-lane PE)
//     3. 16-Point Streaming Fast Walsh-Hadamard Transform (FWHT, 0 DSPs)
//     4. 16-Lane Zero-DSP BitLinear Ternary Adder-Tree Engine
//     5. 16-Segment PWL SiLU + CPoT Mamba-2 Selective Scan Engine (O(1) State)
//     6. Memory-Mapped Slave Interface + Hardware Cycle Counter for HPS ARM AXI Bridge
// =============================================================================

`timescale 1ns / 1ps

module bitmamba_axi_top #(
    parameter integer P_LANES    = 16,
    parameter integer D_STATE    = 16,
    parameter integer M10K_DEPTH = 4096
) (
    input  wire         clk,
    input  wire         rst_n,
    // Lightweight HPS-to-FPGA Avalon-MM / AXI-Lite Slave Interface
    input  wire         avs_write,
    input  wire         avs_read,
    input  wire [5:0]   avs_address,
    input  wire [31:0]  avs_writedata,
    output reg  [31:0]  avs_readdata,
    output wire         irq_done
);

    // On-chip M10K BRAM storing Radix-3 compressed weights (4 bytes = 20 trits per row)
    (* ramstyle = "M10K" *) reg [31:0] weight_m10k_ram [0:M10K_DEPTH-1];

    // Control & status registers
    reg         ctrl_start;
    reg         ctrl_clear_acc;
    reg         cfg_cpot_mode;
    reg  [3:0]  cfg_cpot_shift;
    reg  [11:0] weight_rd_addr;
    reg  [31:0] hw_cycle_counter;
    reg  [127:0] act_in_reg;
    reg  [127:0] b_vec_reg;
    reg  [127:0] c_vec_reg;
    reg  signed [15:0] z_gate_reg;

    // Fetch 4 packed Radix-3 bytes (32 bits -> 20 trits, using 16 lanes)
    reg [31:0] m10k_rd_word;
    always @(posedge clk) begin
        if (avs_write && avs_address == 6'h10) begin
            weight_m10k_ram[weight_rd_addr] <= avs_writedata;
        end
        m10k_rd_word <= weight_m10k_ram[weight_rd_addr];
    end

    wire [19:0] dec_nz;
    wire [19:0] dec_neg;
    genvar di;
    generate
        for (di = 0; di < 4; di = di + 1) begin : gen_r3_dec
            radix3_decoder u_dec (
                .packed_byte (m10k_rd_word[di*8 +: 8]),
                .w_nz        (dec_nz[di*5 +: 5]),
                .w_neg       (dec_neg[di*5 +: 5])
            );
        end
    endgenerate

    // Stage 1: 16-Point Streaming Fast Walsh-Hadamard Transform (0 DSPs)
    wire         fwht_valid;
    wire [127:0] fwht_out;
    fwht_16_stream u_fwht (
        .clk        (clk),
        .rst_n      (rst_n),
        .valid_in   (ctrl_start),
        .x_in_flat  (act_in_reg),
        .valid_out  (fwht_valid),
        .y_out_flat (fwht_out)
    );

    // Stage 2: 16-Lane Zero-DSP BitLinear Engine
    wire               pe_valid;
    wire signed [23:0] pe_acc_out;
    bitlinear_pe #(
        .P_LANES   (P_LANES),
        .ACT_WIDTH (8),
        .ACC_WIDTH (24)
    ) u_bitlinear (
        .clk        (clk),
        .rst_n      (rst_n),
        .clear_acc  (ctrl_clear_acc),
        .valid_in   (fwht_valid),
        .x_vec_flat (fwht_out),
        .w_nz_vec   (dec_nz[P_LANES-1:0]),
        .w_neg_vec  (dec_neg[P_LANES-1:0]),
        .valid_out  (pe_valid),
        .acc_out    (pe_acc_out)
    );

    // Stage 3: Mamba-2 Selective Scan with CPoT Discretization
    wire               ssm_valid;
    wire signed [15:0] ssm_y_out;
    cpot_selective_scan #(
        .D_STATE (D_STATE)
    ) u_ssm (
        .clk           (clk),
        .rst_n         (rst_n),
        .valid_in      (pe_valid),
        .cpot_mode_bit (cfg_cpot_mode),
        .cpot_shift_k  (cfg_cpot_shift),
        .x_in          (pe_acc_out[11:4]),
        .z_gate_q88    (z_gate_reg),
        .b_vec_flat    (b_vec_reg),
        .c_vec_flat    (c_vec_reg),
        .valid_out     (ssm_valid),
        .y_gated_q88   (ssm_y_out)
    );

    assign irq_done = ssm_valid;

    // Avalon-MM / AXI-Lite Register Interface + Cycle Counter
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ctrl_start       <= 1'b0;
            ctrl_clear_acc   <= 1'b0;
            cfg_cpot_mode    <= 1'b1;
            cfg_cpot_shift   <= 4'd5;
            weight_rd_addr   <= 12'd0;
            hw_cycle_counter <= 32'd0;
            act_in_reg       <= 128'd0;
            b_vec_reg        <= 128'd0;
            c_vec_reg        <= 128'd0;
            z_gate_reg       <= 16'sd256;
            avs_readdata     <= 32'd0;
        end else begin
            ctrl_start     <= 1'b0;
            ctrl_clear_acc <= 1'b0;
            hw_cycle_counter <= hw_cycle_counter + 32'd1;

            if (avs_write) begin
                case (avs_address)
                    6'h00: begin
                        ctrl_start     <= avs_writedata[0];
                        ctrl_clear_acc <= avs_writedata[1];
                        cfg_cpot_mode  <= avs_writedata[2];
                        cfg_cpot_shift <= avs_writedata[7:4];
                    end
                    6'h01: weight_rd_addr       <= avs_writedata[11:0];
                    6'h02: act_in_reg[31:0]     <= avs_writedata;
                    6'h03: act_in_reg[63:32]    <= avs_writedata;
                    6'h04: act_in_reg[95:64]    <= avs_writedata;
                    6'h05: act_in_reg[127:96]   <= avs_writedata;
                    6'h06: b_vec_reg[31:0]      <= avs_writedata;
                    6'h07: c_vec_reg[31:0]      <= avs_writedata;
                    6'h08: z_gate_reg           <= avs_writedata[15:0];
                    default: ;
                endcase
            end

            if (avs_read) begin
                case (avs_address)
                    6'h00: avs_readdata <= {15'd0, ssm_valid, ssm_y_out};
                    6'h01: avs_readdata <= {{8{pe_acc_out[23]}}, pe_acc_out};
                    6'h02: avs_readdata <= hw_cycle_counter;
                    default: avs_readdata <= 32'd0;
                endcase
            end
        end
    end

endmodule

