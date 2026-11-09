// Matrix-vector engine y[M] = W[M,K] * x[K] with P parallel lanes.
//   ENC = 0 : INT8 weights, products on DSP blocks          (8   bit/weight)
//   ENC = 1 : ternary, 2-bit code {nz,neg}, ALM mux + adders (2   bit/weight)
//   ENC = 2 : ternary, Radix-3 (5 trits/byte), ALM decoders  (1.6 bit/weight)
// Weights live in on-chip RAM (M10K), word w holds P lanes of flat row r=m*KC+chunk
// (KC = K/P). For ENC=2 one 8P-bit word holds 5 consecutive flat rows.
// Weights are loaded word by word through the w_* port (host / HPS), activations
// (int8) are written into a K-entry buffer before `start`.
// Outputs: y_valid pulses once per output row m (in order), golden: matvec_int.
`timescale 1ns / 1ps

module bitlinear_engine #(
    parameter integer P         = 16,
    parameter integer K         = 64,
    parameter integer M         = 32,
    parameter integer ENC       = 2,
    parameter integer ACC_W     = 24,
    // derived, exposed only so the port widths below can use them
    parameter integer WW        = (ENC == 1) ? 2 * P : 8 * P,
    parameter integer DEPTH     = (ENC == 2) ? ((M * (K / P)) + 4) / 5 : M * (K / P)
) (
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    w_we,
    input  wire [$clog2(DEPTH)-1:0] w_addr,
    input  wire [WW-1:0]           w_data,
    input  wire                    x_we,
    input  wire [$clog2(K)-1:0]    x_addr,
    input  wire [7:0]              x_data,
    input  wire                    start,
    output reg                     y_valid,
    output reg  signed [ACC_W-1:0] y_data,
    output wire                    busy
);
    localparam integer KC     = K / P;
    localparam integer ROWS   = M * KC;                        // flat P-wide rows
    localparam integer LV     = $clog2(P);
    localparam integer LEAF_W = (ENC == 0) ? 16 : 9;
    localparam integer LAT    = 2 + LV;                        // issue -> tree output

    // ---------------- activation buffer ----------------
    reg [8*P-1:0] xbuf [0:KC-1];
    always @(posedge clk)
        if (x_we) xbuf[x_addr / P][(x_addr % P)*8 +: 8] <= x_data;

    // ---------------- weight memory ----------------
    (* ramstyle = "M10K" *) reg [WW-1:0] wmem [0:DEPTH-1];
    always @(posedge clk)
        if (w_we) wmem[w_addr] <= w_data;

    // ---------------- issue stage ----------------
    reg                       run;
    reg [$clog2(DEPTH+1)-1:0] waddr;
    reg [$clog2(KC+1)-1:0]    chunk;
    reg [2:0]                 sub;
    reg [$clog2(ROWS+1)-1:0]  issued;
    wire last_issue = (issued == ROWS - 1);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            run <= 1'b0; waddr <= 0; chunk <= 0; sub <= 0; issued <= 0;
        end else if (start && !run) begin
            run <= 1'b1; waddr <= 0; chunk <= 0; sub <= 0; issued <= 0;
        end else if (run) begin
            issued <= issued + 1'b1;
            chunk  <= (chunk == KC - 1) ? 0 : chunk + 1'b1;
            if (ENC == 2) begin
                sub   <= (sub == 3'd4) ? 3'd0 : sub + 3'd1;
                waddr <= (sub == 3'd4) ? waddr + 1'b1 : waddr;
            end else begin
                waddr <= waddr + 1'b1;
            end
            if (last_issue) run <= 1'b0;
        end
    end

    // ---------------- stage 1: RAM + activation read ----------------
    reg [WW-1:0]  w_word;
    reg [8*P-1:0] x_word;
    reg [2:0]     sub1;
    always @(posedge clk) begin
        w_word <= wmem[waddr];
        x_word <= xbuf[chunk];
        sub1   <= sub;
    end

    // ---------------- weight decode (combinational) ----------------
    wire [2*P-1:0] tern;        // ENC 1/2: 2-bit trit per lane
    genvar i;
    generate
        if (ENC == 2) begin : g_r3
            wire [10*P-1:0] all_trits;  // 5P trits decoded from P bytes
            for (i = 0; i < P; i = i + 1) begin : g_dec
                radix3_decoder u_dec (.packed_byte(w_word[8*i +: 8]), .trits(all_trits[10*i +: 10]));
            end
            assign tern = all_trits[sub1 * 2 * P +: 2 * P];
        end else begin : g_t2
            assign tern = w_word[2*P-1:0];
        end
    endgenerate

    // ---------------- stage 2: lane products (tree leaves) ----------------
    reg [P*LEAF_W-1:0] leaves;
    generate
        for (i = 0; i < P; i = i + 1) begin : g_lane
            wire signed [7:0] xl = x_word[8*i +: 8];
            if (ENC == 0) begin : g_mul
                wire signed [7:0] wl = w_word[8*i +: 8];
                (* multstyle = "dsp" *) wire signed [15:0] prod = xl * wl;
                always @(posedge clk) leaves[LEAF_W*i +: LEAF_W] <= prod;
            end else begin : g_tern
                wire signed [8:0] xe = {xl[7], xl};
                always @(posedge clk)
                    leaves[LEAF_W*i +: LEAF_W] <= !tern[2*i] ? 9'sd0 : (tern[2*i+1] ? -xe : xe);
            end
        end
    endgenerate

    wire signed [ACC_W-1:0] tree_sum;
    adder_tree #(.P(P), .IN_W(LEAF_W), .OUT_W(ACC_W)) u_tree (
        .clk(clk), .in_flat(leaves), .sum(tree_sum));

    // ---------------- control pipeline + accumulator ----------------
    reg [LAT-1:0] v_pipe, first_pipe, last_pipe;
    wire first_issue = (chunk == 0);
    wire lastk_issue = (chunk == KC - 1);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v_pipe <= 0; first_pipe <= 0; last_pipe <= 0;
        end else begin
            v_pipe     <= {v_pipe[LAT-2:0],     run};
            first_pipe <= {first_pipe[LAT-2:0], first_issue};
            last_pipe  <= {last_pipe[LAT-2:0],  lastk_issue};
        end
    end

    reg signed [ACC_W-1:0] acc;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc <= 0; y_valid <= 1'b0; y_data <= 0;
        end else begin
            y_valid <= 1'b0;
            if (v_pipe[LAT-1]) begin
                acc <= first_pipe[LAT-1] ? tree_sum : acc + tree_sum;
                if (last_pipe[LAT-1]) begin
                    y_valid <= 1'b1;
                    y_data  <= first_pipe[LAT-1] ? tree_sum : acc + tree_sum;
                end
            end
        end
    end

    assign busy = run | (|v_pipe);
endmodule
