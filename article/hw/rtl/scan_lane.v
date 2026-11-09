// Mamba-2 selective-scan lane: one channel x_t, N parallel states (golden: scan_lane_step)
//   h_t[n] = sat( decay(h_{t-1}[n]) + ((B_t[n]*x_t) << GUARD) >>> j_t )
//   y_t    = sat( (sum_n C_t[n]*h_t[n]) >>> GUARD )
// decay (DECAY=0, CPoT, 0 multipliers): s = (h + rnd) >>> k ; m ? h - s : s
// decay (DECAY=1, baseline):            (h * a_q15 + 2^14) >>> 15   (DSP multiplier)
// Per-token codes (m,k,j / a_q15) come with the token: the scan is input-selective.
// Pipeline: S0 B*x (DSP) | S1 recurrence (1-cycle loop) | S2 C*h (DSP) | tree | out.
`timescale 1ns / 1ps

module scan_lane #(
    parameter integer N       = 16,
    parameter integer STATE_W = 32,
    parameter integer GUARD   = 10,
    parameter integer ROUND   = 1,
    parameter integer DECAY   = 0,
    parameter integer Y_W     = 24
) (
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  clear,       // zero the state (new sequence)
    input  wire                  valid_in,
    input  wire signed [7:0]     x_in,
    input  wire [N*8-1:0]        b_flat,
    input  wire [N*8-1:0]        c_flat,
    input  wire                  m_in,
    input  wire [3:0]            k_in,
    input  wire [3:0]            j_in,
    input  wire [15:0]           a_q15_in,
    output reg                   valid_out,
    output reg  signed [Y_W-1:0] y_out
);
    localparam integer LV     = $clog2(N);
    localparam integer PROD_W = STATE_W + 8;
    localparam integer SUM_W  = PROD_W + LV;
    // exact-arithmetic width for decay + injection (no overflow before saturation)
    localparam integer W2     = ((STATE_W > 16 + GUARD) ? STATE_W : 16 + GUARD) + 3;

    function signed [STATE_W-1:0] sat_state(input signed [W2-1:0] v);
        begin
            if (v > $signed({{(W2-STATE_W+1){1'b0}}, {(STATE_W-1){1'b1}}}))
                sat_state = {1'b0, {(STATE_W-1){1'b1}}};
            else if (v < -$signed({{(W2-STATE_W+1){1'b0}}, {(STATE_W-1){1'b1}}}) - 1)
                sat_state = {1'b1, {(STATE_W-1){1'b0}}};
            else
                sat_state = v[STATE_W-1:0];
        end
    endfunction

    // ---------------- S0: input products ----------------
    reg [N*16-1:0] bx0;
    reg [N*8-1:0]  c0;
    reg            v0, m0, clr0;
    reg [3:0]      k0, j0;
    reg [15:0]     a0;
    genvar n;
    generate
        for (n = 0; n < N; n = n + 1) begin : g_bx
            wire signed [7:0] bn = b_flat[8*n +: 8];
            (* multstyle = "dsp" *) wire signed [15:0] p = bn * x_in;
            always @(posedge clk) bx0[16*n +: 16] <= p;
        end
    endgenerate
    always @(posedge clk) begin
        c0 <= c_flat; m0 <= m_in; k0 <= k_in; j0 <= j_in; a0 <= a_q15_in;
    end
    always @(posedge clk or negedge rst_n)
        if (!rst_n) begin v0 <= 1'b0; clr0 <= 1'b0; end
        else        begin v0 <= valid_in; clr0 <= clear; end

    // ---------------- S1: recurrence ----------------
    reg [N*STATE_W-1:0] h;
    reg [N*8-1:0]       c1;
    reg                 v1;
    generate
        for (n = 0; n < N; n = n + 1) begin : g_state
            wire signed [STATE_W-1:0] hn  = h[STATE_W*n +: STATE_W];
            wire signed [W2-1:0]      hw  = hn;
            wire signed [15+GUARD:0]  bxg = $signed(bx0[16*n +: 16]) <<< GUARD;
            wire signed [15+GUARD:0]  injn = bxg >>> j0;
            wire signed [W2-1:0]      inj = injn;
            wire signed [W2-1:0]      dec;
            if (DECAY == 0) begin : g_cpot
                wire signed [STATE_W:0]   hx  = hn;
                wire        [STATE_W-1:0] rnd = (ROUND != 0 && k0 != 0) ? ({{(STATE_W-1){1'b0}}, 1'b1} << (k0 - 1)) : 0;
                wire signed [STATE_W:0]   hr  = hx + $signed({1'b0, rnd});
                wire signed [STATE_W:0]   s   = hr >>> k0;
                wire signed [W2-1:0]      sw  = s;
                assign dec = m0 ? hw - sw : sw;
            end else begin : g_mult
                (* multstyle = "dsp" *) wire signed [STATE_W+16:0] prod = hn * $signed({1'b0, a0});
                wire signed [STATE_W+16:0] dsh = (prod + 16384) >>> 15;   // |dsh| <= |h|
                assign dec = dsh;
            end
            always @(posedge clk or negedge rst_n)
                if (!rst_n)        h[STATE_W*n +: STATE_W] <= 0;
                else if (clr0)     h[STATE_W*n +: STATE_W] <= 0;
                else if (v0)       h[STATE_W*n +: STATE_W] <= sat_state(dec + inj);
        end
    endgenerate
    always @(posedge clk) c1 <= c0;
    always @(posedge clk or negedge rst_n)
        if (!rst_n) v1 <= 1'b0; else v1 <= v0;

    // ---------------- S2: readout products + tree ----------------
    reg [N*PROD_W-1:0] ch;
    generate
        for (n = 0; n < N; n = n + 1) begin : g_ch
            wire signed [STATE_W-1:0] hn = h[STATE_W*n +: STATE_W];
            wire signed [7:0]         cn = c1[8*n +: 8];
            (* multstyle = "dsp" *) wire signed [PROD_W-1:0] p = hn * cn;
            always @(posedge clk) ch[PROD_W*n +: PROD_W] <= p;
        end
    endgenerate

    wire signed [SUM_W-1:0] acc;
    adder_tree #(.P(N), .IN_W(PROD_W), .OUT_W(SUM_W)) u_tree (.clk(clk), .in_flat(ch), .sum(acc));

    reg [LV:0] v_pipe;   // S2 product reg + LV tree levels
    always @(posedge clk or negedge rst_n)
        if (!rst_n) v_pipe <= 0; else v_pipe <= {v_pipe[LV-1:0], v1};

    wire signed [SUM_W-1:0] ys = acc >>> GUARD;
    localparam signed [SUM_W-1:0] YMAX = (1 <<< (Y_W - 1)) - 1;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin valid_out <= 1'b0; y_out <= 0; end
        else begin
            valid_out <= v_pipe[LV];
            y_out <= (ys > YMAX) ? YMAX[Y_W-1:0] : (ys < -YMAX - 1) ? -YMAX - 1 : ys[Y_W-1:0];
        end
    end
endmodule
