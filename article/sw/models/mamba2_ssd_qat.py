"""
Mamba-2 (Structured State Space Duality - SSD) with Hardware-Aware QAT:
- H-BitLinear: In-graph Fast Walsh-Hadamard Transform (FWHT) + 1.58-bit Ternary STE
  (Solves the rotation vs. ternary weight preservation conflict, cf. BitNet v2).
- Bimodal Complementary Power-of-Two (CPoT) Discretization for scalar-per-head A_bar:
  A_bar in {2^-k} U {1 - 2^-k}, preserving long-memory eigenvalues near 1.0 without multipliers.
- Piecewise-Linear (PWL) SiLU approximation aligned to power-of-two knot grids for M10K BRAM.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Literal, Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


WeightMode = Literal["fp32", "int8", "int4", "ternary"]
AbarMode = Literal["exact", "fixed16", "pot", "cpot", "cpot2"]
SiluMode = Literal["exact", "pwl16"]


@dataclass
class BitMamba2Config:
    d_model: int = 64
    expand: int = 2
    n_heads: int = 4
    d_state: int = 16
    d_conv: int = 4
    fwht_block_size: int = 16
    weight_mode: WeightMode = "ternary"
    use_fwht: bool = True
    abar_mode: AbarMode = "cpot"
    silu_mode: SiluMode = "pwl16"
    act_bits: int = 8

    @property
    def d_inner(self) -> int:
        return self.d_model * self.expand

    @property
    def head_dim(self) -> int:
        assert self.d_inner % self.n_heads == 0
        return self.d_inner // self.n_heads


def fast_walsh_hadamard_1d(x: torch.Tensor, block_size: int = 16) -> torch.Tensor:
    """
    Applies a block-diagonal Fast Walsh-Hadamard Transform (FWHT) along the last dimension.
    For block_size = 16 (2^4), 1/sqrt(16) = 1/4 = 2^-2, which in hardware is an exact
    2-bit arithmetic right shift with ZERO multipliers.
    """
    orig_shape = x.shape
    d = orig_shape[-1]
    assert d % block_size == 0, f"Dimension {d} must be divisible by fwht_block_size={block_size}"
    num_stages = int(round(math.log2(block_size)))
    assert (1 << num_stages) == block_size, "block_size must be a power of 2"

    h = x.reshape(*orig_shape[:-1], d // block_size, block_size)
    # Iterative butterfly stages using pure additions and subtractions
    step = 1
    for _ in range(num_stages):
        h = h.reshape(*orig_shape[:-1], d // block_size, block_size // (2 * step), 2, step)
        a = h[..., 0, :]
        b = h[..., 1, :]
        h = torch.stack([a + b, a - b], dim=-2)
        step *= 2
    h = h.reshape(orig_shape)
    # Exact power-of-two scaling when log2(block_size) is even (e.g., B=16 -> divide by 4)
    scale = 1.0 / math.sqrt(block_size)
    return h * scale


def ste_round(x: torch.Tensor) -> torch.Tensor:
    """Straight-Through Estimator for rounding."""
    return (x.round() - x).detach() + x


def quantize_weight(w: torch.Tensor, mode: WeightMode) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes weight matrix W using Straight-Through Estimator (STE).
    Returns (w_dequantized, gamma_scale).
    """
    if mode == "fp32":
        return w, torch.ones(1, device=w.device, dtype=w.dtype)

    if mode == "ternary":
        # BitNet b1.58 AbsMean quantization: gamma = mean(|W|)
        gamma = w.abs().mean().clamp(min=1e-6)
        w_scaled = w / gamma
        w_q = ste_round(w_scaled).clamp(-1.0, 1.0)
        return w_q * gamma, gamma

    if mode == "int4":
        qmax = 7.0
        gamma = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / qmax
        w_q = ste_round(w / gamma).clamp(-8.0, 7.0)
        return w_q * gamma, gamma

    if mode == "int8":
        qmax = 127.0
        gamma = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / qmax
        w_q = ste_round(w / gamma).clamp(-128.0, 127.0)
        return w_q * gamma, gamma

    raise ValueError(f"Unknown weight_mode: {mode}")


def quantize_activation_per_token(x: torch.Tensor, bits: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-token symmetric AbsMax activation quantization to INT8 ([-127, 127]) with STE.
    """
    if bits >= 32:
        return x, torch.ones_like(x[..., :1])
    qmax = float((1 << (bits - 1)) - 1)
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / qmax
    x_q = ste_round(x / scale).clamp(-qmax, qmax)
    return x_q * scale, scale


class HBitLinear(nn.Module):
    """
    H-BitLinear Layer (In-Graph Hadamard + Ternary/Low-Bit Linear Projection).
    Because the FWHT is applied inside the QAT forward pass before quantization,
    the ternary weights W_q in {-1, 0, +1} are learned directly in the Hadamard domain,
    preserving 100% ternary weights at inference time.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        weight_mode: WeightMode = "ternary",
        use_fwht: bool = True,
        fwht_block_size: int = 16,
        act_bits: int = 8,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mode = weight_mode
        self.use_fwht = use_fwht
        self.fwht_block_size = fwht_block_size
        self.act_bits = act_bits

        self.norm = nn.RMSNorm(in_features, eps=1e-5)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    @torch.no_grad()
    def get_ternary_weights(self) -> Tuple[torch.Tensor, float]:
        """Returns integer ternary weights in {-1, 0, +1} and scalar scale gamma."""
        gamma = self.weight.abs().mean().clamp(min=1e-6)
        w_int = (self.weight / gamma).round().clamp(-1, 1).to(torch.int8)
        return w_int, float(gamma.item())

    @torch.no_grad()
    def get_zero_sparsity(self) -> float:
        """Returns fraction of zero weights (w == 0) in ternary mode."""
        w_int, _ = self.get_ternary_weights()
        return float((w_int == 0).float().mean().item())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        if self.use_fwht:
            x_rot = fast_walsh_hadamard_1d(x_norm, self.fwht_block_size)
        else:
            x_rot = x_norm

        if self.weight_mode == "fp32":
            x_quant = x_rot
        else:
            x_quant, _ = quantize_activation_per_token(x_rot, bits=self.act_bits)

        w_quant, _ = quantize_weight(self.weight, self.weight_mode)
        return F.linear(x_quant, w_quant, self.bias)


def quantize_abar(abar: torch.Tensor, mode: AbarMode, max_shift: int = 10) -> torch.Tensor:
    """
    Discretizes the recurrent decay factor A_bar = exp(-dt * A) in (0, 1).
    Modes:
    - 'exact': FP32 exact
    - 'fixed16': 16-bit uniform fixed-point (Q1.15)
    - 'pot': Standard Power-of-Two 2^-k (levels: 0.5, 0.25, 0.125...; loses long memory near 1.0!)
    - 'cpot': Bimodal Complementary Power-of-Two {2^-k} U {1 - 2^-k}
              Near 1.0, 1 - 2^-k gives {0.75, 0.875, 0.9375, 0.96875, 0.984375, 0.9921875, 0.99902}!
    - 'cpot2': Dual-shift Complementary PoT (1 - 2^-k1 - 2^-k2)
    """
    if mode == "exact":
        return abar

    abar_clamped = abar.clamp(1e-5, 1.0 - 1e-5)

    if mode == "fixed16":
        scale = 32768.0
        q = (abar_clamped * scale).round() / scale
        return (q - abar).detach() + abar

    if mode == "pot":
        # Standard PoT: 2^-k for k in {0, 1, ..., max_shift}
        # Note: if k=0 is allowed, A_bar=1.0 causes marginal stability/explosion;
        # if k>=1, max value is 0.5, erasing long memory! We include k in [1..max_shift]
        # or round in log2 domain:
        k = (-torch.log2(abar_clamped)).round().clamp(1, max_shift)
        q = torch.pow(2.0, -k)
        return (q - abar).detach() + abar

    if mode == "cpot":
        # Branch 0 (abar <= 0.5): 2^-k, k in [1..max_shift]
        k_low = (-torch.log2(abar_clamped)).round().clamp(1, max_shift)
        q_low = torch.pow(2.0, -k_low)

        # Branch 1 (abar > 0.5): 1 - 2^-k, where 2^-k approximates (1 - abar)
        comp = (1.0 - abar_clamped).clamp(1e-5, 0.5)
        k_high = (-torch.log2(comp)).round().clamp(1, max_shift)
        q_high = 1.0 - torch.pow(2.0, -k_high)

        q = torch.where(abar_clamped > 0.625, q_high, q_low)
        return (q - abar).detach() + abar

    if mode == "cpot2":
        # Two-term Canonical Signed-Digit Complementary PoT: 1 - 2^-k1 +/- 2^-k2
        comp = (1.0 - abar_clamped).clamp(1e-5, 0.5)
        k1 = (-torch.log2(comp)).round().clamp(1, max_shift)
        rem = comp - torch.pow(2.0, -k1)
        sign_rem = torch.sign(rem)
        k2 = (-torch.log2(rem.abs().clamp(1e-7, 0.5))).round().clamp(1, max_shift + 3)
        q_high = (1.0 - torch.pow(2.0, -k1) - sign_rem * torch.pow(2.0, -k2)).clamp(
            0.5, 1.0 - 2.0 ** (-(max_shift + 2))
        )

        k_low = (-torch.log2(abar_clamped)).round().clamp(1, max_shift)
        q_low = torch.pow(2.0, -k_low)
        q = torch.where(abar_clamped > 0.625, q_high, q_low)
        return (q - abar).detach() + abar

    raise ValueError(f"Unknown abar_mode: {mode}")


def pwl_silu(x: torch.Tensor, mode: SiluMode = "pwl16") -> torch.Tensor:
    """
    16-segment Piecewise-Linear (PWL) approximation of SiLU(x) = x * sigmoid(x)
    over [-4, 4] with uniform step delta = 0.5 (power-of-two grid for direct bit-indexing in M10K).
    Outside [-4, 4]: 0 for x <= -4, x for x >= 4.
    """
    if mode == "exact":
        return F.silu(x)

    # Precomputed knot points on [-4.0, 4.0] with step 0.5 (17 knots -> 16 segments)
    x_clamped = x.clamp(-4.0, 3.999)
    idx_float = (x_clamped + 4.0) * 2.0  # step = 0.5 -> multiply by 2 (1-bit shift!)
    idx0 = torch.floor(idx_float)
    frac = idx_float - idx0

    x0 = idx0 * 0.5 - 4.0
    x1 = x0 + 0.5
    y0 = x0 * torch.sigmoid(x0)
    y1 = x1 * torch.sigmoid(x1)
    y_pwl = y0 + frac * (y1 - y0)

    out = torch.where(x <= -4.0, torch.zeros_like(x), torch.where(x >= 4.0, x, y_pwl))
    # Straight-through gradient from exact SiLU during training
    if x.requires_grad:
        exact = F.silu(x)
        return (out - exact).detach() + exact
    return out


class BitMamba2Block(nn.Module):
    """
    Hardware-Co-Designed Mamba-2 (SSD) Block:
    - Input projection (`in_proj`) and Output projection (`out_proj`) use H-BitLinear (1.58-bit + FWHT).
    - Scalar per-head state decay A_h with Complementary Power-of-Two (CPoT) discretization.
    - Depthwise Conv1D (k=4) + PWL SiLU + O(1) Recurrent Selective Scan.
    """

    def __init__(self, cfg: BitMamba2Config):
        super().__init__()
        self.cfg = cfg
        d_inner = cfg.d_inner
        n_heads = cfg.n_heads
        d_state = cfg.d_state

        # Total output of in_proj: z (d_inner) + x (d_inner) + B (n_heads * d_state) + C (n_heads * d_state) + dt (n_heads)
        self.d_proj_out = 2 * d_inner + 2 * n_heads * d_state + n_heads

        self.in_proj = HBitLinear(
            cfg.d_model,
            self.d_proj_out,
            bias=False,
            weight_mode=cfg.weight_mode,
            use_fwht=cfg.use_fwht,
            fwht_block_size=cfg.fwht_block_size,
            act_bits=cfg.act_bits,
        )

        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=cfg.d_conv,
            groups=d_inner,
            padding=cfg.d_conv - 1,
            bias=True,
        )

        # Scalar per-head log(A) parameter initialized so exp(-dt * A) spans (0.90, 0.995) for long memory
        A_init = torch.arange(1, n_heads + 1, dtype=torch.float32) * 0.5
        self.A_log = nn.Parameter(torch.log(A_init))
        self.dt_bias = nn.Parameter(torch.full((n_heads,), -2.2))  # softplus(-2.2) ~ 0.10
        self.D = nn.Parameter(torch.ones(n_heads))

        self.out_proj = HBitLinear(
            d_inner,
            cfg.d_model,
            bias=False,
            weight_mode=cfg.weight_mode,
            use_fwht=cfg.use_fwht,
            fwht_block_size=cfg.fwht_block_size,
            act_bits=cfg.act_bits,
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        u: (B, L, d_model)
        Returns: (B, L, d_model)
        """
        batch_size, seq_len, _ = u.shape
        cfg = self.cfg
        d_inner = cfg.d_inner
        n_heads = cfg.n_heads
        head_dim = cfg.head_dim
        d_state = cfg.d_state

        proj = self.in_proj(u)
        z, x_conv_in, B_raw, C_raw, dt_raw = torch.split(
            proj,
            [d_inner, d_inner, n_heads * d_state, n_heads * d_state, n_heads],
            dim=-1,
        )

        # Depthwise causal Conv1D + PWL SiLU
        x_conv = self.conv1d(x_conv_in.transpose(1, 2))[..., :seq_len].transpose(1, 2)
        x_act = pwl_silu(x_conv, mode=cfg.silu_mode)
        z_act = pwl_silu(z, mode=cfg.silu_mode)

        if cfg.weight_mode != "fp32":
            x_act, _ = quantize_activation_per_token(x_act, bits=cfg.act_bits)
            B_raw, _ = quantize_activation_per_token(B_raw, bits=cfg.act_bits)
            C_raw, _ = quantize_activation_per_token(C_raw, bits=cfg.act_bits)

        # Reshape into multi-head SSD layout:
        # x_heads: (B, L, H, P_head)
        # B_heads, C_heads: (B, L, H, N)
        x_heads = x_act.view(batch_size, seq_len, n_heads, head_dim)
        B_heads = B_raw.view(batch_size, seq_len, n_heads, d_state)
        C_heads = C_raw.view(batch_size, seq_len, n_heads, d_state)

        # Discretization of scalar per-head decay A_bar_t = exp(-dt_t * A_h)
        dt = F.softplus(dt_raw + self.dt_bias).clamp(min=1e-4, max=2.0)  # (B, L, H)
        A = torch.exp(self.A_log).view(1, 1, n_heads)  # (1, 1, H)
        abar_exact = torch.exp(-dt * A)  # (B, L, H) in (0, 1)
        abar = quantize_abar(abar_exact, mode=cfg.abar_mode)  # (B, L, H)

        # Vectorized exact associative scan for training speed:
        # Since A_bar is scalar per head (B, L, H), we can compute the exact recurrence
        # h_t = sum_{s=0}^t (prod_{r=s+1}^t abar_r) u_s via causal decay matrix M_{t,s}
        dt_expanded = dt.unsqueeze(-1).unsqueeze(-1)  # (B, L, H, 1, 1)
        u_ssm = dt_expanded * (x_heads.unsqueeze(-1) * B_heads.unsqueeze(-2))  # (B, L, H, P_head, N)

        # Log-space cumulative sum for exact parallel recurrence across L:
        log_abar = torch.log(abar.clamp(min=1e-10))  # (B, L, H)
        csum = torch.cumsum(log_abar, dim=1)  # (B, L, H)
        # diff[b, t, s, h] = csum[b, t, h] - csum[b, s, h] for t >= s
        diff = csum.unsqueeze(2) - csum.unsqueeze(1)  # (B, L, L, H)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=u.device, dtype=torch.bool))
        diff = diff.masked_fill(~causal_mask.view(1, seq_len, seq_len, 1), -1e9)
        decay_matrix = torch.exp(diff)  # (B, L, L, H)

        # Contract over s: h_all (B, L, H, P_head, N)
        h_all = torch.einsum("btsh,bshpn->bthpn", decay_matrix, u_ssm)
        y = torch.einsum("bthpn,bthn->bthp", h_all, C_heads) + self.D.view(1, 1, n_heads, 1) * x_heads
        y = y.reshape(batch_size, seq_len, d_inner)
        y_gated = y * z_act
        out = u + self.out_proj(y_gated)
        return out


class BitMamba2SequenceModel(nn.Module):
    """
    Compact multi-layer Mamba-2 model for sequence classification and long-range memory tasks.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        cfg: BitMamba2Config,
        n_layers: int = 2,
        pool: Literal["last", "mean"] = "mean",
    ):
        super().__init__()
        self.cfg = cfg
        self.pool = pool
        self.embed = nn.Linear(input_dim, cfg.d_model)
        self.layers = nn.ModuleList([BitMamba2Block(cfg) for _ in range(n_layers)])
        self.norm_f = nn.RMSNorm(cfg.d_model, eps=1e-5)
        self.head = nn.Linear(cfg.d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, input_dim)
        """
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        h = self.norm_f(h)
        if self.pool == "last":
            feat = h[:, -1, :]
        else:
            feat = h.mean(dim=1)
        return self.head(feat)

    @torch.no_grad()
    def model_summary(self) -> Dict[str, float]:
        """Computes parameter counts, memory footprint (bytes), and average ternary zero sparsity."""
        total_params = sum(p.numel() for p in self.parameters())
        bitlinear_params = 0
        other_params = 0
        zero_ratios = []

        for m in self.modules():
            if isinstance(m, HBitLinear):
                bitlinear_params += m.weight.numel()
                zero_ratios.append(m.get_zero_sparsity())

        other_params = total_params - bitlinear_params
        avg_sparsity = sum(zero_ratios) / max(len(zero_ratios), 1)

        return {
            "total_params": total_params,
            "bitlinear_params": bitlinear_params,
            "bitlinear_ratio": bitlinear_params / max(total_params, 1),
            "other_params": other_params,
            "avg_ternary_zero_sparsity": avg_sparsity,
            "size_fp16_bytes": total_params * 2.0,
            "size_int8_bytes": total_params * 1.0,
            "size_2bit_bytes": (bitlinear_params * 2.0 / 8.0) + other_params * 1.0,
            "size_radix3_bytes": (bitlinear_params * 1.6 / 8.0) + other_params * 1.0,
        }
