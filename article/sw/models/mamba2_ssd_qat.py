"""
Mamba-2 (SSD, scalar decay per head) with hardware-aware QAT.

- H-BitLinear: RMSNorm -> block FWHT (B=16) -> per-token INT8 activation quant ->
  ternary {-1,0,+1} weights (absmean, STE). Weights are learned in the Hadamard
  domain, so they stay ternary at inference (cf. BitNet v2).
- Hardware scan (cfg.hw_scan=True), matching article/hw/rtl/scan_lane.v:
    * x, B, C quantised to INT8 with *static* power-of-two scales (EMA-calibrated),
      so the integer state never mixes per-token scales;
    * Delta quantised to a power of two 2^-j, j in [0, 15] (injection = shift);
    * Abar = exp(-Delta_q * A_h) quantised by cfg.abar_mode
      (fixed16 = Q1.15 multiplier, pot = 2^-k (k>=1), pot0 = 2^-k (k>=0),
       cpot = {2^-k, 1-2^-k}, cpot2 = 1 - 2^-k1 -/+ 2^-k2).
- int_emulation(): replaces the float scan by the bit-exact integer recurrence of the
  golden model (state width / guard bits / rounding configurable) for evaluation.
Embedding, Conv1D and classifier head stay in floating point (not in the RTL).
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Literal, Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

WeightMode = Literal["fp32", "int8", "ternary"]
AbarMode = Literal["exact", "fixed16", "pot", "pot0", "cpot", "cpot2"]

K_MAX = 15          # 4-bit shift field in the RTL
CPOT_THRESHOLD = 0.625


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
    hw_scan: bool = False
    abar_mode: AbarMode = "exact"
    act_bits: int = 8
    chunk: int = 64

    @property
    def d_inner(self) -> int:
        return self.d_model * self.expand

    @property
    def head_dim(self) -> int:
        return self.d_inner // self.n_heads


# ------------------------------------------------------------------------------
# quantisation primitives
# ------------------------------------------------------------------------------

def ste(q: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return (q - x).detach() + x


def fwht(x: torch.Tensor, block: int = 16) -> torch.Tensor:
    """Orthonormal block Walsh-Hadamard transform on the last axis."""
    shp = x.shape
    h = x.reshape(-1, shp[-1] // block, block)
    s = 1
    while s < block:
        h = h.reshape(h.shape[0], h.shape[1], block // (2 * s), 2, s)
        h = torch.stack([h[..., 0, :] + h[..., 1, :], h[..., 0, :] - h[..., 1, :]], dim=-2)
        s *= 2
    return h.reshape(shp) / math.sqrt(block)


def quant_weight(w: torch.Tensor, mode: WeightMode) -> torch.Tensor:
    if mode == "fp32":
        return w
    if mode == "ternary":
        g = w.abs().mean().clamp(min=1e-6)
        return ste((w / g).round().clamp(-1, 1) * g, w)
    g = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / 127.0
    return ste((w / g).round().clamp(-128, 127) * g, w)


def quant_act_token(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
    qmax = float(2 ** (bits - 1) - 1)
    s = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / qmax
    return ste((x / s).round().clamp(-qmax, qmax) * s, x)


class StaticPoTQuant(nn.Module):
    """INT8 quantiser with a static power-of-two scale from an EMA of |x|max."""

    def __init__(self, momentum: float = 0.99):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("running_max", torch.tensor(0.0))

    def scale(self) -> torch.Tensor:
        return torch.pow(2.0, torch.ceil(torch.log2(self.running_max.clamp(min=1e-8) / 127.0)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            with torch.no_grad():
                m = x.detach().abs().amax()
                if self.running_max == 0:
                    self.running_max.copy_(m)
                else:
                    self.running_max.mul_(self.momentum).add_((1 - self.momentum) * m)
        s = self.scale()
        return ste((x / s).round().clamp(-127, 127) * s, x)


def quant_delta_pot(dt: torch.Tensor) -> torch.Tensor:
    j = (-torch.log2(dt.clamp(min=1e-9))).round().clamp(0, K_MAX)
    return ste(torch.pow(2.0, -j), dt)


def quant_abar(a: torch.Tensor, mode: AbarMode) -> torch.Tensor:
    if mode == "exact":
        return a
    a_c = a.clamp(1e-9, 1 - 1e-9)
    if mode == "fixed16":
        q = (a_c * 32768).round().clamp(0, 32767) / 32768
    elif mode in ("pot", "pot0"):
        kmin = 1 if mode == "pot" else 0
        q = torch.pow(2.0, -(-torch.log2(a_c)).round().clamp(kmin, K_MAX))
    else:
        k_lo = (-torch.log2(a_c)).round().clamp(1, K_MAX)
        comp = 1 - a_c
        k1 = (-torch.log2(comp)).round().clamp(1, K_MAX)
        q_hi = 1 - torch.pow(2.0, -k1)
        if mode == "cpot2":
            rem = comp - torch.pow(2.0, -k1)
            k2 = (-torch.log2(rem.abs().clamp(min=1e-12))).round().clamp(1, K_MAX)
            q_hi = (q_hi - torch.sign(rem) * torch.pow(2.0, -k2)).clamp(0.5, 1 - 2.0 ** -K_MAX)
        q = torch.where(a_c > CPOT_THRESHOLD, q_hi, torch.pow(2.0, -k_lo))
    return ste(q, a)


# ------------------------------------------------------------------------------
# layers
# ------------------------------------------------------------------------------

class HBitLinear(nn.Module):
    def __init__(self, d_in, d_out, weight_mode: WeightMode, use_fwht: bool, block: int, act_bits: int):
        super().__init__()
        self.weight_mode, self.use_fwht, self.block, self.act_bits = weight_mode, use_fwht, block, act_bits
        self.norm = nn.RMSNorm(d_in, eps=1e-5)
        self.weight = nn.Parameter(torch.empty(d_out, d_in))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    @torch.no_grad()
    def ternary_weights(self) -> torch.Tensor:
        g = self.weight.abs().mean().clamp(min=1e-6)
        return (self.weight / g).round().clamp(-1, 1).to(torch.int8)

    def forward(self, x):
        x = self.norm(x)
        if self.use_fwht:
            x = fwht(x, self.block)
        if self.weight_mode != "fp32":
            x = quant_act_token(x, self.act_bits)
        return F.linear(x, quant_weight(self.weight, self.weight_mode))


class BitMamba2Block(nn.Module):
    def __init__(self, cfg: BitMamba2Config):
        super().__init__()
        self.cfg = cfg
        H, N, Di = cfg.n_heads, cfg.d_state, cfg.d_inner
        self.in_proj = HBitLinear(cfg.d_model, 2 * Di + 2 * H * N + H, cfg.weight_mode,
                                  cfg.use_fwht, cfg.fwht_block_size, cfg.act_bits)
        self.conv1d = nn.Conv1d(Di, Di, cfg.d_conv, groups=Di, padding=cfg.d_conv - 1)
        # Standard Mamba-2 init: A ~ U[1, 16], dt ~ logU[1e-3, 1e-1]
        self.A_log = nn.Parameter(torch.log(torch.empty(H).uniform_(1, 16)))
        dt = torch.exp(torch.empty(H).uniform_(math.log(1e-3), math.log(1e-1)))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))   # softplus^-1
        self.D = nn.Parameter(torch.ones(H))
        self.out_proj = HBitLinear(Di, cfg.d_model, cfg.weight_mode, cfg.use_fwht,
                                   cfg.fwht_block_size, cfg.act_bits)
        self.qx, self.qb, self.qc = StaticPoTQuant(), StaticPoTQuant(), StaticPoTQuant()
        self.int_emul = None    # ScanCfg -> bit-exact integer scan at eval

    # -- float (fake-quant) scan, chunked SSD form ------------------------------
    def _scan_float(self, x, Bm, Cm, dt, abar):
        Bsz, L, H, P = x.shape
        T = min(self.cfg.chunk, L)
        xd = x * dt[..., None]                                               # (B,L,H,P)
        loga = torch.log(abar.clamp(min=1e-30))
        h_prev = x.new_zeros(Bsz, H, P, Bm.shape[-1])
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        ys = []
        for s in range(0, L, T):
            e = min(s + T, L)
            n = e - s
            cs = torch.cumsum(loga[:, s:e], dim=1)                           # (B,n,H)
            decay = torch.exp((cs[:, :, None] - cs[:, None, :]).masked_fill(~mask[:n, :n, None], -float("inf")))
            cb = torch.einsum("bthn,bshn->btsh", Cm[:, s:e], Bm[:, s:e])     # (B,n,n,H)
            y = torch.einsum("btsh,bshp->bthp", cb * decay, xd[:, s:e])
            y = y + torch.exp(cs)[..., None] * torch.einsum("bthn,bhpn->bthp", Cm[:, s:e], h_prev)
            ys.append(y)
            w_end = torch.exp(cs[:, -1:] - cs)                               # (B,n,H)
            h_prev = torch.exp(cs[:, -1])[..., None, None] * h_prev + \
                torch.einsum("bsh,bshp,bshn->bhpn", w_end, xd[:, s:e], Bm[:, s:e])
        return torch.cat(ys, dim=1)

    # -- bit-exact integer scan (golden model semantics) -----------------------
    @torch.no_grad()
    def _scan_int(self, x, Bm, Cm, dt_q, abar_q):
        from models.golden_model_bitexact import CPOT_THRESHOLD as THR
        cfg = self.int_emul
        sx, sb, sc = self.qx.scale(), self.qb.scale(), self.qc.scale()
        xi = (x / sx).round().clamp(-127, 127).long()
        bi = (Bm / sb).round().clamp(-127, 127).long()
        ci = (Cm / sc).round().clamp(-127, 127).long()
        j = (-torch.log2(dt_q)).round().long()
        a = abar_q.double()
        if cfg.decay == "cpot":
            if self.cfg.abar_mode != "cpot":
                raise ValueError("cpot integer scan requires abar_mode='cpot'")
            m = (a > THR).long()
            k = torch.where(m == 1, -torch.log2(1 - a), -torch.log2(a)).round().long()
        else:
            a_q15 = (a * 32768).round().clamp(0, 32767).long()
        lo, hi = -(1 << (cfg.state_w - 1)), (1 << (cfg.state_w - 1)) - 1
        ylo, yhi = -(1 << (cfg.y_w - 1)), (1 << (cfg.y_w - 1)) - 1
        Bsz, L, H, P = x.shape
        h = torch.zeros(Bsz, H, P, Bm.shape[-1], dtype=torch.long, device=x.device)
        ys, sat = [], 0
        for t in range(L):
            if cfg.decay == "cpot":
                kt, mt = k[:, t, :, None, None], m[:, t, :, None, None]
                rnd = torch.where(kt > 0, 1 << (kt - 1).clamp(min=0), 0) if cfg.round_shift else 0
                s = (h + rnd) >> kt
                h_dec = torch.where(mt == 1, h - s, s)
            else:
                h_dec = (h * a_q15[:, t, :, None, None] + (1 << 14)) >> 15
            bx = xi[:, t, :, :, None] * bi[:, t, :, None, :]
            h_new = h_dec + ((bx << cfg.guard) >> j[:, t, :, None, None])
            sat += int(((h_new < lo) | (h_new > hi)).sum())
            h = h_new.clamp(lo, hi)
            acc = (h * ci[:, t, :, None, :]).sum(-1)
            ys.append((acc >> cfg.guard).clamp(ylo, yhi))
        self.last_sat_events = sat
        return torch.stack(ys, dim=1).double().float() * (sx * sb * sc)

    def forward(self, u):
        cfg = self.cfg
        Bsz, L, _ = u.shape
        H, N, Di, P = cfg.n_heads, cfg.d_state, cfg.d_inner, cfg.head_dim
        z, xc, Bm, Cm, dt_raw = torch.split(self.in_proj(u), [Di, Di, H * N, H * N, H], dim=-1)
        x = F.silu(self.conv1d(xc.transpose(1, 2))[..., :L].transpose(1, 2))
        dt = F.softplus(dt_raw + self.dt_bias)
        if cfg.hw_scan:
            x, Bm, Cm = self.qx(x), self.qb(Bm), self.qc(Cm)
            dt = quant_delta_pot(dt.clamp(max=1.0))
        elif cfg.weight_mode != "fp32":
            x, Bm, Cm = (quant_act_token(t, cfg.act_bits) for t in (x, Bm, Cm))
        abar = quant_abar(torch.exp(-dt * torch.exp(self.A_log)), cfg.abar_mode)
        xh, Bh, Ch = x.view(Bsz, L, H, P), Bm.view(Bsz, L, H, N), Cm.view(Bsz, L, H, N)
        if self.int_emul is not None and not self.training:
            y = self._scan_int(xh, Bh, Ch, dt, abar)
        else:
            y = self._scan_float(xh, Bh, Ch, dt, abar)
        y = (y + self.D.view(1, 1, H, 1) * xh).reshape(Bsz, L, Di)
        return u + self.out_proj(y * F.silu(z))


class BitMamba2SequenceModel(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, cfg: BitMamba2Config, n_layers: int = 2):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Linear(input_dim, cfg.d_model)
        self.layers = nn.ModuleList([BitMamba2Block(cfg) for _ in range(n_layers)])
        self.norm_f = nn.RMSNorm(cfg.d_model, eps=1e-5)
        self.head = nn.Linear(cfg.d_model, num_classes)

    def forward(self, x):
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        return self.head(self.norm_f(h)[:, -1])

    def set_int_emulation(self, scan_cfg: Optional[object]):
        for layer in self.layers:
            layer.int_emul = scan_cfg

    @torch.no_grad()
    def summary(self) -> Dict[str, float]:
        lin = [m for m in self.modules() if isinstance(m, HBitLinear)]
        n_lin = sum(m.weight.numel() for m in lin)
        n_tot = sum(p.numel() for p in self.parameters())
        zeros = sum(int((m.ternary_weights() == 0).sum()) for m in lin)
        return {
            "total_params": n_tot,
            "projection_params": n_lin,
            "projection_ratio": n_lin / n_tot,
            "ternary_zero_fraction": zeros / max(n_lin, 1),
            "proj_kib_fp16": n_lin * 2 / 1024, "proj_kib_int8": n_lin / 1024,
            "proj_kib_2bit": n_lin * 2 / 8 / 1024, "proj_kib_radix3": math.ceil(n_lin / 5) / 1024,
        }


if __name__ == "__main__":
    # Self-check: torch integer scan == numpy golden scan on random codes.
    import numpy as np
    from models.golden_model_bitexact import ScanCfg, scan_lane_run, LEGACY_16
    torch.manual_seed(0)
    cfg = BitMamba2Config(d_model=32, n_heads=2, d_state=4, hw_scan=True, abar_mode="cpot")
    blk = BitMamba2Block(cfg).eval()
    for q in (blk.qx, blk.qb, blk.qc):
        q.running_max.fill_(4.0)
    L, P, N = 50, cfg.head_dim, cfg.d_state
    x = torch.randn(1, L, 2, P)
    Bm, Cm = torch.randn(1, L, 2, N), torch.randn(1, L, 2, N)
    dt = quant_delta_pot(torch.rand(1, L, 2) * 0.1)
    abar = quant_abar(torch.exp(-dt * 3.0), "cpot")
    for sc in (ScanCfg(), LEGACY_16):
        blk.int_emul = sc
        y = blk._scan_int(x, Bm, Cm, dt, abar) / (blk.qx.scale() * blk.qb.scale() * blk.qc.scale())
        s = lambda t, q: (t / q.scale()).round().clamp(-127, 127).long().numpy()
        xi, bi, ci = s(x, blk.qx), s(Bm, blk.qb), s(Cm, blk.qc)
        a = abar.double().numpy()
        m = (a > CPOT_THRESHOLD).astype(np.int64)
        k = np.round(np.where(m == 1, -np.log2(1 - a), -np.log2(a))).astype(np.int64)
        j = np.round(-np.log2(dt.numpy())).astype(np.int64)
        ref = scan_lane_run(xi[0, :, 1, 3], bi[0, :, 1], ci[0, :, 1], m[0, :, 1], k[0, :, 1],
                            j[0, :, 1], np.zeros(L, np.int64), sc)
        assert np.array_equal(y[0, :, 1, 3].round().long().numpy(), ref), sc
    print("[OK] torch integer scan == golden scan_lane_run")
    # Self-check: chunked SSD scan == naive sequential recurrence (L > chunk).
    blk.cfg.chunk = 16
    xr, Br, Cr = x.double(), Bm.double(), Cm.double()
    dtr, ar = torch.rand(1, L, 2).double() * 0.1, torch.rand(1, L, 2).double() * 0.2 + 0.8
    yc = blk._scan_float(xr, Br, Cr, dtr, ar)
    h = torch.zeros(1, 2, P, N, dtype=torch.double)
    for t in range(L):
        h = ar[:, t, :, None, None] * h + dtr[:, t, :, None, None] * xr[:, t, :, :, None] * Br[:, t, :, None, :]
        assert torch.allclose(yc[:, t], torch.einsum("bhpn,bhn->bhp", h, Cr[:, t]), atol=1e-9)
    print("[OK] chunked scan == sequential recurrence")
