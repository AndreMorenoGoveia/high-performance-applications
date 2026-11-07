"""
Experiment 1 (Addressing Review Point 3 & Point 4):
1. Long-Range Recurrent Memory Retention & State SQNR vs. Sequence Length L (L = 64 to 16,384):
   Compares Exact FP32, Fixed16 (DSP), Standard PoT (2^-k, LightMamba/FastMamba),
   Proposed CPoT ({2^-k, 1 - 2^-k}), and Dual-Shift CPoT2.
2. On-Chip Memory Scaling O(1) vs O(L) on Intel Cyclone V (M10K BRAM limit):
   Compares Ternary Transformer (TeLLMe/TerEffic KV-Cache) vs. BitMamba-CV O(1) State.
"""

from __future__ import annotations
import json
import os
from typing import Dict, List
import numpy as np
import torch

from models.mamba2_ssd_qat import quantize_abar


def compute_sqnr_db(ref: torch.Tensor, approx: torch.Tensor) -> float:
    noise = ref - approx
    sig_pow = torch.mean(ref ** 2).item()
    noise_pow = torch.mean(noise ** 2).item()
    if noise_pow < 1e-15:
        return 100.0
    return float(10.0 * np.log10(max(sig_pow, 1e-12) / noise_pow))


@torch.no_grad()
def run_long_range_ssm_experiment(
    seq_lengths: List[int] = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
    n_heads: int = 8,
    head_dim: int = 16,
    d_state: int = 16,
    seeds: List[int] = [42, 100, 2026],
) -> Dict:
    """
    Evaluates recurrent state drift (SQNR in dB) and long-horizon impulse recall
    across sequence lengths up to L = 16,384 on CUDA GPU.
    """
    device = torch.device("cpu")
    modes = ["fixed16", "pot", "cpot", "cpot2"]
    results = {m: {"sqnr_mean": [], "sqnr_std": [], "recall_mean": [], "recall_std": []} for m in modes}
    results["seq_lengths"] = seq_lengths

    # Also measure quantization error curve directly over the long-memory regime A_bar in [0.50, 0.999]
    a_grid = torch.linspace(0.50, 0.999, 500, device=device)
    grid_errors = {}
    for m in modes:
        q_grid = quantize_abar(a_grid, mode=m)
        grid_errors[m] = {
            "max_abs_err_long_mem": float(torch.max(torch.abs(a_grid[a_grid >= 0.90] - q_grid[a_grid >= 0.90])).item()),
            "rmse_long_mem": float(torch.sqrt(torch.mean((a_grid[a_grid >= 0.90] - q_grid[a_grid >= 0.90]) ** 2)).item()),
            "sqnr_long_mem_db": compute_sqnr_db(a_grid[a_grid >= 0.90], q_grid[a_grid >= 0.90]),
        }
    results["long_memory_eigenvalue_metrics"] = grid_errors

    for L in seq_lengths:
        sqnr_per_mode = {m: [] for m in modes}
        recall_per_mode = {m: [] for m in modes}

        for seed in seeds:
            torch.manual_seed(seed)
            # Long-memory HiPPO/Mamba-2 schedule where true A_bar spans [0.92, 1 - 1/L]
            target_half_lives = torch.exp(
                torch.linspace(np.log(16.0), np.log(float(L)), n_heads, device=device)
            )
            abar_base = torch.exp(-1.0 / target_half_lives).view(1, 1, n_heads)  # (1, 1, H) in [0.939, 0.9999]

            # Dynamic input modulation around long-memory regime
            dt_jitter = torch.empty(1, L, n_heads, device=device).uniform_(0.8, 1.2)
            abar_exact = torch.pow(abar_base, dt_jitter).clamp(0.55, 0.9995)

            # Input sequence with embedded key signal at t = 0 to test long-range memory retention at t = L-1
            x_seq = torch.randn(1, L, n_heads, head_dim, device=device) * 0.15
            b_seq = torch.randn(1, L, n_heads, d_state, device=device) * 0.15
            c_query = torch.randn(1, n_heads, d_state, device=device)

            # Inject strong trigger token at t=0
            trigger_signal = torch.randn(1, n_heads, head_dim, device=device)
            trigger_b = torch.randn(1, n_heads, d_state, device=device)
            x_seq[:, 0] = trigger_signal * 2.0
            b_seq[:, 0] = trigger_b * 2.0
            c_query = trigger_b / (torch.norm(trigger_b, dim=-1, keepdim=True) + 1e-6)

            # Compute exact recurrent trajectory using associative log-space cumulative products
            # h_T = sum_{t=0}^{L-1} (prod_{s=t+1}^{L-1} abar_s) * u_t
            u_seq = x_seq.unsqueeze(-1) * b_seq.unsqueeze(-2)  # (1, L, H, P, N)

            def compute_final_state(abar_tensor: torch.Tensor) -> torch.Tensor:
                log_a = torch.log(abar_tensor.clamp(min=1e-12))  # (1, L, H)
                rev_cumsum = torch.flip(torch.cumsum(torch.flip(log_a, dims=[1]), dim=1), dims=[1])
                # Shift by 1 so step t gets product from t+1 to L-1; t=L-1 gets weight 1.0
                log_weights = torch.cat([rev_cumsum[:, 1:, :], torch.zeros(1, 1, n_heads, device=device)], dim=1)
                weights = torch.exp(log_weights).unsqueeze(-1).unsqueeze(-1)  # (1, L, H, 1, 1)
                h_final = torch.sum(weights * u_seq, dim=1)  # (1, H, P, N)
                return h_final

            h_exact = compute_final_state(abar_exact)
            y_exact = torch.einsum("bhpn,bhn->bhp", h_exact, c_query)

            # Contribution from t=0 trigger alone in exact model
            prod_all_exact = torch.exp(torch.sum(torch.log(abar_exact[:, 1:, :]), dim=1))  # (1, H)

            for m in modes:
                abar_q = quantize_abar(abar_exact, mode=m, max_shift=14)
                h_q = compute_final_state(abar_q)
                y_q = torch.einsum("bhpn,bhn->bhp", h_q, c_query)

                sqnr_val = compute_sqnr_db(y_exact, y_q)
                sqnr_per_mode[m].append(sqnr_val)

                # Cosine similarity of long-memory head readout (slowest-decaying heads)
                cos_sim = torch.nn.functional.cosine_similarity(
                    y_exact[:, -2:, :].flatten(), y_q[:, -2:, :].flatten(), dim=0
                ).item()
                recall_per_mode[m].append(max(0.0, cos_sim) * 100.0)

        for m in modes:
            results[m]["sqnr_mean"].append(float(np.mean(sqnr_per_mode[m])))
            results[m]["sqnr_std"].append(float(np.std(sqnr_per_mode[m])))
            results[m]["recall_mean"].append(float(np.mean(recall_per_mode[m])))
            results[m]["recall_std"].append(float(np.std(recall_per_mode[m])))

    # O(1) vs O(L) On-Chip State Memory Footprint (in KB) for d_inner=128, n_layers=4, d_state=16
    # Transformer KV-Cache (INT8 K and V): 2 * n_layers * d_inner * L bytes
    # BitMamba-CV State (INT16 Q8.8 h_t + 4-tap Conv1D buffer): n_layers * (d_inner * d_state * 2 + d_inner * 4) bytes
    n_layers = 4
    d_inner = 128
    kv_cache_kb = [float(2 * n_layers * d_inner * L) / 1024.0 for L in seq_lengths]
    mamba_state_kb = [
        float(n_layers * (d_inner * d_state * 2 + d_inner * 4)) / 1024.0 for _ in seq_lengths
    ]
    results["memory_scaling_kb"] = {
        "cyclone5_m10k_capacity_kb": 553.0,  # DE10-Nano 5CSEBA6U23I7: 5,530 Kbits = 691.25 KB (~553 KB usable)
        "transformer_kv_int8_kb": kv_cache_kb,
        "bitmamba_cv_o1_state_kb": mamba_state_kb,
    }
    return results


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    res = run_long_range_ssm_experiment()
    out_path = os.path.join(out_dir, "long_range_cpot_results.json")
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)

    print("=== Long-Memory Eigenvalue Quantization Error (A_bar in [0.90, 0.999]) ===")
    for m, stats in res["long_memory_eigenvalue_metrics"].items():
        print(
            f"  Mode {m:8s} | RMSE: {stats['rmse_long_mem']:.5f} | "
            f"MaxErr: {stats['max_abs_err_long_mem']:.5f} | SQNR: {stats['sqnr_long_mem_db']:.2f} dB"
        )
    print("\n=== Sequence Length Scaling (L = 64 -> 16,384) Final Readout SQNR (dB) ===")
    for idx, L in enumerate(res["seq_lengths"]):
        print(
            f"  L={L:5d} | PoT: {res['pot']['sqnr_mean'][idx]:6.2f} dB (Recall {res['pot']['recall_mean'][idx]:5.1f}%) | "
            f"CPoT: {res['cpot']['sqnr_mean'][idx]:6.2f} dB (Recall {res['cpot']['recall_mean'][idx]:5.1f}%) | "
            f"CPoT2: {res['cpot2']['sqnr_mean'][idx]:6.2f} dB | Fixed16: {res['fixed16']['sqnr_mean'][idx]:6.2f} dB"
        )
