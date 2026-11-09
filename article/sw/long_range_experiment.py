"""
Long-memory numerics of the scan (no training involved).

E1. Quantisation error of Abar in the long-memory band [0.90, 0.999]: absolute error and
    relative error of the time constant tau = -1/ln(Abar) for each scheme.
E2. Float readout fidelity vs. sequence length (L = 64..16384): Abar quantised, all
    else exact; SQNR of y_L and cosine similarity of the slow heads.
E3. Integer state (golden model = RTL): impulse response of h under 1-2^-k for the
    legacy 16-bit/no-guard/truncating state vs. the corrected 32-bit/10-guard/rounded one.
E4. Integer state with random INT8 inputs: SQNR of y_t (integer RTL arithmetic) against
    the float recurrence with the *same* quantised codes -> pure datapath error.
Writes results/long_range_results.json
"""

from __future__ import annotations
import json
import os
import numpy as np
import torch

from models.mamba2_ssd_qat import quant_abar
from models.golden_model_bitexact import (
    LEGACY_16, PROPOSED_32, MULT_Q15, scan_lane_run, encode_cpot, decode_cpot, _shr)

MODES = ["fixed16", "pot", "pot0", "cpot", "cpot2"]


def sqnr_db(ref, approx):
    ref, approx = np.asarray(ref, float), np.asarray(approx, float)
    noise = np.mean((ref - approx) ** 2)
    return float(10 * np.log10(np.mean(ref ** 2) / noise)) if noise > 0 else float("inf")


def e1_grid():
    a = torch.linspace(0.90, 0.999, 2000, dtype=torch.float64)
    tau = -1 / torch.log(a)
    out = {}
    for m in MODES:
        q = quant_abar(a, m)
        tau_q = torch.where(q >= 1, torch.tensor(float("inf"), dtype=torch.float64), -1 / torch.log(q))
        rel = ((tau_q - tau).abs() / tau)
        out[m] = {
            "max_abs_err": float((q - a).abs().max()),
            "rmse": float(torch.sqrt(((q - a) ** 2).mean())),
            "tau_rel_err_median": float(rel.median()),
            "tau_rel_err_max": float(rel.max()),
            "distinct_levels": int(torch.unique(q).numel()),
        }
    return out


def e2_readout(seq_lengths=(64, 256, 1024, 4096, 16384), H=8, P=16, N=16, seeds=(42, 100, 2026, 7, 31)):
    res = {m: {"sqnr_mean": [], "sqnr_std": [], "cos_slow_mean": []} for m in MODES}
    for L in seq_lengths:
        acc = {m: ([], []) for m in MODES}
        for s in seeds:
            g = torch.Generator().manual_seed(s)
            tau = torch.exp(torch.linspace(np.log(16.0), np.log(float(L)), H, dtype=torch.float64))
            abar = torch.exp(-1 / tau).view(1, H) ** torch.empty(L, H, dtype=torch.float64).uniform_(0.8, 1.2, generator=g)
            x = torch.randn(L, H, P, generator=g, dtype=torch.float64) * 0.15
            b = torch.randn(L, H, N, generator=g, dtype=torch.float64) * 0.15
            x[0] *= 13.0
            b[0] *= 13.0
            c = b[0] / b[0].norm(dim=-1, keepdim=True)

            def final_y(a):
                w = torch.exp(torch.flip(torch.cumsum(torch.flip(torch.log(a), [0]), 0), [0]))
                w = torch.cat([w[1:], torch.ones(1, H, dtype=torch.float64)])
                h = torch.einsum("lh,lhp,lhn->hpn", w, x, b)
                return torch.einsum("hpn,hn->hp", h, c)

            y = final_y(abar)
            for m in MODES:
                yq = final_y(quant_abar(abar, m).clamp(min=1e-300))
                acc[m][0].append(sqnr_db(y.numpy(), yq.numpy()))
                acc[m][1].append(float(torch.nn.functional.cosine_similarity(y[-2:].flatten(), yq[-2:].flatten(), dim=0)))
        for m in MODES:
            res[m]["sqnr_mean"].append(float(np.mean(acc[m][0])))
            res[m]["sqnr_std"].append(float(np.std(acc[m][0])))
            res[m]["cos_slow_mean"].append(float(np.mean(acc[m][1])))
    res["seq_lengths"] = list(seq_lengths)
    return res


def e3_impulse(ks=(4, 6, 8, 10, 12), bx=64 * 64):
    out = {}
    for name, cfg in (("legacy_s16_g0_trunc", LEGACY_16), ("proposed_s32_g10_round", PROPOSED_32)):
        rows = {}
        for k in ks:
            tau = 2 ** k
            h = np.array([bx << cfg.guard], dtype=np.int64)
            traj = [h[0]]
            for _ in range(4 * tau):
                h = np.clip(h - _shr(h, k, cfg.round_shift), -(1 << (cfg.state_w - 1)), (1 << (cfg.state_w - 1)) - 1)
                traj.append(h[0])
            traj = np.array(traj, float) / (1 << cfg.guard)
            ideal = bx * (1 - 2.0 ** -k) ** np.arange(len(traj))
            rows[str(k)] = {
                "tau_steps": tau,
                "h_over_ideal_at_1tau": float(traj[tau] / ideal[tau]),
                "h_over_ideal_at_4tau": float(traj[4 * tau] / ideal[4 * tau]),
                "final_h": float(traj[-1]), "final_ideal": float(ideal[-1]),
            }
        out[name] = rows
    return out


def e4_random(T=4096, N=16, seeds=(42, 100, 2026), k_range=(4, 12)):
    out = {}
    for name, cfg in (("legacy_s16_g0_trunc", LEGACY_16), ("proposed_s32_g10_round", PROPOSED_32),
                      ("mult_q15_s32_g10", MULT_Q15)):
        vals = []
        for s in seeds:
            rng = np.random.default_rng(s)
            k = rng.integers(k_range[0], k_range[1] + 1, size=T)
            m = np.ones(T, np.int64)
            a = decode_cpot(m, k)
            j = np.clip(k - 1, 0, 15)                      # Delta ~ 2 * (1 - Abar)  -> bounded state
            x = rng.integers(-64, 64, size=T)
            b = rng.integers(-64, 64, size=(T, N))
            c = rng.integers(-64, 64, size=(T, N))
            a_q15 = np.round(a * 32768).clip(0, 32767).astype(np.int64)
            y_int = scan_lane_run(x, b, c, m, k, j, a_q15, cfg)
            h = np.zeros(N)
            a_used = a if cfg.decay == "cpot" else a_q15 / 32768
            y_ref = []
            for t in range(T):
                h = a_used[t] * h + 2.0 ** -j[t] * x[t] * b[t]
                y_ref.append(h @ c[t])
            vals.append(sqnr_db(y_ref, y_int))
        out[name] = {"sqnr_mean_db": float(np.mean(vals)), "sqnr_std_db": float(np.std(vals)),
                     "k_range": list(k_range), "T": T}
    return out


if __name__ == "__main__":
    res = {"E1_grid_0.90_0.999": e1_grid(), "E2_readout_vs_L": e2_readout(),
           "E3_integer_impulse": e3_impulse(), "E4_integer_random_sqnr": e4_random()}
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "long_range_results.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print("E1 (Abar in [0.90,0.999])")
    for m, r in res["E1_grid_0.90_0.999"].items():
        print(f"  {m:8s} max|err|={r['max_abs_err']:.5f} rmse={r['rmse']:.5f} "
              f"tau rel err median={r['tau_rel_err_median']:.3f} max={r['tau_rel_err_max']:.3f} levels={r['distinct_levels']}")
    e2 = res["E2_readout_vs_L"]
    print("E2 readout SQNR (dB) for L =", e2["seq_lengths"])
    for m in MODES:
        print(f"  {m:8s}", [round(v, 1) for v in e2[m]["sqnr_mean"]], "cos", [round(v, 3) for v in e2[m]["cos_slow_mean"]])
    print("E3 impulse h/ideal at 1 tau and 4 tau")
    for n, rows in res["E3_integer_impulse"].items():
        print(f"  {n:24s}", {k: (round(r['h_over_ideal_at_1tau'], 3), round(r['h_over_ideal_at_4tau'], 3)) for k, r in rows.items()})
    print("E4 integer datapath SQNR (dB):", {k: round(v["sqnr_mean_db"], 1) for k, v in res["E4_integer_random_sqnr"].items()})
    print("[OK] ->", out)
