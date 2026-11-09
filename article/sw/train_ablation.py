"""
QAT ablation (A0..A8) on ECG5000 (real, L=140) and a delayed-cue recall task
(synthetic, L in {256, 1024}); 5 seeds (3 for L=1024). For the hardware-scan configs the trained
model is re-evaluated with the bit-exact integer scan of the RTL (golden model).

Usage: python3 train_ablation.py [--tasks ecg5000 cue256 cue1024] [--seeds N] [--configs ...]
Writes results/ablation_<task>.json
"""

from __future__ import annotations
import argparse
import io
import json
import os
import time
import urllib.request
import zipfile
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from models.mamba2_ssd_qat import BitMamba2Config, BitMamba2SequenceModel
from models.golden_model_bitexact import ScanCfg, LEGACY_16, PROPOSED_32, MULT_Q15

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
ECG_URL = "https://www.timeseriesclassification.com/aeon-toolkit/ECG5000.zip"


# ------------------------------------------------------------------------------
# data
# ------------------------------------------------------------------------------

def load_ecg5000():
    d = os.path.join(DATA, "ECG5000")
    if not os.path.exists(os.path.join(d, "ECG5000_TRAIN.txt")):
        os.makedirs(d, exist_ok=True)
        with urllib.request.urlopen(ECG_URL, timeout=120) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(d)

    def read(split):
        a = np.loadtxt(os.path.join(d, f"ECG5000_{split}.txt"))
        x = torch.tensor(a[:, 1:], dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor(a[:, 0], dtype=torch.long) - 1
        return TensorDataset(x, y)

    return read("TRAIN"), read("TEST"), 1, 5


def make_delayed_cue(L: int, n_train=4000, n_test=1000, seed=7):
    """
    4 classes. The class is shown only during t < 8 (one-hot on channels 0..3,
    amplitude 2); afterwards only Gaussian noise (std 0.1) on all 8 channels.
    The label is read at t = L-1 -> requires memory of ~L steps.
    """
    g = torch.Generator().manual_seed(seed)
    n = n_train + n_test
    y = torch.randint(0, 4, (n,), generator=g)
    x = torch.randn(n, L, 8, generator=g) * 0.1
    x[torch.arange(n), :8, y] += 2.0
    return (TensorDataset(x[:n_train], y[:n_train]),
            TensorDataset(x[n_train:], y[n_train:]), 8, 4)


TASKS = {
    "ecg5000": dict(load=load_ecg5000, epochs=40, batch=32, seeds=5),
    "cue256": dict(load=lambda: make_delayed_cue(256), epochs=20, batch=64, seeds=5),
    "cue1024": dict(load=lambda: make_delayed_cue(1024, 2000, 500), epochs=20, batch=64, seeds=3),
}

BASE = dict(d_model=64, n_heads=4, d_state=16, act_bits=8)
CONFIGS = {
    "A0_fp32":            dict(weight_mode="fp32", use_fwht=False),
    "A1_w8a8":            dict(weight_mode="int8", use_fwht=False),
    "A2_tern_nofwht":     dict(weight_mode="ternary", use_fwht=False),
    "A3_tern_hbitlinear": dict(weight_mode="ternary", use_fwht=True),
    "A4_hwscan_fixed16":  dict(weight_mode="ternary", use_fwht=True, hw_scan=True, abar_mode="fixed16"),
    "A5_hwscan_pot":      dict(weight_mode="ternary", use_fwht=True, hw_scan=True, abar_mode="pot"),
    "A6_hwscan_pot0":     dict(weight_mode="ternary", use_fwht=True, hw_scan=True, abar_mode="pot0"),
    "A7_hwscan_cpot":     dict(weight_mode="ternary", use_fwht=True, hw_scan=True, abar_mode="cpot"),
    "A8_hwscan_cpot2":    dict(weight_mode="ternary", use_fwht=True, hw_scan=True, abar_mode="cpot2"),
}
# integer (bit-exact RTL) re-evaluation of the trained model
INT_EVAL = {
    "A4_hwscan_fixed16": {"int_mult_q15_s32": MULT_Q15},
    "A7_hwscan_cpot": {"int_cpot_s32_g10_round": PROPOSED_32, "int_cpot_s16_g0_trunc": LEGACY_16},
}


# ------------------------------------------------------------------------------
# train / eval
# ------------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, dev, n_cls):
    model.eval()
    preds, ys = [], []
    for bx, by in loader:
        preds.append(model(bx.to(dev)).argmax(-1).cpu())
        ys.append(by)
    p, y = torch.cat(preds), torch.cat(ys)
    acc = float((p == y).float().mean()) * 100
    recalls = [float((p[y == c] == c).float().mean()) for c in range(n_cls) if (y == c).any()]
    return acc, float(np.mean(recalls)) * 100


def train_one(cfg, train_ds, test_ds, in_dim, n_cls, seed, epochs, batch, dev):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = BitMamba2SequenceModel(in_dim, n_cls, cfg).to(dev)
    tl = DataLoader(train_ds, batch_size=batch, shuffle=True,
                    generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs * len(tl))
    lossf = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for bx, by in tl:
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(bx.to(dev)), by.to(dev))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
    return model


def run_task(task, seeds, dev):
    spec = TASKS[task]
    train_ds, test_ds, in_dim, n_cls = spec["load"]()
    test_l = DataLoader(test_ds, batch_size=256)
    out = {"task": task, "seq_len": int(train_ds[0][0].shape[0]), "n_train": len(train_ds),
           "n_test": len(test_ds), "n_classes": n_cls, "epochs": spec["epochs"],
           "seeds": seeds, "model": BASE, "configs": {}}
    print(f"=== {task}: L={out['seq_len']} train={len(train_ds)} test={len(test_ds)} ===", flush=True)
    for name, over in CONFIGS.items():
        cfg = BitMamba2Config(**BASE, **over)
        rec = {"cfg": over, "acc": [], "bal_acc": [], "zero_frac": [], "int_eval": {}}
        t0 = time.time()
        for s in seeds:
            model = train_one(cfg, train_ds, test_ds, in_dim, n_cls, s, spec["epochs"], spec["batch"], dev)
            acc, bacc = evaluate(model, test_l, dev, n_cls)
            rec["acc"].append(acc)
            rec["bal_acc"].append(bacc)
            summ = model.summary()
            rec["zero_frac"].append(summ["ternary_zero_fraction"])
            rec["summary"] = summ
            for iname, scfg in INT_EVAL.get(name, {}).items():
                model.set_int_emulation(scfg)
                iacc, ibacc = evaluate(model, test_l, dev, n_cls)
                sat = sum(getattr(l, "last_sat_events", 0) for l in model.layers)
                model.set_int_emulation(None)
                r = rec["int_eval"].setdefault(iname, {"acc": [], "bal_acc": [], "sat_events_last_batch": []})
                r["acc"].append(iacc)
                r["bal_acc"].append(ibacc)
                r["sat_events_last_batch"].append(sat)
        rec["acc_mean"], rec["acc_std"] = float(np.mean(rec["acc"])), float(np.std(rec["acc"]))
        rec["bal_acc_mean"], rec["bal_acc_std"] = float(np.mean(rec["bal_acc"])), float(np.std(rec["bal_acc"]))
        for r in rec["int_eval"].values():
            r["acc_mean"], r["acc_std"] = float(np.mean(r["acc"])), float(np.std(r["acc"]))
        rec["elapsed_s"] = time.time() - t0
        out["configs"][name] = rec
        ints = " ".join(f"{k}={v['acc_mean']:.2f}" for k, v in rec["int_eval"].items())
        print(f"  {name:20s} acc {rec['acc_mean']:6.2f} +- {rec['acc_std']:4.2f} | "
              f"bal {rec['bal_acc_mean']:6.2f} | zeros {np.mean(rec['zero_frac'])*100:4.1f}% "
              f"| {ints} ({rec['elapsed_s']:.0f}s)", flush=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", f"ablation_{task}.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--seeds", type=int, default=None, help="override per-task seed count")
    ap.add_argument("--configs", nargs="+", default=None)
    a = ap.parse_args()
    if a.configs:
        CONFIGS = {k: v for k, v in CONFIGS.items() if k in a.configs}
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for t in a.tasks:
        run_task(t, [42, 100, 2026, 7, 31][: a.seeds or TASKS[t]["seeds"]], dev)
