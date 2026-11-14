"""Append the per-seed results of results/ablation_<task>_extra.json into
results/ablation_<task>.json (only the configs present in the extra file) and
recompute means/stds. Each merged config gets its own "seeds" list.

Usage: python3 merge_seeds.py win512 cue1024
"""
import json
import os
import sys

import numpy as np

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def stats(r):
    for k in ("acc", "bal_acc"):
        if k in r:
            r[k + "_mean"], r[k + "_std"] = float(np.mean(r[k])), float(np.std(r[k]))


for task in sys.argv[1:]:
    base_p, extra_p = (os.path.join(R, f"ablation_{task}{t}.json") for t in ("", "_extra"))
    base, extra = json.load(open(base_p)), json.load(open(extra_p))
    for name, e in extra["configs"].items():
        b = base["configs"][name]
        seeds = b.get("seeds", base["seeds"])
        assert not set(seeds) & set(extra["seeds"]), f"{task}/{name}: seeds already merged"
        b["seeds"] = seeds + extra["seeds"]
        for k in ("acc", "bal_acc", "zero_frac"):
            b[k] += e[k]
        for iname, ie in e["int_eval"].items():
            for k in ("acc", "bal_acc", "sat_events_last_batch"):
                b["int_eval"][iname][k] += ie[k]
            stats(b["int_eval"][iname])
        stats(b)
        print(f"{task}/{name}: {len(b['acc'])} seeds, acc {b['acc_mean']:.2f} +- {b['acc_std']:.2f}")
    json.dump(base, open(base_p, "w"), indent=2)
