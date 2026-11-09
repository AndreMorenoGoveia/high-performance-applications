"""
Pre-Quartus resource estimate of every RTL variant with Yosys (synth_intel_alm,
Cyclone V cell library). Yosys 0.33's ABC9 step aborts on these designs
(giaMfs.c assertion), so the flow runs synth_intel_alm up to map_luts and then
maps LUTs with classic `abc -lut 6`.

These are *technology-mapped cell counts* (6-LUTs, carry-chain cells, FFs, M10K,
9x9 multipliers), not Quartus fitter ALMs/Fmax. Writes results/yosys_dse.json.
Env YOSYS: yosys binary (default: system yosys, else ~/opt/eda/usr/bin/yosys).
"""

from __future__ import annotations
import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(os.path.dirname(HERE), "hw", "rtl")
YOSYS = os.environ.get("YOSYS") or shutil.which("yosys") or os.path.expanduser("~/opt/eda/usr/bin/yosys")

K, M = 256, 512          # projection size used for the engine DSE (131,072 weights)
ENGINE_SRC = ["bitlinear_engine.v", "adder_tree.v", "radix3_decoder.v"]
JOBS = (
    [dict(name=f"engine_ENC{e}_P{p}", top="bitlinear_engine", src=ENGINE_SRC,
          params=dict(P=p, K=K, M=M, ENC=e)) for e in (0, 1, 2) for p in (16, 32, 64)]
    + [dict(name="scan_cpot_s32_g10_round", top="scan_lane", src=["scan_lane.v", "adder_tree.v"],
            params=dict(N=16, STATE_W=32, GUARD=10, ROUND=1, DECAY=0)),
       dict(name="scan_cpot_s16_g0_trunc", top="scan_lane", src=["scan_lane.v", "adder_tree.v"],
            params=dict(N=16, STATE_W=16, GUARD=0, ROUND=0, DECAY=0)),
       dict(name="scan_mult_q15_s32_g10", top="scan_lane", src=["scan_lane.v", "adder_tree.v"],
            params=dict(N=16, STATE_W=32, GUARD=10, ROUND=1, DECAY=1)),
       dict(name="fwht16", top="fwht16", src=["fwht16.v"], params={}),
       dict(name="radix3_decoder", top="radix3_decoder", src=["radix3_decoder.v"], params={})]
)


def run(job):
    chp = " ".join(f"-set {k} {v}" for k, v in job["params"].items())
    script = (f"read_verilog -sv {' '.join(job['src'])}; "
              + (f"chparam {chp} {job['top']}; " if chp else "")
              + f"synth_intel_alm -family cyclonev -top {job['top']} -noiopad -run begin:map_luts; "
              "abc -lut 6; techmap -map +/intel_alm/common/alm_map.v; opt -fast; clean; stat")
    out = subprocess.run([YOSYS, "-q", "-p", script.replace("; stat", "; tee -o /dev/stdout stat")],
                         cwd=RTL, capture_output=True, text=True, check=True).stdout
    cells = {m.group(1): int(m.group(2)) for m in re.finditer(r"(MISTRAL_\w+)\s+(\d+)", out)}
    luts = sum(v for k, v in cells.items() if re.fullmatch(r"MISTRAL_ALUT\d", k))
    mul9 = cells.get("MISTRAL_MUL9X9", 0)
    mul18 = cells.get("MISTRAL_MUL18X18", 0)
    mul27 = cells.get("MISTRAL_MUL27X27", 0)
    r = {"name": job["name"], "params": job["params"], "cells": cells,
         "luts": luts, "carry_cells": cells.get("MISTRAL_ALUT_ARITH", 0),
         "ffs": cells.get("MISTRAL_FF", 0), "m10k": cells.get("MISTRAL_M10K", 0),
         "mlab": cells.get("MISTRAL_MLAB", 0), "mul9x9": mul9, "mul18x18": mul18, "mul27x27": mul27,
         # Cyclone V variable-precision DSP block = 3 x 9x9 | 2 x 18x18 | 1 x 27x27
         "dsp_blocks_est": -(-mul9 // 3) + -(-mul18 // 2) + mul27}
    if job["top"] == "bitlinear_engine":
        p, e = job["params"]["P"], job["params"]["ENC"]
        rows = M * (K // p)
        ww = 2 * p if e == 1 else 8 * p
        depth = (rows + 4) // 5 if e == 2 else rows
        r.update(weight_bits=ww * depth, bits_per_weight=ww * depth / (M * K),
                 cycles_per_matvec=M * K // p + (p.bit_length() - 1) + 3)
    print(f"  {r['name']:26s} LUT={luts:5d} carry={r['carry_cells']:5d} FF={r['ffs']:5d} "
          f"M10K={r['m10k']:4d} MLAB={r['mlab']:4d} 9x9={mul9:3d} 18x18={mul18:3d} 27x27={mul27:3d}", flush=True)
    return r


if __name__ == "__main__":
    print(f"=== Yosys ({YOSYS}) synth_intel_alm cyclonev ===")
    with ProcessPoolExecutor(max_workers=6) as ex:
        res = list(ex.map(run, JOBS))
    ver = subprocess.run([YOSYS, "-V"], capture_output=True, text=True).stdout.strip()
    out = os.path.join(HERE, "results", "yosys_dse.json")
    with open(out, "w") as f:
        json.dump({"tool": ver, "flow": "synth_intel_alm -family cyclonev (to map_luts) + abc -lut 6",
                   "engine_K": K, "engine_M": M, "results": res}, f, indent=2)
    print("[OK] ->", out)
