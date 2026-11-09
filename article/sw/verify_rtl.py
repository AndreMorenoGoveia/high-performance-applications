"""
RTL vs. golden-model verification. Generates stimulus/expected vectors from
models/golden_model_bitexact.py, simulates every RTL module with Icarus Verilog and
requires 0 mismatches. Writes results/rtl_verification.json.

Usage: python3 verify_rtl.py      (env IVL_ROOT: prefix holding bin/iverilog, default:
                                    system iverilog, else ~/opt/eda/usr)
"""

from __future__ import annotations
import json
import os
import re
import shutil
import subprocess
import numpy as np

from models.golden_model_bitexact import (
    RADIX3_LUT, fwht16_int, matvec_int, radix3_memory_words,
    ScanCfg, scan_lane_run, encode_cpot, decode_cpot,
)

HERE = os.path.dirname(os.path.abspath(__file__))
HW = os.path.join(os.path.dirname(HERE), "hw")
RTL = os.path.join(HW, "rtl")
TB = os.path.join(HW, "tb")
VEC = os.path.join(TB, "vectors")
BUILD = os.path.join(TB, "build")


def ivl_cmds():
    if shutil.which("iverilog") and not os.environ.get("IVL_ROOT"):
        return ["iverilog"], ["vvp"]
    root = os.environ.get("IVL_ROOT", os.path.expanduser("~/opt/eda/usr"))
    lib = os.path.join(root, "lib", "x86_64-linux-gnu", "ivl")
    return ([os.path.join(root, "bin", "iverilog"), "-B", lib],
            [os.path.join(root, "bin", "vvp"), "-M", lib])


def hexw(v: int, bits: int) -> str:
    return format(int(v) & ((1 << bits) - 1), "0%dx" % ((bits + 3) // 4))


def pack_fields(fields) -> int:
    """fields: list of (value, bits) MSB first -> int."""
    out = 0
    for v, b in fields:
        out = (out << b) | (int(v) & ((1 << b) - 1))
    return out


def write_lines(path, lines):
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def simulate(tb: str, srcs, params: dict, tag: str) -> dict:
    iverilog, vvp = ivl_cmds()
    out = os.path.join(BUILD, tag + ".vvp")
    cmd = iverilog + ["-g2012", "-o", out, "-s", tb]
    for k, v in params.items():
        cmd += ["-P", f"{tb}.{k}={json.dumps(v) if isinstance(v, str) else v}"]
    cmd += [os.path.join(TB, tb + ".v")] + [os.path.join(RTL, s) for s in srcs]
    subprocess.run(cmd, check=True, cwd=TB)
    res = subprocess.run(vvp + [out], check=True, cwd=TB, capture_output=True, text=True).stdout
    line = [l for l in res.splitlines() if l.startswith("RESULT")][-1]
    checked = int(re.search(r"checked=(\d+)", line).group(1))
    mism = int(re.search(r"mismatches=(\d+)", line).group(1))
    r = {"test": tag, "checked": checked, "mismatches": mism}
    cyc = re.search(r"cycles=(\d+)", line)
    if cyc:
        r["cycles_per_matvec"] = int(cyc.group(1))
    print(f"  {tag:36s} checked={checked:6d} mismatches={mism}" + (f" cycles={r['cycles_per_matvec']}" if cyc else ""))
    return r


def test_radix3():
    exp = []
    for b in range(256):
        word = 0
        for i, t in enumerate(RADIX3_LUT[b]):
            code = {0: 0b00, 1: 0b01, -1: 0b11}[int(t)]
            word |= code << (2 * i)
        exp.append(hexw(word, 10))
    write_lines(os.path.join(VEC, "radix3_expected.hex"), exp)
    return simulate("tb_radix3", ["radix3_decoder.v"], {}, "radix3_exhaustive")


def test_fwht(rng, nv=2000):
    x = rng.integers(-128, 128, size=(nv, 16))
    x[:4] = [[-128] * 16, [127] * 16, [-128, 127] * 8, [0] * 16]   # extremes
    y = fwht16_int(x)
    xl = [hexw(pack_fields([(v, 8) for v in row[::-1]]), 128) for row in x]
    yl = [hexw(pack_fields([(v, 12) for v in row[::-1]]), 192) for row in y]
    write_lines(os.path.join(VEC, "fwht_x.hex"), xl)
    write_lines(os.path.join(VEC, "fwht_y.hex"), yl)
    return simulate("tb_fwht16", ["fwht16.v"], {"NV": nv}, "fwht16_random")


def test_engine(rng, P, K, M, enc, nx=4):
    if enc == 0:
        w = rng.integers(-128, 128, size=(M, K))
    else:
        w = rng.choice([-1, 0, 1], size=(M, K), p=[0.35, 0.30, 0.35])
    x = rng.integers(-128, 128, size=(nx, K))
    y = np.stack([matvec_int(w, xv) for xv in x])
    KC = K // P
    rows = w.reshape(M, KC, P).reshape(M * KC, P)          # flat row r = m*KC + chunk
    if enc == 0:
        words = [hexw(pack_fields([(v, 8) for v in r[::-1]]), 8 * P) for r in rows]
    elif enc == 1:
        code = {0: 0b00, 1: 0b01, -1: 0b11}
        words = [hexw(pack_fields([(code[int(v)], 2) for v in r[::-1]]), 2 * P) for r in rows]
    else:
        mem = radix3_memory_words(rows)
        words = [hexw(pack_fields([(v, 8) for v in r[::-1]]), 8 * P) for r in mem]
    tag = f"engine_P{P}_K{K}_M{M}_ENC{enc}"
    wf, xf, yf = (os.path.join(VEC, f"{tag}_{s}.hex") for s in ("w", "x", "y"))
    write_lines(wf, words)
    write_lines(xf, [hexw(v, 8) for v in x.reshape(-1)])
    write_lines(yf, [hexw(v, 24) for v in y.reshape(-1)])
    return simulate("tb_engine", ["bitlinear_engine.v", "adder_tree.v", "radix3_decoder.v"],
                    {"P": P, "K": K, "M": M, "ENC": enc, "NX": nx,
                     "WF": wf, "XF": xf, "YF": yf}, tag)


def test_scan(rng, cfg: ScanCfg, name, N=16, ns=4, T=512):
    toks, ys = [], []
    for _ in range(ns):
        x = rng.integers(-128, 128, size=T)
        b = rng.integers(-128, 128, size=(T, N))
        c = rng.integers(-128, 128, size=(T, N))
        abar = 1 - 10 ** rng.uniform(-3.5, -0.2, size=T)     # mostly long-memory decays
        abar[rng.random(T) < 0.15] = rng.uniform(0.02, 0.6)  # some fast-decay tokens
        m, k = encode_cpot(abar)
        j = rng.integers(0, 16, size=T)
        a_q15 = np.round(decode_cpot(m, k) * 32768).clip(0, 32767).astype(np.int64)
        y = scan_lane_run(x, b, c, m, k, j, a_q15, cfg)
        ys += [hexw(v, cfg.y_w) for v in y]
        for t in range(T):
            f = [(x[t], 8)] + [(v, 8) for v in b[t][::-1]] + [(v, 8) for v in c[t][::-1]]
            f += [(m[t], 1), (k[t], 4), (j[t], 4), (a_q15[t], 16)]
            toks.append(hexw(pack_fields(f), 8 + 16 * N + 25))
    tf, yf = (os.path.join(VEC, f"scan_{name}_{s}.hex") for s in ("tok", "y"))
    write_lines(tf, toks)
    write_lines(yf, ys)
    return simulate("tb_scan", ["scan_lane.v", "adder_tree.v"],
                    {"N": N, "STATE_W": cfg.state_w, "GUARD": cfg.guard,
                     "ROUND": int(cfg.round_shift), "DECAY": int(cfg.decay == "mult"),
                     "Y_W": cfg.y_w, "NS": ns, "T": T, "TF": tf, "YF": yf}, f"scan_{name}")


def main():
    os.makedirs(VEC, exist_ok=True)
    os.makedirs(BUILD, exist_ok=True)
    rng = np.random.default_rng(2026)
    print("=== RTL vs golden model (Icarus Verilog) ===")
    results = [test_radix3(), test_fwht(rng)]
    for enc in (0, 1, 2):
        for P, K, M in ((16, 64, 40), (32, 128, 25), (64, 256, 17)):
            results.append(test_engine(rng, P, K, M, enc))
    results.append(test_scan(rng, ScanCfg(), "cpot_s32_g10_round"))
    results.append(test_scan(rng, ScanCfg(state_w=16, guard=0, round_shift=False), "cpot_s16_g0_trunc"))
    results.append(test_scan(rng, ScanCfg(decay="mult"), "mult_q15_s32_g10"))
    total = sum(r["mismatches"] for r in results)
    out = os.path.join(HERE, "results", "rtl_verification.json")
    with open(out, "w") as f:
        json.dump({"simulator": "Icarus Verilog 12.0", "tests": results,
                   "total_checked": sum(r["checked"] for r in results),
                   "total_mismatches": total}, f, indent=2)
    print(f"[{'OK' if total == 0 else 'FAIL'}] total mismatches = {total} -> {out}")
    raise SystemExit(0 if total == 0 else 1)


if __name__ == "__main__":
    main()
