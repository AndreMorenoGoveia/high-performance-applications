"""
Vectorless power estimate (Quartus Power Analyzer, default 12.5% toggle rate) of the
scan-lane variants at 50 MHz (DE10-Nano board clock). Each project is copied from the
synth_quartus.py build dir, recompiled with a 20 ns clock and analysed.
(Re-using the 200 MHz fit directly made quartus_pow 25.1 crash on scan_cpot_s32_g10_round.)

Usage: QUARTUS_BIN=... QUARTUS_BUILD=... python3 quartus_power.py [--parse]
Writes results/quartus_power.json.
"""
import json
import os
import re
import shutil
import subprocess
import sys

from synth_quartus import BUILD, QBIN
from synth_yosys import HERE

NAMES = ["scan_cpot_s32_g10_round", "scan_cpot_s16_g0_trunc", "scan_mult_q15_s32_g10"]
POW = BUILD.rstrip("/") + "_pow"
exe = lambda t: os.path.join(QBIN, t + (".exe" if os.path.exists(os.path.join(QBIN, t + ".exe")) else ""))


def run(n):
    d = os.path.join(POW, n)
    shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(os.path.join(BUILD, n), d, ignore=shutil.ignore_patterns("db", "incremental_db", "*.rpt", "*.summary"))
    open(os.path.join(d, "clk.sdc"), "w").write("create_clock -name clk -period 20.000 [get_ports {clk}]\nderive_clock_uncertainty\n")
    for tool, args in (("quartus_sh", ["--flow", "compile", n]), ("quartus_pow", [n])):
        subprocess.run([exe(tool), *args], cwd=d, check=True, stdout=subprocess.DEVNULL)


def parse(n):
    rpt = open(os.path.join(POW, n, f"{n}.pow.rpt")).read()
    g = lambda pat: float(re.search(pat, rpt).group(1))
    blocks = {b: g(rf";\s*{b}\s*;\s*([\d.]+) mW") for b in ("DSP block", "Combinational cell", "Register cell")}
    return {"name": n, "clock_mhz": 50, "toggle_rate": 0.125,
            "total_mw": g(r"Total Thermal Power Dissipation\s*;\s*([\d.]+) mW"),
            "core_dynamic_mw": g(r"Core Dynamic Thermal Power Dissipation\s*;\s*([\d.]+) mW"),
            "core_static_mw": g(r"Core Static Thermal Power Dissipation\s*;\s*([\d.]+) mW"),
            "dynamic_by_block_mw": blocks,
            "confidence": re.search(r"Power Estimation Confidence\s*;\s*([^;]+);", rpt).group(1).strip()}


if __name__ == "__main__":
    if "--parse" not in sys.argv:
        for n in NAMES:
            run(n)
    res = [parse(n) for n in NAMES]
    json.dump({"tool": "Quartus Prime Lite 25.1std Power Analyzer (vectorless)", "results": res},
              open(os.path.join(HERE, "results", "quartus_power.json"), "w"), indent=2)
    for r in res:
        print(r)
