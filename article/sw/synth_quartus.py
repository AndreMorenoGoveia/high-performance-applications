"""
Quartus Prime Lite synthesis + fit + timing of every RTL variant (same list as
synth_yosys.py) on the DE10-Nano device 5CSEBA6U23I7. Each variant is compiled
stand-alone with all I/O as virtual pins and a 200 MHz clock constraint.

Usage: python3 synth_quartus.py [name ...]     (env QUARTUS_BIN: dir with quartus_sh)
Writes results/quartus_dse.json (ALMs, registers, M10K, DSP blocks, Fmax slow 1100mV 100C (I7 grade)).

NOTE: written before Quartus was available on this machine; the report parsing
follows the documented .fit.summary / .sta.rpt formats and must be checked on the
first run.
"""

from __future__ import annotations
import json
import os
import re
import shutil
import subprocess
import sys

from synth_yosys import JOBS, RTL, HERE

DEVICE = "5CSEBA6U23I7"
# QUARTUS_BUILD: build dir; under WSL with a Windows Quartus it must live on /mnt/c (no \\wsl$ paths)
BUILD = os.environ.get("QUARTUS_BUILD", os.path.join(os.path.dirname(HERE), "hw", "quartus", "build"))
QBIN = os.environ.get("QUARTUS_BIN", "")
QSH = next((os.path.join(QBIN, e) for e in ("quartus_sh", "quartus_sh.exe")
            if QBIN and os.path.exists(os.path.join(QBIN, e))), shutil.which("quartus_sh") or "quartus_sh")


def write_project(job, d):
    os.makedirs(d, exist_ok=True)
    n = job["name"]
    q = [f'set_global_assignment -name FAMILY "Cyclone V"',
         f"set_global_assignment -name DEVICE {DEVICE}",
         f"set_global_assignment -name TOP_LEVEL_ENTITY {job['top']}",
         "set_global_assignment -name SDC_FILE clk.sdc",
         'set_global_assignment -name OPTIMIZATION_MODE "HIGH PERFORMANCE EFFORT"',
         "set_global_assignment -name NUM_PARALLEL_PROCESSORS 4"]
    for s in job["src"]:   # local copies: relative paths work for both Linux and Windows Quartus
        shutil.copy(os.path.join(RTL, s), d)
    q += [f"set_global_assignment -name VERILOG_FILE {s}" for s in job["src"]]
    q += [f"set_parameter -name {k} {v}" for k, v in job["params"].items()]
    # every port except the clock becomes a virtual pin (block-level characterisation);
    # listed by name because a "*" wildcard also catches clk (no override possible)
    q += [f"set_instance_assignment -name VIRTUAL_PIN ON -to {p}" for p in ports(job) if p != "clk"]
    with open(os.path.join(d, f"{n}.qsf"), "w") as f:
        f.write("\n".join(q) + "\n")
    with open(os.path.join(d, f"{n}.qpf"), "w") as f:
        f.write(f'PROJECT_REVISION = "{n}"\n')
    with open(os.path.join(d, "clk.sdc"), "w") as f:
        f.write("create_clock -name clk -period 5.000 [get_ports {clk}]\nderive_clock_uncertainty\n")


def ports(job):
    """Port names of the top module (ANSI header); buses get [*]."""
    src = open(os.path.join(RTL, next(x for x in job["src"] if x.startswith(job["top"])))).read()
    hdr = re.search(r"module\s+" + job["top"] + r"\b.*?\)\s*;", src, re.S).group(0)
    hdr = re.sub(r"//.*", "", hdr)
    return [n + ("[*]" if bus else "") for bus, n in
            re.findall(r"(?:input|output)\s+(?:wire|reg)?\s*(?:signed)?\s*(\[[^\]]+\])?\s*(\w+)", hdr)]


def num(s):
    return int(s.replace(",", "")) if s else None


def parse(d, n):
    fit = open(os.path.join(d, "output_files", f"{n}.fit.summary")).read() \
        if os.path.exists(os.path.join(d, "output_files")) else open(os.path.join(d, f"{n}.fit.summary")).read()
    g = lambda pat: (re.search(pat, fit) or [None, None])[1]
    r = {"alms": num(g(r"Logic utilization \(in ALMs\)\s*:\s*([\d,]+)")),
         "registers": num(g(r"Total registers\s*:\s*([\d,]+)")),
         "block_memory_bits": num(g(r"Total block memory bits\s*:\s*([\d,]+)")),
         "m10k": num(g(r"Total RAM Blocks\s*:\s*([\d,]+)")),
         "dsp_blocks": num(g(r"Total DSP Blocks\s*:\s*([\d,]+)"))}
    sta_path = next(p for p in (os.path.join(d, "output_files", f"{n}.sta.rpt"), os.path.join(d, f"{n}.sta.rpt"))
                    if os.path.exists(p))
    sta = open(sta_path).read()
    sec = sta.split("Slow 1100mV 100C Model Fmax Summary", 1)
    m = re.search(r";\s*([\d.]+)\s*MHz\s*;\s*([\d.]+)\s*MHz\s*;\s*clk", sec[1]) if len(sec) > 1 else None
    r["fmax_mhz_slow100"] = float(m.group(1)) if m else None
    r["restricted_fmax_mhz_slow100"] = float(m.group(2)) if m else None
    return r


if __name__ == "__main__":
    sel = set(sys.argv[1:])
    jobs = [j for j in JOBS if not sel or j["name"] in sel]
    ver = subprocess.run([QSH, "--version"], capture_output=True, text=True).stdout.strip().splitlines()
    res = []
    for job in jobs:
        d = os.path.join(BUILD, job["name"])
        write_project(job, d)
        print(f"=== {job['name']} ===", flush=True)
        subprocess.run([QSH, "--flow", "compile", job["name"]], cwd=d, check=True,
                       stdout=open(os.path.join(d, "compile.log"), "w"), stderr=subprocess.STDOUT)
        r = {"name": job["name"], "params": job["params"], **parse(d, job["name"])}
        print("  ", r, flush=True)
        res.append(r)
    out = os.path.join(HERE, "results", "quartus_dse.json")
    with open(out, "w") as f:
        json.dump({"tool": ver, "device": DEVICE, "clock_constraint_mhz": 200, "results": res}, f, indent=2)
    print("[OK] ->", out)
