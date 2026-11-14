"""
Open-source ASIC flow (Yosys + OpenROAD, Nangate45 / FreePDK45) for the scan-lane
variants. Synthesis -> floorplan -> global placement -> repair_design -> CTS ->
repair_timing -> detailed placement -> global routing; timing, area and power are
reported with global-route parasitics (no detailed routing / extraction).

Usage: python3 asic_flow.py [period_ns ...]          (default: 2.5)
  env ASIC_BIN (dir with yosys/openroad), NANGATE45 (ORFS platforms/nangate45 dir)
Writes results/asic_nangate45.json (one entry per variant and clock period).
"""
from __future__ import annotations
import json
import os
import re
import subprocess
import sys

from synth_yosys import JOBS, RTL, HERE

BIN = os.environ.get("ASIC_BIN", os.path.expanduser("~/opt/micromamba/root/envs/asic/bin"))
PDK = os.environ.get("NANGATE45", os.path.expanduser("~/opt/pdk/orfs/flow/platforms/nangate45"))
LIB = f"{PDK}/lib/NangateOpenCellLibrary_typical.lib"
BUILD = os.path.join(os.path.dirname(HERE), "hw", "asic", "build")
SCAN = [j for j in JOBS if j["name"].startswith("scan_")]
ACTIVITY = 0.125      # input toggle rate (same default as the Quartus power analysis)


def synth(job, period, d):
    net = os.path.join(d, "netlist.v")
    chp = " ".join(f"-set {k} {v}" for k, v in job["params"].items())
    ys = (f"read_verilog -defer {' '.join(os.path.join(RTL, s) for s in job['src'])}\n"
          f"chparam {chp} {job['top']}\n"
          f"synth -flatten -top {job['top']}\n"
          f"dfflibmap -liberty {LIB}\n"
          f"abc -D {int(period * 1000)} -liberty {LIB} -dont_use AOI211_X1 -dont_use OAI211_X1\n"
          "opt_clean -purge\n"
          "hilomap -singleton -hicell LOGIC1_X1 Z -locell LOGIC0_X1 Z\n"
          "setundef -zero\nsplitnets -ports\nopt_clean -purge\n"
          f"stat -liberty {LIB}\n"
          f"write_verilog -noattr -noexpr -nohex -nodec {net}\n")
    open(os.path.join(d, "synth.ys"), "w").write(ys)
    subprocess.run([f"{BIN}/yosys", "-q", "-l", os.path.join(d, "synth.log"), "-s", os.path.join(d, "synth.ys")],
                   check=True)
    return net


def pnr(job, period, d, net):
    tcl = f"""
read_lef {PDK}/lef/NangateOpenCellLibrary.tech.lef
read_lef {PDK}/lef/NangateOpenCellLibrary.macro.mod.lef
read_liberty {LIB}
read_verilog {net}
link_design {job['top']}
create_clock -name clk -period {period} [get_ports clk]
set io [expr {{0.2 * {period}}}]
set_input_delay $io -clock clk [delete_from_list [all_inputs] [get_ports clk]]
set_output_delay $io -clock clk [all_outputs]
set_load 0.005 [all_outputs]
set_driving_cell -lib_cell BUF_X2 -pin Z [delete_from_list [all_inputs] [get_ports clk]]
source {PDK}/setRC.tcl
initialize_floorplan -utilization 45 -aspect_ratio 1 -core_space 4 -site FreePDK45_38x28_10R_NP_162NW_34O
source {PDK}/make_tracks.tcl
place_pins -hor_layers metal3 -ver_layers metal4
global_placement -density 0.65
estimate_parasitics -placement
repair_design
detailed_placement
clock_tree_synthesis -root_buf CLKBUF_X3 -buf_list CLKBUF_X3 -sink_clustering_enable
set_propagated_clock [all_clocks]
estimate_parasitics -placement
repair_timing -setup
detailed_placement
check_placement
set_routing_layers -signal metal2-metal10 -clock metal2-metal10
global_route -allow_congestion
estimate_parasitics -global_routing
set_power_activity -input -activity {ACTIVITY}
puts "RES wns [sta::worst_slack -max]"
puts "RES tns [sta::total_negative_slack -max]"
report_checks -path_delay max -to [all_registers -data_pins] -fields {{slew cap}} -digits 3
report_design_area
report_power
exit
"""
    open(os.path.join(d, "pnr.tcl"), "w").write(tcl)
    log = os.path.join(d, "pnr.log")
    with open(log, "w") as f:
        subprocess.run([f"{BIN}/openroad", "-no_init", "-exit", os.path.join(d, "pnr.tcl")],
                       stdout=f, stderr=subprocess.STDOUT, check=True)
    return open(log).read()


def parse(txt, synth_log):
    f = lambda pat, s=txt: float(re.search(pat, s).group(1))
    wns = min(0.0, f(r"RES wns (-?[\d.eE+-]+)"))
    tot = re.search(r"^Total\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)", txt, re.M)
    return {"synth_area_um2": f(r"Chip area for (?:top )?module.*?:\s*([\d.]+)", synth_log),
            "design_area_um2": f(r"Design area ([\d.]+) u\^2"),
            "utilization": f(r"Design area [\d.]+ u\^2 ([\d.]+)% utilization") / 100,
            "wns_ns": wns, "tns_ns": f(r"RES tns (-?[\d.eE+-]+)"),
            "power_internal_w": float(tot.group(1)), "power_switching_w": float(tot.group(2)),
            "power_leakage_w": float(tot.group(3)), "power_total_w": float(tot.group(4))}


if __name__ == "__main__":
    periods = [float(p) for p in sys.argv[1:]] or [2.5]
    out_p = os.path.join(HERE, "results", "asic_nangate45.json")
    res = json.load(open(out_p))["results"] if os.path.exists(out_p) else []
    for period in periods:
        for job in SCAN:
            d = os.path.join(BUILD, f"{job['name']}_T{period}")
            os.makedirs(d, exist_ok=True)
            net = synth(job, period, d)
            r = parse(pnr(job, period, d, net), open(os.path.join(d, "synth.log")).read())
            r.update(name=job["name"], params=job["params"], period_ns=period,
                     fmax_mhz=1000 / (period - r["wns_ns"]),
                     energy_pj_per_cycle=r["power_total_w"] * period * 1e3)
            print(json.dumps(r), flush=True)
            res = [x for x in res if not (x["name"] == r["name"] and x["period_ns"] == period)] + [r]
            json.dump({"pdk": "Nangate45 (FreePDK45), typical corner", "flow": "Yosys + OpenROAD, global-route parasitics",
                       "input_activity": ACTIVITY, "results": res}, open(out_p, "w"), indent=2)
