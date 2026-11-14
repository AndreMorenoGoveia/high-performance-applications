"""
Builds every number-bearing table and figure of the paper from results/*.json,
so the manuscript contains no hand-typed results.
Outputs: paper/generated/*.tex and paper/figures/*.pdf
"""

from __future__ import annotations
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
PAPER = os.path.join(os.path.dirname(HERE), "paper")
GEN = os.path.join(PAPER, "generated")
FIG = os.path.join(PAPER, "figures")

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # validated, fixed order
MARKERS = ["o", "s", "^", "D", "v"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MODE_LABEL = {"fixed16": "Q1.15 (mult.)", "pot": "PoT $2^{-k}$, $k\\geq1$",
              "pot0": "PoT $2^{-k}$, $k\\geq0$", "cpot": "CPoT", "cpot2": "CPoT-2"}


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def br(x, d=2):
    """Brazilian decimal comma."""
    s = f"{x:.{d}f}"
    if float(s) == 0:
        s = s.lstrip("-")
    return s.replace(".", "{,}")


def thousands(n):
    return f"{n:,}".replace(",", ".")


def write(name, s):
    os.makedirs(GEN, exist_ok=True)
    with open(os.path.join(GEN, name), "w") as f:
        f.write(s)
    print("  wrote", name)


def style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK2)
        ax.spines[sp].set_linewidth(0.6)
    ax.tick_params(colors=INK2, labelsize=7.5, width=0.6)
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------------------

def verification_table():
    v = load("rtl_verification.json")
    lab = {"scan_cpot_s32_g10_round": "CPoT, 32 bits, 10 bits de guarda, arredondamento (proposta)",
           "scan_cpot_s16_g0_trunc": "CPoT, 16 bits, truncamento (legado)",
           "scan_mult_q15_s32_g10": "Multiplicador Q1.15, 32 bits (baseline)"}
    by = {t["test"]: t for t in v["tests"]}
    rows = [f"{l} & {thousands(by[n]['checked'])} & {by[n]['mismatches']} \\\\" for n, l in lab.items()]
    tot = sum(by[n]["checked"] for n in lab)
    mis = sum(by[n]["mismatches"] for n in lab)
    rows += ["\\midrule", f"\\textbf{{Total}} & \\textbf{{{thousands(tot)}}} & \\textbf{{{mis}}} \\\\"]
    write("tab_verification.tex", "\n".join(rows) + "\n\\bottomrule\n")
    return tot, mis


def abar_table():
    r = load("long_range_results.json")
    e1, e2 = r["E1_grid_0.90_0.999"], r["E2_readout_vs_L"]
    Ls = e2["seq_lengths"]
    idx = [Ls.index(L) for L in (256, 4096, 16384)]
    rows = []
    hw = {"fixed16": "1 mult.", "pot": "1 shift", "pot0": "1 shift", "cpot": "1 shift + 1 sub.",
          "cpot2": "2 shifts + 2 sub."}
    for m in ["fixed16", "pot", "pot0", "cpot", "cpot2"]:
        g = e1[m]
        tau = "$\\infty$" if not np.isfinite(g["tau_rel_err_median"]) else br(100 * g["tau_rel_err_median"], 1) + "\\%"
        sq = " & ".join(br(e2[m]["sqnr_mean"][i], 1) for i in idx)
        name = MODE_LABEL[m] + (" (só SW)" if m == "cpot2" else "")
        rows.append(f"{name} & {hw[m]} & {g['distinct_levels']} & {br(g['max_abs_err'], 4)} & {tau} & {sq} \\\\")
    write("tab_abar.tex", "\n".join(rows) + "\n\\bottomrule\n")

    e3, e4 = r["E3_integer_impulse"], r["E4_integer_random_sqnr"]
    lab = {"legacy_s16_g0_trunc": "Legado: 16 bits, sem guarda, truncamento",
           "proposed_s32_g10_round": "Corrigido: 32 bits, 10 bits de guarda, arredondamento",
           "mult_q15_s32_g10": "Baseline: multiplicador Q1.15, 32 bits"}
    rows = []
    for name in ["legacy_s16_g0_trunc", "proposed_s32_g10_round", "mult_q15_s32_g10"]:
        imp = e3.get(name)
        cells = " & ".join(br(imp[k]["h_over_ideal_at_4tau"], 2) for k in ("8", "10", "12")) if imp else "--- & --- & ---"
        rows.append(f"{lab[name]} & {cells} & {br(e4[name]['sqnr_mean_db'], 1)} \\\\")
    write("tab_state.tex", "\n".join(rows) + "\n\\bottomrule\n")


def long_range_figure():
    r = load("long_range_results.json")
    e2, e3 = r["E2_readout_vs_L"], r["E3_integer_impulse"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.9))
    Ls = e2["seq_lengths"]
    for i, m in enumerate(["fixed16", "pot", "pot0", "cpot", "cpot2"]):
        a.plot(Ls, e2[m]["sqnr_mean"], color=SERIES[i], marker=MARKERS[i], ms=4, lw=1.4, label=MODE_LABEL[m])
    a.set_xscale("log", base=2)
    a.set_xlabel("Comprimento $L$", color=INK, fontsize=8)
    a.set_ylabel("SQNR de $y_L$ (dB)", color=INK, fontsize=8)
    a.set_title("(a) Quantização de $\\bar{A}_t$ (estado em float)", fontsize=8.5, color=INK)
    style(a)
    a.legend(fontsize=6.5, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.28))

    k = 10
    tau = 2 ** k
    t = np.arange(4 * tau + 1)
    ideal = (1 - 2.0 ** -k) ** t
    # re-simulate trajectories with the golden model for the plot
    from models.golden_model_bitexact import LEGACY_16, PROPOSED_32, _shr
    for i, (cfg, lab) in enumerate(((LEGACY_16, "16 bits, trunc. (legado)"), (PROPOSED_32, "32 bits, guarda 10, arred."))):
        h = np.array([4096 << cfg.guard], dtype=np.int64)
        tr = [h[0]]
        for _ in t[1:]:
            h = h - _shr(h, k, cfg.round_shift)
            tr.append(h[0])
        b.plot(t / tau, np.array(tr) / (4096 << cfg.guard), color=SERIES[[1, 3][i]], lw=1.6, label=lab)
    b.plot(t / tau, ideal, color=INK, lw=1.0, ls=(0, (3, 3)), label="ideal $(1-2^{-10})^t$", zorder=3)
    b.set_yscale("log")
    b.set_xlabel("Tempo $t/\\tau$ ($\\tau = 2^{10}$ passos)", color=INK, fontsize=8)
    b.set_ylabel("$h_t / h_0$", color=INK, fontsize=8)
    b.set_title("(b) Resposta ao impulso do estado inteiro (RTL)", fontsize=8.5, color=INK)
    style(b)
    b.legend(fontsize=6.5, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, "fig_long_memory.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig_long_memory.pdf")


ABL_LABEL = {
    "S0_fp32": "S0 & FP32 (referência) & FP32 & float exato",
    "S1_w8a8": "S1 & W8A8, \\emph{scan} ideal & INT8 & float exato",
    "S2_q15": "S2 & \\emph{scan} HW & INT8 & Q1.15 (mult.)",
    "S3_pot1": "S3 & \\emph{scan} HW & INT8 & PoT, $k\\geq1$",
    "S4_pot0": "S4 & \\emph{scan} HW & INT8 & PoT, $k\\geq0$",
    "S5_cpot": "\\textbf{S5} & \\textbf{\\emph{scan} HW} & INT8 & \\textbf{CPoT}",
    "S6_cpot2": "S6 & \\emph{scan} HW & INT8 & CPoT-2 (só SW)",
}
INT_LABEL = {
    "int_mult_q15_s32": "S2$^\\dagger$ & \\multicolumn{2}{l}{S2 com \\emph{scan} inteiro bit-exato} & Q1.15, 32b",
    "int_cpot_s32_g10_round": "\\textbf{S5$^\\dagger$} & \\multicolumn{2}{l}{\\textbf{S5 com \\emph{scan} inteiro bit-exato}} & \\textbf{CPoT, 32b}",
    "int_cpot_s16_g0_trunc": "S5$^\\ddagger$ & \\multicolumn{2}{l}{S5 com estado legado (16b, trunc.)} & CPoT, 16b",
}
TASK_ORDER = ("ecg5000", "cue256", "cue1024", "win512")
SHORT = {"ecg5000": "ecg", "cue256": "cueA", "cue1024": "cueB", "win512": "win"}
TAG = {"ecg5000": "Ecg", "cue256": "CueA", "cue1024": "CueB", "win512": "Win"}


def ablation_table():
    tasks = [t for t in TASK_ORDER if load(f"ablation_{t}.json")]
    data = {t: load(f"ablation_{t}.json") for t in tasks}
    pm = lambda vals: f"${br(np.mean(vals), 1)} \\pm {br(np.std(vals), 1)}$"
    # recall is all-or-nothing per seed: report seeds solved (acc > 90%) instead of mean +- std
    cell = lambda vals, t="": f"{sum(a > 90 for a in vals)}/{len(vals)}" if t.startswith("cue") else pm(vals)
    rows = []
    for name, label in ABL_LABEL.items():
        cs = []
        for t in tasks:
            c = data[t]["configs"].get(name)
            cs.append(cell(c["acc"], t) if c else "---")
            if t == "ecg5000":
                cs.append(cell(c["bal_acc"]) if c else "---")
        rows.append(label + " & " + " & ".join(cs) + " \\\\")
        if name == "S6_cpot2":
            rows.append("\\midrule")
    for parent, ints in (("S2_q15", ["int_mult_q15_s32"]),
                         ("S5_cpot", ["int_cpot_s32_g10_round", "int_cpot_s16_g0_trunc"])):
        for iname in ints:
            cs = []
            for t in tasks:
                c = data[t]["configs"].get(parent, {}).get("int_eval", {}).get(iname)
                cs.append(cell(c["acc"], t) if c else "---")
                if t == "ecg5000":
                    cs.append(cell(c["bal_acc"]) if c else "---")
            rows.append(INT_LABEL[iname] + " & " + " & ".join(cs) + " \\\\")
    body = "\n".join(rows) + "\n"

    head = {"ecg5000": "\\multicolumn{2}{c}{ECG5000 ($L=140$)}", "cue256": "Record. $L=256$",
            "cue1024": "Record. $L=1024$", "win512": "Janela $L=512$"}
    sub = {"ecg5000": "Acur. & Acur. bal.", "cue256": "Resolve", "cue1024": "Resolve", "win512": "Acur."}
    ncols = sum(2 if t == "ecg5000" else 1 for t in tasks)
    write("tab_ablation.tex",
          f"\\begin{{tabular}}{{llll{'c' * ncols}}}\n\\toprule\n"
          f" & & & & {' & '.join(head[t] for t in tasks)} \\\\\n"
          f"ID & Configuração & Projeções & $\\bar{{A}}_t$ & {' & '.join(sub[t] for t in tasks)} \\\\\n\\midrule\n"
          + body + "\\bottomrule\n\\end{tabular}\n")
    meta = []
    for t in tasks:
        d = data[t]
        meta.append(f"{t}: L={d['seq_len']}, treino={d['n_train']}, teste={d['n_test']}, "
                    f"épocas={d['epochs']}, seeds={len(d['seeds'])}")
    write("ablation_meta.tex", "% " + " | ".join(meta) + "\n")

    s = data[tasks[0]]["configs"]["S5_cpot"]["summary"]
    macros = {"NParams": thousands(s["total_params"])}
    for t in TASK_ORDER:                            # missing tasks stay visibly pending
        for a_ in ("Seeds", "Epochs", "Train", "Test"):
            macros[f"{a_}{TAG[t]}"] = "\\textbf{??}"
    for t in tasks:
        d = data[t]
        macros[f"Seeds{TAG[t]}"] = str(len(d["seeds"]))
        macros[f"Epochs{TAG[t]}"] = str(d["epochs"])
        macros[f"Train{TAG[t]}"] = thousands(d["n_train"])
        macros[f"Test{TAG[t]}"] = thousands(d["n_test"])
    # per task/config results: \acc{task}{id}, \solved{task}{id} (seeds with acc > 90%)
    lines = ["\\providecommand{\\acc}[2]{\\csname acc@#1@#2\\endcsname}",
             "\\providecommand{\\solved}[2]{\\csname solved@#1@#2\\endcsname}"]
    short = SHORT
    for t in tasks:
        for name, c in data[t]["configs"].items():
            cid = name.split("_")[0]
            entries = [(cid, c["acc"])] + [(cid + iname.split("_")[1] + iname.split("_")[2], r["acc"])
                                           for iname, r in c["int_eval"].items()]
            for key, accs in entries:
                lines.append(f"\\expandafter\\def\\csname acc@{short[t]}@{key}\\endcsname{{{br(np.mean(accs), 1)}}}")
                lines.append(f"\\expandafter\\def\\csname solved@{short[t]}@{key}\\endcsname"
                             f"{{{sum(a > 90 for a in accs)}/{len(accs)}}}")
    write("macros_ablation.tex", "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items())
          + "\n".join(lines) + "\n")


SCAN_HW = {"scan_cpot_s32_g10_round": "CPoT, 32b + guarda (proposta)",
           "scan_cpot_s16_g0_trunc": "CPoT, 16b, trunc. (legado, incorreto)",
           "scan_mult_q15_s32_g10": "Multiplicador Q1.15, 32b (baseline)"}


def quartus_table():
    q = load("quartus_dse.json")
    by = {r["name"]: r for r in q["results"]}
    pw = {r["name"]: r for r in (load("quartus_power.json") or {"results": []})["results"]}
    for n in by:
        by[n]["dyn_mw"] = pw.get(n, {}).get("core_dynamic_mw")
    rows = [f"{l} & {thousands(by[n]['alms'])} & {thousands(by[n]['registers'])} & {by[n]['dsp_blocks']} & "
            f"{br(by[n]['fmax_mhz_slow100'], 1)} & {br(by[n]['dyn_mw'], 1)} \\\\" for n, l in SCAN_HW.items()]
    write("tab_quartus.tex", "\n".join(rows) + "\n\\bottomrule\n")
    ver = q["tool"][1].split("Build")[0].replace("Version", "").strip()
    write("macros_quartus.tex", f"\\newcommand{{\\QuartusVersion}}{{Quartus Prime Lite {ver}}}\n")
    return by


def asic_table():
    """Nangate45 OpenROAD results: fmax from the tightest run, area/power at the iso-frequency run."""
    a = load("asic_nangate45.json")
    if not a:
        return {}
    res = a["results"]
    iso = max(r["period_ns"] for r in res)                      # every variant meets the slowest target
    by = {}
    for n in SCAN_HW:
        rs = [r for r in res if r["name"] == n]
        it = next(r for r in rs if r["period_ns"] == iso)
        tight = min(rs, key=lambda r: r["period_ns"])
        by[n] = dict(it, fmax_best=tight["fmax_mhz"], fmax_met=tight["wns_ns"] >= 0, iso_mhz=1000 / iso)
    ge = lambda n: "$\\geq$" if by[n]["fmax_met"] else ""        # target met: fmax is only a lower bound
    rows = [f"{l} & {thousands(round(by[n]['design_area_um2']))} & {ge(n)}{br(by[n]['fmax_best'], 0)} & "
            f"{br(by[n]['power_total_w'] * 1e3, 1)} & {br(by[n]['energy_pj_per_cycle'], 1)} \\\\"
            for n, l in SCAN_HW.items()]
    write("tab_asic.tex", "\n".join(rows) + "\n\\bottomrule\n")
    return by


def number_macros(ver_tot, ver_mis, qby, aby):
    r = load("long_range_results.json")
    e1, e3, e4 = r["E1_grid_0.90_0.999"], r["E3_integer_impulse"], r["E4_integer_random_sqnr"]
    c, m_ = qby["scan_cpot_s32_g10_round"], qby["scan_mult_q15_s32_g10"]
    m = {
        "VerTests": "3", "VerTotal": thousands(ver_tot), "VerMism": str(ver_mis),
        "SqnrLegacy": br(e4["legacy_s16_g0_trunc"]["sqnr_mean_db"], 1),
        "SqnrProposed": br(e4["proposed_s32_g10_round"]["sqnr_mean_db"], 1),
        "SqnrMult": br(e4["mult_q15_s32_g10"]["sqnr_mean_db"], 1),
        "ImpLegacyTwelve": br(e3["legacy_s16_g0_trunc"]["12"]["h_over_ideal_at_4tau"], 1),
        "ImpLegacyEight": br(e3["legacy_s16_g0_trunc"]["8"]["h_over_ideal_at_4tau"], 2),
        "TauErrCpot": br(100 * e1["cpot"]["tau_rel_err_median"], 1) + "\\%",
        "TauErrCpotTwo": br(100 * e1["cpot2"]["tau_rel_err_median"], 1) + "\\%",
        "TauErrPot": br(100 * e1["pot"]["tau_rel_err_median"], 1) + "\\%",
        "LevelsCpot": str(e1["cpot"]["distinct_levels"]),
        "DspCpot": str(c["dsp_blocks"]), "DspMult": str(m_["dsp_blocks"]),
        "AlmCpot": thousands(c["alms"]), "AlmMult": thousands(m_["alms"]),
        "FmaxCpot": br(c["fmax_mhz_slow100"], 1), "FmaxMult": br(m_["fmax_mhz_slow100"], 1),
        "DspSaving": br(100 * (1 - c["dsp_blocks"] / m_["dsp_blocks"]), 0) + "\\%",
        "LanesCpot": str(112 // c["dsp_blocks"]), "LanesMult": str(112 // m_["dsp_blocks"]),
        "QPowCpot": br(c["dyn_mw"], 1), "QPowMult": br(m_["dyn_mw"], 1),
    }
    if aby:
        ac, am = aby["scan_cpot_s32_g10_round"], aby["scan_mult_q15_s32_g10"]
        m.update({
            "AsicAreaCpot": thousands(round(ac["design_area_um2"])), "AsicAreaMult": thousands(round(am["design_area_um2"])),
            "AsicAreaSaving": br(100 * (1 - ac["design_area_um2"] / am["design_area_um2"]), 0) + "\\%",
            "AsicFmaxCpot": br(ac["fmax_best"], 0), "AsicFmaxMult": br(am["fmax_best"], 0),
            "AsicIsoMhz": br(ac["iso_mhz"], 0),
            "AsicPowCpot": br(ac["power_total_w"] * 1e3, 1), "AsicPowMult": br(am["power_total_w"] * 1e3, 1),
            "AsicEnergyCpot": br(ac["energy_pj_per_cycle"], 1), "AsicEnergyMult": br(am["energy_pj_per_cycle"], 1),
            "AsicEnergySaving": br(100 * (1 - ac["energy_pj_per_cycle"] / am["energy_pj_per_cycle"]), 0) + "\\%",
        })
    write("macros_numbers.tex", "".join(f"\\newcommand{{\\{k}}}{{{val}}}\n" for k, val in m.items()))


if __name__ == "__main__":
    tot, mis = verification_table()
    abar_table()
    long_range_figure()
    number_macros(tot, mis, quartus_table(), asic_table())
    if any(load(f"ablation_{t}.json") for t in TASK_ORDER):
        ablation_table()
    else:   # placeholders so the paper still compiles; visibly marked as missing
        print("  (no ablation results yet -> placeholders)")
        keys = ["NParams"] + [f"{a}{TAG[t]}" for a in ("Seeds", "Epochs", "Train", "Test") for t in TASK_ORDER]
        write("macros_ablation.tex", "".join(f"\\newcommand{{\\{k}}}{{\\textbf{{??}}}}\n" for k in keys)
              + "\\providecommand{\\acc}[2]{\\textbf{??}}\\providecommand{\\solved}[2]{\\textbf{??}}\n")
        write("tab_ablation.tex", "\\begin{tabular}{c}\\toprule\n\\textbf{resultados pendentes} \\\\\n\\bottomrule\n\\end{tabular}\n")
