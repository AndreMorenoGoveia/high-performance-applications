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
    rows = []
    groups = {"radix3": "Decodificador Radix-3 (exaustivo, 256 códigos)",
              "fwht16": "FWHT 16 pontos (vetores aleatórios + extremos)",
              "engine": "Motor matriz-vetor (INT8/2-bit/Radix-3, $P\\in\\{16,32,64\\}$)",
              "scan": "Lane de \\emph{scan} (CPoT 32b, CPoT 16b legado, Q1.15)"}
    for key, label in groups.items():
        ts = [t for t in v["tests"] if t["test"].startswith(key)]
        rows.append(f"{label} & {len(ts)} & {thousands(sum(t['checked'] for t in ts))} & "
                    f"{sum(t['mismatches'] for t in ts)} \\\\")
    rows.append("\\midrule")
    rows.append(f"\\textbf{{Total}} & {len(v['tests'])} & \\textbf{{{thousands(v['total_checked'])}}} & "
                f"\\textbf{{{v['total_mismatches']}}} \\\\")
    write("tab_verification.tex", "\n".join(rows) + "\n\\bottomrule\n")


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
    "A0_fp32": "A0 & FP32 (referência) & FP32 & --- & float exato",
    "A1_w8a8": "A1 & W8A8 & INT8 & não & float exato",
    "A2_tern_nofwht": "A2 & Ternário & 1,58b & não & float exato",
    "A3_tern_hbitlinear": "A3 & + H-BitLinear & 1,58b & sim & float exato",
    "A4_hwscan_fixed16": "A4 & + \\emph{scan} HW & 1,58b & sim & Q1.15 (mult.)",
    "A5_hwscan_pot": "A5 & + \\emph{scan} HW & 1,58b & sim & PoT, $k\\geq1$",
    "A6_hwscan_pot0": "A6 & + \\emph{scan} HW & 1,58b & sim & PoT, $k\\geq0$",
    "A7_hwscan_cpot": "\\textbf{A7} & \\textbf{+ \\emph{scan} HW} & 1,58b & sim & \\textbf{CPoT}",
    "A8_hwscan_cpot2": "A8 & + \\emph{scan} HW & 1,58b & sim & CPoT-2 (só SW)",
}
INT_LABEL = {
    "int_mult_q15_s32": "A4$^\\dagger$ & \\multicolumn{3}{l}{A4 avaliado com \\emph{scan} inteiro bit-exato} & Q1.15, 32b",
    "int_cpot_s32_g10_round": "\\textbf{A7$^\\dagger$} & \\multicolumn{3}{l}{\\textbf{A7 avaliado com \\emph{scan} inteiro bit-exato}} & \\textbf{CPoT, 32b}",
    "int_cpot_s16_g0_trunc": "A7$^\\ddagger$ & \\multicolumn{3}{l}{A7 com estado legado (16b, trunc.)} & CPoT, 16b",
}


def ablation_table():
    tasks = [t for t in ("ecg5000", "cue256", "cue1024") if load(f"ablation_{t}.json")]
    data = {t: load(f"ablation_{t}.json") for t in tasks}
    cell = lambda vals: f"${br(np.mean(vals), 1)} \\pm {br(np.std(vals), 1)}$"
    rows = []
    for name, label in ABL_LABEL.items():
        cs = []
        for t in tasks:
            c = data[t]["configs"].get(name)
            cs.append(cell(c["acc"]) if c else "---")
            if t == "ecg5000":
                cs.append(cell(c["bal_acc"]) if c else "---")
        rows.append(label + " & " + " & ".join(cs) + " \\\\")
        if name == "A8_hwscan_cpot2":
            rows.append("\\midrule")
    for parent, ints in (("A4_hwscan_fixed16", ["int_mult_q15_s32"]),
                         ("A7_hwscan_cpot", ["int_cpot_s32_g10_round", "int_cpot_s16_g0_trunc"])):
        for iname in ints:
            cs = []
            for t in tasks:
                c = data[t]["configs"].get(parent, {}).get("int_eval", {}).get(iname)
                cs.append(cell(c["acc"]) if c else "---")
                if t == "ecg5000":
                    cs.append(cell(c["bal_acc"]) if c else "---")
            rows.append(INT_LABEL[iname] + " & " + " & ".join(cs) + " \\\\")
    body = "\n".join(rows) + "\n"

    head = {"ecg5000": "\\multicolumn{2}{c}{ECG5000 ($L=140$)}", "cue256": "Cue $L=256$", "cue1024": "Cue $L=1024$"}
    sub = {"ecg5000": "Acur. & Acur. bal.", "cue256": "Acur.", "cue1024": "Acur."}
    ncols = sum(2 if t == "ecg5000" else 1 for t in tasks)
    write("tab_ablation.tex",
          f"\\begin{{tabular}}{{lllll{'c' * ncols}}}\n\\toprule\n"
          f" & & & & & {' & '.join(head[t] for t in tasks)} \\\\\n"
          f"ID & Configuração & Pesos & FWHT & $\\bar{{A}}_t$ & {' & '.join(sub[t] for t in tasks)} \\\\\n\\midrule\n"
          + body + "\\bottomrule\n\\end{tabular}\n")
    meta = []
    for t in tasks:
        d = data[t]
        meta.append(f"{t}: L={d['seq_len']}, treino={d['n_train']}, teste={d['n_test']}, "
                    f"épocas={d['epochs']}, seeds={len(d['seeds'])}")
    write("ablation_meta.tex", "% " + " | ".join(meta) + "\n")

    # model facts used in the text
    any_t = data[tasks[0]]["configs"]["A7_hwscan_cpot"]
    s = any_t["summary"]
    zeros = [z for t in tasks for z in data[t]["configs"]["A7_hwscan_cpot"]["zero_frac"]]
    macros = {
        "NParams": thousands(s["total_params"]),
        "NProjParams": thousands(s["projection_params"]),
        "ProjRatio": br(100 * s["projection_ratio"], 1) + "\\%",
        "ZeroFrac": br(100 * np.mean(zeros), 1) + "\\%",
        "ProjKiBfp": br(s["proj_kib_fp16"], 1), "ProjKiBint": br(s["proj_kib_int8"], 1),
        "ProjKiBtwo": br(s["proj_kib_2bit"], 1), "ProjKiBrthree": br(s["proj_kib_radix3"], 1),
    }
    for tag in ("Ecg", "CueA", "CueB"):            # missing tasks stay visibly pending
        for a_ in ("Seeds", "Epochs", "Train", "Test"):
            macros[f"{a_}{tag}"] = "\\textbf{??}"
    for t in tasks:
        d = data[t]
        tag = {"ecg5000": "Ecg", "cue256": "CueA", "cue1024": "CueB"}[t]
        macros[f"Seeds{tag}"] = str(len(d["seeds"]))
        macros[f"Epochs{tag}"] = str(d["epochs"])
        macros[f"Train{tag}"] = thousands(d["n_train"])
        macros[f"Test{tag}"] = thousands(d["n_test"])
    # per task/config results: \acc{task}{id}, \solved{task}{id} (seeds with acc > 90%)
    lines = ["\\providecommand{\\acc}[2]{\\csname acc@#1@#2\\endcsname}",
             "\\providecommand{\\solved}[2]{\\csname solved@#1@#2\\endcsname}"]
    short = {"ecg5000": "ecg", "cue256": "cueA", "cue1024": "cueB"}
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


def yosys_table():
    y = load("yosys_dse.json")
    by = {r["name"]: r for r in y["results"]}
    enc = {0: "INT8 (DSP)", 1: "Ternário 2-bit", 2: "Ternário Radix-3"}
    rows = []
    for e in (0, 1, 2):
        for p in (16, 32, 64):
            r = by[f"engine_ENC{e}_P{p}"]
            rows.append(f"{enc[e] if p == 16 else ''} & {p} & {br(r['bits_per_weight'], 2)} & "
                        f"{thousands(r['luts'])} & {thousands(r['carry_cells'])} & {thousands(r['ffs'])} & "
                        f"{r['m10k']} & {r['mul9x9']} & {thousands(r['cycles_per_matvec'])} \\\\")
        if e < 2:
            rows.append("\\midrule")
    write("tab_yosys_engine.tex", "\n".join(rows) + "\n\\bottomrule\n")
    lab = {"scan_cpot_s32_g10_round": "CPoT, 32b + guarda (proposto)",
           "scan_cpot_s16_g0_trunc": "CPoT, 16b (legado, incorreto)",
           "scan_mult_q15_s32_g10": "Multiplicador Q1.15, 32b",
           "fwht16": "FWHT 16 pontos", "radix3_decoder": "Decodificador Radix-3 (1 byte)"}
    rows = []
    for n, l in lab.items():
        r = by[n]
        rows.append(f"{l} & {thousands(r['luts'])} & {thousands(r['carry_cells'])} & {thousands(r['ffs'])} & "
                    f"{r['mul9x9']} & {r['mul18x18']} & {r['mul27x27']} & {r['dsp_blocks_est']} \\\\")
    write("tab_yosys_blocks.tex", "\n".join(rows) + "\n\\bottomrule\n")
    write("macros_yosys.tex", f"\\newcommand{{\\YosysVersion}}{{{y['tool'].split('(')[0].strip()}}}\n"
                              f"\\newcommand{{\\EngK}}{{{y['engine_K']}}}\n\\newcommand{{\\EngM}}{{{y['engine_M']}}}\n")


def number_macros():
    v, r, y = load("rtl_verification.json"), load("long_range_results.json"), load("yosys_dse.json")
    by = {x["name"]: x for x in y["results"]}
    e1, e3, e4 = r["E1_grid_0.90_0.999"], r["E3_integer_impulse"], r["E4_integer_random_sqnr"]
    m = {
        "VerTests": str(len(v["tests"])), "VerTotal": thousands(v["total_checked"]),
        "VerMism": str(v["total_mismatches"]),
        "SqnrLegacy": br(e4["legacy_s16_g0_trunc"]["sqnr_mean_db"], 1),
        "SqnrProposed": br(e4["proposed_s32_g10_round"]["sqnr_mean_db"], 1),
        "SqnrMult": br(e4["mult_q15_s32_g10"]["sqnr_mean_db"], 1),
        "ImpLegacyTwelve": br(e3["legacy_s16_g0_trunc"]["12"]["h_over_ideal_at_4tau"], 1),
        "ImpLegacyEight": br(e3["legacy_s16_g0_trunc"]["8"]["h_over_ideal_at_4tau"], 2),
        "TauErrCpot": br(100 * e1["cpot"]["tau_rel_err_median"], 1) + "\\%",
        "TauErrCpotTwo": br(100 * e1["cpot2"]["tau_rel_err_median"], 1) + "\\%",
        "TauErrPot": br(100 * e1["pot"]["tau_rel_err_median"], 1) + "\\%",
        "LevelsCpot": str(e1["cpot"]["distinct_levels"]),
        "DspCpot": str(by["scan_cpot_s32_g10_round"]["dsp_blocks_est"]),
        "DspMult": str(by["scan_mult_q15_s32_g10"]["dsp_blocks_est"]),
        "LutCpot": thousands(by["scan_cpot_s32_g10_round"]["luts"]),
        "LutMult": thousands(by["scan_mult_q15_s32_g10"]["luts"]),
        "LutDec": str(by["radix3_decoder"]["luts"]),
    }
    for p_ in (16, 64):
        for e, tag in ((0, "Int"), (1, "Two"), (2, "Rthree")):
            x = by[f"engine_ENC{e}_P{p_}"]
            sfx = {16: "A", 64: "C"}[p_]
            m[f"Mtenk{tag}{sfx}"] = str(x["m10k"])
            m[f"Lut{tag}{sfx}"] = thousands(x["luts"])
            m[f"Mul{tag}{sfx}"] = str(x["mul9x9"])
    write("macros_numbers.tex", "".join(f"\\newcommand{{\\{k}}}{{{val}}}\n" for k, val in m.items()))


if __name__ == "__main__":
    number_macros()
    verification_table()
    abar_table()
    long_range_figure()
    yosys_table()
    if any(load(f"ablation_{t}.json") for t in ("ecg5000", "cue256", "cue1024")):
        ablation_table()
    else:   # placeholders so the paper still compiles; visibly marked as missing
        print("  (no ablation results yet -> placeholders)")
        keys = ["NParams", "NProjParams", "ProjRatio", "ZeroFrac", "ProjKiBfp", "ProjKiBint", "ProjKiBtwo",
                "ProjKiBrthree"] + [f"{a}{t}" for a in ("Seeds", "Epochs", "Train", "Test") for t in ("Ecg", "CueA", "CueB")]
        write("macros_ablation.tex", "".join(f"\\newcommand{{\\{k}}}{{\\textbf{{??}}}}\n" for k in keys))
        write("tab_ablation.tex", "\\begin{tabular}{c}\\toprule\n\\textbf{resultados pendentes} \\\\\n\\bottomrule\n\\end{tabular}\n")
