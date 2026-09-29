# -*- coding: utf-8 -*-
"""Dense-budget robustness outputs (spec section 8.6).

Reads the EXISTING budget sweeps (no re-run):
  results/02_budget_sweep/amini/amini_optimistic_sweep.csv       (12 seeds)
  results/02_budget_sweep/sensodat/sensodat_optimistic_sweep.csv (10 splits x 3 reps)

Writes:
  results/paper_dense_budget/dense_budget_results.csv
  results/paper_dense_budget/dense_budget_paired.csv
  results/paper_dense_budget/fig_dense_budget_sensodat.pdf
  results/paper_dense_budget/fig_dense_budget_amini.pdf

Statistics per (benchmark, budget, method): mean recall, 95% CI at the
statistical unit, paired difference vs ReproAlloc (ReproAlloc vs the
strongest external of the cell), wins/losses/ties, Wilcoxon p.

Naming (paper): ReproAlloc-Optimistic -> ReproAlloc; ReproAlloc -> Mean
Planning; ReproAlloc-OA excluded from comparison (kept in raw files).
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")
OUT = os.path.join(R, "paper_dense_budget")
os.makedirs(OUT, exist_ok=True)

RENAME = {"ReproAlloc-Optimistic": "ReproAlloc", "ReproAlloc": "Mean Planning"}
METHODS = ["Uniform", "FailRerun", "ScoreCost", "APGAI",
           "Mean Planning", "ReproAlloc"]
EXTERNAL = ["Uniform", "FailRerun", "ScoreCost", "APGAI"]
BUDGETS = [0.03, 0.05, 0.06, 0.07, 0.08, 0.10, 0.15, 0.20]


def load_units():
    units = {}
    am = pd.read_csv(os.path.join(R, "02_budget_sweep", "amini",
                                  "amini_optimistic_sweep.csv"))
    am = am[am.method != "ReproAlloc-OA"].copy()
    am["paper_method"] = am.method.map(lambda m: RENAME.get(m, m))
    for (pm, tf), g in am.groupby(["paper_method", "budget_frac"]):
        units[("amini", float(tf), pm)] = \
            g.set_index("seed")["recall_refpos"].sort_index()

    sd = pd.read_csv(os.path.join(R, "02_budget_sweep", "sensodat",
                                  "sensodat_optimistic_sweep.csv"))
    sd = sd[sd.method != "ReproAlloc-OA"].copy()
    sd["paper_method"] = sd.method.map(lambda m: RENAME.get(m, m))
    per_split = (sd.groupby(["paper_method", "time_frac", "split_id"],
                            as_index=False)["recall"].mean())
    for (pm, tf), g in per_split.groupby(["paper_method", "time_frac"]):
        units[("sensodat", float(tf), pm)] = \
            g.set_index("split_id")["recall"].sort_index()
    return units


def paired(a, b):
    common = a.index.intersection(b.index)
    d = a[common] - b[common]
    n = len(d)
    m_ = float(d.mean())
    sd_ = float(d.std(ddof=1))
    ci = 1.96 * sd_ / math.sqrt(n) if sd_ == sd_ else 0.0
    try:
        p = float(wilcoxon(d).pvalue) if (d != 0).any() else 1.0
    except Exception:
        p = float("nan")
    wins = int((d > 1e-12).sum())
    losses = int((d < -1e-12).sum())
    return m_, m_ - ci, m_ + ci, p, wins, losses, n - wins - losses


def main():
    units = load_units()
    res_rows, pair_rows = [], []
    for bench in ["sensodat", "amini"]:
        for tf in BUDGETS:
            rp = units[(bench, tf, "ReproAlloc")]
            # strongest external at this cell (by mean)
            ext_means = {m: float(units[(bench, tf, m)].mean())
                         for m in EXTERNAL}
            strongest = max(ext_means, key=ext_means.get)
            for pm in METHODS:
                s = units[(bench, tf, pm)]
                n = len(s)
                mean = float(s.mean())
                sd_ = float(s.std(ddof=1))
                ci = 1.96 * sd_ / math.sqrt(n)
                res_rows.append({"benchmark": bench, "budget": tf,
                                 "method": pm, "n_units": n,
                                 "mean_recall": mean, "sd": sd_,
                                 "ci_low": mean - ci, "ci_high": mean + ci})
                if pm == "ReproAlloc":
                    ref = units[(bench, tf, strongest)]
                    (d, lo, hi, p, w_, l_, t_) = paired(rp, ref)
                    pair_rows.append({"benchmark": bench, "budget": tf,
                                      "method": pm,
                                      "paired_reference": strongest,
                                      "paired_delta": d,
                                      "paired_ci_low": lo,
                                      "paired_ci_high": hi,
                                      "wilcoxon_p": p, "wins": w_,
                                      "losses": l_, "ties": t_})
                else:
                    (d, lo, hi, p, w_, l_, t_) = paired(s, rp)
                    pair_rows.append({"benchmark": bench, "budget": tf,
                                      "method": pm,
                                      "paired_reference": "ReproAlloc",
                                      "paired_delta": d,
                                      "paired_ci_low": lo,
                                      "paired_ci_high": hi,
                                      "wilcoxon_p": p, "wins": w_,
                                      "losses": l_, "ties": t_})
    res = pd.DataFrame(res_rows)
    prs = pd.DataFrame(pair_rows)
    res.to_csv(os.path.join(OUT, "dense_budget_results.csv"), index=False,
               float_format="%.10g")
    prs.to_csv(os.path.join(OUT, "dense_budget_paired.csv"), index=False,
               float_format="%.10g")

    # ---------------- figures (paper style, cf. fig3_main_effectiveness) --
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8.5, "axes.titlesize": 9,
        "axes.labelsize": 8.5, "legend.fontsize": 7.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.6})
    STYLE = {
        "Uniform":       {"c": "#999999", "m": "o", "ls": "--", "lw": 0.9,
                          "z": 1, "alpha": 0.65},
        "FailRerun":     {"c": "#8c564b", "m": "s", "ls": "--", "lw": 1.0,
                          "z": 1, "alpha": 1.0},
        "ScoreCost":     {"c": "#378ADD", "m": "d", "ls": "--", "lw": 1.2,
                          "z": 2, "alpha": 1.0},
        "APGAI":         {"c": "#639922", "m": "^", "ls": "--", "lw": 1.2,
                          "z": 2, "alpha": 1.0},
        "Mean Planning": {"c": "#7F77DD", "m": "v", "ls": ":",  "lw": 1.0,
                          "z": 2, "alpha": 0.85},
        "ReproAlloc":    {"c": "#D85A30", "m": "*", "ls": "-",  "lw": 1.8,
                          "z": 5, "alpha": 1.0},
    }
    TITLES = {"sensodat": "SensoDat replay", "amini": "Amini TransFuser traces"}
    for bench in ["sensodat", "amini"]:
        fig, ax = plt.subplots(figsize=(3.6, 2.6))
        sub = res[res.benchmark == bench]
        for m in ["Uniform", "FailRerun", "ScoreCost", "APGAI",
                  "Mean Planning", "ReproAlloc"]:
            g = sub[sub.method == m].sort_values("budget")
            st = STYLE[m]
            ci = np.vstack([g.mean_recall - g.ci_low,
                            g.ci_high - g.mean_recall])
            ax.errorbar(g.budget * 100, g.mean_recall, yerr=ci, label=m,
                        color=st["c"], marker=st["m"], ls=st["ls"],
                        lw=st["lw"], ms=4.0 if m != "ReproAlloc" else 6.5,
                        capsize=1.4, zorder=st["z"], alpha=st["alpha"])
        ax.set_xticks([3, 5, 6, 7, 8, 10, 15, 20])
        ax.set_xticklabels(["3", "5", "6", "7", "8", "10", "15", "20%"])
        ax.set_xlabel("budget (% of total recorded execution time)")
        ax.set_ylabel("confirmed test recall")
        ax.set_title(TITLES[bench])
        ax.grid(alpha=0.25, lw=0.4)
        ax.legend(frameon=False, loc="upper left", fontsize=6.8)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"fig_dense_budget_{bench}.pdf"))
        plt.close(fig)

    # verification printout
    pd.set_option("display.width", 220)
    for bench in ["sensodat", "amini"]:
        print(f"\n=== {bench} (dense sweep) ===")
        sub = res[res.benchmark == bench]
        for tf in BUDGETS:
            row_rp = sub[(sub.budget == tf) & (sub.method == "ReproAlloc")].iloc[0]
            ext = {m: sub[(sub.budget == tf) & (sub.method == m)].iloc[0].mean_recall
                   for m in EXTERNAL}
            bs = max(ext, key=ext.get)
            pr = prs[(prs.benchmark == bench) & (prs.budget == tf)
                     & (prs.method == "ReproAlloc")].iloc[0]
            flag = "BEST" if row_rp.mean_recall >= ext[bs] - 1e-12 else "below"
            print(f" {tf*100:>4.0f}%  ReproAlloc {row_rp.mean_recall:.4f}  "
                  f"strongest-ext {bs} {ext[bs]:.4f}  [{flag}]  "
                  f"paired d={pr.paired_delta:+.4f} "
                  f"CI[{pr.paired_ci_low:+.4f},{pr.paired_ci_high:+.4f}] "
                  f"p={pr.wilcoxon_p:.4f} W/L/T={pr.wins}/{pr.losses}/{pr.ties}")
    print(f"\nwrote {OUT} ({len(res)} + {len(prs)} rows, 2 PDFs)")


if __name__ == "__main__":
    main()
