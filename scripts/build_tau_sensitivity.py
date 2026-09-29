# -*- coding: utf-8 -*-
"""Tau sensitivity analysis (spec section 10.5).

Combines the frozen main runs (tau = 0.3) with the tau-sensitivity runs
(tau = 0.2, 0.4) and produces:
  results/tau_sensitivity/tau_sensitivity_raw.csv
  results/tau_sensitivity/tau_sensitivity_summary.csv
  results/tau_sensitivity/tau_sensitivity_paired.csv
  results/tau_sensitivity/fig_tau_sensitivity_sensodat.pdf
  results/tau_sensitivity/fig_tau_sensitivity_amini.pdf

Amini note: the frozen harness couples the reference-positive set to tau
(p_hat > tau): 26 / 23 / 20 reference positives at tau = 0.2 / 0.3 / 0.4.
SensoDat's reference set (observed failures) is tau-free.

Naming: ReproAlloc-Optimistic -> ReproAlloc; ReproAlloc -> Mean Planning;
ReproAlloc-OA excluded. Datasets with missing runs are skipped gracefully.
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
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "tau_sensitivity")
os.makedirs(OUT, exist_ok=True)

RENAME = {"ReproAlloc-Optimistic": "ReproAlloc", "ReproAlloc": "Mean Planning"}
METHODS = ["Uniform", "FailRerun", "ScoreCost", "APGAI",
           "Mean Planning", "ReproAlloc"]
EXTERNAL = ["Uniform", "FailRerun", "ScoreCost", "APGAI"]
TAUS = [0.2, 0.3, 0.4]
BUDGETS = [0.05, 0.10, 0.20]


def load():
    frames = []
    am_dir = os.path.join(RES, "01_optimistic_core", "amini")
    am_tau = os.path.join(RES, "tau_sensitivity", "amini")
    for tau, path in [(0.3, os.path.join(am_dir, "amini_optimistic_main.csv")),
                      (0.2, os.path.join(am_tau,
                                         "amini_tau_sensitivity_tau0.20.csv")),
                      (0.4, os.path.join(am_tau,
                                         "amini_tau_sensitivity_tau0.40.csv"))]:
        if not os.path.exists(path):
            print(f"  [skip] missing {path}")
            continue
        d = pd.read_csv(path)
        d = d[d.method != "ReproAlloc-OA"].copy()
        d["tau"] = tau
        d["benchmark"] = "amini"
        d = d.rename(columns={"budget_frac": "budget",
                              "recall_refpos": "recall"})
        d["unit"] = d.seed
        frames.append(d[["benchmark", "tau", "budget", "unit", "method",
                         "recall", "executions", "n_refpos"]])

    sd_dir = os.path.join(RES, "01_optimistic_core", "sensodat")
    sd_tau = os.path.join(RES, "tau_sensitivity", "sensodat")
    for tau, path in [(0.3, os.path.join(sd_dir,
                                         "sensodat_optimistic_main.csv")),
                      (0.2, os.path.join(sd_tau,
                                         "sensodat_tau_sensitivity_tau0.20.csv")),
                      (0.4, os.path.join(sd_tau,
                                         "sensodat_tau_sensitivity_tau0.40.csv"))]:
        if not os.path.exists(path):
            print(f"  [skip] missing {path}")
            continue
        d = pd.read_csv(path)
        d = d[d.method != "ReproAlloc-OA"].copy()
        d["tau"] = tau
        d["benchmark"] = "sensodat"
        d = d.rename(columns={"time_frac": "budget"})
        # nested: average reps within (split, tau, budget, method)
        d = (d.groupby(["benchmark", "tau", "budget", "split_id", "method"],
                       as_index=False)["recall"].mean())
        d["unit"] = d.split_id
        d["n_refpos"] = -1
        frames.append(d[["benchmark", "tau", "budget", "unit", "method",
                         "recall", "n_refpos"]])
    if not frames:
        raise SystemExit("no tau data found")
    raw = pd.concat(frames, ignore_index=True)
    raw["paper_method"] = raw.method.map(lambda m: RENAME.get(m, m))
    return raw


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
    losses = int(d < -1e-12) if False else int((d < -1e-12).sum())
    return m_, m_ - ci, m_ + ci, p, wins, losses, n - wins - losses


def main():
    raw = load()
    raw.to_csv(os.path.join(OUT, "tau_sensitivity_raw.csv"), index=False,
               float_format="%.10g")

    sum_rows, pair_rows = [], []
    for (bench, tau, tf), g in raw.groupby(["benchmark", "tau", "budget"]):
        units = {pm: gg.set_index("unit")["recall"].sort_index()
                 for pm, gg in g.groupby("paper_method")}
        rp = units["ReproAlloc"]
        ext_means = {m: float(units[m].mean()) for m in EXTERNAL
                     if m in units}
        strongest = max(ext_means, key=ext_means.get)
        n_ref = int(g.n_refpos.max())
        for pm, s in units.items():
            n = len(s)
            mean = float(s.mean())
            sd_ = float(s.std(ddof=1))
            ci = 1.96 * sd_ / math.sqrt(n)
            sum_rows.append({"benchmark": bench, "tau": tau, "budget": tf,
                             "method": pm, "n_units": n, "n_refpos": n_ref,
                             "mean_recall": mean, "sd": sd_,
                             "ci_low": mean - ci, "ci_high": mean + ci,
                             "strongest_external": strongest,
                             "is_best_mean": bool(mean >= max(
                                 ext_means.values()) - 1e-12)})
            if pm == "ReproAlloc":
                (d, lo, hi, p, w_, l_, t_) = paired(rp, units[strongest])
                ref = strongest
            else:
                (d, lo, hi, p, w_, l_, t_) = paired(s, rp)
                ref = "ReproAlloc"
            pair_rows.append({"benchmark": bench, "tau": tau, "budget": tf,
                              "method": pm, "paired_reference": ref,
                              "paired_delta": d, "paired_ci_low": lo,
                              "paired_ci_high": hi, "wilcoxon_p": p,
                              "wins": w_, "losses": l_, "ties": t_})
    summ = pd.DataFrame(sum_rows)
    prs = pd.DataFrame(pair_rows)
    summ.to_csv(os.path.join(OUT, "tau_sensitivity_summary.csv"),
                index=False, float_format="%.10g")
    prs.to_csv(os.path.join(OUT, "tau_sensitivity_paired.csv"),
               index=False, float_format="%.10g")

    # ---------------- figures: 3 tau panels per benchmark -------------------
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8.5, "axes.titlesize": 9,
        "axes.labelsize": 8.5, "legend.fontsize": 7.0,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.6})
    STYLE = {
        "Uniform":       {"c": "#999999", "m": "o", "ls": "--", "lw": 0.9},
        "FailRerun":     {"c": "#8c564b", "m": "s", "ls": "--", "lw": 1.0},
        "ScoreCost":     {"c": "#378ADD", "m": "d", "ls": "--", "lw": 1.2},
        "APGAI":         {"c": "#639922", "m": "^", "ls": "--", "lw": 1.2},
        "Mean Planning": {"c": "#7F77DD", "m": "v", "ls": ":",  "lw": 1.0},
        "ReproAlloc":    {"c": "#D85A30", "m": "*", "ls": "-",  "lw": 1.8},
    }
    TITLES = {"sensodat": "SensoDat replay", "amini": "Amini TransFuser traces"}
    for bench in summ.benchmark.unique():
        sub = summ[summ.benchmark == bench]
        taus = sorted(sub.tau.unique())
        fig, axes = plt.subplots(1, len(taus), figsize=(6.8, 2.4),
                                 sharey=True)
        if len(taus) == 1:
            axes = [axes]
        for ax, tau in zip(axes, taus):
            ss = sub[sub.tau == tau]
            for pm in METHODS:
                g = ss[ss.method == pm].sort_values("budget")
                if not len(g):
                    continue
                st = STYLE[pm]
                ci = np.vstack([g.mean_recall - g.ci_low,
                                g.ci_high - g.mean_recall])
                ax.errorbar(g.budget * 100, g.mean_recall, yerr=ci,
                            label=pm, color=st["c"], marker=st["m"],
                            ls=st["ls"], lw=st["lw"],
                            ms=4.0 if pm != "ReproAlloc" else 6.5,
                            capsize=1.4,
                            zorder=5 if pm == "ReproAlloc" else 2)
            ax.set_title(rf"$\tau={tau}$")
            ax.set_xticks([5, 10, 20])
            ax.set_xticklabels(["5%", "10%", "20%"])
            ax.grid(alpha=0.25, lw=0.4)
            if ax is axes[0]:
                ax.set_ylabel("confirmed test recall")
        axes[0].set_xlabel("budget")
        for ax in axes[1:]:
            ax.set_xlabel("budget")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, loc="upper center",
                   ncol=6, bbox_to_anchor=(0.5, 1.10))
        fig.suptitle(TITLES.get(bench, bench), y=0.99, fontsize=9)
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        fig.savefig(os.path.join(OUT, f"fig_tau_sensitivity_{bench}.pdf"))
        plt.close(fig)

    # -------- console summary: ordering stability check (spec 10.6) --------
    pd.set_option("display.width", 240)
    print("\n=== ReproAlloc vs strongest external per (benchmark, tau, budget)")
    for (bench, tau, tf), g in summ.groupby(["benchmark", "tau", "budget"]):
        rp = g[g.method == "ReproAlloc"].iloc[0]
        pr = prs[(prs.benchmark == bench) & (prs.tau == tau)
                 & (prs.budget == tf) & (prs.method == "ReproAlloc")].iloc[0]
        print(f"{bench:9s} tau={tau} b={tf*100:>2.0f}%  "
              f"RA {rp.mean_recall:.4f} vs {rp.strongest_external}  "
              f"best={rp.is_best_mean}  d={pr.paired_delta:+.4f} "
              f"CI[{pr.paired_ci_low:+.4f},{pr.paired_ci_high:+.4f}] "
              f"p={pr.wilcoxon_p:.4f}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
