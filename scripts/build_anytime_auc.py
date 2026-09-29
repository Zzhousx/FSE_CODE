# -*- coding: utf-8 -*-
"""Anytime behaviour: AUC + paper-style curves (spec section 11).

Reuses the EXISTING anytime traces (no re-run):
  results/diagnostics/amini_anytime_traces.csv    (12 seeds x 4 methods)
  results/diagnostics/sensodat_anytime_traces.csv (10 splits x 4 methods)
Both harnesses were run at a 20% budget; time_frac is the consumed fraction
of total recorded execution time, so each trace spans 0 -> ~0.20.

AUC is the trapezoidal area under the run's recall-vs-time-fraction step
curve, normalized by the interval width (i.e. the mean recall held over the
interval). Computed per run over [0, 0.20] and [0.05, 0.20]; aggregated at
the statistical unit (seed / split) with 95% CI, paired difference vs
ReproAlloc, Wilcoxon p. AUC is a SECONDARY metric only (spec 11.4).

Naming: ReproAlloc-Optimistic -> ReproAlloc; ReproAlloc -> Mean Planning.

Outputs:
  results/anytime_auc/anytime_auc.csv
  results/anytime_auc/fig_anytime_sensodat.pdf
  results/anytime_auc/fig_anytime_amini.pdf
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
OUT = os.path.join(RES, "anytime_auc")
os.makedirs(OUT, exist_ok=True)

RENAME = {"ReproAlloc-Optimistic": "ReproAlloc", "ReproAlloc": "Mean Planning"}
B_END = 0.20
B_LO = 0.05


def run_auc(t, r, lo, hi):
    """Area under the step curve (t, r) restricted to [lo, hi], normalized
    by width. t, r are the trace points (recall after each execution);
    recall is a left-continuous step function starting at 0."""
    t = np.asarray(t, float)
    r = np.asarray(r, float)
    # build step samples: recall at time x = r at the last exec <= x (0 if none)
    tt = np.concatenate([[0.0], t, [B_END]])
    rr = np.concatenate([[0.0], r, [r[-1] if len(r) else 0.0]])
    # restrict to [lo, hi]
    xs = tt[(tt >= lo) & (tt <= hi)]
    ys = rr[(tt >= lo) & (tt <= hi)]
    for edge in (lo, hi):
        if edge not in xs:
            idx = np.searchsorted(tt, edge, side="right") - 1
            xs = np.append(xs, edge)
            ys = np.append(ys, rr[idx])
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    if xs[-1] - xs[0] <= 0:
        return 0.0
    return float(np.trapezoid(ys, xs) / (xs[-1] - xs[0]))


def analyze(bench, path, idcol, rcol):
    df = pd.read_csv(path)
    df["paper_method"] = df.method.map(lambda m: RENAME.get(m, m))
    rows = []
    for (uid, pm), g in df.groupby([idcol, "paper_method"]):
        g = g.sort_values("time_frac")
        rows.append({"unit": uid, "method": pm,
                     "auc_0_20": run_auc(g.time_frac.values, g[rcol].values,
                                         0.0, B_END),
                     "auc_5_20": run_auc(g.time_frac.values, g[rcol].values,
                                         B_LO, B_END)})
    auc = pd.DataFrame(rows)

    out = []
    methods = sorted(auc.method.unique())
    rp = auc[auc.method == "ReproAlloc"].set_index("unit")
    for metric in ["auc_0_20", "auc_5_20"]:
        for pm in methods:
            s = auc[auc.method == pm].set_index("unit")[metric]
            n = len(s)
            mean = float(s.mean())
            sd_ = float(s.std(ddof=1))
            ci = 1.96 * sd_ / math.sqrt(n)
            common = s.index.intersection(rp.index)
            d = rp.loc[common, metric] - s[common]
            try:
                p = float(wilcoxon(d).pvalue) if (d != 0).any() else 1.0
            except Exception:
                p = float("nan")
            out.append({"benchmark": bench, "metric": metric, "method": pm,
                        "n_units": n, "mean_auc": mean,
                        "ci_low": mean - ci, "ci_high": mean + ci,
                        "paired_delta_ReproAlloc_minus_method": float(d.mean()),
                        "wilcoxon_p": (p if pm != "ReproAlloc" else float("nan"))})
    return pd.DataFrame(out)


def main():
    res = pd.concat([
        analyze("amini", os.path.join(RES, "diagnostics",
                                      "amini_anytime_traces.csv"),
                "seed", "recall_refpos"),
        analyze("sensodat", os.path.join(RES, "diagnostics",
                                         "sensodat_anytime_traces.csv"),
                "split_id", "recall"),
    ], ignore_index=True)
    res.to_csv(os.path.join(OUT, "anytime_auc.csv"), index=False,
               float_format="%.10g")

    # -------- paper-style anytime curves from the existing curve CSVs ------
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8.5, "axes.titlesize": 9,
        "axes.labelsize": 8.5, "legend.fontsize": 7.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.6})
    STYLE = {
        "FailRerun":     {"c": "#8c564b", "ls": "--", "lw": 1.0},
        "ScoreCost":     {"c": "#378ADD", "ls": "--", "lw": 1.2},
        "APGAI":         {"c": "#639922", "ls": "--", "lw": 1.2},
        "Mean Planning": {"c": "#7F77DD", "ls": ":",  "lw": 1.0},
        "ReproAlloc":    {"c": "#D85A30", "ls": "-",  "lw": 1.8},
    }
    for bench, title in [("sensodat", "SensoDat replay"),
                         ("amini", "Amini TransFuser traces")]:
        curve = pd.read_csv(os.path.join(RES, "diagnostics",
                                         f"{bench}_anytime_curve.csv"))
        curve["paper_method"] = curve.method.map(
            lambda m: RENAME.get(m, m))
        fig, ax = plt.subplots(figsize=(3.6, 2.6))
        for pm, g in curve.groupby("paper_method"):
            g = g.sort_values("time_frac")
            st = STYLE[pm]
            ax.plot(g.time_frac * 100, g.recall_mean, label=pm,
                    color=st["c"], ls=st["ls"], lw=st["lw"],
                    zorder=5 if pm == "ReproAlloc" else 2)
            ax.fill_between(g.time_frac * 100, g.recall_lo, g.recall_hi,
                            color=st["c"], alpha=0.12, lw=0)
        ax.set_xlabel("consumed budget (% of total recorded execution time)")
        ax.set_ylabel("confirmed test recall")
        ax.set_title(title)
        ax.grid(alpha=0.25, lw=0.4)
        ax.legend(frameon=False, loc="upper left", fontsize=6.8)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"fig_anytime_{bench}.pdf"))
        plt.close(fig)

    pd.set_option("display.width", 220)
    print(res.to_string(index=False))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
