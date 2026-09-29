# -*- coding: utf-8 -*-
"""Figure 3 (paper `fig:dense`): confirmed-test recall across eight budgets
(3%-20%).

Two panels sharing one legend:
  (a) SensoDat-replay
  (b) Amini-TransFuser

External schedulers + ReproAlloc only (Mean Planning and OptimisticScoreCost
are internal component controls and appear in the RQ3 component table, not in
this effectiveness figure). Light vertical guides mark the primary operating
points 5/10/20.

Reads : results/paper_dense_budget/dense_budget_results.csv
        (produced by scripts/build_dense_budget.py; no re-run needed)
Writes: figures/fig3_budget_effectiveness.pdf
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "paper_dense_budget",
                   "dense_budget_results.csv")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
OUT = os.path.join(FIG, "fig3_budget_effectiveness.pdf")

res = pd.read_csv(RES)

METHODS = ["Uniform", "FailRerun", "ScoreCost", "APGAI", "ReproAlloc"]
STYLE = {
    "Uniform":    {"c": "#999999", "m": "o", "ls": "--", "lw": 0.9,
                   "z": 1, "alpha": 0.65},
    "FailRerun":  {"c": "#8c564b", "m": "s", "ls": "--", "lw": 1.0,
                   "z": 1, "alpha": 1.0},
    "ScoreCost":  {"c": "#378ADD", "m": "d", "ls": "--", "lw": 1.2,
                   "z": 2, "alpha": 1.0},
    "APGAI":      {"c": "#639922", "m": "^", "ls": "--", "lw": 1.2,
                   "z": 2, "alpha": 1.0},
    "ReproAlloc": {"c": "#D85A30", "m": "*", "ls": "-",  "lw": 1.8,
                   "z": 5, "alpha": 1.0},
}
PANELS = [("sensodat", "(a) SensoDat-replay"),
          ("amini", "(b) Amini-TransFuser")]
PRIMARY = [5, 10, 20]

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.titlesize": 9,
    "axes.labelsize": 8.5, "legend.fontsize": 7.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.6})

fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.55), sharey=True)
handles, labels = None, None
for ax, (bench, title) in zip(axes, PANELS):
    sub = res[res.benchmark == bench]
    for m in METHODS:
        g = sub[sub.method == m].sort_values("budget")
        st = STYLE[m]
        ci = np.vstack([g.mean_recall - g.ci_low,
                        g.ci_high - g.mean_recall])
        ax.errorbar(g.budget * 100, g.mean_recall, yerr=ci, label=m,
                    color=st["c"], marker=st["m"], ls=st["ls"],
                    lw=st["lw"], ms=4.0 if m != "ReproAlloc" else 6.5,
                    capsize=1.4, zorder=st["z"], alpha=st["alpha"])
    for xp in PRIMARY:
        ax.axvline(xp, color="gray", lw=0.7, alpha=0.18, zorder=0)
    ax.set_xticks([3, 5, 6, 7, 8, 10, 15, 20])
    ax.set_xticklabels(["3", "5", "6", "7", "8", "10", "15", "20%"])
    ax.set_xlabel("budget (% of total recorded execution time)")
    ax.set_title(title)
    ax.grid(alpha=0.25, lw=0.4)
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
axes[0].set_ylabel("confirmed-test recall")
fig.tight_layout(rect=[0, 0, 1, 0.86])
fig.legend(handles, labels, frameon=False, loc="upper center",
           ncol=len(METHODS), bbox_to_anchor=(0.5, 1.0),
           columnspacing=1.1, handletextpad=0.5)
fig.savefig(OUT)
plt.close(fig)
print("wrote", OUT)
