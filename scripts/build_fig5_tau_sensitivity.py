# -*- coding: utf-8 -*-
"""Figure 5 (paper `fig:tau`): confirmation-threshold sensitivity,
2 rows x 3 cols.

Each mini-panel plots confirmed-test recall against tau for ReproAlloc and
the best external scheduler of that benchmark-budget-threshold cell (means
with 95% intervals). Rows use the unified benchmark names
SensoDat-replay / Amini-TransFuser.

Reads : results/tau_sensitivity/tau_sensitivity_summary.csv
        (produced by scripts/build_tau_sensitivity.py; no re-run needed)
Writes: figures/fig5_tau_sensitivity.pdf
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUM = os.path.join(ROOT, "results", "tau_sensitivity",
                   "tau_sensitivity_summary.csv")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
OUT = os.path.join(FIG, "fig5_tau_sensitivity.pdf")

df = pd.read_csv(SUM)
BENCH = [("sensodat", "SensoDat-replay"), ("amini", "Amini-TransFuser")]
BUDGETS = [0.05, 0.10, 0.20]
TAUS = [0.2, 0.3, 0.4]
EXT = ["Uniform", "FailRerun", "ScoreCost", "APGAI"]

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.0, "axes.titlesize": 8.5,
    "axes.labelsize": 8.0, "legend.fontsize": 7.0, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "axes.linewidth": 0.6})

fig, axes = plt.subplots(2, 3, figsize=(6.8, 3.65), sharex=True)
for r, (bk, btitle) in enumerate(BENCH):
    for c, bud in enumerate(BUDGETS):
        ax = axes[r, c]
        sub = df[(df.benchmark == bk) & (np.isclose(df.budget, bud))]
        for tau in TAUS:
            st = sub[np.isclose(sub.tau, tau)]
            ext = st[st.method.isin(EXT)]
            strongest = ext.loc[ext.mean_recall.idxmax(), "method"]
            ra = st[st.method == "ReproAlloc"].iloc[0]
            ex = ext[ext.method == strongest].iloc[0]
            ax.errorbar([tau - 0.006], [ra.mean_recall],
                        yerr=[[ra.mean_recall - ra.ci_low],
                              [ra.ci_high - ra.mean_recall]],
                        color="#D85A30", marker="*", ms=7, ls="none",
                        capsize=1.6, zorder=5,
                        label="ReproAlloc" if (r == 0 and c == 0
                                               and tau == TAUS[0]) else None)
            ax.errorbar([tau + 0.006], [ex.mean_recall],
                        yerr=[[ex.mean_recall - ex.ci_low],
                              [ex.ci_high - ex.mean_recall]],
                        color="#378ADD", marker="d", ms=4, ls="none",
                        capsize=1.6, zorder=4,
                        label="best external in each cell" if (r == 0 and c == 0
                                                             and tau == TAUS[0])
                        else None)
        ax.set_xticks(TAUS)
        ax.set_xticklabels(["0.2", "0.3", "0.4"])
        ax.grid(alpha=0.25, lw=0.4)
        if r == 0:
            ax.set_title(f"budget {int(bud*100)}%")
        if c == 0:
            ax.set_ylabel(f"{btitle}\nconfirmed-test recall")
        if r == 1:
            ax.set_xlabel(r"confirmation threshold $\tau$")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.tight_layout(rect=[0, 0.08, 1, 1])
fig.legend(handles, labels, frameon=False, loc="lower center", ncol=2,
           bbox_to_anchor=(0.5, 0.005))
fig.savefig(OUT)
plt.close(fig)
print("wrote", OUT)
