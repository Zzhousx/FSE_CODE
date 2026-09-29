"""Render the SensoDat figures for the final report.

  results/diagnostics/fig_sensodat_budget_sweep.png  - recall vs budget (7 methods)
  results/diagnostics/fig_sensodat_anytime.png       - anytime recall(t) curves

Style mirrors make_amini_figures.py.  Nested aggregation convention matches
analyze_sensodat_optimistic.py: average reps within (split, budget, method)
first, then mean/sd over the 10 splits.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
NEW = os.path.dirname(HERE)
DIAG = os.path.join(NEW, "results", "diagnostics")
SWEEP = os.path.join(NEW, "results", "02_budget_sweep", "sensodat",
                     "sensodat_optimistic_sweep.csv")

ORDER = ["Uniform", "FailRerun", "APGAI", "ScoreCost",
         "ReproAlloc", "ReproAlloc-OA", "ReproAlloc-Optimistic"]
STYLE = {
    "Uniform": {"c": "#999999", "m": "o", "ls": "--", "lw": 1.2, "z": 1},
    "FailRerun": {"c": "#8c564b", "m": "s", "ls": "--", "lw": 1.2, "z": 1},
    "APGAI": {"c": "#2ca02c", "m": "^", "ls": "--", "lw": 1.4, "z": 2},
    "ScoreCost": {"c": "#1f77b4", "m": "d", "ls": "--", "lw": 1.4, "z": 2},
    "ReproAlloc": {"c": "#9467bd", "m": "v", "ls": "-", "lw": 1.8, "z": 3},
    "ReproAlloc-OA": {"c": "#c5b0d5", "m": "v", "ls": ":", "lw": 1.4, "z": 3},
    "ReproAlloc-Optimistic": {"c": "#d62728", "m": "*", "ls": "-", "lw": 2.6,
                              "z": 5},
}

# ---- sweep figure ------------------------------------------------------
df = pd.read_csv(SWEEP)
# nested: average reps within split first, then mean/sd over splits
per_split = (df.groupby(["method", "time_frac", "split_id"], as_index=False)
             ["recall"].mean())
g = per_split.groupby(["method", "time_frac"])["recall"].agg(["mean", "std"])
fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
for m in ORDER:
    if m not in g.index.get_level_values(0):
        continue
    s = g.loc[m].sort_index()
    st = STYLE[m]
    ax.errorbar(s.index * 100, s["mean"], yerr=s["std"], label=m,
                color=st["c"], marker=st["m"], ls=st["ls"], lw=st["lw"],
                ms=6 if m != "ReproAlloc-Optimistic" else 10,
                capsize=2, zorder=st["z"])
ax.set_xlabel("budget (% of total recorded duration)")
ax.set_ylabel("recall (confirmed true failures / all target failures)")
ax.set_title("SensoDat budget sweep (10 splits x 3 reps, mean +/- sd over splits)")
ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(DIAG, "fig_sensodat_budget_sweep.png"))
plt.close(fig)

# ---- anytime figure ----------------------------------------------------
c = pd.read_csv(os.path.join(DIAG, "sensodat_anytime_curve.csv"))
fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
for m in ["FailRerun", "ScoreCost", "ReproAlloc", "ReproAlloc-Optimistic"]:
    s = c[c.method == m]
    st = STYLE[m]
    ax.plot(s.time_frac * 100, s.recall_mean, label=m, color=st["c"],
            ls=st["ls"], lw=st["lw"], zorder=st["z"])
    ax.fill_between(s.time_frac * 100, s.recall_lo, s.recall_hi,
                    color=st["c"], alpha=0.12, zorder=st["z"] - 1)
for tf in [5, 10, 20]:
    ax.axvline(tf, color="#cccccc", lw=0.8, zorder=0)
ax.set_xlabel("wall-clock budget consumed (% of total recorded duration)")
ax.set_ylabel("recall (confirmed true failures / all target failures)")
ax.set_title("SensoDat anytime curves (mean over 10 splits, 95% band)")
ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(DIAG, "fig_sensodat_anytime.png"))
plt.close(fig)
print("wrote fig_sensodat_budget_sweep.png and fig_sensodat_anytime.png")
