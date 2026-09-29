# -*- coding: utf-8 -*-
"""Figure 4 (paper `fig:cases`): two representative Amini allocation traces
side by side.

  (a) Rescued confirmation : RouteScenario_33, seed 2004
  (b) Shared confirmation  : RouteScenario_8,  seed 2002

Both panels use one identical visual specification (same y-axis method order,
same markers: x for failures, o for passes, star for confirmation, same fonts)
and share one x-range (0 to 60 global cumulative wall-clock hours) and one
legend. Failing executions carry no text label; the cross marker alone encodes
them. Panel (a) carries a continuation note: no further OptimisticScoreCost
execution through the 108.64 h budget end. The 20% budget line is NOT drawn
(it lies outside the shown range).

Reads : results/engineering_value_check/route_case_{A,B}_timeline.csv
        (produced by scripts/build_engineering_value.py; no re-run needed)
Writes: figures/fig4_allocation_traces.pdf
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EV = os.path.join(ROOT, "results", "engineering_value_check")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
OUT = os.path.join(FIG, "fig4_allocation_traces.pdf")

PANELS = [("a", "RouteScenario_33", 2004, "route_case_A_timeline.csv"),
          ("b", "RouteScenario_8", 2002, "route_case_B_timeline.csv")]
XMIN, XMAX = 0.0, 60.0
BUDGET_H = 108.64  # outside the shown range; used only in the panel-(a) note

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.titlesize": 9,
    "axes.labelsize": 8.5, "legend.fontsize": 7.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.6})

fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.5), sharey=True)
for ax, (tag, rid, sdx, fname) in zip(axes, PANELS):
    sub = pd.read_csv(os.path.join(EV, fname))
    sub = sub.sort_values(["method", "exec_idx"])
    ymap = {"ReproAlloc": 1, "OptimisticScoreCost": 0}
    for meth in ["ReproAlloc", "OptimisticScoreCost"]:
        g = sub[sub.method == meth]
        for r in g.itertuples(index=False):
            y = ymap[meth]
            is_conf = bool(r.confirmed_after)
            is_fail = r.outcome == "fail"
            ax.scatter(r.cum_time_h, y,
                       marker=("*" if is_conf else
                               ("x" if is_fail else "o")),
                       s=(140 if is_conf else 45),
                       color=("#d62728" if is_fail else "#1f77b4"),
                       zorder=3)
            if not is_fail:
                ax.annotate("P", (r.cum_time_h, y),
                            textcoords="offset points", xytext=(0, 7),
                            ha="center", fontsize=7, color="#1f77b4")
    if tag == "a":
        ax.annotate("No further OptimisticScoreCost\n"
                    "execution before budget end\n"
                    f"({BUDGET_H:.2f} h) $\\rightarrow$",
                    xy=(59.0, 1.38), ha="right", va="center", fontsize=7,
                    color="#555555")
    ax.set_xlim(XMIN, XMAX)
    ax.set_xticks(range(0, 61, 10))
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["OptimisticScoreCost", "ReproAlloc"])
    ax.set_ylim(-0.6, 1.75)
    ax.set_xlabel("global cumulative wall-clock hours (this seed)")
    ax.set_title(f"({tag}) {rid}, seed {sdx}")
    ax.grid(axis="x", alpha=0.3)

handles = [
    Line2D([0], [0], marker="x", color="#d62728", ls="none", ms=6,
           label="failing execution"),
    Line2D([0], [0], marker="*", color="#d62728", ls="none", ms=11,
           label="confirmation"),
]
fig.tight_layout(rect=[0, 0, 1, 0.87])
fig.legend(handles=handles, frameon=False, loc="upper center", ncol=2,
           bbox_to_anchor=(0.5, 1.0), columnspacing=1.4, handletextpad=0.5)
fig.savefig(OUT)
plt.close(fig)
print("wrote", OUT)
