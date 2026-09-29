# -*- coding: utf-8 -*-
"""Figure 2 (paper `fig:mechanism`): controlled concentration analysis.

The confirmation evidence is FIXED at (s,n)=(0,0), tau=0.3, delta=0.05. Only
the concentration c of the symmetric Beta(c,c) scheduling distribution varies;
its mean is 0.5 for every c.

  (a) planning rates: q_mean stays at 0.5 (flat), q_O approaches it as the
      distribution concentrates;
  (b) planned remaining executions: m* under q_mean stays at 176 (flat),
      m* under q_O approaches it.

All values are computed with the real implementation in src/:
  reproalloc_v6.est_remaining_execs  - the m* search used by the selectors
  optimistic_ercc.posterior_mean / posterior_upper_quantile

Sanity checks assert q_mean == 0.5 and m*_mean constant across the sweep.

Reads : nothing (pure computation from src/)
Writes: results/mechanism_concentration_sweep.csv
        figures/fig2_optimistic_planning_mechanism.pdf
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import reproalloc_v6 as v6  # noqa: E402
import optimistic_ercc as oe  # noqa: E402

RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

TAU = 0.3
DELTA = 0.05
M_CAP = 10000  # numerical safety cap for the illustration (real runs derive
               # m_cap from the budget); every m* below is far smaller.

# ------------------------------------------------- concentration sweep (fig 2)
# Controlled setup: confirmation evidence fixed at (s,n)=(0,0); only the
# concentration c of Beta(c,c) changes. q_mean is 0.5 for every c.
cs = np.unique(np.concatenate([
    np.arange(1, 21), np.arange(22, 52, 2), np.arange(55, 205, 5)]))
qm, qo, qa, mm, mo = [], [], [], [], []
for c in cs:
    a = b = float(c)
    q_mean = oe.posterior_mean(a, b)
    q_o = oe.posterior_upper_quantile(a, b, DELTA)
    qm.append(q_mean)
    qo.append(q_o)
    qa.append(q_o - q_mean)
    mm.append(v6.est_remaining_execs(0, 0, q_mean, TAU, M_CAP, DELTA))
    mo.append(v6.est_remaining_execs(0, 0, q_o, TAU, M_CAP, DELTA))

# sanity checks: the sweep must be concentration-only
assert all(abs(oe.posterior_mean(float(c), float(c)) - 0.5) < 1e-15
           for c in cs), "q_mean is not 0.5 for every c"
assert len(set(mm)) == 1, f"m*_mean is not constant across the sweep: {sorted(set(mm))}"

sweep = pd.DataFrame({"concentration_c": cs, "total_pseudo_count": 2 * cs,
                      "q_mean": qm, "q_upper": qo,
                      "q_upper_minus_mean": qa,
                      "m_star_mean": mm, "m_star_orcc": mo})
sweep["ratio_mean_over_orcc"] = sweep.m_star_mean / sweep.m_star_orcc
sweep.to_csv(os.path.join(RES, "mechanism_concentration_sweep.csv"),
             index=False)
print("controlled sweep: q_mean==0.5 for all c; m*_mean==%d for all c"
      % mm[0])
for c0 in (1, 50):
    row = sweep[sweep.concentration_c == c0].iloc[0]
    print("  c=%3d: gap=%.4f  m*_mean=%d  m*_orcc=%d"
          % (c0, row.q_upper_minus_mean, row.m_star_mean, row.m_star_orcc))

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.titlesize": 9,
    "axes.labelsize": 8.5, "legend.fontsize": 7.5, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.linewidth": 0.6,
})
C_MEAN = "#7F77DD"
C_ORCC = "#D85A30"

fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5))
ax = axes[0]
ax.plot(sweep.total_pseudo_count, sweep.q_mean, color=C_MEAN, lw=1.6,
        label=r"$q_i^{\mathrm{mean}}$")
ax.plot(sweep.total_pseudo_count, sweep.q_upper, color=C_ORCC, lw=1.6,
        ls="--", label=r"$q_i^{O}$")
ax.set_xscale("log")
ax.set_xlabel(r"concentration $a+b$ of $\mathrm{Beta}(c,c)$ (log scale)")
ax.set_ylabel("planning rate")
ax.set_title("(a) planning rates at fixed mean 0.5")
ax.grid(alpha=0.25, lw=0.4)
ax.legend(frameon=False, loc="upper right")

ax = axes[1]
ax.plot(sweep.total_pseudo_count, sweep.m_star_mean, color=C_MEAN, lw=1.6,
        label=r"$m^{*}$ under $q^{\mathrm{mean}}$")
ax.plot(sweep.total_pseudo_count, sweep.m_star_orcc, color=C_ORCC, lw=1.6,
        ls="--", label=r"$m^{*}$ under $q^{O}$")
ax.set_xscale("log")
ax.set_xlabel(r"concentration $a+b$ of $\mathrm{Beta}(c,c)$ (log scale)")
ax.set_ylabel(r"remaining executions $m_i^{*}$")
ax.set_title("(b) planned remaining executions")
ax.grid(alpha=0.25, lw=0.4)
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig2_optimistic_planning_mechanism.pdf"))
plt.close(fig)
print("wrote", os.path.join(FIG, "fig2_optimistic_planning_mechanism.pdf"))
