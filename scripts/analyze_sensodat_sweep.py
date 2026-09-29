"""Analyze the SensoDat budget sweep (7 methods x 8 budgets x 10 splits x 3 reps).

Nested aggregation (reps averaged within split, then over splits), paired
diffs of ReproAlloc-Optimistic vs the frozen methods at every budget, and a
per-budget "best frozen" comparison table for the final report.

Output: results/diagnostics/sensodat_sweep_analysis.txt
"""
import io
import math
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
NEW = os.path.dirname(HERE)
SWEEP = os.path.join(NEW, "results", "02_budget_sweep", "sensodat",
                     "sensodat_optimistic_sweep.csv")
DIAG = os.path.join(NEW, "results", "diagnostics")
METHODS = ["Uniform", "FailRerun", "APGAI", "ScoreCost",
           "ReproAlloc", "ReproAlloc-OA", "ReproAlloc-Optimistic"]

os.makedirs(DIAG, exist_ok=True)
out = io.open(os.path.join(DIAG, "sensodat_sweep_analysis.txt"), "w",
              encoding="utf-8")


def w(s=""):
    print(s, flush=True)
    out.write(s + "\n")


df = pd.read_csv(SWEEP)
per_split = (df.groupby(["method", "time_frac", "split_id"], as_index=False)
             ["recall"].mean())
budgets = sorted(per_split.time_frac.unique())

w("=" * 78)
w("SENSODAT BUDGET SWEEP - nested means (10 splits), recall")
w("=" * 78)
hdr = f"{'budget':>7s} | " + " | ".join(f"{m:>22s}" for m in METHODS)
w(hdr)
for tf in budgets:
    sub = per_split[np.isclose(per_split.time_frac, tf)]
    g = sub.groupby("method")["recall"].mean().reindex(METHODS)
    w(f"{tf*100:6.0f}% | " + " | ".join(
        f"{g[m]:22.4f}" if g[m] == g[m] else f"{'-':>22s}" for m in METHODS))

w("\n" + "=" * 78)
w("PAIRED DIFFS vs ReproAlloc-Optimistic (matched on split)")
w("=" * 78)
for tf in budgets:
    sub = per_split[np.isclose(per_split.time_frac, tf)]
    opt = sub[sub.method == "ReproAlloc-Optimistic"].set_index("split_id")["recall"]
    w(f"\n-- budget {tf*100:.0f}% --")
    best_frozen, best_val = None, -1.0
    for base in METHODS[:-1]:
        b = sub[sub.method == base].set_index("split_id")["recall"]
        if b.mean() > best_val:
            best_val, best_frozen = b.mean(), base
        common = opt.index.intersection(b.index)
        d = opt[common] - b[common]
        m_, sd_ = d.mean(), d.std(ddof=1)
        ci = 1.96 * sd_ / math.sqrt(len(d)) if sd_ == sd_ else 0.0
        try:
            from scipy.stats import wilcoxon
            p = wilcoxon(d).pvalue if (d != 0).any() else 1.0
        except Exception:
            p = float("nan")
        w(f"  OPT - {base:14s}: mean={m_:+.4f} CI=[{m_-ci:+.4f},{m_+ci:+.4f}] "
          f"p={p:.4f}")
    w(f"  best frozen at this budget: {best_frozen} {best_val:.4f} "
      f"(OPT {opt.mean():.4f})")

out.close()
print(f"\nwrote {os.path.join(DIAG, 'sensodat_sweep_analysis.txt')}")
