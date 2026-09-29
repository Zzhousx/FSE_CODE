"""Analyze the Amini optimistic-vs-frozen main experiment.

Produces (to results/diagnostics/ + printed summary):
  * per-(method,budget) recall_refpos means with 95% CI over the 12 seeds
  * paired diffs (matched on seed) vs ReproAlloc and vs each baseline
  * decision-log statistics: how often optimism changes the pick
    (would_differ), fallback escape counts, m* reduction stats
  * route-level: which routes gained/lost confirmations under optimism
"""
import io
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
NEW = os.path.dirname(HERE)
RES = os.path.join(NEW, "results", "01_optimistic_core", "amini")
DIAG = os.path.join(NEW, "results", "diagnostics")
LOGS = os.path.join(NEW, "logs", "optimistic_ercc", "amini", "main")
METHODS = ["Uniform", "FailRerun", "APGAI", "ScoreCost",
           "ReproAlloc", "ReproAlloc-OA", "ReproAlloc-Optimistic"]

os.makedirs(DIAG, exist_ok=True)
out = io.open(os.path.join(DIAG, "amini_optimistic_analysis.txt"), "w",
              encoding="utf-8")


def w(s=""):
    print(s, flush=True)
    out.write(s + "\n")


df = pd.read_csv(os.path.join(RES, "amini_optimistic_main.csv"))
alloc = pd.read_csv(os.path.join(RES, "amini_optimistic_alloc_main.csv"))

w("=" * 78)
w("AMINI-TRANSFUSER: OPTIMISTIC ERCC vs FROZEN (12 matched seeds)")
w("=" * 78)

for tf in [0.05, 0.10, 0.20]:
    sub = df[np.isclose(df.budget_frac, tf)]
    w(f"\n-- budget {tf*100:.0f}% --")
    g = sub.groupby("method")["recall_refpos"].agg(["mean", "std", "count"])
    g = g.reindex([m for m in METHODS if m in g.index])
    for m, r in g.iterrows():
        ci = 1.96 * r["std"] / math.sqrt(r["count"]) if r["std"] == r["std"] else 0
        w(f"   {m:24s} {r['mean']:.4f}  (sd={r['std']:.4f}, 95%CI pm {ci:.4f})")
    w("   paired diffs (matched on seed), recall_refpos:")
    opt = sub[sub.method == "ReproAlloc-Optimistic"].set_index("seed")["recall_refpos"]
    for base in METHODS:
        if base == "ReproAlloc-Optimistic":
            continue
        b = sub[sub.method == base].set_index("seed")["recall_refpos"]
        common = opt.index.intersection(b.index)
        d = opt[common] - b[common]
        m_, sd_ = d.mean(), d.std(ddof=1)
        ci = 1.96 * sd_ / math.sqrt(len(d)) if sd_ == sd_ else 0.0
        # Wilcoxon signed-rank (scipy) as robustness
        try:
            from scipy.stats import wilcoxon
            p = wilcoxon(d).pvalue if (d != 0).any() else 1.0
        except Exception:
            p = float("nan")
        w(f"     OPT - {base:14s}: mean={m_:+.4f} 95%CI=[{m_-ci:+.4f},{m_+ci:+.4f}] "
          f"wilcoxon p={p:.4f} n={len(d)}")

# ---------------- decision-log statistics ----------------
w("\n" + "=" * 78)
w("DECISION LOGS (ReproAlloc-Optimistic): what optimism actually does")
w("=" * 78)
frames = []
for f in sorted(os.listdir(LOGS)):
    if f.startswith("decisions_seed") and f.endswith(".csv"):
        parts = f[:-4].split("_")
        seed = int(parts[1].replace("seed", ""))
        tf = float(parts[2].replace("b", ""))
        d = pd.read_csv(os.path.join(LOGS, f))
        d["seed"] = seed
        d["budget_frac"] = tf
        frames.append(d)
logs = pd.concat(frames, ignore_index=True)
logs["m_mean_inf"] = logs.m_mean < 0
logs["m_opt_inf"] = logs.m_opt < 0
for tf in [0.05, 0.10, 0.20]:
    L = logs[np.isclose(logs.budget_frac, tf)]
    n = len(L)
    ndiff = int(L.would_differ.sum())
    esc = int((L.m_mean_inf & ~L.m_opt_inf).sum())
    fb = int((L.sel_kind == "fallback").sum())
    w(f"\n  budget {tf*100:.0f}%: executions={n}")
    w(f"    picks where mean-index would choose differently: {ndiff} "
      f"({100*ndiff/max(n,1):.1f}%)")
    w(f"    executed arms with m_mean=inf but m_opt finite (fallback escape): "
      f"{esc} ({100*esc/max(n,1):.1f}%)")
    w(f"    optimistic fallback executions: {fb} ({100*fb/max(n,1):.1f}%)")
    fin = L[~L.m_mean_inf & ~L.m_opt_inf]
    if len(fin):
        ratio = fin.m_mean / fin.m_opt
        w(f"    m* reduction on executed arms (m_mean/m_opt): "
          f"median={ratio.median():.2f} mean={ratio.mean():.2f} "
          f"max={ratio.max():.1f}")
    w(f"    q_opt - q_mean on executed arms: "
      f"median={(L.q_opt-L.q_mean).median():.3f} mean={(L.q_opt-L.q_mean).mean():.3f}")
logs.to_csv(os.path.join(DIAG, "amini_decision_logs_all.csv"), index=False)

# ---------------- route-level wins/losses ----------------
w("\n" + "=" * 78)
w("ROUTE-LEVEL: confirmations gained/lost vs frozen ReproAlloc")
w("=" * 78)
for tf in [0.05, 0.10, 0.20]:
    a = alloc[np.isclose(alloc.budget_frac, tf)]
    ra = a[a.method == "ReproAlloc"].set_index(["seed", "route_id"])
    op = a[a.method == "ReproAlloc-Optimistic"].set_index(["seed", "route_id"])
    joined = ra[["confirmed_sticky", "runs_used"]].join(
        op[["confirmed_sticky", "runs_used"]], lsuffix="_mean", rsuffix="_opt")
    gained = joined[(joined.confirmed_sticky_opt == 1) & (joined.confirmed_sticky_mean == 0)]
    lost = joined[(joined.confirmed_sticky_opt == 0) & (joined.confirmed_sticky_mean == 1)]
    w(f"\n  budget {tf*100:.0f}%: gained={len(gained)} lost={len(lost)} "
      f"(seed,route) cells")
    if len(gained):
        rg = gained.reset_index().route_id.value_counts()
        w(f"    most gained routes: {dict(rg.head(6))}")
    if len(lost):
        rl = lost.reset_index().route_id.value_counts()
        w(f"    most lost routes  : {dict(rl.head(6))}")

out.close()
print(f"\nwrote {os.path.join(DIAG, 'amini_optimistic_analysis.txt')}")
