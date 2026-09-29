"""Analyze the SensoDat optimistic-vs-frozen main experiment.

Mirror of analyze_amini_optimistic.py for the SensoDat harness:
  * per-(method,budget) recall means (nested: reps averaged within split,
    then mean/sd over the 10 splits)
  * paired diffs (matched on split) vs ReproAlloc and baselines
  * decision-log statistics from logs/optimistic_ercc/sensodat/main/
  * writes results/diagnostics/sensodat_optimistic_analysis.txt
"""
import io
import math
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
NEW = os.path.dirname(HERE)
RES = os.path.join(NEW, "results", "01_optimistic_core", "sensodat")
DIAG = os.path.join(NEW, "results", "diagnostics")
LOGS = os.path.join(NEW, "logs", "optimistic_ercc", "sensodat", "main")
METHODS = ["Uniform", "FailRerun", "APGAI", "ScoreCost",
           "ReproAlloc", "ReproAlloc-OA", "ReproAlloc-Optimistic"]

os.makedirs(DIAG, exist_ok=True)
out = io.open(os.path.join(DIAG, "sensodat_optimistic_analysis.txt"), "w",
              encoding="utf-8")


def w(s=""):
    print(s, flush=True)
    out.write(s + "\n")


df = pd.read_csv(os.path.join(RES, "sensodat_optimistic_main.csv"))
# nested design: average reps within (split, budget, method) first
per_split = (df.groupby(["method", "time_frac", "split_id"], as_index=False)
             ["recall"].mean())

w("=" * 78)
w("SENSODAT (AmbieGen->Frenetic): OPTIMISTIC ERCC vs FROZEN (10 splits)")
w("=" * 78)

for tf in [0.05, 0.10, 0.20]:
    sub = per_split[np.isclose(per_split.time_frac, tf)]
    w(f"\n-- budget {tf*100:.0f}% --")
    g = sub.groupby("method")["recall"].agg(["mean", "std", "count"])
    g = g.reindex([m for m in METHODS if m in g.index])
    for m, r in g.iterrows():
        ci = 1.96 * r["std"] / math.sqrt(r["count"]) if r["std"] == r["std"] else 0
        w(f"   {m:24s} {r['mean']:.4f}  (sd={r['std']:.4f}, 95%CI pm {ci:.4f})")
    w("   paired diffs (matched on split), recall:")
    opt = sub[sub.method == "ReproAlloc-Optimistic"].set_index("split_id")["recall"]
    for base in METHODS:
        if base == "ReproAlloc-Optimistic":
            continue
        b = sub[sub.method == base].set_index("split_id")["recall"]
        common = opt.index.intersection(b.index)
        d = opt[common] - b[common]
        m_, sd_ = d.mean(), d.std(ddof=1)
        ci = 1.96 * sd_ / math.sqrt(len(d)) if sd_ == sd_ else 0.0
        try:
            from scipy.stats import wilcoxon
            p = wilcoxon(d).pvalue if (d != 0).any() else 1.0
        except Exception:
            p = float("nan")
        w(f"     OPT - {base:14s}: mean={m_:+.4f} 95%CI=[{m_-ci:+.4f},{m_+ci:+.4f}] "
          f"wilcoxon p={p:.4f} n={len(d)}")

# ---------------- decision-log statistics ----------------
w("\n" + "=" * 78)
w("DECISION LOGS (ReproAlloc-Optimistic) - SensoDat")
w("=" * 78)
frames = []
if os.path.isdir(LOGS):
    for f in sorted(os.listdir(LOGS)):
        if f.startswith("decisions_split") and f.endswith(".csv"):
            parts = f[:-4].split("_")
            sid = int(parts[1].replace("split", ""))
            tf = float(parts[2].replace("b", ""))
            rep = int(parts[3].replace("r", ""))
            d = pd.read_csv(os.path.join(LOGS, f))
            d["split_id"] = sid
            d["time_frac"] = tf
            d["rep"] = rep
            frames.append(d)
if frames:
    logs = pd.concat(frames, ignore_index=True)
    logs["m_mean_inf"] = logs.m_mean < 0
    logs["m_opt_inf"] = logs.m_opt < 0
    for tf in [0.05, 0.10, 0.20]:
        L = logs[np.isclose(logs.time_frac, tf)]
        n = len(L)
        if n == 0:
            continue
        ndiff = int(L.would_differ.sum())
        esc = int((L.m_mean_inf & ~L.m_opt_inf).sum())
        fb = int((L.sel_kind == "fallback").sum())
        w(f"\n  budget {tf*100:.0f}%: optimistic executions={n}")
        w(f"    picks where mean-index would choose differently: {ndiff} "
          f"({100*ndiff/n:.1f}%)")
        w(f"    executed arms with m_mean=inf but m_opt finite (fallback escape): "
          f"{esc} ({100*esc/n:.1f}%)")
        w(f"    optimistic fallback executions: {fb} ({100*fb/n:.1f}%)")
        fin = L[~L.m_mean_inf & ~L.m_opt_inf]
        if len(fin):
            ratio = fin.m_mean / fin.m_opt
            w(f"    m* reduction on executed arms (m_mean/m_opt): "
              f"median={ratio.median():.2f} mean={ratio.mean():.2f}")
        w(f"    q_opt - q_mean on executed arms: "
          f"median={(L.q_opt-L.q_mean).median():.3f} mean={(L.q_opt-L.q_mean).mean():.3f}")
    logs.to_csv(os.path.join(DIAG, "sensodat_decision_logs_all.csv"), index=False)
else:
    w("  (no decision logs found)")

out.close()
print(f"\nwrote {os.path.join(DIAG, 'sensodat_optimistic_analysis.txt')}")
