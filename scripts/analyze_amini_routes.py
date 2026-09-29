"""Amini route-level diagnostic: WHERE does optimism help and hurt?

For every route x budget: confirmation rate over the 12 seeds under frozen
ReproAlloc vs ReproAlloc-Optimistic, joined with the route's confirmability
context (p_hat, pool size, k_best/k_worst, mean duration).

Outputs:
  results/diagnostics/amini_route_level_optimistic.csv
  results/diagnostics/amini_route_level_report.txt
"""
import io
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
NEW = os.path.dirname(HERE)
sys.path.insert(0, SRC)

import experiment_amini_finite_trace as A  # noqa: E402

DIAG = os.path.join(NEW, "results", "diagnostics")
RES = os.path.join(NEW, "results", "01_optimistic_core", "amini")

routes = A.load_routes()
ctx = []
for r in routes:
    ctx.append({
        "route_id": r.route_id, "n_pool": r.n, "s_total": r.s,
        "p_hat": r.p_hat, "ref_pos": int(r.p_hat > A.TAU),
        "L_full": A.cs_lower(r.s, r.n),
        "k_best": A.k_best_order(r.s, r.n),
        "k_worst": A.k_worst_order(r.s, r.n),
        "mean_dur": float(np.mean(r.durations)),
    })
ctx = pd.DataFrame(ctx)

alloc = pd.read_csv(os.path.join(RES, "amini_optimistic_alloc_main.csv"))
piv = []
for (m, tf, rid), sub in alloc.groupby(["method", "budget_frac", "route_id"]):
    piv.append({"method": m, "budget_frac": tf, "route_id": rid,
                "conf_rate": sub.confirmed_sticky.mean(),
                "runs_mean": sub.runs_used.mean(),
                "fails_mean": sub.fails_used.mean()})
piv = pd.DataFrame(piv)

ra = piv[piv.method == "ReproAlloc"].set_index(["budget_frac", "route_id"])
op = piv[piv.method == "ReproAlloc-Optimistic"].set_index(["budget_frac", "route_id"])
ra = ra.rename(columns={"conf_rate": "conf_rate_mean", "runs_mean": "runs_mean_mean",
                        "fails_mean": "fails_mean_mean"}).drop(columns=["method"])
op = op.rename(columns={"conf_rate": "conf_rate_opt", "runs_mean": "runs_mean_opt",
                        "fails_mean": "fails_mean_opt"}).drop(columns=["method"])
j = op.join(ra).reset_index()
j["d_conf"] = j.conf_rate_opt - j.conf_rate_mean
j["d_runs"] = j.runs_mean_opt - j.runs_mean_mean
j = j.merge(ctx, on="route_id", how="left")
j = j.sort_values(["budget_frac", "d_conf"], ascending=[True, False])
j.to_csv(os.path.join(DIAG, "amini_route_level_optimistic.csv"), index=False)

out = io.open(os.path.join(DIAG, "amini_route_level_report.txt"), "w",
              encoding="utf-8")
for tf in [0.05, 0.10, 0.20]:
    sub = j[np.isclose(j.budget_frac, tf)]
    out.write(f"\n===== budget {tf*100:.0f}% =====\n")
    out.write("net confirmation-rate change per route (opt - mean), only "
              "non-zero:\n")
    nz = sub[np.abs(sub.d_conf) > 1e-9].sort_values("d_conf", ascending=False)
    for _, r in nz.iterrows():
        out.write(f"  {r.route_id:20s} d_conf={r.d_conf:+.3f} "
                  f"(mean {r.conf_rate_mean:.2f} -> opt {r.conf_rate_opt:.2f}) "
                  f"p_hat={r.p_hat:.2f} pool={r.n_pool} s={r.s_total} "
                  f"k_best={r.k_best} k_worst={r.k_worst} "
                  f"d_runs={r.d_runs:+.1f}\n")
    g = sub[sub.d_conf > 1e-9]
    l = sub[sub.d_conf < -1e-9]
    out.write(f"  gained routes: {len(g)} (ref_pos {int(g.ref_pos.sum())}), "
              f"lost routes: {len(l)} (ref_pos {int(l.ref_pos.sum())})\n")
    if len(g):
        out.write(f"  gained: mean p_hat={g.p_hat.mean():.2f} "
                  f"mean k_best={g.k_best.mean():.1f}\n")
    if len(l):
        out.write(f"  lost  : mean p_hat={l.p_hat.mean():.2f} "
                  f"mean k_best={l.k_best.mean():.1f}\n")
out.close()
print("wrote amini_route_level_optimistic.csv + report")
