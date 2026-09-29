# -*- coding: utf-8 -*-
"""Runtime overhead aggregation + Beta-quantile micro-benchmark
(spec section 12).

Scheduler CPU time (sched_overhead_s) is recorded per run for every method
in the existing main CSVs; no experiment is re-run. Per-decision latency is
the per-run average (sched_overhead_s / executions); true per-decision
percentiles are not logged, so a micro-benchmark times the Beta quantile
computation directly.

Naming: ReproAlloc-Optimistic -> ReproAlloc; ReproAlloc -> Mean Planning.

Outputs:
  results/runtime_overhead/runtime_overhead.csv
  results/runtime_overhead/runtime_microbench.txt
"""
import os
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "runtime_overhead")
os.makedirs(OUT, exist_ok=True)

RENAME = {"ReproAlloc-Optimistic": "ReproAlloc", "ReproAlloc": "Mean Planning"}
KEEP = ["ReproAlloc", "Mean Planning", "ScoreCost"]


def agg(bench, path, bcol, tcol):
    df = pd.read_csv(path)
    df["paper_method"] = df.method.map(lambda m: RENAME.get(m, m))
    df = df[df.paper_method.isin(KEEP)]
    df["per_decision_ms"] = df.sched_overhead_s / df.executions * 1000.0
    rows = []
    for (tf, pm), g in df.groupby([bcol, "paper_method"]):
        rows.append({
            "benchmark": bench, "budget": float(tf), "method": pm,
            "n_runs": len(g),
            "scheduler_cpu_per_run_s_mean": float(g.sched_overhead_s.mean()),
            "scheduler_cpu_per_run_s_median":
                float(g.sched_overhead_s.median()),
            "scheduler_cpu_per_run_s_p95":
                float(g.sched_overhead_s.quantile(0.95)),
            "per_decision_ms_mean": float(g.per_decision_ms.mean()),
            "per_decision_ms_median": float(g.per_decision_ms.median()),
            "per_decision_ms_p95":
                float(g.per_decision_ms.quantile(0.95)),
            "simulated_exec_time_per_run_s_mean": float(g[tcol].mean()),
            "overhead_share_of_simulated_time":
                float((g.sched_overhead_s / g[tcol]).mean()),
        })
    return pd.DataFrame(rows)


def microbench():
    from scipy.stats import beta as beta_dist
    rng = np.random.default_rng(0)
    a = rng.uniform(1.0, 60.0, 200_000)
    b = rng.uniform(1.0, 60.0, 200_000)
    t0 = time.perf_counter()
    q = beta_dist.ppf(0.95, a, b)
    el = time.perf_counter() - t0
    return el, len(q)


def main():
    res = pd.concat([
        agg("amini", os.path.join(RES, "01_optimistic_core", "amini",
                                  "amini_optimistic_main.csv"),
            "budget_frac", "time_used_s"),
        agg("sensodat", os.path.join(RES, "01_optimistic_core", "sensodat",
                                     "sensodat_optimistic_main.csv"),
            "time_frac", "time_used"),
    ], ignore_index=True)
    res.to_csv(os.path.join(OUT, "runtime_overhead.csv"), index=False,
               float_format="%.10g")

    el, n = microbench()
    lines = [
        "Beta-quantile micro-benchmark (scipy.stats.beta.ppf, 1-delta=0.95)",
        f"calls: {n} vectorized over realistic (a,b) ranges [1,60]",
        f"total: {el:.3f}s  ->  {el / n * 1e6:.2f} us per quantile",
        "",
        "Context: the optimistic selector evaluates est_remaining_execs once",
        "per executed arm per step (cached by (s,n,q,tau)); the planning-rate",
        "quantile itself is one beta.ppf call per evaluation.",
        "Per-run scheduler CPU and per-decision latency are in",
        "runtime_overhead.csv (from sched_overhead_s recorded in the main",
        "experiment CSVs; per-decision values are per-run averages).",
    ]
    with open(os.path.join(OUT, "runtime_microbench.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines))

    pd.set_option("display.width", 240)
    print(res.to_string(index=False))
    print("\n".join(lines))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
