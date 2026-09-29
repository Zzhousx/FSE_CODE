"""Amini anytime curves: per-execution (time, recall) traces.

Runs 4 methods (ReproAlloc, ReproAlloc-Optimistic, ScoreCost, APGAI) at the
maximum canonical budget (20%) on all 12 seeds, recording after EVERY
execution the wall-clock time used and the reference-positive recall so far.
From these traces, recall(t) can be read at ANY budget t (anytime behaviour),
which subsumes the discrete budget points.

Output: results/diagnostics/amini_anytime_traces.csv
        results/diagnostics/amini_anytime_curve.csv  (mean over seeds on a
        common time grid)
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
NEW = os.path.dirname(HERE)
sys.path.insert(0, SRC)

import experiment_amini_finite_trace as A  # noqa: E402
import optimistic_ercc_amini as OA  # noqa: E402

DIAG = os.path.join(NEW, "results", "diagnostics")
os.makedirs(DIAG, exist_ok=True)

BUDGET_FRAC = 0.20
BUDGET_IDX = 2          # canonical index of 0.20 (keeps frozen streams)
METHOD_IDX = {"ScoreCost": 3, "ReproAlloc": 4, "ReproAlloc-Optimistic": 6,
              "APGAI": 2}


def traced_run(routes, orders, ref_pos_mask, budget, sel, rng, gs, trace):
    """A.run_one variant that also records the anytime trace."""
    K = len(routes)
    pools = [A.Pool(r.outcomes, r.durations, o) for r, o in zip(routes, orders)]
    arms = A.Arms(K, gs)
    st = sel.init(arms, pools, rng)
    time_used = 0.0
    execs = 0
    n_ref = int(ref_pos_mask.sum())
    while time_used < budget:
        i = sel.select(arms, pools, rng, st)
        if i < 0:
            break
        fail, cost = pools[i].draw()
        arms.update(i, fail, cost)
        sel.update(arms, i, st)
        time_used += cost
        execs += 1
        tp = int((arms.confirmed & ref_pos_mask).sum())
        trace.append((time_used, execs, tp / n_ref if n_ref else 0.0,
                      tp, int(arms.confirmed.sum())))


def main():
    routes = A.load_routes()
    gs = A.build_gs(routes)
    total_dur = sum(sum(r.durations) for r in routes)
    p_hat = np.array([r.p_hat for r in routes])
    ref_pos_mask = p_hat > A.TAU
    budget = total_dur * BUDGET_FRAC

    rows = []
    for sd in range(2000, 2012):
        orders = A.make_orders(routes, sd)
        m_cap = A.m_cap_from_budget(budget, gs)
        sels = {}
        frozen = A.build_selectors(m_cap)
        sels["ScoreCost"] = frozen[3]
        sels["ReproAlloc"] = frozen[4]
        sels["APGAI"] = frozen[2]
        sels["ReproAlloc-Optimistic"] = OA.ReproAllocOptAdapter(
            False, m_cap, diagnostics=False)
        for name, sel in sels.items():
            mi = METHOD_IDX[name]
            rng = A.method_rng(sd, mi, BUDGET_IDX)
            trace = []
            traced_run(routes, orders, ref_pos_mask, budget, sel, rng, gs,
                       trace)
            for (t, ex, rec, tp, conf) in trace:
                rows.append({"seed": sd, "method": name, "exec_idx": ex,
                             "time_used_s": t, "time_frac": t / total_dur,
                             "recall_refpos": rec, "tp": tp,
                             "confirmed": conf})
        print(f"seed {sd} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DIAG, "amini_anytime_traces.csv"), index=False)

    # pooled curve on a common time-fraction grid
    grid = np.linspace(0, BUDGET_FRAC, 201)
    curve = []
    for name in df.method.unique():
        sub = df[df.method == name]
        vals = []
        for sd in sub.seed.unique():
            tr = sub[sub.seed == sd].sort_values("time_frac")
            # step function: recall holds between executions
            v = np.interp(grid, tr.time_frac, tr.recall_refpos)
            vals.append(v)
        vals = np.array(vals)
        curve.append(pd.DataFrame({
            "time_frac": grid,
            "method": name,
            "recall_mean": vals.mean(axis=0),
            "recall_sd": vals.std(axis=0),
            "recall_lo": np.percentile(vals, 2.5, axis=0),
            "recall_hi": np.percentile(vals, 97.5, axis=0),
        }))
    c = pd.concat(curve)
    c.to_csv(os.path.join(DIAG, "amini_anytime_curve.csv"), index=False)
    print(f"wrote traces={len(df)} curve={len(c)}", flush=True)


if __name__ == "__main__":
    main()
