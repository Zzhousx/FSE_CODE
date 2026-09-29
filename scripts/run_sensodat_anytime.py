"""SensoDat anytime curves: per-execution (time, recall) traces at 20% budget.

4 methods (ReproAlloc, ReproAlloc-Optimistic, ScoreCost, FailRerun) x 10
splits x 1 rep.  Traced variant of E.run_task with identical semantics
(same per-cell tie-break stream formula), recording (time_used, recall)
after every execution.

Run with the ANACONDA python (sklearn 1.5.1) after the sweep finishes.

Output: results/diagnostics/sensodat_anytime_traces.csv
        results/diagnostics/sensodat_anytime_curve.csv
"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
NEW = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, SRC)

import original_env_patch  # noqa: F401  (before sklearn)

import sensodat_final_experiment as S  # noqa: E402
import v7_experiment as E  # noqa: E402
import optimistic_ercc_sensodat as OS  # noqa: E402

DIAG = os.path.join(NEW, "results", "diagnostics")
os.makedirs(DIAG, exist_ok=True)

BUDGET_FRAC = 0.20
METHODS = ["ReproAlloc", "ReproAlloc-Optimistic", "ScoreCost", "FailRerun"]


def traced_run_task(K, draw_fn, gt_good, budget, prior_score, gs, tau, sel,
                    rng, trace, kappa=S.KAPPA):
    """E.run_task variant that records the anytime trace (same semantics)."""
    arms = E.Arms(K, prior_score, gs, tau, kappa)
    st = sel.init(arms, rng)
    time_used = 0.0
    execs = 0
    n_good = int(gt_good.sum())
    while time_used < budget:
        i = sel.select(arms, st, rng)
        if i < 0:
            break
        fail, actual_cost = draw_fn(i, rng)
        arms.update(i, fail, actual_cost)
        sel.update(arms, i, st)
        time_used += actual_cost
        execs += 1
        tp = int((arms.confirmed & gt_good).sum())
        trace.append((time_used, execs, tp / n_good if n_good else 0.0, tp,
                      int(arms.confirmed.sum())))


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier
    M = S.make_splits()
    X, y, dur = M["X"], M["y"], M["dur"]
    rows = []
    t_start = time.time()
    for sp in M["splits"]:
        sid, seed = sp["split_id"], sp["split_seed"]
        train_idx = np.array(sp["train"])
        target_idx = np.array(sp["target"])
        ytr, dtr = y[train_idx], dur[train_idx]
        gs = E.GS(float(dtr[ytr == 1].mean()) if (ytr == 1).any() else float(dtr.mean()),
                  float(dtr[ytr == 0].mean()) if (ytr == 0).any() else float(dtr.mean()),
                  float(dtr.mean()))
        clf = HistGradientBoostingClassifier(random_state=seed, max_iter=200)
        clf.fit(X[train_idx], ytr)
        scores = clf.predict_proba(X[target_idx])[:, 1]
        y_t, d_t = y[target_idx], dur[target_idx]
        K = len(target_idx)
        gt_good = (y_t == 1)
        total_time = float(d_t.sum())

        def draw(i, rng, _yt=y_t, _dt=d_t):
            return int(_yt[i]), float(_dt[i])

        budget = total_time * BUDGET_FRAC
        m_cap = E.m_cap_from_budget(budget, gs)
        for pretty in METHODS:
            if pretty == "ReproAlloc":
                sf = E.ERCCSel(False, m_cap)
            elif pretty == "ReproAlloc-Optimistic":
                sf = OS.ERCCSelOptimistic(False, m_cap, diagnostics=False)
            elif pretty == "ScoreCost":
                sf = E.ScoreCostSel(False)
            else:
                sf = E.FnSel("failrerun")
            # rep 0 stream, identical formula as the frozen main run
            rng = np.random.default_rng(seed * S.TIEBREAK_BASE + 0 * 17
                                        + int(BUDGET_FRAC * 1000))
            trace = []
            traced_run_task(K, draw, gt_good, budget, scores, gs, S.TAU, sf,
                            rng, trace)
            for (t, ex, rec, tp, conf) in trace:
                rows.append({"split_id": sid, "method": pretty,
                             "exec_idx": ex, "time_used_s": t,
                             "time_frac": t / total_time,
                             "recall": rec, "tp": tp, "confirmed": conf})
        print(f"split {sid} done elapsed={time.time()-t_start:.0f}s",
              flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DIAG, "sensodat_anytime_traces.csv"), index=False)

    grid = np.linspace(0, BUDGET_FRAC, 201)
    curve = []
    for name in METHODS:
        sub = df[df.method == name]
        vals = []
        for sid in sub.split_id.unique():
            tr = sub[sub.split_id == sid].sort_values("time_frac")
            v = np.interp(grid, tr.time_frac, tr.recall)
            vals.append(v)
        vals = np.array(vals)
        curve.append(pd.DataFrame({
            "time_frac": grid, "method": name,
            "recall_mean": vals.mean(axis=0),
            "recall_sd": vals.std(axis=0),
            "recall_lo": np.percentile(vals, 2.5, axis=0),
            "recall_hi": np.percentile(vals, 97.5, axis=0),
        }))
    c = pd.concat(curve)
    c.to_csv(os.path.join(DIAG, "sensodat_anytime_curve.csv"), index=False)
    print(f"wrote traces={len(df)} curve={len(c)}", flush=True)


if __name__ == "__main__":
    main()
