# -*- coding: utf-8 -*-
"""OptimisticScoreCost check runner (spec: OptimisticScoreCost_validation_prompt).

Runs the frozen seven methods + the new OptimisticScoreCost (8th, appended)
on BOTH benchmarks with exactly the paper's main-experiment settings, and
writes raw results to results/optimistic_scorecost_check/ (never mixed with
the official 01_optimistic_core outputs).

  SensoDat : 10 splits x 3 tie-break reps x budgets 5/10/20%, tau=0.3,
             delta=0.05, HGB priors (anaconda python for bit-exactness).
  Amini    : 12 matched trace-order seeds (2000..2011) x budgets 5/10/20%,
             Beta(1,1), without-replacement pools, tau=0.3, delta=0.05.

Regression safety by construction:
  * SensoDat: every (split, budget, rep, method) cell builds a fresh rng with
    the SAME seed material for every method, so the 8th method cannot perturb
    the streams of the existing seven.
  * Amini: method_rng(seed, method_idx, budget_idx) depends on list position;
    OptimisticScoreCost is appended at index 7, indices 0..6 unchanged.

Behavior fields: SensoDat rows get visit statistics via a _VisitTracker
wrapper (update-hook only; no scheduling/rng change).  Amini keeps the
per-route alloc rows, from which the same fields are derived downstream.

Usage: python run_optimistic_scorecost.py [sensodat|amini|all] [budgets]
       [n_splits] [n_reps] [n_seeds] [diag]
"""
import csv
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
NEW = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, SRC)

TAG = "osc_check"
OUTDIR = os.path.join(NEW, "results", "optimistic_scorecost_check")
LOGDIR = os.path.join(NEW, "logs", "optimistic_scorecost_check")


# ----------------------------------------------------------------------
# SensoDat branch (mirrors scripts/run_optimistic_sensodat.py)
# ----------------------------------------------------------------------
class _VisitTracker:
    """Wraps a v7-style selector: counts visits per arm and snapshots the
    final arms.n / arms.confirmed.  Forwards the diagnostic counters and
    last_diag so v7_experiment.run_task behaves exactly as with the bare
    selector.  No scheduling or rng change."""

    def __init__(self, inner):
        self.inner = inner
        self.visits = Counter()
        self.final_n = None
        self.final_conf = None

    # forwarded diagnostic counters (run_task resets/reads them on `sel`)
    @property
    def n_main_pos(self):
        return getattr(self.inner, "n_main_pos", 0)

    @n_main_pos.setter
    def n_main_pos(self, v):
        if hasattr(self.inner, "n_main_pos"):
            self.inner.n_main_pos = v

    @property
    def n_fallback(self):
        return getattr(self.inner, "n_fallback", 0)

    @n_fallback.setter
    def n_fallback(self, v):
        if hasattr(self.inner, "n_fallback"):
            self.inner.n_fallback = v

    @property
    def last_diag(self):
        return getattr(self.inner, "last_diag", None)

    def init(self, arms, rng):
        self.visits = Counter()
        self.final_n = None
        self.final_conf = None
        return self.inner.init(arms, rng)

    def select(self, arms, st, rng):
        return self.inner.select(arms, st, rng)

    def update(self, arms, i, st):
        self.visits[int(i)] += 1
        r = self.inner.update(arms, i, st)
        self.final_n = arms.n.copy()
        self.final_conf = arms.confirmed.copy()
        return r


def run_sensodat(budget_fracs=(0.05, 0.10, 0.20), n_splits=None, n_reps=None,
                 diag=True):
    import original_env_patch  # noqa: F401  (before sklearn)
    from sklearn.ensemble import HistGradientBoostingClassifier
    import sensodat_final_experiment as S
    import v7_experiment as E
    import optimistic_ercc_sensodat as OS

    if n_splits is None:
        n_splits = S.N_SPLITS
    if n_reps is None:
        n_reps = S.N_TIEBREAK_REPS
    os.makedirs(OUTDIR, exist_ok=True)
    logdir = os.path.join(LOGDIR, "sensodat")
    os.makedirs(logdir, exist_ok=True)

    t_start = time.time()
    M = S.make_splits()
    X, y, dur = M["X"], M["y"], M["dur"]
    rows, split_rows = [], []

    for sp in M["splits"][:n_splits]:
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
        n_good = int(gt_good.sum())
        total_time = float(d_t.sum())

        def draw(i, rng, _yt=y_t, _dt=d_t):
            return int(_yt[i]), float(_dt[i])

        split_rows.append({"split_id": sid, "split_seed": seed,
                           "K_target": K, "n_target_failures": n_good,
                           "total_target_time_s": total_time})
        for tf in budget_fracs:
            budget = total_time * tf
            m_cap = E.m_cap_from_budget(budget, gs)
            for rep in range(n_reps):
                sels = OS.make_sensodat_selectors_8(m_cap, diagnostics=diag)
                for pretty, mkey in OS.METHODS_8:
                    sf = _VisitTracker(sels[mkey])
                    rng = np.random.default_rng(
                        seed * S.TIEBREAK_BASE + rep * 17 + int(tf * 1000))
                    r = E.run_task(K, draw, gt_good, budget, scores, gs,
                                   S.TAU, sf, rng, kappa=S.KAPPA)
                    nv = len(sf.visits)
                    exc = int(r["executions"])
                    if sf.final_n is not None and sf.final_conf is not None:
                        e_conf = int(sf.final_n[sf.final_conf].sum())
                    else:
                        e_conf = 0
                    r.update({"dataset": "sensodat", "split_id": sid,
                              "split_seed": seed, "rep": rep,
                              "method": pretty, "method_key": mkey,
                              "time_frac": tf, "budget_s": budget,
                              "m_cap": int(m_cap), "K": K,
                              "n_ref_good": n_good,
                              "unique_tests_visited": nv,
                              "mean_exec_per_visited": (exc / nv if nv else 0.0),
                              "exec_finally_confirmed": e_conf,
                              "exec_finally_unconfirmed": exc - e_conf})
                    rows.append(r)
                    if diag and mkey == "ercc_optimistic" and sf.last_diag:
                        _dump_diag(sf.last_diag, logdir, sid, tf, rep)
        el = time.time() - t_start
        print(f"  split {sid} (seed={seed}) done K={K} fails={n_good} "
              f"elapsed={el:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    p_main = os.path.join(OUTDIR, "sensodat_osc_check.csv")
    df.to_csv(p_main, index=False)
    pd.DataFrame(split_rows).to_csv(
        os.path.join(OUTDIR, "sensodat_osc_check_splits.csv"), index=False)
    meta = {"tag": TAG, "budget_fracs": list(budget_fracs),
            "methods": [p for p, _ in OS.METHODS_8],
            "n_splits": n_splits, "n_reps": n_reps,
            "tau": S.TAU, "delta": S.DELTA, "kappa": S.KAPPA,
            "optimistic_delta": 0.05, "planning_rate": "Beta.ppf(0.95;a,b)"}
    with open(os.path.join(OUTDIR, "sensodat_osc_check_meta.json"),
              "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"wrote {p_main} ({len(df)} rows) elapsed "
          f"{time.time()-t_start:.0f}s", flush=True)
    return df


def _dump_diag(diag, logdir, sid, tf, rep):
    p = os.path.join(logdir, f"decisions_split{sid}_b{tf:.2f}_r{rep}.csv")
    cols = ["arm", "s", "n", "post_a", "post_b", "q_mean", "q_opt",
            "m_mean", "m_opt", "confirmed", "sel_kind", "would_differ"]
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for d in diag:
            w.writerow({k: d.get(k, "") for k in cols})


# ----------------------------------------------------------------------
# Amini branch (mirrors scripts/run_optimistic_amini.py)
# ----------------------------------------------------------------------
BUDGET_INDEX = {0.05: 0, 0.10: 1, 0.20: 2,
                0.03: 3, 0.06: 4, 0.07: 5, 0.08: 6, 0.15: 7}


def run_amini(budget_fracs=(0.05, 0.10, 0.20), n_seeds=12, diag=True):
    import experiment_amini_finite_trace as A
    import optimistic_ercc_amini as OA

    os.makedirs(OUTDIR, exist_ok=True)
    logdir = os.path.join(LOGDIR, "amini")
    os.makedirs(logdir, exist_ok=True)

    routes = A.load_routes()
    gs = A.build_gs(routes)
    total_dur = sum(sum(r.durations) for r in routes)
    p_hat = np.array([r.p_hat for r in routes])
    ref_pos_mask = p_hat > A.TAU
    n_ref = int(ref_pos_mask.sum())

    seeds = [2000 + k for k in range(n_seeds)]
    rows, alloc_rows = [], []
    for sd in seeds:
        orders = A.make_orders(routes, sd)
        ceiling, _info = A.oracle_ceiling_for_order(routes, orders)
        for tf in budget_fracs:
            bi = BUDGET_INDEX[round(tf, 2)]
            budget = total_dur * tf
            m_cap = A.m_cap_from_budget(budget, gs)
            sels = OA.build_selectors_8(m_cap, diagnostics=diag)
            for mi, sel in enumerate(sels):
                rng = A.method_rng(sd, mi, bi)
                res = A.run_one(routes, orders, ref_pos_mask, budget, sel,
                                rng, gs, ceiling)
                alloc = res.pop("_alloc")
                rec = {"tag": TAG, "seed": sd, "budget_frac": tf,
                       "budget_s": budget, "method": sel.name,
                       "m_cap": m_cap, "ceiling_seed": ceiling,
                       "n_refpos": n_ref}
                rec.update(res)
                rows.append(rec)
                for (rid, ni, si, L, cf, kf, cap, left, rp) in alloc:
                    alloc_rows.append({"tag": TAG, "seed": sd,
                                       "budget_frac": tf, "method": sel.name,
                                       "route_id": rid, "runs_used": ni,
                                       "fails_used": si, "pool_size": cap,
                                       "pool_remaining": left,
                                       "pool_exhausted": int(left == 0),
                                       "L_final": L, "confirmed_sticky": int(cf),
                                       "k_first_confirm": kf,
                                       "reference_positive": int(rp)})
                if diag and getattr(sel, "inner", None) is not None \
                        and hasattr(sel.inner, "last_diag"):
                    _dump_diag_amini(sel.inner.last_diag, logdir, sd, tf)
            print(f"  seed {sd} budget {tf}: done", flush=True)
        print(f"seed {sd} complete (ceiling={ceiling})", flush=True)

    main_cols = ["tag", "seed", "budget_frac", "budget_s", "method", "m_cap",
                 "ceiling_seed", "n_refpos", "tp", "fp", "fn", "confirmed",
                 "recall_refpos", "precision", "recall_ceiling_seed",
                 "nonsticky_confirmed", "executions", "time_used_s",
                 "budget_util", "overtime_s", "overrun_flag",
                 "pools_exhausted", "runs_left_in_pools",
                 "stopped_by_exhaustion", "sched_overhead_s"]
    p_main = _wcsv(os.path.join(OUTDIR, "amini_osc_check.csv"), rows, main_cols)
    p_alloc = _wcsv(os.path.join(OUTDIR, "amini_osc_check_alloc.csv"),
                    alloc_rows)
    meta = {"tag": TAG, "seeds": seeds, "budget_fracs": list(budget_fracs),
            "methods": [s.name for s in OA.build_selectors_8(1, False)],
            "tau": A.TAU, "delta": A.DELTA,
            "optimistic_delta": 0.05, "planning_rate": "Beta.ppf(0.95;a,b)",
            "total_duration_s": total_dur}
    with open(os.path.join(OUTDIR, "amini_osc_check_meta.json"), "w",
              encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"wrote {p_main} ({len(rows)} rows)", flush=True)
    print(f"wrote {p_alloc} ({len(alloc_rows)} rows)", flush=True)
    return rows


def _wcsv(path, data, cols=None):
    if not data:
        return path
    cols = cols or list(data[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for d in data:
            w.writerow({k: d.get(k, "") for k in cols})
    return path


def _dump_diag_amini(diag, logdir, seed, tf):
    if not diag:
        return
    p = os.path.join(logdir, f"decisions_seed{seed}_b{tf:.2f}.csv")
    cols = ["arm", "s", "n", "post_a", "post_b", "q_mean", "q_opt",
            "m_mean", "m_opt", "confirmed", "sel_kind", "would_differ"]
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for d in diag:
            w.writerow({k: d.get(k, "") for k in cols})


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    budgets = (tuple(float(x) for x in sys.argv[2].split(","))
               if len(sys.argv) > 2 else (0.05, 0.10, 0.20))
    n_splits = int(sys.argv[3]) if len(sys.argv) > 3 else None
    n_reps = int(sys.argv[4]) if len(sys.argv) > 4 else None
    n_seeds = int(sys.argv[5]) if len(sys.argv) > 5 else 12
    diag = (sys.argv[6] != "0") if len(sys.argv) > 6 else True
    if which in ("sensodat", "all"):
        kw = {}
        if n_splits is not None:
            kw["n_splits"] = n_splits
        if n_reps is not None:
            kw["n_reps"] = n_reps
        run_sensodat(budget_fracs=budgets, diag=diag, **kw)
    if which in ("amini", "all"):
        run_amini(budget_fracs=budgets, n_seeds=n_seeds, diag=diag)
