"""Run T2/T3/T4 variants through the frozen SensoDat and Amini loops."""

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import original_env_patch  # noqa: F401
import sensodat_final_experiment as S
import v7_experiment as E
import experiment_amini_finite_trace as A
from revision_selectors import (
    SensoRevisionERCC, AminiRevisionERCC,
    SensoRerunK, AminiRerunK,
    SensoBayesUCBCost, AminiBayesUCBCost,
)
from revision_cap_diagnostics import SensoCapDiagnostic, AminiCapDiagnostic

BUDGETS = (0.03, 0.05, 0.06, 0.07, 0.08, 0.10, 0.15, 0.20)
PRIMARY = (0.05, 0.10, 0.20)
BI = {0.05: 0, 0.10: 1, 0.20: 2, 0.03: 3, 0.06: 4,
      0.07: 5, 0.08: 6, 0.15: 7}


def variants(benchmark, task, cap, tau, budget):
    s = benchmark == "sensodat"
    orcc = SensoRevisionERCC if s else AminiRevisionERCC
    rerun = SensoRerunK if s else AminiRerunK
    bayes = SensoBayesUCBCost if s else AminiBayesUCBCost
    if task == "T2diag":
        if tau != .3 or budget not in PRIMARY:
            return []
        diagnostic = SensoCapDiagnostic if s else AminiCapDiagnostic
        return [("Mean Planning", diagnostic(cap, mode="mean"), 4),
                ("ReproAlloc", diagnostic(cap), 6)]
    if task == "T2":
        if tau != .3 or budget not in PRIMARY:
            return []
        return [("MeanPlanning-uncapped", orcc(cap, mode="mean", planning_cap="none"), 6),
                ("ReproAlloc-uncapped", orcc(cap, planning_cap="none"), 6)]
    if task == "T3":
        if tau != .3 or budget not in PRIMARY:
            return []
        return [(f"ReproAlloc-rho{rho:.2f}", orcc(cap, planning_quantile=rho), 6)
                for rho in (.50, .75, .90, .99)]
    if task == "T4":
        if tau != .3 and budget not in PRIMARY:
            return []
        return [("RerunK-5", rerun(5), 8),
                ("RerunK-10", rerun(10), 8),
                ("RerunK-20", rerun(20), 8),
                ("BayesUCB-Cost", bayes(), 9)]
    raise ValueError(task)


def run_sensodat(split_id, task):
    from sklearn.ensemble import HistGradientBoostingClassifier

    S.DATA = str(ROOT.parent / "data")
    data = S.make_splits()
    sp = data["splits"][split_id]
    X, y, dur = data["X"], data["y"], data["dur"]
    train = np.asarray(sp["train"])
    target = np.asarray(sp["target"])
    ytr, dtr = y[train], dur[train]
    gs = E.GS(float(dtr[ytr == 1].mean()), float(dtr[ytr == 0].mean()), float(dtr.mean()))
    clf = HistGradientBoostingClassifier(random_state=sp["split_seed"], max_iter=200)
    clf.fit(X[train], ytr)
    scores = clf.predict_proba(X[target])[:, 1]
    yt, dt = y[target], dur[target]
    good = yt == 1

    def draw(i, rng):
        return int(yt[i]), float(dt[i])

    rows = []
    start = time.time()
    for tau in (.2, .3, .4):
        for budget_frac in BUDGETS:
            budget = float(dt.sum()) * budget_frac
            cap = E.m_cap_from_budget(budget, gs)
            items = variants("sensodat", task, cap, tau, budget_frac)
            if not items:
                continue
            for rep in range(S.N_TIEBREAK_REPS):
                seed = sp["split_seed"] * S.TIEBREAK_BASE + rep * 17 + int(budget_frac * 1000)
                for name, selector, _index in items:
                    rng = np.random.default_rng(seed)
                    r = E.run_task(len(target), draw, good, budget, scores, gs,
                                   tau, selector, rng, kappa=S.KAPPA)
                    rows.append({"benchmark": "SensoDat", "unit": split_id,
                                 "rep": rep, "method": name, "budget": budget_frac,
                                 "tau": tau, "n_reference": int(good.sum()),
                                 "count": r["tp"], "recall": r["recall"],
                                 "executions": r["executions"], "time_used_s": r["time_used"],
                                 "m_cap": cap, "sched_overhead_s": r["sched_overhead_s"],
                                 "cap_bound_steps": getattr(selector, "cap_bound_steps", None),
                                 "changed_steps": getattr(selector, "changed_steps", None),
                                 "diagnostic_steps": getattr(selector, "steps", None)})
        print(f"split {split_id} task {task} tau={tau} elapsed={time.time()-start:.1f}s", flush=True)
    out = ROOT / "results" / "revision" / task / "sensodat"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"split_{split_id}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"wrote {path}, rows={len(rows)}", flush=True)


def run_amini(task):
    routes = A.load_routes()
    gs = A.build_gs(routes)
    total = sum(sum(r.durations) for r in routes)
    rows = []
    start = time.time()
    for tau in (.2, .3, .4):
        A.TAU = tau
        reference = np.array([r.p_hat > tau for r in routes])
        for seed in range(2000, 2012):
            orders = A.make_orders(routes, seed)
            ceiling, _ = A.oracle_ceiling_for_order(routes, orders)
            for budget_frac in BUDGETS:
                budget = total * budget_frac
                cap = A.m_cap_from_budget(budget, gs)
                for name, selector, method_index in variants("amini", task, cap, tau, budget_frac):
                    r = A.run_one(routes, orders, reference, budget, selector,
                                  A.method_rng(seed, method_index, BI[budget_frac]), gs, ceiling)
                    rows.append({"benchmark": "Amini", "unit": seed, "rep": 0,
                                 "method": name, "budget": budget_frac, "tau": tau,
                                 "n_reference": int(reference.sum()), "count": r["tp"],
                                 "recall": r["recall_refpos"], "executions": r["executions"],
                                 "time_used_s": r["time_used_s"], "m_cap": cap,
                                 "sched_overhead_s": r["sched_overhead_s"],
                                 "cap_bound_steps": getattr(selector, "cap_bound_steps", None),
                                 "changed_steps": getattr(selector, "changed_steps", None),
                                 "diagnostic_steps": getattr(selector, "steps", None)})
        print(f"Amini task {task} tau={tau} elapsed={time.time()-start:.1f}s", flush=True)
    A.TAU = .3
    out = ROOT / "results" / "revision" / task / "amini"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "raw.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"wrote {path}, rows={len(rows)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", choices=("sensodat", "amini"))
    parser.add_argument("task", choices=("T2", "T2diag", "T3", "T4"))
    parser.add_argument("--split", type=int)
    args = parser.parse_args()
    if args.benchmark == "sensodat":
        if args.split is None:
            parser.error("SensoDat requires --split")
        run_sensodat(args.split, args.task)
    else:
        run_amini(args.task)
