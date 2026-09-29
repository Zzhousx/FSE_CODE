"""Tau-sensitivity runs for SensoDat (AmbieGen->Frenetic).

Replicates scripts/run_optimistic_sensodat.py exactly (same 10 splits,
same GS-from-train construction, same HGB fits, same per-cell tie-break
streams seed*TIEBREAK_BASE + rep*17 + int(tf*1000)) but passes an
alternative confirmation threshold tau into E.run_task. Arms propagates
tau to confirmation, ScoreCost and ERCC planning at call time; the
optimistic selector reads arms.tau at call time as well. The SensoDat
reference set (observed failures, y == 1) is tau-free.

Everything else stays frozen: delta = 0.05, kappa = 5 HGB prior, budgets
5/10/20%, cost model, quantile rule (Beta.ppf(0.95; a, b)).

MUST be run with the original environment (anaconda python 3.12.7 /
sklearn 1.5.1) so the HGB priors are bit-identical; original_env_patch
pre-seeds the physical-core count (no wmic spawn).

Outputs (tag = tau<T>, e.g. tau0.20):
  results/tau_sensitivity/sensodat/sensodat_tau_sensitivity_<tag>.csv
  results/tau_sensitivity/sensodat/sensodat_tau_sensitivity_splits_<tag>.csv
  results/tau_sensitivity/sensodat/sensodat_tau_sensitivity_<tag>_meta.json
  logs/optimistic_ercc/sensodat/<tag>/decisions_split<S>_b<frac>_r<rep>.csv

Usage: python run_tau_sensitivity_sensodat.py TAU [budgets] [n_splits] [n_reps] [diag]
"""
import csv
import json
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


def run(tau, budget_fracs=(0.05, 0.10, 0.20), n_splits=S.N_SPLITS,
        n_reps=S.N_TIEBREAK_REPS, diag=True):
    tau = float(tau)
    if abs(tau - S.TAU) < 1e-12:
        raise SystemExit("tau=0.3 is the frozen main setting; use "
                         "scripts/run_optimistic_sensodat.py instead")
    from sklearn.ensemble import HistGradientBoostingClassifier
    tag = f"tau{tau:.2f}"
    outdir = os.path.join(NEW, "results", "tau_sensitivity", "sensodat")
    logdir = os.path.join(NEW, "logs", "optimistic_ercc", "sensodat", tag)
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)

    t_start = time.time()
    M = S.make_splits()
    X, y, dur = M["X"], M["y"], M["dur"]
    rows, split_rows = [], []
    print(f"tau sensitivity SensoDat: tau={tau} (frozen main is {S.TAU})",
          flush=True)

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
                sels = OS.make_sensodat_selectors(m_cap, diagnostics=diag)
                for pretty, mkey in OS.METHODS_7:
                    sf = sels[mkey]
                    rng = np.random.default_rng(
                        seed * S.TIEBREAK_BASE + rep * 17 + int(tf * 1000))
                    r = E.run_task(K, draw, gt_good, budget, scores, gs,
                                   tau, sf, rng, kappa=S.KAPPA)
                    r.update({"dataset": "sensodat", "split_id": sid,
                              "split_seed": seed, "rep": rep,
                              "method": pretty, "method_key": mkey,
                              "time_frac": tf, "budget_s": budget,
                              "m_cap": int(m_cap), "K": K,
                              "n_ref_good": n_good})
                    rows.append(r)
                    if diag and mkey == "ercc_optimistic" and hasattr(sf, "last_diag"):
                        _dump_diag(sf.last_diag, logdir, sid, tf, rep)
        el = time.time() - t_start
        print(f"  split {sid} (seed={seed}) done K={K} fails={n_good} "
              f"elapsed={el:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    p_main = os.path.join(outdir, f"sensodat_tau_sensitivity_{tag}.csv")
    df.to_csv(p_main, index=False)
    pd.DataFrame(split_rows).to_csv(
        os.path.join(outdir,
                     f"sensodat_tau_sensitivity_splits_{tag}.csv"),
        index=False)
    meta = {"tag": tag, "tau": tau, "budget_fracs": list(budget_fracs),
            "methods": [p for p, _ in OS.METHODS_7],
            "n_splits": n_splits, "n_reps": n_reps,
            "optimistic_delta": 0.05,
            "planning_rate": "Beta.ppf(0.95;a,b)"}
    with open(os.path.join(outdir,
                           f"sensodat_tau_sensitivity_{tag}_meta.json"),
              "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"wrote {p_main} ({len(df)} rows) elapsed "
          f"{time.time()-t_start:.0f}s", flush=True)
    return df


def _dump_diag(diag, logdir, sid, tf, rep):
    if not diag:
        return
    p = os.path.join(logdir, f"decisions_split{sid}_b{tf:.2f}_r{rep}.csv")
    cols = ["arm", "s", "n", "post_a", "post_b", "q_mean", "q_opt",
            "m_mean", "m_opt", "confirmed", "sel_kind", "would_differ"]
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for d in diag:
            w.writerow({k: d.get(k, "") for k in cols})


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    tau_ = float(sys.argv[1])
    budgets_ = (tuple(float(x) for x in sys.argv[2].split(","))
                if len(sys.argv) > 2 else (0.05, 0.10, 0.20))
    n_splits_ = int(sys.argv[3]) if len(sys.argv) > 3 else S.N_SPLITS
    n_reps_ = int(sys.argv[4]) if len(sys.argv) > 4 else S.N_TIEBREAK_REPS
    diag_ = (sys.argv[5] != "0") if len(sys.argv) > 5 else True
    run(tau_, budget_fracs=budgets_, n_splits=n_splits_, n_reps=n_reps_,
        diag=diag_)
