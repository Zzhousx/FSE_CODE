"""Run the Optimistic-ERCC experiment on Amini-TransFuser (finite-trace).

Seven methods per (seed, budget): the frozen six (identical objects, identical
matched design as the frozen full run) + ReproAlloc-Optimistic.

Outputs
  results/01_optimistic_core/amini/amini_optimistic_<tag>.csv       (run metrics)
  results/01_optimistic_core/amini/amini_optimistic_alloc_<tag>.csv (route-level)
  logs/optimistic_ercc/amini/<tag>/decisions_seed<S>_b<frac>.csv   (optimistic
                                                                    decision log)

The frozen six MUST reproduce results/00_baseline_reproduction/amini_raw
exactly - verified afterwards by scripts/check_frozen_overlap.py.

Usage: python run_optimistic_amini.py [tag] [budgets] [n_seeds] [diag]
  tag     default "main"
  budgets default "0.05,0.10,0.20"  (comma list; canonical fracs keep the
          frozen stream indices so overlap cells stay matched)
  n_seeds default 12 (seeds 2000..)
  diag    default 1 (write decision logs for the optimistic method)
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
NEW = os.path.dirname(HERE)
sys.path.insert(0, SRC)

import experiment_amini_finite_trace as A  # noqa: E402
import optimistic_ercc_amini as OA  # noqa: E402

# canonical budget->index mapping preserves the frozen method_rng streams
BUDGET_INDEX = {0.05: 0, 0.10: 1, 0.20: 2,
                0.03: 3, 0.06: 4, 0.07: 5, 0.08: 6, 0.15: 7,
                0.40: 8}


def run(tag="main", budget_fracs=(0.05, 0.10, 0.20), n_seeds=12, diag=True):
    outdir = os.path.join(NEW, "results", "01_optimistic_core", "amini")
    logdir = os.path.join(NEW, "logs", "optimistic_ercc", "amini", tag)
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)

    routes = A.load_routes()
    gs = A.build_gs(routes)
    K = len(routes)
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
            sels = OA.build_selectors_7(m_cap, diagnostics=diag)
            for mi, sel in enumerate(sels):
                rng = A.method_rng(sd, mi, bi)
                res = A.run_one(routes, orders, ref_pos_mask, budget, sel,
                                rng, gs, ceiling)
                alloc = res.pop("_alloc")
                rec = {"tag": tag, "seed": sd, "budget_frac": tf,
                       "budget_s": budget, "method": sel.name,
                       "m_cap": m_cap, "ceiling_seed": ceiling,
                       "n_refpos": n_ref}
                rec.update(res)
                rows.append(rec)
                for (rid, ni, si, L, cf, kf, cap, left, rp) in alloc:
                    alloc_rows.append({"tag": tag, "seed": sd,
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
                    _dump_diag(sel.inner.last_diag, logdir, sd, tf, tag)
            print(f"  seed {sd} budget {tf}: done", flush=True)
        print(f"seed {sd} complete (ceiling={ceiling})", flush=True)

    main_cols = ["tag", "seed", "budget_frac", "budget_s", "method", "m_cap",
                 "ceiling_seed", "n_refpos", "tp", "fp", "fn", "confirmed",
                 "recall_refpos", "precision", "recall_ceiling_seed",
                 "nonsticky_confirmed", "executions", "time_used_s",
                 "budget_util", "overtime_s", "overrun_flag",
                 "pools_exhausted", "runs_left_in_pools",
                 "stopped_by_exhaustion", "sched_overhead_s"]
    p_main = _wcsv(os.path.join(outdir, f"amini_optimistic_{tag}.csv"),
                   rows, main_cols)
    p_alloc = _wcsv(os.path.join(outdir, f"amini_optimistic_alloc_{tag}.csv"),
                    alloc_rows)
    meta = {"tag": tag, "seeds": seeds, "budget_fracs": list(budget_fracs),
            "methods": [s.name for s in OA.build_selectors_7(1, False)],
            "tau": A.TAU, "delta": A.DELTA,
            "optimistic_delta": 0.05, "planning_rate": "Beta.ppf(0.95;a,b)",
            "total_duration_s": total_dur}
    with open(os.path.join(outdir, f"amini_optimistic_{tag}_meta.json"), "w",
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


def _dump_diag(diag, logdir, seed, tf, tag):
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
    tag = sys.argv[1] if len(sys.argv) > 1 else "main"
    budgets = (tuple(float(x) for x in sys.argv[2].split(","))
               if len(sys.argv) > 2 else (0.05, 0.10, 0.20))
    n_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    diag = (sys.argv[4] != "0") if len(sys.argv) > 4 else True
    run(tag=tag, budget_fracs=budgets, n_seeds=n_seeds, diag=diag)
