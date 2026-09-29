"""Run LUCB-G and Murphy Sampling on matched SensoDat/Amini replay units."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(Path(__file__).resolve().parent)]

import sensodat_final_experiment as S
import v7_experiment as E
import experiment_amini_finite_trace as A
from run_hdoc_existing import BUDGET_INDEX, FixedReplay, AminiReplay
from gai_published_baselines import PublishedGAISampling

METHODS = ("LUCB-G", "Murphy Sampling")


def row(benchmark, unit, rep, frac, budget, ref, arms, method, result):
    tp = int(np.sum(arms.confirmed & ref))
    return {"benchmark": benchmark, "unit": unit, "rep": rep,
            "budget_frac": frac, "method": method, "count": tp,
            "recall": tp / int(np.sum(ref)),
            "confirmed_total": int(np.sum(arms.confirmed)),
            "false_conf": int(np.sum(arms.confirmed & ~ref)),
            "n_reference": int(np.sum(ref)), "B": budget,
            "n_candidates": arms.K, **result}


def run_sensodat(split_ids, fracs):
    S.DATA = str(ROOT / "data")
    data = S.make_splits()
    y, dur = data["y"], data["dur"]
    for sp in data["splits"]:
        unit = int(sp["split_id"])
        if unit not in split_ids:
            continue
        target = np.asarray(sp["target"])
        yt, dt = y[target], dur[target]
        k, seed = len(target), int(sp["split_seed"])
        ref = yt == 1
        for frac in fracs:
            budget = float(np.sum(dt)) * frac
            for rep in range(S.N_TIEBREAK_REPS):
                for method in METHODS:
                    rng = np.random.default_rng(np.random.SeedSequence(
                        [seed, BUDGET_INDEX[frac], rep, 947,
                         METHODS.index(method)]))
                    arms = E.Arms(k, None, E.GS(1., 1., 1.), .3,
                                  kappa=S.KAPPA)
                    result = PublishedGAISampling(method).run(
                        arms, FixedReplay(yt, dt), budget, rng, False)
                    yield row("SensoDat", unit, rep, frac, budget, ref,
                              arms, method, result)


def run_amini(seeds, fracs):
    A.TAU = .3
    routes = A.load_routes()
    total = sum(sum(r.durations) for r in routes)
    ref = np.asarray([r.p_hat > .3 for r in routes], dtype=bool)
    for seed in seeds:
        orders = A.make_orders(routes, seed)
        for frac in fracs:
            budget = total * frac
            for method in METHODS:
                rng = A.method_rng(seed, 12 + METHODS.index(method),
                                   BUDGET_INDEX[frac])
                arms = A.Arms(len(routes), A.GS(1., 1., 1.))
                result = PublishedGAISampling(method).run(
                    arms, AminiReplay(routes, orders), budget, rng, True)
                yield row("Amini", seed, 0, frac, budget, ref,
                          arms, method, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("amini", "sensodat", "both"),
                        default="both")
    parser.add_argument("--budgets", default="0.05,0.10,0.20")
    parser.add_argument("--splits", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--seeds", default=",".join(map(str, range(2000, 2012))))
    parser.add_argument("--output", default="published_gai_runs.csv")
    args = parser.parse_args()
    fracs = tuple(float(x) for x in args.budgets.split(","))
    if not fracs or any(x not in (.05, .10, .20) for x in fracs):
        parser.error("budgets must be drawn from 0.05,0.10,0.20")
    rows = []
    started = time.perf_counter()
    if args.benchmark in ("sensodat", "both"):
        rows.extend(run_sensodat(set(map(int, args.splits.split(","))), fracs))
    if args.benchmark in ("amini", "both"):
        rows.extend(run_amini(map(int, args.seeds.split(",")), fracs))
    outdir = ROOT / "results" / "published_gai"
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / args.output
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {len(rows)} rows to {out}; elapsed {time.perf_counter()-started:.1f}s")
    print(pd.DataFrame(rows).groupby(["benchmark", "budget_frac", "method"])
          [["count", "recall", "executions", "init_completed"]]
          .mean().to_string())


if __name__ == "__main__":
    main()
