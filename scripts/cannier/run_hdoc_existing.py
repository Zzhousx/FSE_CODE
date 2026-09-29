"""Add HDoC-Sampling to the unchanged SensoDat and Amini experiment units."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sensodat_final_experiment as S
import v7_experiment as E
import experiment_amini_finite_trace as A
from hdoc_sampling import HDoCSampling

BUDGET_INDEX = {.05: 0, .10: 1, .20: 2, .40: 3}


class FixedReplay:
    def __init__(self, y, c):
        self.y, self.c = np.asarray(y, int), np.asarray(c, float)

    def execute(self, i):
        return int(self.y[i]), float(self.c[i])

    def has_remaining(self, i):
        return True


class AminiReplay:
    def __init__(self, routes, orders):
        self.pools = [A.Pool(r.outcomes, r.durations, o)
                      for r, o in zip(routes, orders)]

    def has_remaining(self, i):
        return not self.pools[int(i)].exhausted()

    def execute(self, i):
        return self.pools[int(i)].draw()


def _row(benchmark, unit, rep, budget, reference, arms, result):
    conf = arms.confirmed
    ref = np.asarray(reference, dtype=bool)
    tp = int((conf & ref).sum()); fp = int((conf & ~ref).sum())
    return {"benchmark": benchmark, "unit": unit, "rep": rep,
            "budget": budget, "tau": .3, "method": "HDoC-Sampling",
            "count": tp, "recall": tp / int(ref.sum()) if ref.sum() else 0.,
            "n_reference": int(ref.sum()), "confirmed_total": int(conf.sum()),
            "false_conf": fp, "n_exec": result["executions"],
            "B": budget, "B_used": result["B_used"],
            "overtime": result["overtime"], "sched_cpu_s": result["sched_cpu_s"],
            "n_eliminated": result["n_eliminated"],
            "init_completed": result["init_completed"],
            "init_exec": result["init_exec"], "init_used": result["init_used"]}


def run_sensodat(budget_fracs=(.05, .10, .20)):
    S.DATA = str(ROOT / "data")
    M = S.make_splits(); y, dur = M["y"], M["dur"]
    rows, init_rows = [], []
    for sp in M["splits"]:
        train, target = np.asarray(sp["train"]), np.asarray(sp["target"])
        seed, unit = int(sp["split_seed"]), int(sp["split_id"])
        # HDoC has no prior or cost term; neutral state defaults are unused by it.
        gs = E.GS(1., 1., 1.)
        yt, dt = y[target], dur[target]
        K = len(target); ref = yt == 1; budget_base = float(dt.sum())
        for frac in budget_fracs:
            bi = BUDGET_INDEX[frac]
            B = budget_base * frac
            for rep in range(S.N_TIEBREAK_REPS):
                rng = np.random.default_rng(np.random.SeedSequence(
                    [seed, bi, rep, 731]))
                arms = E.Arms(K, None, gs, .3, kappa=S.KAPPA)
                res = HDoCSampling(K).run(arms, FixedReplay(yt, dt), B, rng,
                                          finite_pool=False)
                rows.append(_row("SensoDat", unit, rep, B, ref, arms, res)
                            | {"budget_frac": frac, "split_seed": seed,
                               "n_candidates": K, "init_share_of_budget":
                               (res["init_used"] / B if B else 0.)})
                init_rows.append({"benchmark": "SensoDat", "unit": unit,
                                  "r": frac, "init_completed": res["init_completed"],
                                  "n_eliminated": res["n_eliminated"],
                                  "init_share_of_budget": res["init_used"] / B if B else 0.})
    return rows, init_rows


def run_amini(budget_fracs=(.05, .10, .20)):
    A.TAU = .3
    routes = A.load_routes(); gs = A.GS(1., 1., 1.)
    total = sum(sum(r.durations) for r in routes)
    rows, init_rows = [], []
    for seed in range(2000, 2012):
        orders = A.make_orders(routes, seed)
        ref = np.array([r.p_hat > .3 for r in routes])
        for frac in budget_fracs:
            bi = BUDGET_INDEX[frac]
            B = total * frac
            # Index 8 is reserved for the appended baseline, preserving all frozen method RNGs.
            rng = A.method_rng(seed, 8, bi)
            arms = A.Arms(len(routes), gs)
            env = AminiReplay(routes, orders)
            res = HDoCSampling(len(routes)).run(arms, env, B, rng, finite_pool=True)
            rows.append(_row("Amini", seed, 0, B, ref, arms, res)
                        | {"unit_seed": seed, "budget_frac": frac,
                           "n_candidates": len(routes), "init_share_of_budget":
                           (res["init_used"] / B if B else 0.)})
            init_rows.append({"benchmark": "Amini", "unit": seed, "r": frac,
                              "init_completed": res["init_completed"],
                              "n_eliminated": res["n_eliminated"],
                              "init_share_of_budget": res["init_used"] / B if B else 0.})
    return rows, init_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-fracs", default="0.05,0.10,0.20")
    parser.add_argument("--output-suffix", default="")
    args = parser.parse_args()
    budget_fracs = tuple(float(x) for x in args.budget_fracs.split(","))
    if not budget_fracs or any(x not in BUDGET_INDEX for x in budget_fracs):
        parser.error(f"budget fractions must be in {tuple(BUDGET_INDEX)}")
    if args.output_suffix and not args.output_suffix.replace("_", "").isalnum():
        parser.error("output suffix must contain only letters, digits, and underscores")
    started = time.perf_counter()
    out = ROOT / "results/hdoc"; out.mkdir(parents=True, exist_ok=True)
    sr, si = run_sensodat(budget_fracs); ar, ai = run_amini(budget_fracs)
    df = pd.DataFrame(sr + ar)
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    df.to_csv(out / f"sensodat_amini{suffix}.csv", index=False)
    pd.DataFrame(si + ai).to_csv(out / f"init_elim_existing{suffix}.csv", index=False)
    print(f"wrote {len(df)} HDoC rows in {time.perf_counter()-started:.1f}s")
    print(df.groupby(["benchmark", "budget_frac"])[["count", "recall", "n_exec",
          "n_eliminated", "init_completed"]].mean().to_string())


if __name__ == "__main__":
    main()
