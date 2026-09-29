"""Part C - FINAL SensoDat-replay experiment.

Statistical redesign (the ONLY change vs the frozen v8 protocol):

  before : 3 train/target splits x 10 tie-break repetitions  -> n=30 "units"
  after  : 10 INDEPENDENT train/target splits                -> n=10 units

The 10 repetitions on one split were never independent statistical units:
SensoDat-replay is deterministic (one recorded outcome+duration per scenario),
so the only thing a repetition changes is scheduler tie-breaking. Treating them
as independent inflated n from 3 to 30 and produced implausibly tight intervals
(e.g. .074 [.074,.075]).

New design, per instruction Part C:
  * 10 independently generated train/target splits (fresh 75/25 train/history
    draw over the source generator + fresh classifier fit per split);
  * the source/target information boundary is preserved exactly;
  * no target label and no target duration is used for scheduling or for the
    global cost scalars (GS comes from the TRAIN split only);
  * the same split is used for all six compared methods;
  * R=3 internal tie-break repetitions are run per split and AVERAGED WITHIN
    the split -> nested repetitions, never top-level units;
  * budgets 5% / 10% / 20% of total target replay duration;
  * tau=0.3, delta=0.05, kappa=5.0, manuscript Hoeffding CS, ERCC definitions
    and the whole selector set are imported UNCHANGED from the frozen modules.

Nothing about the method is new. No new variant, threshold, CS or dataset.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reproalloc_v6 as v6
import v7_experiment as E          # frozen Arms / GS / selectors / run_task / m_cap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "results", "00_baseline_reproduction", "sensodat_raw")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- frozen knobs
TAU = 0.3
DELTA = 0.05
KAPPA = E.KAPPA                    # 5.0, unchanged
BUDGET_FRACS = [0.05, 0.10, 0.20]
SOURCE_GEN, TARGET_GEN = 0, 1      # AmbieGen -> Frenetic (frozen direction)
GEN_SD = {0: "AmbieGen", 1: "Frenetic", 2: "FreneticV"}
TRAIN_FRAC = 0.75                  # matches the frozen pipeline (9377/12496)

N_SPLITS = 10
SPLIT_SEEDS = [20261001 + i for i in range(N_SPLITS)]   # fresh, independent
N_TIEBREAK_REPS = 3                # nested within split, averaged

METHODS = [("Uniform", "uniform"),
           ("FailRerun", "failrerun"),
           ("APGAI", "apgai"),
           ("ScoreCost", "scorecost"),
           ("ReproAlloc", "ercc_mean"),
           ("ReproAlloc-OA", "ercc_outcome")]

# Deterministic per-method stream offsets are deliberately NOT used. The frozen
# v7/v8 protocol hands every method the SAME tie-break stream within a
# (split, budget, rep) cell so that the paired design is matched and any
# difference is attributable to the scheduling policy alone.
# Note: never use hash() for seeding -- Python randomizes str hashing per
# process (PYTHONHASHSEED), which would make results irreproducible.
TIEBREAK_BASE = 1000               # same constant as v8_final.py


def make_sels(m_cap):
    """The frozen selector set, same objects as v8_final.py."""
    return {"uniform": E.FnSel("uniform"),
            "failrerun": E.FnSel("failrerun"),
            "apgai": E.FnSel("apgai"),
            "scorecost": E.ScoreCostSel(False),
            "ercc_mean": E.ERCCSel(False, m_cap),
            "ercc_outcome": E.ERCCSel(True, m_cap)}


def make_splits():
    """10 independent train/target splits.

    target  = every scenario of the target generator (fixed pool, as frozen);
    train   = an independent 75% draw over the source generator;
    history = the complementary 25% (held out, never used).

    Independence comes from the freshly drawn train subsample and the fresh
    classifier fit, i.e. from the source-domain prior that drives scheduling.
    """
    d = np.load(os.path.join(DATA, "sensodat_fresh.npz"), allow_pickle=True)
    X = d["X_spec"]
    y = d["y"].astype(int)
    gen = d["generator"].astype(int)
    dur_raw = d["duration_s"].astype(float)
    dur = np.where(np.isnan(dur_raw) | (dur_raw <= 0),
                   np.nanmedian(dur_raw), dur_raw).astype(float)

    src_idx = np.flatnonzero(gen == SOURCE_GEN)
    tgt_idx = np.flatnonzero(gen == TARGET_GEN)

    splits = []
    for k, seed in enumerate(SPLIT_SEEDS):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(src_idx)
        n_tr = int(round(TRAIN_FRAC * len(perm)))
        train = np.sort(perm[:n_tr])
        history = np.sort(perm[n_tr:])
        assert len(set(train.tolist()) & set(tgt_idx.tolist())) == 0
        splits.append({"split_id": k, "split_seed": int(seed),
                       "source": GEN_SD[SOURCE_GEN], "target": GEN_SD[TARGET_GEN],
                       "train": train.tolist(), "history": history.tolist(),
                       "target": np.sort(tgt_idx).tolist()})
    meta = {"X": X, "y": y, "dur": dur, "splits": splits}
    return meta


def run(n_splits=N_SPLITS, n_reps=N_TIEBREAK_REPS, tag="final"):
    from sklearn.ensemble import HistGradientBoostingClassifier
    t_start = time.time()
    print("=" * 72, flush=True)
    print(f"### Part C SensoDat FINAL: {n_splits} independent splits "
          f"x {n_reps} nested tie-break reps", flush=True)
    print("=" * 72, flush=True)

    M = make_splits()
    X, y, dur = M["X"], M["y"], M["dur"]
    rows, split_rows = [], []

    for sp in M["splits"][:n_splits]:
        sid, seed = sp["split_id"], sp["split_seed"]
        train_idx = np.array(sp["train"])
        target_idx = np.array(sp["target"])

        ytr, dtr = y[train_idx], dur[train_idx]
        # global cost scalars from TRAIN ONLY -> no target duration leakage
        gs = E.GS(float(dtr[ytr == 1].mean()) if (ytr == 1).any() else float(dtr.mean()),
                  float(dtr[ytr == 0].mean()) if (ytr == 0).any() else float(dtr.mean()),
                  float(dtr.mean()))

        clf = HistGradientBoostingClassifier(random_state=seed, max_iter=200)
        clf.fit(X[train_idx], ytr)
        # prior uses target FEATURES only, never target labels
        scores = clf.predict_proba(X[target_idx])[:, 1]

        y_t, d_t = y[target_idx], dur[target_idx]
        K = len(target_idx)
        gt_good = (y_t == 1)
        n_good = int(gt_good.sum())
        total_time = float(d_t.sum())

        def draw(i, rng, _yt=y_t, _dt=d_t):
            # deterministic replay of the single recorded outcome/duration
            return int(_yt[i]), float(_dt[i])

        split_rows.append({
            "split_id": sid, "split_seed": seed,
            "source": sp["source"], "target": sp["target"],
            "n_train": len(train_idx), "n_history": len(sp["history"]),
            "K_target": K, "n_target_failures": n_good,
            "target_failure_rate": n_good / K,
            "total_target_time_s": total_time,
            "gs_mean_fail": gs.mean_fail_cost, "gs_mean_pass": gs.mean_pass_cost,
            "gs_mean_overall": gs.mean_overall,
        })

        for tf in BUDGET_FRACS:
            budget = total_time * tf
            m_cap = E.m_cap_from_budget(budget, gs)
            for rep in range(n_reps):
                sels = make_sels(m_cap)
                for pretty, mkey in METHODS:
                    sf = sels[mkey]
                    # Frozen v8 tie-break stream: identical for every method
                    # inside one (split, budget, rep) cell, so the design stays
                    # matched and differences come only from the policy.
                    rng = np.random.default_rng(
                        seed * TIEBREAK_BASE + rep * 17 + int(tf * 1000))
                    r = E.run_task(K, draw, gt_good, budget, scores, gs,
                                   TAU, sf, rng, kappa=KAPPA)
                    r.update({"dataset": "sensodat", "split_id": sid,
                              "split_seed": seed, "source": sp["source"],
                              "target": sp["target"], "rep": rep,
                              "method": pretty, "method_key": mkey,
                              "time_frac": tf, "budget_s": budget,
                              "m_cap": int(m_cap), "K": K,
                              "n_ref_good": n_good})
                    rows.append(r)
        el = time.time() - t_start
        print(f"  split {sid} (seed={seed}) done  K={K} fails={n_good} "
              f"elapsed={el:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    sm = pd.DataFrame(split_rows)
    csv_path = os.path.join(OUT, f"sensodat_raw_{tag}.csv")
    df.to_csv(csv_path, index=False)
    sm.to_csv(os.path.join(OUT, f"sensodat_splits_{tag}.csv"), index=False)
    with open(os.path.join(OUT, f"sensodat_splits_{tag}.json"), "w",
              encoding="utf-8") as fh:
        json.dump([{k: v for k, v in s.items()} for s in M["splits"][:n_splits]], fh)
    print(f"\nsaved {len(df)} rows -> {csv_path}", flush=True)
    print(f"total elapsed {time.time()-t_start:.0f}s", flush=True)
    return df


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if mode == "smoke":
        run(n_splits=2, n_reps=1, tag="smoke")
    elif mode == "final":
        run(n_splits=N_SPLITS, n_reps=N_TIEBREAK_REPS, tag="final")
    else:
        raise SystemExit(f"unknown mode {mode}")
