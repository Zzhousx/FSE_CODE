"""Resume CANNIER cells from checkpointed run tables and write a new log shard."""
from __future__ import annotations

import csv
import gzip
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import run_cannier as RC

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/cannier"


def atomic_csv(frame, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def key(project, seed, ratio, method):
    return (str(project), int(seed), float(ratio), str(method))


def main():
    start = time.perf_counter()
    runs_path = OUT / "runs.csv"
    oracle_path = OUT / "oracle.csv"
    confirm_path = OUT / "confirmability.csv"
    runs = pd.read_csv(runs_path) if runs_path.exists() else pd.DataFrame()
    oracles = pd.read_csv(oracle_path) if oracle_path.exists() else pd.DataFrame()
    confirms = pd.read_csv(confirm_path) if confirm_path.exists() else pd.DataFrame()
    done = set(key(r.project, r.seed, r.r, r.method)
               for r in runs.itertuples(index=False))
    oracle_done = set((str(r.project), int(r.seed), float(r.r))
                      for r in oracles.itertuples(index=False)) if len(oracles) else set()
    confirm_done = set((str(r.project), int(r.seed), float(r.r))
                       for r in confirms.itertuples(index=False)) if len(confirms) else set()

    shard_num = 1
    while (OUT / f"served_resume_{shard_num:02d}.csv.gz").exists():
        shard_num += 1
    log_path = OUT / f"served_resume_{shard_num:02d}.csv.gz"
    pool_dir = OUT / "pools"
    projects = sorted(p.stem for p in pool_dir.glob("*.npz"))
    print(f"resume: {len(done)} completed run cells; writing {log_path.name}", flush=True)

    with gzip.open(log_path, "wt", newline="", encoding="utf-8",
                   compresslevel=1) as logf:
        writer = csv.writer(logf)
        writer.writerow(["project", "seed", "r", "method", "item_id", "k", "y"])
        logf.flush()
        for project in projects:
            p = RC.pool_read(pool_dir / f"{project}.npz")
            for seed in RC.SEEDS:
                all_seed_keys = {
                    key(project, seed, ratio, method)
                    for ratio in RC.RATIOS for method in RC.METHOD_ORDER
                }
                if all_seed_keys.issubset(done):
                    continue

                bank = RC.SequenceBank(project, seed, p["item_id"], p["F"])
                bank.audit_all()
                need_oracle = any((project, int(seed), float(ratio)) not in oracle_done
                                  for ratio in RC.RATIOS)
                if need_oracle:
                    seqs = np.stack([bank.sequence(i) for i in range(len(p["item_id"]))])
                    prefix = seqs.cumsum(axis=1, dtype=np.int32)
                    nn = np.arange(1, 2501, dtype=np.float64)
                    width = np.sqrt(np.log(2.0 * nn * (nn + 1.0) / RC.DELTA)
                                    / (2.0 * nn))
                    crossing = prefix / nn[None, :] - width[None, :] > RC.TAU
                    has_crossed = crossing.any(axis=1)
                    first_k = np.where(has_crossed, crossing.argmax(axis=1) + 1, -1)
                    positives = int(np.sum(p["F"] / 2500 > RC.TAU))
                    never = int(np.sum((p["F"] / 2500 > RC.TAU) & ~has_crossed))
                for ratio in RC.RATIOS:
                    cell = (project, int(seed), float(ratio))
                    if cell not in oracle_done:
                        budget = ratio * float(p["B_pass"])
                        count, _ = RC.exact_oracle(p, project, seed, budget, bank,
                                                   first_k, never)
                        oracles = pd.concat([oracles, pd.DataFrame([{
                            "project": project, "seed": seed, "r": ratio,
                            "oracle_count": int(count),
                            "oracle_recall": count / positives if positives else 0.,
                        }])], ignore_index=True)
                        oracle_done.add(cell)
                        atomic_csv(oracles, oracle_path)
                    if cell not in confirm_done:
                        if need_oracle:
                            n_never = never
                        else:
                            row = oracles[(oracles.project == project)
                                          & (oracles.seed == seed)
                                          & (oracles.r == ratio)].iloc[0]
                            n_never = int(confirms.loc[
                                (confirms.project == project)
                                & (confirms.seed == seed)
                                & (confirms.r == ratio),
                                "n_positive_never_cross"].iloc[0]) if len(confirms) else 0
                        confirms = pd.concat([confirms, pd.DataFrame([{
                            "project": project, "seed": seed, "r": ratio,
                            "n_positive_never_cross": n_never,
                        }])], ignore_index=True)
                        confirm_done.add(cell)
                        atomic_csv(confirms, confirm_path)

                    missing = [m for m in RC.METHOD_ORDER
                               if key(project, seed, ratio, m) not in done]
                    if not missing:
                        continue
                    cap = int(math.ceil((ratio * p["B_pass"]) / p["cost"].mean()) + 1)
                    sels = RC.selectors(cap)
                    cell_served = {}
                    for method in missing:
                        sel = None if method == "HDoC-Sampling" else sels[method]
                        oracle_count = int(oracles.loc[
                            (oracles.project == project) & (oracles.seed == seed)
                            & (oracles.r == ratio), "oracle_count"].iloc[0])
                        result, served = RC.run_one(p, project, seed, ratio, method,
                                                    sel, oracle_count, bank)
                        writer.writerows((project, seed, ratio, method, item, k, y)
                                         for item, k, y in served)
                        logf.flush()
                        cell_served[method] = {(item, k): y for item, k, y in served}
                        rows = pd.DataFrame([result])
                        runs = pd.concat([runs, rows], ignore_index=True)
                        done.add(key(project, seed, ratio, method))
                        atomic_csv(runs, runs_path)
                        # The latest completed cells are durable after each method.
                        if len(runs) % 24 == 0:
                            print(f"checkpoint runs={len(runs)} last={project}/{seed}/"
                                  f"{ratio}/{method}", flush=True)
                    methods = list(cell_served)
                    for ai, a in enumerate(methods):
                        for b in methods[ai + 1:]:
                            common = cell_served[a].keys() & cell_served[b].keys()
                            if any(cell_served[a][k] != cell_served[b][k] for k in common):
                                raise AssertionError(
                                    f"realization mismatch: {project}/{seed}/{ratio}/{a}/{b}")

    df = pd.read_csv(runs_path)
    df.groupby(["project", "method", "r"], as_index=False).agg(
        {"confirmed_total": "mean", "confirmed_pos": "mean",
         "confirmed_pos_int": "mean", "false_conf": "mean", "recall": "mean",
         "recall_int": "mean", "n_exec": "mean", "B": "mean",
         "B_used": "mean", "overtime": "mean", "sched_cpu_s": "mean",
         "n_eliminated": "mean", "n_fallback": "mean",
         "oracle_count": "mean"}).to_csv(OUT / "project_means.csv", index=False)
    print(f"resume complete: runs={len(df)} elapsed={time.perf_counter()-start:.1f}s",
          flush=True)


if __name__ == "__main__":
    main()
