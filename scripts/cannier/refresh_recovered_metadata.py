"""Recompute selector-only diagnostics omitted while recovering a partial log."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

import run_cannier as RC

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/cannier"


def main():
    path = OUT / "runs.csv"
    runs = pd.read_csv(path)
    todo = runs[((runs.method == "HDoC-Sampling") & runs.n_eliminated.isna())
                | (runs.method.isin(["ReproAlloc", "ReproAlloc (uncapped)"])
                   & runs.n_fallback.isna())]
    if todo.empty:
        print("no recovered metadata to refresh")
        return
    pool_cache = {}
    bank_cache = {}
    selector_cache = {}
    for idx, row in todo.iterrows():
        project, seed, ratio, method = (str(row.project), int(row.seed),
                                        float(row.r), str(row.method))
        if project not in pool_cache:
            pool_cache[project] = RC.pool_read(OUT / "pools" / f"{project}.npz")
        p = pool_cache[project]
        seed_key = (project, seed)
        if seed_key not in bank_cache:
            bank = RC.SequenceBank(project, seed, p["item_id"], p["F"])
            bank.audit_all()
            bank_cache[seed_key] = bank
        bank = bank_cache[seed_key]
        cell_key = (project, seed, ratio)
        if cell_key not in selector_cache:
            cap = int(math.ceil((ratio * p["B_pass"]) / p["cost"].mean()) + 1)
            selector_cache[cell_key] = RC.selectors(cap)
        selector = selector_cache[cell_key].get(method)
        if method == "HDoC-Sampling":
            selector = None
        result, _served = RC.run_one(p, project, seed, ratio, method, selector,
                                     int(row.oracle_count), bank)
        runs.loc[idx, "sched_cpu_s"] = result["sched_cpu_s"]
        if method == "HDoC-Sampling":
            runs.loc[idx, "n_eliminated"] = result["n_eliminated"]
        else:
            runs.loc[idx, "n_fallback"] = result["n_fallback"]
        runs.to_csv(path.with_suffix(".tmp"), index=False)
        path.with_suffix(".tmp").replace(path)
        if (idx + 1) % 12 == 0:
            print(f"refreshed {idx + 1}/{len(todo)} metadata rows", flush=True)
    runs.groupby(["project", "method", "r"], as_index=False).agg(
        {"confirmed_total": "mean", "confirmed_pos": "mean",
         "confirmed_pos_int": "mean", "false_conf": "mean", "recall": "mean",
         "recall_int": "mean", "n_exec": "mean", "B": "mean",
         "B_used": "mean", "overtime": "mean", "sched_cpu_s": "mean",
         "n_eliminated": "mean", "n_fallback": "mean",
         "oracle_count": "mean"}).to_csv(OUT / "project_means.csv", index=False)
    print(f"refreshed {len(todo)} recovered rows")


if __name__ == "__main__":
    main()
