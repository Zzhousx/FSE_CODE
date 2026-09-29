"""Compute the hindsight ceiling when only intermittent positives count."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from replay_env import SequenceBank

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/cannier"
RATIOS = (.5, 1., 2., 4.)
SEEDS = range(2000, 2012)


def soft_budget_count(first_k, costs, budget):
    keep = first_k > 0
    confirmation_cost = first_k[keep] * costs[keep]
    last_cost = costs[keep]
    if not len(confirmation_cost):
        return 0
    order = np.argsort(confirmation_cost, kind="stable")
    best = 0
    for last in range(len(confirmation_cost)):
        threshold = budget - (confirmation_cost[last] - last_cost[last])
        spent = 0.0
        count = 1
        for i in order:
            if i == last:
                continue
            if spent + confirmation_cost[i] < threshold:
                spent += confirmation_cost[i]
                count += 1
            else:
                break
        best = max(best, count)
    return best


def main():
    n = np.arange(1, 2501, dtype=np.float64)
    width = np.sqrt(np.log(2.0 * n * (n + 1.0) / .05) / (2.0 * n))
    rows = []
    for file in sorted((OUT / "pools").glob("*.npz")):
        with np.load(file) as pool:
            ids = pool["item_id"]
            failures = pool["F"]
            cost = pool["cost"]
            B_pass = float(pool["B_pass"])
        subset = np.flatnonzero((failures / 2500 > .3) & (failures < 2500))
        for seed in SEEDS:
            bank = SequenceBank(file.stem, seed, ids, failures)
            first_k = np.full(len(subset), -1, dtype=np.int32)
            for j, i in enumerate(subset):
                sequence = bank.sequence(int(i))
                prefix = sequence.cumsum(dtype=np.int32)
                crossed = np.flatnonzero(prefix / n - width > .3)
                if crossed.size:
                    first_k[j] = int(crossed[0]) + 1
            for ratio in RATIOS:
                count = soft_budget_count(first_k, cost[subset], ratio * B_pass)
                rows.append({"project": file.stem, "seed": seed, "r": ratio,
                             "oracle_int_count": count,
                             "oracle_int_recall": count / len(subset) if len(subset) else 0.})
        print(file.stem, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "oracle_int.csv", index=False)
    print(f"wrote {len(rows)} intermittent oracle cells")


if __name__ == "__main__":
    main()
