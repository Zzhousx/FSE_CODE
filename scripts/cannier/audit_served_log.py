"""Stream-audit served realization logs without loading the large CSV in memory."""
import csv
import gzip
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/cannier"
FINAL_LOG = OUT / "served.csv.gz"
SOURCES = ([FINAL_LOG] if FINAL_LOG.exists() else
           [OUT / "served_legacy.csv.gz"] + sorted(OUT.glob("served_resume_*.csv.gz")))


def all_rows():
    for source in SOURCES:
        with gzip.open(source, "rt", newline="", encoding="utf-8") as f:
            yield from csv.DictReader(f)


def main():
    event_count = pair_count = 0
    mismatch_count = 0
    cell_rows = []
    init_rows = []
    last_cell = None
    seen = {}
    checks_in_cell = 0
    mismatches_in_cell = 0
    init_cost = 0.0
    init_n = 0
    init_target = 0
    current_costs = {}
    for row in all_rows():
        cell = (row["project"], int(row["seed"]), float(row["r"]))
        if cell != last_cell:
            if last_cell is not None:
                cell_rows.append({"project": last_cell[0], "seed": last_cell[1],
                                  "r": last_cell[2], "checked_pairs": checks_in_cell,
                                  "mismatches": mismatches_in_cell})
                init_rows.append({"project": last_cell[0], "seed": last_cell[1],
                                  "r": last_cell[2], "init_exec": init_n,
                                  "n_candidates": init_target,
                                  "init_used": init_cost})
            last_cell = cell; seen = {}; checks_in_cell = 0; mismatches_in_cell = 0
            init_cost = 0.; init_n = 0
            poolfile = ROOT / "results/cannier/pools" / f"{cell[0]}.npz"
            with __import__("numpy").load(poolfile) as data:
                ids, costs = data["item_id"], data["cost"]
                init_target = len(ids)
                current_costs = {int(i): float(c) for i, c in zip(ids, costs)}
        key = (int(row["item_id"]), int(row["k"]))
        y = int(row["y"])
        prior = seen.get(key)
        if prior is None:
            seen[key] = (y, 1)
        else:
            prior_y, methods = prior
            if prior_y != y:
                mismatch_count += 1
                mismatches_in_cell += 1
            checks_in_cell += methods
            pair_count += methods
            seen[key] = (prior_y, methods + 1)
        event_count += 1
        if row["method"] == "HDoC-Sampling" and init_n < init_target:
            init_cost += current_costs[int(row["item_id"])]
            init_n += 1
    if last_cell is not None:
        cell_rows.append({"project": last_cell[0], "seed": last_cell[1],
                          "r": last_cell[2], "checked_pairs": checks_in_cell,
                          "mismatches": mismatches_in_cell})
        init_rows.append({"project": last_cell[0], "seed": last_cell[1],
                          "r": last_cell[2], "init_exec": init_n,
                          "n_candidates": init_target, "init_used": init_cost})
    pd.DataFrame(cell_rows).to_csv(OUT / "realization_audit.csv", index=False)
    init_df = pd.DataFrame(init_rows)
    init_df["B"] = init_df.r * init_df.project.map(
        pd.read_csv(OUT / "pool_stats.csv").set_index("project").B_pass)
    init_df["init_share_of_budget"] = init_df.init_used / init_df.B
    init_df.to_csv(OUT / "hdoc_init_audit.csv", index=False)
    runs = pd.read_csv(OUT / "runs.csv")
    expected = int(runs.n_exec.sum())
    print(f"events={event_count}; expected_from_runs={expected}; "
          f"checked_method_pairs={pair_count}; mismatches={mismatch_count}")
    if event_count != expected or mismatch_count:
        raise SystemExit("served log audit failed")


if __name__ == "__main__": main()
