"""Compare uncapped ReproAlloc outcomes against its frozen capped runs."""

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "results" / "revision"

execution_differences = 0
for split in range(10):
    frozen = pd.read_csv(ROOT / f"t0_sensodat/shard_{split}/sensodat_osc_check.csv")
    frozen = frozen[frozen.method == "ReproAlloc-Optimistic"].rename(
        columns={"split_id": "unit", "time_frac": "budget", "tp": "count",
                 "time_used": "time_used_s"})
    changed = pd.read_csv(ROOT / f"T2/sensodat/split_{split}.csv")
    changed = changed[changed.method == "ReproAlloc-uncapped"]
    merged = frozen.merge(changed, on=["unit", "rep", "budget"],
                          suffixes=("_capped", "_uncapped"), validate="one_to_one")
    assert len(merged) == 9
    for col in ("count", "recall"):
        assert np.allclose(merged[f"{col}_capped"], merged[f"{col}_uncapped"],
                           rtol=0, atol=1e-8), (split, col)
    execution_differences += int((merged.executions_capped != merged.executions_uncapped).sum())

frozen = pd.read_csv(ROOT / "t0_amini_8/amini_osc_check.csv")
frozen = frozen[frozen.method == "ReproAlloc-Optimistic"].rename(
    columns={"seed": "unit", "budget_frac": "budget", "tp": "count",
             "recall_refpos": "recall"})
changed = pd.read_csv(ROOT / "T2/amini/raw.csv")
changed = changed[changed.method == "ReproAlloc-uncapped"]
merged = frozen.merge(changed, on=["unit", "budget"],
                      suffixes=("_capped", "_uncapped"), validate="one_to_one")
assert len(merged) == 36
for col in ("count", "recall"):
    assert np.allclose(merged[f"{col}_capped"], merged[f"{col}_uncapped"],
                       rtol=0, atol=1e-8), col
execution_differences += int((merged.executions_capped != merged.executions_uncapped).sum())
print("Uncapped ReproAlloc preserves confirmed counts and recall in 126 unit-budget runs; "
      f"execution counts differ in {execution_differences} rows")
