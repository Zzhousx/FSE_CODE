"""Check diagnostic wrappers preserve the frozen primary-run outcomes."""

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "results" / "revision"
pieces = []
for split in range(10):
    old = pd.read_csv(ROOT / f"t0_sensodat/shard_{split}/sensodat_osc_check.csv")
    old = old[old.method.isin(("ReproAlloc", "ReproAlloc-Optimistic"))].copy()
    old = old.rename(columns={"split_id": "unit", "time_frac": "budget",
                              "tp": "count", "time_used": "time_used_s"})
    old["method"] = old.method.map({"ReproAlloc": "Mean Planning",
                                     "ReproAlloc-Optimistic": "ReproAlloc"})
    new = pd.read_csv(ROOT / f"T2diag/sensodat/split_{split}.csv")
    keys = ["unit", "rep", "method", "budget"]
    merged = old.merge(new, on=keys, suffixes=("_old", "_new"), validate="one_to_one")
    assert len(merged) == len(new) == 18
    for col in ("count", "recall", "executions", "time_used_s"):
        assert np.allclose(merged[f"{col}_old"], merged[f"{col}_new"],
                           rtol=0, atol=1e-8), (split, col)
    pieces.append(len(merged))

old = pd.read_csv(ROOT / "t0_amini_8/amini_osc_check.csv")
old = old[old.method.isin(("ReproAlloc", "ReproAlloc-Optimistic"))].copy()
old = old.rename(columns={"seed": "unit", "budget_frac": "budget", "tp": "count",
                          "recall_refpos": "recall"})
old["rep"] = 0
old["method"] = old.method.map({"ReproAlloc": "Mean Planning",
                                 "ReproAlloc-Optimistic": "ReproAlloc"})
new = pd.read_csv(ROOT / "T2diag/amini/raw.csv")
merged = old.merge(new, on=["unit", "rep", "method", "budget"],
                   suffixes=("_old", "_new"), validate="one_to_one")
assert len(merged) == len(new) == 72
for col in ("count", "recall", "executions", "time_used_s"):
    assert np.allclose(merged[f"{col}_old"], merged[f"{col}_new"],
                       rtol=0, atol=1e-8), col
print(f"Diagnostic default parity: {sum(pieces)} SensoDat + {len(merged)} Amini rows matched")
