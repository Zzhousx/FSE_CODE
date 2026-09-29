"""Summarize GAI baselines against ReproAlloc at rho = 0.5."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from exact_wilcoxon import exact_two_sided_wilcoxon


def paired(new, old, key, value="recall"):
    joined = new.merge(old, on=key, suffixes=("_new", "_repro"), validate="one_to_one")
    diff = joined[f"{value}_repro"] - joined[f"{value}_new"]
    return (int((diff > 1e-12).sum()), int((diff < -1e-12).sum()),
            int((abs(diff) <= 1e-12).sum()), exact_two_sided_wilcoxon(diff))


def main():
    out = ROOT / "results" / "published_gai"
    new = pd.concat([pd.read_csv(out / "amini_runs.csv"),
                     pd.read_csv(out / "sensodat_runs.csv")], ignore_index=True)
    a = pd.read_csv(ROOT / "results/01_optimistic_core/amini/amini_optimistic_main.csv")
    s = pd.read_csv(ROOT / "results/01_optimistic_core/sensodat/sensodat_optimistic_main.csv")
    summaries = []
    comparisons = []
    for benchmark in ("Amini", "SensoDat"):
        baseline = a if benchmark == "Amini" else s
        baseline = baseline[baseline.method == "ReproAlloc-Optimistic"].copy()
        if benchmark == "Amini":
            baseline = baseline.rename(columns={"seed": "unit", "budget_frac": "budget",
                                                "recall_refpos": "recall", "tp": "count"})
            baseline["rep"] = 0
        else:
            baseline = baseline.rename(columns={"split_id": "unit", "time_frac": "budget",
                                                "tp": "count"})
        for frac in (.05, .10, .20):
            b = baseline[np.isclose(baseline.budget, frac)]
            for method in ("LUCB-G", "Murphy Sampling"):
                d = new[(new.benchmark == benchmark) &
                        np.isclose(new.budget_frac, frac) &
                        (new.method == method)]
                summaries.append({"benchmark": benchmark, "budget": frac, "method": method,
                                  "count_mean": d["count"].mean(),
                                  "recall_mean": d.recall.mean(), "false_conf": d.false_conf.sum(),
                                  "init_completed": d.init_completed.mean(),
                                  "n_runs": len(d)})
                if benchmark == "SensoDat":
                    d = d.groupby("unit", as_index=False)[["count", "recall"]].mean()
                    b = b.groupby("unit", as_index=False)[["count", "recall"]].mean()
                    key = ["unit"]
                else:
                    key = ["unit"]
                wins, losses, ties, p = paired(d, b, key)
                comparisons.append({"benchmark": benchmark, "budget": frac,
                                    "method": method, "wins_repro": wins,
                                    "losses_repro": losses, "ties": ties,
                                    "p_exact": p})
    cpath = out / "cannier_test_runs.csv"
    if cpath.exists():
        c = pd.read_csv(cpath)
        expected = 8 * 12 * 4 * 2
        if len(c) == expected:
            baseline = pd.read_csv(ROOT / "results/cannier/reproalloc_runs.csv")
            baseline = baseline[np.isclose(baseline.rho, .50)]
            for ratio in (.5, 1., 2., 4.):
                b = baseline[np.isclose(baseline.r, ratio)].groupby(
                    "project", as_index=False).recall.mean()
                for method in ("LUCB-G", "Murphy Sampling"):
                    d = c[np.isclose(c.r, ratio) & (c.method == method)]
                    summaries.append({"benchmark": "CANNIER", "budget": ratio,
                                      "method": method,
                                      "count_mean": d.groupby("project")["count"].mean().mean(),
                                      "recall_mean": d.groupby("project").recall.mean().mean(),
                                      "false_conf": d.false_conf.sum(),
                                      "init_completed": d.init_completed.mean(),
                                      "n_runs": len(d)})
                    d = d.groupby("project", as_index=False).recall.mean()
                    wins, losses, ties, p = paired(d, b, ["project"])
                    comparisons.append({"benchmark": "CANNIER", "budget": ratio,
                                        "method": method, "wins_repro": wins,
                                        "losses_repro": losses, "ties": ties,
                                        "p_exact": p})
        else:
            print(f"CANNIER incomplete: {len(c)}/{expected} cells")
    sd = pd.DataFrame(summaries)
    cp = pd.DataFrame(comparisons)
    sd.to_csv(out / "summary.csv", index=False)
    cp.to_csv(out / "paired.csv", index=False)
    print(sd.to_string(index=False))
    print(cp.to_string(index=False))


if __name__ == "__main__":
    main()
