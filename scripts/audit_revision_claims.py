"""Summarize strict mean winners and exact paired values from revision tables."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "revision"
stats = pd.read_csv(OUT / "statistics_rev.csv")
online = ("Uniform", "FailRerun", "ScoreCost", "APGAI", "RerunK-10",
          "BayesUCB-Cost", "ReproAlloc")
primary = stats[stats.budget.isin((.05, .10, .20)) & stats.method.isin(online)]
pivot = primary.pivot(index=["benchmark", "budget", "tau"],
                      columns="method", values="recall_mean")
pivot["best_other"] = pivot.drop(columns="ReproAlloc").max(axis=1)
pivot["strict_win"] = pivot.ReproAlloc > pivot.best_other + 1e-12
pivot["tie"] = (pivot.ReproAlloc - pivot.best_other).abs() <= 1e-12
print("Mean winners over 18 threshold-budget cells (six external baselines):")
print(pivot.groupby(level="benchmark")[["strict_win", "tie"]].sum().to_string())
print("strict wins:", int(pivot.strict_win.sum()),
      "ties:", int(pivot.tie.sum()),
      "lower mean:", int((~pivot.strict_win & ~pivot.tie).sum()))

pairs = pd.read_csv(OUT / "paired_rev.csv")
primary_pairs = pairs[(pairs.budget.isin((.05, .10, .20))) & (pairs.tau == .3)]
primary_pairs = primary_pairs[primary_pairs.comparator.isin((
    "FailRerun", "ScoreCost", "APGAI", "RerunK-10", "BayesUCB-Cost"))]
print("\nPrimary BayesUCB-Cost matched means and exact p-values:")
print(primary_pairs[primary_pairs.comparator == "BayesUCB-Cost"][
    ["benchmark", "budget", "delta", "wilcoxon_p", "wins", "losses", "ties"]
].to_string(index=False))
print("\nPrimary table 3 strongest external results:")
print(primary_pairs[primary_pairs.comparator == primary_pairs.strongest_external][
    ["benchmark", "budget", "comparator", "delta", "wilcoxon_p", "wins", "losses", "ties"]
].to_string(index=False))
print("\nUncapped Mean Planning exact comparisons:")
print(pd.read_csv(OUT / "T2_uncapped_paired.csv").query(
    "method == 'ReproAlloc'")[
        ["benchmark", "budget", "delta", "wilcoxon_p", "wins", "losses", "ties"]
    ].to_string(index=False))
