# -*- coding: utf-8 -*-
"""Unified final statistics.

Recomputes every paper statistic from the raw matched experiment data in
this tree and writes results/final_paper_statistics.csv.

Experiments covered:
  main            main budgets 5/10/20%, tau = 0.3
                  (from results/01_optimistic_core/)
  dense_budget    budgets 3/5/6/7/8/10/15/20%, tau = 0.3
                  (from results/02_budget_sweep/)
  tau_sensitivity budgets 5/10/20%, tau in {0.2, 0.4}
                  (from results/tau_sensitivity/; tau = 0.3 cells live in
                  "main")

Anytime AUC and runtime overhead keep their own files
(results/anytime_auc/anytime_auc.csv,
results/runtime_overhead/runtime_overhead.csv).

Conventions (identical to the paper pipeline):
  naming: ReproAlloc-Optimistic -> ReproAlloc; ReproAlloc -> Mean Planning;
          ReproAlloc-OA excluded.
  units : Amini = seed (12); SensoDat = split (10, nested reps averaged).
  paired: each method vs ReproAlloc; the ReproAlloc row vs the strongest
          external of its cell. CI = mean +/- 1.96 sd/sqrt(n) on the paired
          differences; Wilcoxon signed-rank; wins/losses/ties at 1e-12.
"""
import math
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "final_paper_statistics.csv")

RENAME = {"ReproAlloc-Optimistic": "ReproAlloc", "ReproAlloc": "Mean Planning"}
METHODS = ["Uniform", "FailRerun", "ScoreCost", "APGAI",
           "Mean Planning", "ReproAlloc"]
EXTERNAL = ["Uniform", "FailRerun", "ScoreCost", "APGAI"]

FIELDS = ["experiment", "benchmark", "tau", "budget", "method", "n_units",
          "mean_recall", "ci_low", "ci_high", "paired_reference",
          "paired_delta", "paired_ci_low", "paired_ci_high", "wilcoxon_p",
          "wins", "losses", "ties"]


def paired(a, b):
    common = a.index.intersection(b.index)
    d = a[common] - b[common]
    n = len(d)
    m_ = float(d.mean())
    sd_ = float(d.std(ddof=1))
    ci = 1.96 * sd_ / math.sqrt(n) if sd_ == sd_ else 0.0
    try:
        p = float(wilcoxon(d).pvalue) if (d != 0).any() else 1.0
    except Exception:
        p = float("nan")
    wins = int((d > 1e-12).sum())
    losses = int((d < -1e-12).sum())
    return m_, m_ - ci, m_ + ci, p, wins, losses, n - wins - losses


def load_amini(path, tau, experiment, budget_filter=None):
    d = pd.read_csv(path)
    d = d[d.method != "ReproAlloc-OA"].copy()
    d["paper_method"] = d.method.map(lambda m: RENAME.get(m, m))
    if budget_filter is not None:
        d = d[d.budget_frac.isin(budget_filter)]
    units = {}
    for (pm, tf), g in d.groupby(["paper_method", "budget_frac"]):
        units[("amini", tau, float(tf), pm, experiment)] = \
            g.set_index("seed")["recall_refpos"].sort_index()
    return units


def load_sensodat(path, tau, experiment, budget_filter=None):
    d = pd.read_csv(path)
    d = d[d.method != "ReproAlloc-OA"].copy()
    d["paper_method"] = d.method.map(lambda m: RENAME.get(m, m))
    if budget_filter is not None:
        d = d[d.time_frac.isin(budget_filter)]
    per_split = (d.groupby(["paper_method", "time_frac", "split_id"],
                           as_index=False)["recall"].mean())
    units = {}
    for (pm, tf), g in per_split.groupby(["paper_method", "time_frac"]):
        units[("sensodat", tau, float(tf), pm, experiment)] = \
            g.set_index("split_id")["recall"].sort_index()
    return units


def main():
    units = {}
    # main (tau = 0.3, budgets 5/10/20)
    units.update(load_amini(os.path.join(
        RES, "01_optimistic_core", "amini", "amini_optimistic_main.csv"),
        0.3, "main", [0.05, 0.10, 0.20]))
    units.update(load_sensodat(os.path.join(
        RES, "01_optimistic_core", "sensodat", "sensodat_optimistic_main.csv"),
        0.3, "main", [0.05, 0.10, 0.20]))
    # dense budget (tau = 0.3, 8 budgets)
    units.update(load_amini(os.path.join(
        RES, "02_budget_sweep", "amini", "amini_optimistic_sweep.csv"),
        0.3, "dense_budget"))
    units.update(load_sensodat(os.path.join(
        RES, "02_budget_sweep", "sensodat", "sensodat_optimistic_sweep.csv"),
        0.3, "dense_budget"))
    # tau sensitivity (tau = 0.2 / 0.4, budgets 5/10/20)
    for tau, tag in [(0.2, "tau0.20"), (0.4, "tau0.40")]:
        p_am = os.path.join(RES, "tau_sensitivity", "amini",
                            f"amini_tau_sensitivity_{tag}.csv")
        p_sd = os.path.join(RES, "tau_sensitivity", "sensodat",
                            f"sensodat_tau_sensitivity_{tag}.csv")
        if os.path.exists(p_am):
            units.update(load_amini(p_am, tau, "tau_sensitivity",
                                    [0.05, 0.10, 0.20]))
        else:
            print(f"  [skip] {p_am}")
        if os.path.exists(p_sd):
            units.update(load_sensodat(p_sd, tau, "tau_sensitivity",
                                       [0.05, 0.10, 0.20]))
        else:
            print(f"  [skip] {p_sd}")

    rows = []
    keys = sorted({(b, t, tf, e) for (b, t, tf, _m, e) in units},
                  key=lambda x: (x[3], x[0], x[1], x[2]))
    for (bench, tau, tf, experiment) in keys:
        rp = units[(bench, tau, tf, "ReproAlloc", experiment)]
        ext_means = {m: float(units[(bench, tau, tf, m, experiment)].mean())
                     for m in EXTERNAL
                     if (bench, tau, tf, m, experiment) in units}
        strongest = max(ext_means, key=ext_means.get)
        for pm in METHODS:
            s = units.get((bench, tau, tf, pm, experiment))
            if s is None:
                continue
            n = len(s)
            mean = float(s.mean())
            sd_ = float(s.std(ddof=1))
            ci = 1.96 * sd_ / math.sqrt(n)
            if pm == "ReproAlloc":
                ref_series = units[(bench, tau, tf, strongest, experiment)]
                (d, lo, hi, p, w_, l_, t_) = paired(rp, ref_series)
                ref = strongest
            else:
                (d, lo, hi, p, w_, l_, t_) = paired(s, rp)
                ref = "ReproAlloc"
            rows.append({"experiment": experiment, "benchmark": bench,
                         "tau": tau, "budget": tf, "method": pm,
                         "n_units": n, "mean_recall": mean,
                         "ci_low": mean - ci, "ci_high": mean + ci,
                         "paired_reference": ref, "paired_delta": d,
                         "paired_ci_low": lo, "paired_ci_high": hi,
                         "wilcoxon_p": p, "wins": w_, "losses": l_,
                         "ties": t_})
    df = pd.DataFrame(rows, columns=FIELDS)
    df.to_csv(OUT, index=False, float_format="%.10g")
    print(f"wrote {OUT} ({len(df)} rows)")
    for experiment in ["main", "dense_budget", "tau_sensitivity"]:
        sub = df[df.experiment == experiment]
        if len(sub):
            print(f"  {experiment}: {len(sub)} rows, "
                  f"benchmarks={sorted(sub.benchmark.unique())}, "
                  f"taus={sorted(sub.tau.unique())}, "
                  f"budgets={sorted(sub.budget.unique())}")


if __name__ == "__main__":
    main()
