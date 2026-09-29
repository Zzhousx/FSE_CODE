"""Canonical matched-unit statistics for the FSE revision, with t intervals."""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from exact_wilcoxon import exact_two_sided_wilcoxon
R = ROOT / "results"
OUT = R / "revision"
EXTERNAL = ("Uniform", "FailRerun", "APGAI", "ScoreCost", "RerunK-10", "BayesUCB-Cost")
NAME = {"ReproAlloc-Optimistic": "ReproAlloc", "ReproAlloc": "Mean Planning"}


def old_rows(path, benchmark, tau, main=True):
    d = pd.read_csv(path)
    d = d[d.method != "ReproAlloc-OA"].copy()
    if benchmark == "SensoDat":
        d = d.rename(columns={"split_id": "unit", "time_frac": "budget",
                              "tp": "count", "n_ref_good": "n_reference"})
    else:
        d = d.rename(columns={"seed": "unit", "budget_frac": "budget",
                              "tp": "count", "recall_refpos": "recall",
                              "n_refpos": "n_reference"})
        d["rep"] = 0
    d["benchmark"] = benchmark
    d["tau"] = tau
    d["method"] = d.method.map(lambda x: NAME.get(x, x))
    if not main:
        d = d[~d.budget.round(5).isin((.05, .10, .20))]
    return d[["benchmark", "unit", "rep", "budget", "tau", "method",
              "count", "recall", "n_reference"]]


def load_raw():
    files = []
    files += [old_rows(path, "SensoDat", .3) for path in sorted(
        (OUT / "t0_sensodat").glob("shard_*/sensodat_osc_check.csv"))]
    assert len(files) == 10
    files += [old_rows(OUT / "t0_amini_8/amini_osc_check.csv", "Amini", .3)]
    files += [old_rows(R / "02_budget_sweep/sensodat/sensodat_optimistic_sweep.csv",
                       "SensoDat", .3, main=False)]
    files += [old_rows(R / "02_budget_sweep/amini/amini_optimistic_sweep.csv",
                       "Amini", .3, main=False)]
    for tau in (.2, .4):
        tag = f"tau{tau:.2f}"
        files += [old_rows(R / f"tau_sensitivity/sensodat/sensodat_tau_sensitivity_{tag}.csv",
                           "SensoDat", tau)]
        files += [old_rows(R / f"tau_sensitivity/amini/amini_tau_sensitivity_{tag}.csv",
                           "Amini", tau)]
    for benchmark in ("sensodat", "amini"):
        paths = sorted((OUT / "T4" / benchmark).glob("split_*.csv")) if benchmark == "sensodat" \
            else [OUT / "T4/amini/raw.csv"]
        assert paths and all(path.exists() for path in paths)
        if benchmark == "sensodat":
            assert len(paths) == 10
        for path in paths:
            d = pd.read_csv(path)
            d = d[d.method.isin(("RerunK-10", "BayesUCB-Cost"))]
            files.append(d[["benchmark", "unit", "rep", "budget", "tau", "method",
                            "count", "recall", "n_reference"]])
    d = pd.concat(files, ignore_index=True)
    d["budget"] = d.budget.round(5)
    d["tau"] = d.tau.round(5)
    keys = ["benchmark", "unit", "rep", "budget", "tau", "method"]
    assert not d.duplicated(keys).any(), d[d.duplicated(keys, keep=False)].head()
    return d


def unit_means(raw):
    return raw.groupby(["benchmark", "unit", "budget", "tau", "method"],
                       as_index=False)[["count", "recall", "n_reference"]].mean()


def mean_halfwidth(values):
    x = np.asarray(values, dtype=float)
    if len(x) < 2:
        return float(x.mean()), 0.0
    return float(x.mean()), float(t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def paired(a, b):
    common = a.index.intersection(b.index)
    d = a.loc[common] - b.loc[common]
    mean, half = mean_halfwidth(d)
    p = exact_two_sided_wilcoxon(d)
    wins = int((d > 1e-12).sum())
    losses = int((d < -1e-12).sum())
    return dict(n_units=len(d), delta=mean, halfwidth=half,
                ci_low=mean - half, ci_high=mean + half,
                wilcoxon_p=p, wins=wins, losses=losses,
                ties=len(d) - wins - losses)


def build_main(raw):
    units = unit_means(raw)
    units.to_csv(OUT / "unit_means_rev.csv", index=False)
    summary = []
    for (bench, budget, tau, method), g in units.groupby(
            ["benchmark", "budget", "tau", "method"], sort=True):
        count, count_half = mean_halfwidth(g["count"])
        recall, recall_half = mean_halfwidth(g["recall"])
        summary.append(dict(benchmark=bench, budget=budget, tau=tau, method=method,
                            n_units=len(g), n_reference=g.n_reference.iloc[0],
                            count_mean=count, count_halfwidth=count_half,
                            recall_mean=recall, recall_halfwidth=recall_half,
                            recall_ci_low=recall - recall_half,
                            recall_ci_high=recall + recall_half))
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / "statistics_rev.csv", index=False)

    comparisons = []
    for (bench, budget, tau), cell in units.groupby(["benchmark", "budget", "tau"]):
        series = {name: g.set_index("unit")["recall"].sort_index()
                  for name, g in cell.groupby("method")}
        strongest = max((m for m in EXTERNAL if m in series),
                        key=lambda m: series[m].mean())
        repro = series["ReproAlloc"]
        for method, other in series.items():
            if method == "ReproAlloc":
                continue
            comparisons.append(dict(benchmark=bench, budget=budget, tau=tau,
                                    method="ReproAlloc", comparator=method,
                                    strongest_external=strongest,
                                    **paired(repro, other)))
    pairs = pd.DataFrame(comparisons)
    pairs.to_csv(OUT / "paired_rev.csv", index=False)
    return units, summary, pairs


def build_t1(summary):
    oracle = pd.read_csv(OUT / "T1_oracle_summary.csv")
    rows = []
    for _, o in oracle.iterrows():
        cell = summary[(summary.benchmark == o.benchmark) &
                       (summary.budget.round(5) == round(o.budget, 5)) &
                       (summary.tau == .3)]
        ours = cell[cell.method == "ReproAlloc"].iloc[0]
        ext = cell[cell.method.isin(EXTERNAL)].sort_values("recall_mean", ascending=False).iloc[0]
        row = o.to_dict()
        row.update(repro_count_mean=ours.count_mean,
                   repro_recall_mean=ours.recall_mean,
                   best_external=ext.method,
                   external_count_mean=ext.count_mean,
                   external_recall_mean=ext.recall_mean,
                   repro_percent_of_soft=100 * ours.recall_mean / o.oracle_soft_recall_mean)
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "T1_oracle.csv", index=False)


def build_t2_t3(units):
    extra = []
    for task in ("T2", "T3"):
        for benchmark in ("sensodat", "amini"):
            paths = sorted((OUT / task / benchmark).glob("split_*.csv")) if benchmark == "sensodat" \
                else [OUT / task / "amini/raw.csv"]
            for path in paths:
                extra.append(pd.read_csv(path))
    extra = unit_means(pd.concat(extra, ignore_index=True))
    base = units[(units.tau == .3) & units.budget.isin((.05, .10, .20)) &
                 units.method.isin(("ReproAlloc", "Mean Planning"))]
    all_units = pd.concat([base, extra], ignore_index=True)
    rows = []
    for (bench, budget, method), g in all_units.groupby(["benchmark", "budget", "method"]):
        m, h = mean_halfwidth(g.recall)
        c, ch = mean_halfwidth(g["count"])
        rows.append(dict(benchmark=bench, budget=budget, method=method,
                         recall_mean=m, recall_halfwidth=h,
                         count_mean=c, count_halfwidth=ch))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "T2_T3_variants.csv", index=False)
    pairs = []
    for bench in ("SensoDat", "Amini"):
        for budget in (.05, .10, .20):
            cell = all_units[(all_units.benchmark == bench) & (all_units.budget == budget)]
            ser = {m: g.set_index("unit").recall for m, g in cell.groupby("method")}
            for method, comparator in (("ReproAlloc", "MeanPlanning-uncapped"),
                                       ("ReproAlloc-uncapped", "MeanPlanning-uncapped")):
                pairs.append(dict(benchmark=bench, budget=budget, method=method,
                                  comparator=comparator, **paired(ser[method], ser[comparator])))
    pd.DataFrame(pairs).to_csv(OUT / "T2_uncapped_paired.csv", index=False)


def main():
    raw = load_raw()
    raw.to_csv(OUT / "all_raw_rev.csv", index=False)
    units, summary, pairs = build_main(raw)
    build_t1(summary)
    build_t2_t3(units)
    print(f"{len(raw)} run rows, {len(summary)} summary cells, {len(pairs)} paired cells")


if __name__ == "__main__":
    main()
