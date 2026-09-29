"""Regenerate the requested CSV/TeX tables from revision result files."""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "revision"
TABLE = ROOT.parent / "paper_reproalloc_orcc" / "tables" / "revision"
TABLE.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "scripts"))
from build_revision_statistics import mean_halfwidth, paired

BENCH = {"SensoDat": "SensoDat-replay", "Amini": "Amini-TransFuser"}
METHODS = ("Uniform", "FailRerun", "APGAI", "ScoreCost", "RerunK-10",
           "BayesUCB-Cost", "ReproAlloc")
PRIMARY = (.05, .10, .20)


def pct(value):
    return f"{value:.0%}".replace("%", r"\%")


def write(name, data, lines):
    data.to_csv(TABLE / f"{name}.csv", index=False)
    (TABLE / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt_recall_count(r, oracle=False):
    if oracle:
        return (f"\\shortstack{{{r.oracle_soft_recall_mean:.4f} $\\pm$ "
                f"{r.oracle_soft_recall_halfwidth:.4f}\\\\"
                f"({r.oracle_soft_count_mean:.1f} $\\pm$ {r.oracle_soft_count_halfwidth:.1f})}}")
    return (f"\\shortstack{{{r.recall_mean:.4f} $\\pm$ {r.recall_halfwidth:.4f}\\\\"
            f"({r.count_mean:.1f} $\\pm$ {r.count_halfwidth:.1f})}}")


def table2(stats, oracle):
    data = stats[(stats.tau == .3) & stats.budget.isin(PRIMARY) &
                 stats.method.isin(METHODS)].copy()
    lines = [r"\begin{tabular}{llccc}", r"\toprule",
             r"Benchmark & Method & 5\% & 10\% & 20\%\\", r"\midrule"]
    for bench in ("SensoDat", "Amini"):
        for method in ("Hindsight oracle",) + METHODS:
            cells = []
            for budget in PRIMARY:
                if method == "Hindsight oracle":
                    r = oracle[(oracle.benchmark == bench) & (oracle.budget == budget)].iloc[0]
                    cells.append(fmt_recall_count(r, oracle=True))
                else:
                    r = data[(data.benchmark == bench) & (data.budget == budget) &
                             (data.method == method)].iloc[0]
                    cells.append(fmt_recall_count(r))
            label = BENCH[bench] if method == "Hindsight oracle" else ""
            lines.append(f"{label} & {method} & " + " & ".join(cells) + r"\\")
        lines.append(r"\midrule" if bench == "SensoDat" else r"\bottomrule")
    lines.append(r"\end{tabular}")
    write("Table2_rev", data, lines)


def table3(stats, pairs):
    rows = []
    for bench in ("SensoDat", "Amini"):
        for budget in PRIMARY:
            cell = stats[(stats.benchmark == bench) & (stats.budget == budget) & (stats.tau == .3)]
            ext = cell[cell.method.isin(METHODS[:-1])].sort_values("recall_mean", ascending=False).iloc[0]
            ours = cell[cell.method == "ReproAlloc"].iloc[0]
            comparators = list(dict.fromkeys((ext.method, "RerunK-10", "BayesUCB-Cost")))
            for comparator in comparators:
                other = cell[cell.method == comparator].iloc[0]
                pair = pairs[(pairs.benchmark == bench) & (pairs.budget == budget) &
                             (pairs.tau == .3) & (pairs.comparator == comparator)].iloc[0]
                rows.append(dict(benchmark=bench, budget=budget,
                                 strongest_external=ext.method, comparator=comparator,
                                 external_recall=other.recall_mean, external_count=other.count_mean,
                                 external_count_halfwidth=other.count_halfwidth,
                                 repro_recall=ours.recall_mean, repro_count=ours.count_mean,
                                 repro_count_halfwidth=ours.count_halfwidth,
                                 delta=pair.delta, ci_low=pair.ci_low, ci_high=pair.ci_high,
                                 wilcoxon_p=pair.wilcoxon_p, wins=pair.wins,
                                 losses=pair.losses, ties=pair.ties))
    data = pd.DataFrame(rows)
    lines = [r"\begin{tabular}{lllccll}", r"\toprule",
             r"Benchmark & Budget & Comparator & Comparator R (count) & ReproAlloc R (count) & $\Delta$ [95\% CI] & $p$ (W/L/T)\\",
             r"\midrule"]
    for r in data.itertuples():
        lines.append(f"{BENCH[r.benchmark]} & {pct(r.budget)} & {r.comparator} & "
                     f"{r.external_recall:.4f} ({r.external_count:.1f} $\\pm$ {r.external_count_halfwidth:.1f}) & "
                     f"{r.repro_recall:.4f} ({r.repro_count:.1f} $\\pm$ {r.repro_count_halfwidth:.1f}) & "
                     f"{r.delta:+.4f} [{r.ci_low:+.4f}, {r.ci_high:+.4f}] & "
                     f"{r.wilcoxon_p:.4f} ({r.wins}/{r.losses}/{r.ties})" + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("Table3_rev", data, lines)


def table4():
    raw = pd.read_csv(ROOT / "results/engineering_value_check/time_to_k_raw.csv")
    raw = raw[raw.basis == "reference_positive"].copy()
    assert raw.groupby("method").seed.nunique().eq(12).all()
    methods = ("ReproAlloc", "ScoreCost", "OptimisticScoreCost", "APGAI", "Mean Planning")
    data = []
    for method in methods:
        cell = raw[raw.method == method]
        for k in (1, 2, 3):
            reached = cell[cell[f"reached{k}"] == 1]
            mean, half = mean_halfwidth(reached[f"T{k}_h"])
            data.append(dict(method=method, milestone=k, n_reached=len(reached),
                             hours_mean=mean, hours_halfwidth=half))
    data = pd.DataFrame(data)
    paired_rows = []
    ours = raw[raw.method == "ReproAlloc"].set_index("seed")
    for method in methods[1:]:
        other = raw[raw.method == method].set_index("seed")
        for k in (1, 2, 3):
            both = ours[(ours[f"reached{k}"] == 1)].index.intersection(
                other[(other[f"reached{k}"] == 1)].index)
            savings = other.loc[both, f"T{k}_h"] - ours.loc[both, f"T{k}_h"]
            mean, half = mean_halfwidth(savings)
            paired_rows.append(dict(method=method, milestone=k, n_joint=len(both),
                                    hours_saved_mean=mean, hours_saved_halfwidth=half,
                                    wins=int((savings > 1e-12).sum()),
                                    losses=int((savings < -1e-12).sum()),
                                    ties=int((abs(savings) <= 1e-12).sum())))
    pairs = pd.DataFrame(paired_rows)
    pairs.to_csv(TABLE / "Table4_rev_paired.csv", index=False)
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"Method & 1st & 2nd & 3rd\\", r"\midrule"]
    for method in methods:
        cells = []
        for k in (1, 2, 3):
            r = data[(data.method == method) & (data.milestone == k)].iloc[0]
            cells.append(f"{r.hours_mean:.2f} $\\pm$ {r.hours_halfwidth:.2f} ({int(r.n_reached)}/12)")
        lines.append(method + " & " + " & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", "", r"\begin{tabular}{lccc}",
              r"\toprule", r"Hours saved vs. ReproAlloc & 1st & 2nd & 3rd\\", r"\midrule"]
    for method in methods[1:]:
        cells = []
        for k in (1, 2, 3):
            r = pairs[(pairs.method == method) & (pairs.milestone == k)].iloc[0]
            cells.append(f"{r.hours_saved_mean:+.2f} $\\pm$ {r.hours_saved_halfwidth:.2f} "
                         f"({int(r.wins)}/{int(r.losses)}/{int(r.ties)})")
        lines.append(method + " & " + " & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("Table4_rev", data, lines)


def table5(stats, variants, pairs, t2pairs):
    methods = ("ScoreCost", "OptimisticScoreCost", "Mean Planning",
               "MeanPlanning-uncapped", "ReproAlloc", "ReproAlloc-uncapped")
    base = stats[(stats.tau == .3) & stats.budget.isin(PRIMARY) &
                 stats.method.isin(methods)]
    add = variants[variants.method.isin(("MeanPlanning-uncapped", "ReproAlloc-uncapped"))]
    add = add.rename(columns={"recall_halfwidth": "recall_halfwidth"})
    data = pd.concat([base[["benchmark", "budget", "method", "recall_mean",
                            "recall_halfwidth", "count_mean", "count_halfwidth"]],
                      add[["benchmark", "budget", "method", "recall_mean",
                           "recall_halfwidth", "count_mean", "count_halfwidth"]]], ignore_index=True)
    data = data.sort_values(["benchmark", "budget", "method"])
    controls = pairs[(pairs.tau == .3) & pairs.budget.isin(PRIMARY) &
                     pairs.method.eq("ReproAlloc") &
                     pairs.comparator.isin(("Mean Planning", "OptimisticScoreCost"))].copy()
    uncapped = t2pairs[(t2pairs.method == "ReproAlloc") &
                       (t2pairs.comparator == "MeanPlanning-uncapped")].copy()
    uncapped["tau"] = .3
    controls = pd.concat([controls, uncapped], ignore_index=True)
    controls = controls.sort_values(["benchmark", "budget", "comparator"])
    controls.to_csv(TABLE / "Table5_rev_paired.csv", index=False)
    controls.to_csv(TABLE / "Table5_rev_component_paired.csv", index=False)
    def cell_text(group, method):
        r = group[group.method == method].iloc[0]
        return (f"{r.recall_mean:.4f} $\\pm$ {r.recall_halfwidth:.4f} "
                f"({r.count_mean:.1f} $\\pm$ {r.count_halfwidth:.1f})")

    panel_a = [r"\emph{Panel A: recall (95\% $t$ halfwidth; mean count with halfwidth).}",
               r"\smallskip", r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llcccc@{}}",
               r"\toprule", r"Benchmark & Budget & ScoreCost & OptimisticScoreCost & Mean Planning & ReproAlloc\\",
               r"\midrule"]
    for benchmark in ("SensoDat", "Amini"):
        for budget in PRIMARY:
            group = data[(data.benchmark == benchmark) & (data.budget == budget)]
            panel_a.append(
                f"{BENCH[benchmark]} & {pct(budget)} & {cell_text(group, 'ScoreCost')} & "
                f"{cell_text(group, 'OptimisticScoreCost')} & "
                f"\\shortstack{{cap: {cell_text(group, 'Mean Planning')}\\\\"
                f"uncapped: {cell_text(group, 'MeanPlanning-uncapped')}}} & "
                f"\\shortstack{{cap: {cell_text(group, 'ReproAlloc')}\\\\"
                f"uncapped: {cell_text(group, 'ReproAlloc-uncapped')}}}" + r"\\")
        panel_a.append(r"\midrule" if benchmark == "SensoDat" else r"\bottomrule")
    panel_a.append(r"\end{tabular*}")
    pair_lines = [r"\emph{Panel B: paired ReproAlloc minus comparator differences.}",
                  r"\smallskip", r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lllrll@{}}",
                  r"\toprule",
                  r"Benchmark & Budget & Comparator & $\Delta$ [95\% CI] & exact $p$ & W/L/T\\",
                  r"\midrule"]
    for r in controls.itertuples():
        name = {"MeanPlanning-uncapped": "Mean Planning (uncapped)"}.get(r.comparator, r.comparator)
        pair_lines.append(f"{BENCH[r.benchmark]} & {pct(r.budget)} & {name} & "
                          f"{r.delta:+.4f} [{r.ci_low:+.4f}, {r.ci_high:+.4f}] & "
                          f"{r.wilcoxon_p:.4f} & {r.wins}/{r.losses}/{r.ties}" + r"\\")
    pair_lines += [r"\bottomrule", r"\end{tabular*}"]
    write("Table5_rev", data, panel_a + [r"\medskip"] + pair_lines)
    write("Table5_rev_component_paired", controls, pair_lines)


def table7():
    rows = []
    for bench, path in (("SensoDat", ROOT / "results/01_optimistic_core/sensodat/sensodat_optimistic_main.csv"),
                        ("Amini", ROOT / "results/01_optimistic_core/amini/amini_optimistic_main.csv")):
        d = pd.read_csv(path)
        budget_col = "time_frac" if bench == "SensoDat" else "budget_frac"
        unit_col = "split_id" if bench == "SensoDat" else "seed"
        for budget in PRIMARY:
            for label, code in (("ReproAlloc", "ReproAlloc-Optimistic"),
                                ("Mean Planning", "ReproAlloc"), ("ScoreCost", "ScoreCost")):
                g = d[(d[budget_col].round(5) == budget) & (d.method == code)]
                unit = g.groupby(unit_col).sched_overhead_s.mean()
                mean, half = mean_halfwidth(unit)
                rows.append(dict(benchmark=bench, budget=budget, method=label,
                                 n_units=len(unit), cpu_mean_s=mean, cpu_halfwidth_s=half))
    data = pd.DataFrame(rows)
    lines = [r"\begin{tabular}{llccc}", r"\toprule",
             r"Benchmark & Budget & ReproAlloc CPU/run (s) & Mean Planning & ScoreCost\\", r"\midrule"]
    for bench in ("SensoDat", "Amini"):
        for budget in PRIMARY:
            cell = data[(data.benchmark == bench) & (data.budget == budget)]
            vals = []
            for method in ("ReproAlloc", "Mean Planning", "ScoreCost"):
                r = cell[cell.method == method].iloc[0]
                vals.append(f"{r.cpu_mean_s:.3f} $\\pm$ {r.cpu_halfwidth_s:.3f}")
            lines.append(f"{BENCH[bench]} & {pct(budget)} & " + " & ".join(vals) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("Table7_rev", data, lines)


def oracle_table(oracle):
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"Benchmark & Budget & Hard count & Soft count & ReproAlloc count & Best external count & Repro/soft\\",
             r"\midrule"]
    for r in oracle.itertuples():
        lines.append(f"{BENCH[r.benchmark]} & {pct(r.budget)} & "
                     f"{r.oracle_hard_count_mean:.1f} & {r.oracle_soft_count_mean:.1f} & "
                     f"{r.repro_count_mean:.1f} & {r.external_count_mean:.1f} "
                     f"({r.best_external}) & {r.repro_percent_of_soft:.1f}\\%" + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("T1_oracle", oracle, lines)


def uncapped_table(variants, paired_data):
    data = variants[variants.method.isin(("ReproAlloc", "Mean Planning",
                                        "ReproAlloc-uncapped", "MeanPlanning-uncapped"))].copy()
    lines = [r"\begin{tabular}{lllcc}", r"\toprule",
             r"Benchmark & Budget & Method & Recall (95\% halfwidth) & Count (95\% halfwidth)\\",
             r"\midrule"]
    for r in data.itertuples():
        lines.append(f"{BENCH[r.benchmark]} & {pct(r.budget)} & {r.method} & "
                     f"{r.recall_mean:.4f} $\\pm$ {r.recall_halfwidth:.4f} & "
                     f"{r.count_mean:.1f} $\\pm$ {r.count_halfwidth:.1f}" + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("T2_uncapped", data, lines)
    paired_data.to_csv(TABLE / "T2_uncapped_paired.csv", index=False)


def main():
    stats = pd.read_csv(OUT / "statistics_rev.csv")
    pairs = pd.read_csv(OUT / "paired_rev.csv")
    oracle = pd.read_csv(OUT / "T1_oracle.csv")
    variants = pd.read_csv(OUT / "T2_T3_variants.csv")
    t2pairs = pd.read_csv(OUT / "T2_uncapped_paired.csv")
    table2(stats, oracle)
    table3(stats, pairs)
    table4()
    table5(stats, variants, pairs, t2pairs)
    table7()
    oracle_table(oracle)
    uncapped_table(variants, t2pairs)
    print(f"wrote revision tables to {TABLE}")


if __name__ == "__main__":
    main()
