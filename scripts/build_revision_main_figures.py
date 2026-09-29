"""Build result-dependent Figure 3 and Figure 5 candidates for the revision."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def apply_figure_style(sizes=(8, 7, 6)):
    """Apply a portable, publication-ready style without local skill dependencies."""
    base, label, small = sizes
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": base,
        "axes.labelsize": label,
        "axes.titlesize": base,
        "xtick.labelsize": small,
        "ytick.labelsize": small,
        "legend.fontsize": small,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
    })
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "paper_reproalloc_orcc" / "figs" / "revision"
DATA = ROOT / "results" / "revision"

BENCHES = (("SensoDat", "(a) SensoDat-replay"),
           ("Amini", "(b) Amini-TransFuser"))
BUDGETS = (.03, .05, .06, .07, .08, .10, .15, .20)
EXTERNAL = ("Uniform", "FailRerun", "ScoreCost", "APGAI",
            "RerunK-10", "BayesUCB-Cost")
METHODS = EXTERNAL + ("ReproAlloc",)
STYLES = {
    "Uniform": ("#999999", "o", ":"),
    "FailRerun": ("#8C564B", "s", "--"),
    "ScoreCost": ("#378ADD", "D", "--"),
    "APGAI": ("#639922", "^", "--"),
    "RerunK-10": ("#8064A2", "v", "-."),
    "BayesUCB-Cost": ("#16867D", "P", "-."),
    "ReproAlloc": ("#D85A30", "*", "-"),
}


def figure3(stats, oracle):
    primary = stats[np.isclose(stats.tau, .3) & stats.method.isin(METHODS)]
    assert set(np.round(primary.budget.unique(), 2)) == set(BUDGETS)
    assert len(primary) == 2 * len(BUDGETS) * len(METHODS)
    assert len(oracle) == 2 * len(BUDGETS)
    fig, axes = plt.subplots(1, 2, figsize=(7.35, 2.8))
    for ax, (bench, title) in zip(axes, BENCHES):
        sub = primary[primary.benchmark == bench]
        for method in METHODS:
            g = sub[sub.method == method].sort_values("budget")
            color, marker, ls = STYLES[method]
            ax.errorbar(g.budget * 100, g.recall_mean,
                        yerr=g.recall_halfwidth, label=method,
                        color=color, marker=marker, ls=ls,
                        lw=1.75 if method == "ReproAlloc" else 1.0,
                        ms=5 if method == "ReproAlloc" else 3.1,
                        capsize=1.1, zorder=5 if method == "ReproAlloc" else 2)
        o = oracle[oracle.benchmark == bench].sort_values("budget")
        ax.plot(o.budget * 100, o.oracle_soft_recall_mean, color="#202020",
                ls="--", lw=1.35, marker="x", ms=3.5, label="Hindsight oracle")
        for x in (5, 10, 20):
            ax.axvline(x, color="#999999", lw=.55, alpha=.35, zorder=0)
        ax.set_title(title, loc="left", pad=3)
        ax.set_xticks([3, 5, 6, 7, 8, 10, 15, 20])
        ax.set_xticklabels(["3", "5", "6", "7", "8", "10", "15", "20%"])
        ax.set_xlabel("budget (% of total recorded execution time)")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="#DDDDDD", lw=.4)
    axes[0].set_ylabel("confirmed-test recall")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center",
               bbox_to_anchor=(.5, 1.02), columnspacing=.9, handlelength=1.5)
    fig.subplots_adjust(top=.79, bottom=.20, left=.08, right=.99, wspace=.24)
    fig.savefig(OUT / "fig3_with_oracle.pdf")
    plt.close(fig)


def figure5(stats):
    rows = []
    for bench, _ in BENCHES:
        for budget in (.05, .10, .20):
            for tau in (.2, .3, .4):
                cell = stats[(stats.benchmark == bench) &
                             np.isclose(stats.budget, budget) &
                             np.isclose(stats.tau, tau)]
                ex = cell[cell.method.isin(EXTERNAL)].sort_values(
                    ["recall_mean", "method"], ascending=[False, True]).iloc[0]
                ra = cell[cell.method == "ReproAlloc"].iloc[0]
                for method, source in (("ReproAlloc", ra), ("Best external", ex)):
                    rows.append({"benchmark": bench, "budget": budget, "tau": tau,
                                 "method": method,
                                 "selected_external": ex.method,
                                 "mean_recall": source.recall_mean,
                                 "halfwidth": source.recall_halfwidth})
    selected = pd.DataFrame(rows)
    assert len(selected) == 36
    selected.to_csv(DATA / "T5_fig5_sources.csv", index=False)
    fig, axes = plt.subplots(2, 3, figsize=(7.35, 3.65), sharex=True)
    for r, (bench, title) in enumerate(BENCHES):
        for c, budget in enumerate((.05, .10, .20)):
            ax = axes[r, c]
            cell = selected[(selected.benchmark == bench) &
                            np.isclose(selected.budget, budget)]
            for method, shift, color, marker in (
                ("ReproAlloc", -.006, "#D85A30", "*"),
                ("Best external", .006, "#378ADD", "D"),
            ):
                sub = cell[cell.method == method].sort_values("tau")
                ax.errorbar(sub.tau + shift, sub.mean_recall,
                            yerr=sub.halfwidth, label=method,
                            color=color, marker=marker, ms=5 if marker == "*" else 3.5,
                            ls="-", lw=1.1, capsize=1.5)
            ax.set_ylim(bottom=0)
            ax.set_xticks((.2, .3, .4))
            ax.grid(axis="y", color="#DDDDDD", lw=.4)
            if r == 0:
                ax.set_title(f"{int(budget * 100)}% budget")
            if c == 0:
                ax.set_ylabel(f"{title[4:]}\nconfirmed-test recall")
            if r == 1:
                ax.set_xlabel(r"confirmation threshold $\tau$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="lower center",
               bbox_to_anchor=(.5, .005))
    fig.subplots_adjust(top=.92, bottom=.19, left=.12, right=.99,
                        hspace=.37, wspace=.28)
    fig.savefig(OUT / "fig5_tau_sensitivity_rev.pdf")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    apply_figure_style(sizes=(8, 7, 6))
    stats = pd.read_csv(DATA / "statistics_rev.csv")
    oracle = pd.read_csv(DATA / "T1_oracle.csv")
    figure3(stats, oracle)
    figure5(stats)
    print("wrote", OUT / "fig3_with_oracle.pdf")
    print("wrote", OUT / "fig5_tau_sensitivity_rev.pdf")


if __name__ == "__main__":
    main()
