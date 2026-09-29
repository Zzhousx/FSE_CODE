"""Compact two-panel planning-quantile sweep (T3) for the manuscript."""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT.parent / "paper_reproalloc_orcc"
OUT = PAPER / "figs" / "revision"
DATA = ROOT / "results" / "revision" / "T3"

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


def summarize(x):
    x = np.asarray(x, dtype=float)
    return float(x.mean()), float(t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def inputs():
    pieces = []
    for bench in ("sensodat", "amini"):
        if bench == "sensodat":
            raw = pd.concat([pd.read_csv(p) for p in sorted((DATA / bench).glob("split_*.csv"))])
            assert raw.unit.nunique() == 10
            original = pd.concat([pd.read_csv(p) for p in sorted(
                (ROOT / "results/revision/t0_sensodat").glob("shard_*/sensodat_osc_check.csv"))])
            original = original[original.method.isin(("ReproAlloc-Optimistic", "ReproAlloc"))]
            original = original.rename(columns={"split_id": "unit", "time_frac": "budget"})
            original["method"] = original.method.map({"ReproAlloc-Optimistic": "ReproAlloc-rho0.95",
                                                        "ReproAlloc": "Mean Planning"})
        else:
            raw = pd.read_csv(DATA / bench / "raw.csv")
            assert raw.unit.nunique() == 12
            original = pd.read_csv(ROOT / "results/revision/t0_amini_8/amini_osc_check.csv")
            original = original[original.method.isin(("ReproAlloc-Optimistic", "ReproAlloc"))]
            original = original.rename(columns={"seed": "unit", "budget_frac": "budget",
                                                "recall_refpos": "recall"})
            original["rep"] = 0
            original["method"] = original.method.map({"ReproAlloc-Optimistic": "ReproAlloc-rho0.95",
                                                        "ReproAlloc": "Mean Planning"})
        raw = pd.concat([raw, original[["unit", "rep", "budget", "method", "recall"]]])
        raw["benchmark"] = "SensoDat" if bench == "sensodat" else "Amini"
        pieces.append(raw[["benchmark", "unit", "budget", "method", "recall"]])
    data = pd.concat(pieces, ignore_index=True)
    unit = data.groupby(["benchmark", "unit", "budget", "method"], as_index=False).recall.mean()
    summary = []
    for (bench, budget, method), g in unit.groupby(["benchmark", "budget", "method"]):
        mean, half = summarize(g.recall)
        summary.append({"benchmark": bench, "budget": budget, "method": method,
                        "n_units": len(g), "mean_recall": mean, "halfwidth": half})
    return pd.DataFrame(summary)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = inputs()
    summary.to_csv(DATA / "rho_summary.csv", index=False)
    apply_figure_style(sizes=(8, 7, 6))
    colors = {0.05: "#253A6A", 0.10: "#775482", 0.20: "#B35A28"}
    markers = {0.05: "o", 0.10: "s", 0.20: "^"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.28))
    for ax, bench, label in zip(axes, ("SensoDat", "Amini"),
                                ("(a) SensoDat-replay", "(b) Amini-TransFuser")):
        cell = summary[summary.benchmark == bench]
        for budget in (.05, .10, .20):
            sub = cell[(cell.budget == budget) & cell.method.str.startswith("ReproAlloc-rho")].copy()
            sub["rho"] = sub.method.str.replace("ReproAlloc-rho", "", regex=False).astype(float)
            sub = sub.sort_values("rho")
            ax.errorbar(sub.rho, sub.mean_recall, yerr=sub.halfwidth,
                        color=colors[budget], marker=markers[budget], ms=3.2,
                        lw=1.3, capsize=1.5, label=f"{int(budget*100)}% budget")
            ref = cell[(cell.budget == budget) & (cell.method == "Mean Planning")].iloc[0]
            ax.axhline(ref.mean_recall, color=colors[budget], ls="--", lw=.8, alpha=.65)
        ax.axvline(.95, color="#555555", ls=":", lw=.8)
        ax.set_title(label, loc="left", pad=4)
        ax.set_xlabel("planning quantile $\\rho$")
        ax.set_xticks((.50, .75, .90, .95, .99))
        ax.set_xlim(.47, 1.01)
        ax.set_ylim(bottom=0)
        ax.margins(y=.06)
        ax.grid(axis="y", color="#DADADA", lw=.4)
    axes[0].set_ylabel("confirmed-test recall")
    handles, labels = axes[0].get_legend_handles_labels()
    handles += [plt.Line2D([0], [0], color="#555555", ls="--", lw=.8),
                plt.Line2D([0], [0], color="#555555", ls=":", lw=.8)]
    labels += ["Mean Planning", "fixed $\\rho=0.95$"]
    fig.legend(handles, labels, loc="upper center", ncol=5,
               bbox_to_anchor=(.5, 1.05), columnspacing=.9, handlelength=1.5)
    fig.subplots_adjust(top=.76, bottom=.25, left=.08, right=.99, wspace=.24)
    fig.savefig(OUT / "fig_rho_sweep.pdf")
    fig.savefig(OUT / "fig_rho_sweep.svg")
    plt.close(fig)
    print(f"wrote {OUT / 'fig_rho_sweep.pdf'}")


if __name__ == "__main__":
    main()
