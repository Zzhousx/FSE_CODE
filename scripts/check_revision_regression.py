"""Compare newly rerun primary cells with the shipped, previously reported cells."""

from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "revision" / "T0_regression.csv"
METHODS = {
    "Uniform": "Uniform",
    "FailRerun": "FailRerun",
    "APGAI": "APGAI",
    "ScoreCost": "ScoreCost",
    "ReproAlloc": "ReproAlloc-Optimistic",
    "Mean Planning": "ReproAlloc",
    "OptimisticScoreCost": "OptimisticScoreCost",
}

SENSO = {
    "ReproAlloc": [0.0206, 0.0405, 0.0782],
    "FailRerun": [0.0173, 0.0342, 0.0691],
    "ScoreCost": [0.0170, 0.0344, 0.0663],
    "Mean Planning": [0.0203, 0.0388, 0.0755],
    "OptimisticScoreCost": [0.0147, 0.0294, 0.0539],
    "Uniform": [0, 0, 0],
    "APGAI": [0, 0, 0],
}
AMINI = {
    "ReproAlloc": [0.0652, 0.1413, 0.2464],
    "ScoreCost": [0.0435, 0.1051, 0.2065],
    "APGAI": [0, 0.0652, 0.2283],
    "FailRerun": [0.0217, 0.0435, 0.0833],
    "Mean Planning": [0.0471, 0.0761, 0.1594],
    "OptimisticScoreCost": [0.0399, 0.0906, 0.1775],
}


def compare(benchmark, current_path, old_path, budget_col, recall_col, anchors):
    if current_path.is_dir():
        paths = sorted(current_path.glob("shard_*/sensodat_osc_check.csv"))
        if len(paths) != 10:
            raise ValueError(f"Expected 10 SensoDat shards, found {len(paths)}")
        current = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    else:
        current = pd.read_csv(current_path)
    old = pd.read_csv(old_path)
    rows = []
    for budget_idx, budget in enumerate((0.05, 0.10, 0.20)):
        for paper_name, code_name in METHODS.items():
            key = (budget, code_name)
            actual = current.loc[
                current[budget_col].round(5).eq(budget) & current.method.eq(code_name),
                recall_col,
            ]
            previous = old.loc[
                old[budget_col].round(5).eq(budget) & old.method.eq(code_name),
                recall_col,
            ]
            if actual.empty or previous.empty:
                raise ValueError(f"Missing regression cell: {benchmark} {key}")
            mean = actual.mean()
            shipped = previous.mean()
            headline = anchors.get(paper_name, [None] * 3)[budget_idx]
            # The prompt reports four-decimal means, so compare at that precision.
            shipped_difference = abs(round(mean, 4) - round(shipped, 4))
            headline_difference = (
                abs(round(mean, 4) - headline) if headline is not None else None
            )
            rows.append({
                "benchmark": benchmark,
                "budget": budget,
                "method": paper_name,
                "n_rerun_rows": len(actual),
                "rerun_mean": mean,
                "shipped_mean": shipped,
                "headline_4dp": headline,
                "shipped_difference_4dp": shipped_difference,
                "headline_difference_4dp": headline_difference,
                "pass": shipped_difference <= 0.0001 + 1e-12 and (
                    headline_difference is None or headline_difference <= 0.0001 + 1e-12
                ),
            })
    return rows


def main():
    rows = compare(
        "SensoDat",
        ROOT / "results/revision/t0_sensodat",
        ROOT / "results/optimistic_scorecost_check/sensodat_osc_check.csv",
        "time_frac", "recall", SENSO,
    ) + compare(
        "Amini",
        ROOT / "results/revision/t0_amini_8/amini_osc_check.csv",
        ROOT / "results/optimistic_scorecost_check/amini_osc_check.csv",
        "budget_frac", "recall_refpos", AMINI,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}; {df['pass'].sum()}/{len(df)} cells passed")
    print(df.loc[~df['pass']].to_string(index=False))
    if not df['pass'].all():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
