"""Hindsight hard/soft confirmation ceilings using the frozen replay semantics."""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import reproalloc_v6 as V
import sensodat_final_experiment as S
import experiment_amini_finite_trace as A

BUDGETS = (0.03, 0.05, 0.06, 0.07, 0.08, 0.10, 0.15, 0.20, 0.40)
OUT = ROOT / "results" / "revision"


def hard_soft(costs, last_costs, budget):
    """Greedy hard count and an exact one-final-execution soft count."""
    costs = np.asarray(costs, dtype=float)
    last_costs = np.asarray(last_costs, dtype=float)
    if not len(costs):
        return 0, 0
    order = np.argsort(costs, kind="stable")
    costs, last_costs = costs[order], last_costs[order]
    prefix = np.r_[0.0, np.cumsum(costs)]
    hard = int(np.searchsorted(prefix, budget, side="right") - 1)
    soft = hard
    for j, (cost, last) in enumerate(zip(costs, last_costs)):
        threshold = budget - (cost - last)
        if threshold <= 0:
            continue
        # Prefix cost of the cheapest m items excluding final item j.
        before = np.searchsorted(prefix[: j + 1], threshold, side="left") - 1
        after = np.searchsorted(prefix[j + 2 :] - cost, threshold, side="left")
        others = max(before, j + after if after else before)
        soft = max(soft, 1 + others)
    return hard, min(int(soft), len(costs))


def interval(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return 0.0
    return float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    S.DATA = str(ROOT / "data")
    data = S.make_splits()
    target = np.asarray(data["splits"][0]["target"])
    y, durations = data["y"][target], data["dur"][target]
    failures = durations[y == 1]
    n_ref_s = len(failures)
    assert n_ref_s == 4322
    k = next(k for k in range(1, 100) if V.confirmed_sn(k, k, S.TAU, S.DELTA))
    assert k == 9
    total_s = float(durations.sum())

    routes = A.load_routes()
    reference = np.array([r.p_hat > A.TAU for r in routes])
    assert reference.sum() == 23
    total_a = sum(sum(r.durations) for r in routes)
    raw = []
    for budget_frac in BUDGETS:
        budget = total_s * budget_frac
        hard, soft = hard_soft(k * failures, failures, budget)
        raw.append({"benchmark": "SensoDat", "unit": 0, "budget": budget_frac,
                    "n_reference": n_ref_s, "confirmable": n_ref_s,
                    "oracle_hard_count": hard, "oracle_soft_count": soft,
                    "oracle_hard_recall": hard / n_ref_s,
                    "oracle_soft_recall": soft / n_ref_s})

    for seed in range(2000, 2012):
        orders = A.make_orders(routes, seed)
        costs, last_costs = [], []
        for i, (route, order) in enumerate(zip(routes, orders)):
            s = 0
            for n, idx in enumerate(order, 1):
                s += route.outcomes[idx]
                if A.sticky_confirm_trace(s, n):
                    assert reference[i], f"Non-reference route confirmed: {seed} {route.route_id}"
                    costs.append(sum(route.durations[j] for j in order[:n]))
                    last_costs.append(route.durations[idx])
                    break
        confirmable = len(costs)
        assert confirmable in (8, 9), (seed, confirmable)
        for budget_frac in BUDGETS:
            hard, soft = hard_soft(costs, last_costs, total_a * budget_frac)
            raw.append({"benchmark": "Amini", "unit": seed, "budget": budget_frac,
                        "n_reference": int(reference.sum()), "confirmable": confirmable,
                        "oracle_hard_count": hard, "oracle_soft_count": soft,
                        "oracle_hard_recall": hard / reference.sum(),
                        "oracle_soft_recall": soft / reference.sum()})

    raw_df = pd.DataFrame(raw)
    raw_df.to_csv(OUT / "T1_oracle_raw.csv", index=False)
    # Save summary separately so later baseline additions can update the comparison columns.
    summary = []
    for (benchmark, budget), cell in raw_df.groupby(["benchmark", "budget"], sort=True):
        row = {"benchmark": benchmark, "budget": budget,
               "n_units": len(cell), "confirmable_mean": cell.confirmable.mean()}
        for kind in ("hard", "soft"):
            for metric in ("count", "recall"):
                values = cell[f"oracle_{kind}_{metric}"]
                row[f"oracle_{kind}_{metric}_mean"] = values.mean()
                row[f"oracle_{kind}_{metric}_halfwidth"] = interval(values)
        summary.append(row)
    pd.DataFrame(summary).to_csv(OUT / "T1_oracle_summary.csv", index=False)
    print(f"k={k}, SensoDat reference={n_ref_s}, Amini reference={reference.sum()}")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
