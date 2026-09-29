import itertools
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from revision_oracle import hard_soft, interval


def brute_soft(costs, last_costs, budget):
    n = len(costs)
    best = 0
    for count in range(1, n + 1):
        for chosen in itertools.combinations(range(n), count):
            if sum(costs[i] for i in chosen) <= budget:
                best = max(best, count)
            for last in chosen:
                before = sum(costs[i] for i in chosen if i != last)
                before += costs[last] - last_costs[last]
                if before < budget:
                    best = max(best, count)
    return best


@pytest.mark.parametrize("budget,expected", [(5, (0, 0)), (6, (0, 1)), (10, (1, 1))])
def test_soft_budget_strict_start(budget, expected):
    assert hard_soft([10], [5], budget) == expected


def test_soft_oracle_matches_exhaustive_small_instances():
    for costs in itertools.product((4, 7, 10), repeat=3):
        last = tuple(x / 2 for x in costs)
        for budget in (3, 5, 8, 12, 17):
            hard, soft = hard_soft(costs, last, budget)
            assert soft == brute_soft(costs, last, budget)
            assert hard <= soft


def test_interval_zero_for_single_unit():
    assert interval([1]) == 0
    assert interval([1, 1, 1]) == 0
