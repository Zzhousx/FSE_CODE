import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reproalloc_v6 import compute_mstar, est_remaining_execs


@pytest.mark.parametrize("state,q,expected", [
    ((1, 1), 2 / 3, 39),
    ((1, 1), 0.975, 8),
    ((0, 0), 0.5, 176),
])
def test_uncapped_anchors(state, q, expected):
    assert compute_mstar(*state, q, 0.3) == expected


def test_capped_matches_frozen_grid():
    for s, n in [(0, 0), (0, 1), (1, 1), (1, 2), (3, 5), (8, 10)]:
        for q in (0.2, 0.3, 0.5, 2 / 3, 0.975):
            for cap in (1, 5, 8, 20, 40, 200):
                old = est_remaining_execs(s, n, q, 0.3, cap)
                new = compute_mstar(s, n, q, 0.3, cap=cap)
                assert (math.isinf(old) and math.isinf(new)) or old == new
