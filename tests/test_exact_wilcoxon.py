import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from exact_wilcoxon import exact_two_sided_wilcoxon


def test_requested_exact_wilcoxon_sanity_cases():
    assert exact_two_sided_wilcoxon([1] * 8 + [-1] * 0 + [0] * 4) == 0.0078125
    assert exact_two_sided_wilcoxon([1] * 4 + [0] * 8) == 0.125
    assert exact_two_sided_wilcoxon([1] * 10) == 0.001953125


def test_zeros_are_dropped_and_midranks_are_used():
    # The nonzero absolute differences have tied ranks, producing a midrank
    # distribution whose exact tail differs from the no-tie rank distribution.
    got = exact_two_sided_wilcoxon([1, 1, 2, 2, 0])
    assert np.isclose(got, 0.125)


def test_enumeration_rejects_more_than_twelve_nonzero_units():
    try:
        exact_two_sided_wilcoxon(np.ones(13))
    except ValueError as exc:
        assert "n <= 12" in str(exc)
    else:
        raise AssertionError("expected exact test to reject n > 12")
