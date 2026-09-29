"""Exact two-sided Wilcoxon signed-rank test by sign-flip enumeration."""

from itertools import product

import numpy as np
from scipy.stats import rankdata


def exact_two_sided_wilcoxon(values):
    """Drop zeros, assign midranks, and enumerate every sign assignment.

    The supported sample size is at most 12 nonzero paired differences, which
    keeps enumeration exact and inexpensive for the revision's unit counts.
    """
    d = np.asarray(values, dtype=float).ravel()
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return 1.0
    if n > 12:
        raise ValueError(f"exact sign-flip enumeration supports n <= 12, got {n}")
    ranks = rankdata(np.abs(d), method="average")
    observed = abs(float(np.dot(np.sign(d), ranks)))
    totals = (abs(float(np.dot(signs, ranks)))
              for signs in product((-1.0, 1.0), repeat=n))
    extreme = sum(value >= observed - 1e-12 for value in totals)
    return extreme / (2 ** n)
