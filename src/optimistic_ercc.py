"""Optimistic ERCC - uncertainty-aware planning rate (single-change variant).

Research question (frozen spec):
    ERCC plans the remaining-executions estimate m_i^* with the posterior
    MEAN q_i = a_i/(a_i+b_i) as if it were the true failure rate.  Does that
    point-estimate planning need uncertainty awareness?

This module implements the optimistic variant:
    q_i^opt = BetaQuantile(1 - delta; a_i, b_i)      (delta = 0.05 -> 95th pct)

The optimistic rate enters EXACTLY ONE place: the planning trajectory
    s~(m) = s + m*q,  n~(m) = n + m
inside est_remaining_execs (reproalloc_v6), i.e. only m_i^* (and therefore
only the ERCC scheduling index R_i = m_i^* * C_i).

HARD INVARIANTS - none of the following is touched by this variant:
  * confirmation rule: integer Hoeffding CS, strict L > tau
    (reproalloc_v6.cs_lower / confirmed_sn) - never sees q_opt;
  * fallback ordering: ScoreCost q_mean / C (posterior MEAN);
  * OA cost mixture: q_mean*cF + (1-q_mean)*cP (posterior MEAN);
  * budgets, seeds, tau = 0.3, delta = 0.05, priors, finite pools, tie-breaks.

Note: the frozen verification (verify_partA_frozen_method.py, check A8)
forbids Beta-quantile machinery in the six frozen files.  This is a NEW
file; the frozen files are not modified.
"""
import numpy as np
from scipy.stats import beta as _beta_dist

DELTA = 0.05
MODE_MEAN = "mean"
MODE_OPTIMISTIC = "optimistic"
MODES = (MODE_MEAN, MODE_OPTIMISTIC)


def posterior_mean(a, b):
    """Frozen planning rate: posterior mean q = a/(a+b)."""
    a = float(a)
    b = float(b)
    return a / (a + b)


def posterior_upper_quantile(a, b, delta=DELTA):
    """Optimistic planning rate: q_opt = Beta.ppf(1-delta; a, b).

    With delta = 0.05 this is the 95th percentile of the Beta scheduling
    posterior - the smallest failure rate whose credibility mass to the
    left covers 1-delta.  a, b >= 1 always holds here (priors are
    Beta(1+kappa*pi, 1+kappa*(1-pi)) or Beta(1,1)), so the quantile is
    finite and strictly inside (0, 1).
    """
    a = float(a)
    b = float(b)
    if not (np.isfinite(a) and np.isfinite(b) and a > 0.0 and b > 0.0):
        raise ValueError(f"invalid Beta parameters a={a} b={b}")
    if not (0.0 < delta < 1.0):
        raise ValueError(f"delta must be in (0,1), got {delta}")
    return float(_beta_dist.ppf(1.0 - delta, a, b))


def planning_rate(a, b, mode=MODE_MEAN, delta=DELTA):
    """Single dispatch point for the rate entering the m_i^* planning path.

    mode = "mean"       -> identical to the frozen method (a/(a+b))
    mode = "optimistic" -> posterior_upper_quantile(a, b, delta)
    """
    if mode == MODE_MEAN:
        return posterior_mean(a, b)
    if mode == MODE_OPTIMISTIC:
        return posterior_upper_quantile(a, b, delta)
    raise ValueError(f"unknown planning-rate mode: {mode!r}")
