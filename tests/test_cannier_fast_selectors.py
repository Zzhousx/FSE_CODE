"""Check optimized CANNIER baseline selectors preserve their score maxima."""
import os
import sys

import numpy as np
from scipy.stats import beta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "cannier"))

import experiment_amini_finite_trace as A
from run_cannier import FastBayesUCBCost, FastOptimisticScoreCost, FastERCC


class Available:
    def exhausted(self):
        return False


def _verify(selector, rho):
    arms = A.Arms(7, A.GS(.7, .7, .7))
    pools = [Available() for _ in range(arms.K)]
    rng = np.random.default_rng(29)
    state = selector.init(arms, pools, rng)
    updates = [(0, 1, .2), (1, 0, .4), (2, 1, .3), (0, 0, .6),
               (3, 0, .15), (4, 1, .9), (1, 1, .25), (5, 0, .8)]
    for i, y, cost in updates:
        arms.update(i, y, cost)
        selector.update(arms, i, state)
        cand = np.arange(arms.K)
        cand = cand[~arms.confirmed]
        if not len(cand):
            break
        q = beta.ppf(rho(state), arms.post_a[cand], arms.post_b[cand])
        c = np.asarray([arms.mean_cost(int(j)) for j in cand])
        scores = q / c
        chosen = selector.select(arms, pools, rng, state)
        chosen_score = scores[np.flatnonzero(cand == chosen)[0]]
        assert chosen_score >= scores.max() - 1e-12


def test_fast_bayes_ucb_preserves_exact_beta_quantile_maximum():
    _verify(FastBayesUCBCost(), lambda st: 1. - 1. / (st["t"] + 1.))


def test_fast_optimistic_scorecost_preserves_exact_quantile_maximum():
    _verify(FastOptimisticScoreCost(), lambda st: .95)


class ZeroUtilityInner:
    name = "ReproAlloc"

    def init(self, arms, pools, rng):
        return {"util": np.zeros(arms.K)}

    def update(self, arms, i, st):
        st["util"][i] = 0.


class PositiveUtilityInner(ZeroUtilityInner):
    def init(self, arms, pools, rng):
        return {"util": np.asarray([.1, .5, .5, 0., .2])}


def test_fast_ercc_main_queue_keeps_all_maximizers():
    arms = A.Arms(5, A.GS(.7, .7, .7))
    pools = [Available() for _ in range(arms.K)]
    rng = np.random.default_rng(13)
    selector = FastERCC(PositiveUtilityInner())
    state = selector.init(arms, pools, rng)
    assert selector.select(arms, pools, rng, state) in (1, 2)
    assert selector.fallback_count == 0


def test_fast_ercc_fallback_queue_matches_scorecost_and_readds_bucket():
    arms = A.Arms(5, A.GS(.7, .7, .7))
    pools = [Available() for _ in range(arms.K)]
    rng = np.random.default_rng(7)
    selector = FastERCC(ZeroUtilityInner())
    state = selector.init(arms, pools, rng)
    for i, y, cost in ((0, 1, .1), (1, 0, .4), (2, 1, .2)):
        chosen = selector.select(arms, pools, rng, state)
        q = arms.post_a / (arms.post_a + arms.post_b)
        c = np.asarray([arms.mean_cost(j) for j in range(arms.K)])
        assert q[chosen] / c[chosen] >= np.max(q / c) - 1e-12
        arms.update(i, y, cost)
        selector.update(arms, i, state)
    assert selector.fallback_count == 3
