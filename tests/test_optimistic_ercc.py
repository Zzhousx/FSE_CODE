"""Unit tests for the Optimistic-ERCC single-change variant.

Spec tests 1-7 (plus frozen anchors).  Run from the project root:
    python -m pytest tests/test_optimistic_ercc.py -v
"""
import inspect
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
sys.path.insert(0, SRC)

import reproalloc_v6 as v6  # noqa: E402
import v7_experiment as E  # noqa: E402
import experiment_amini_finite_trace as A  # noqa: E402
from optimistic_ercc import (posterior_mean, posterior_upper_quantile,  # noqa: E402
                             planning_rate, MODE_MEAN, MODE_OPTIMISTIC)
import optimistic_ercc_sensodat as OS  # noqa: E402
import optimistic_ercc_amini as OA  # noqa: E402

TAU, DELTA = 0.3, 0.05


# ----------------------------------------------------------------------
# Test 1: optimistic quantile strictly exceeds the posterior mean
# ----------------------------------------------------------------------
def test1_quantile_exceeds_mean():
    cases = [(1, 1), (1, 5), (5, 1), (2, 6), (6, 2), (1 + 5 * 0.4, 1 + 5 * 0.6),
             (11, 11), (2.5, 7.5), (100, 50), (50, 100)]
    for a, b in cases:
        qm = posterior_mean(a, b)
        qo = posterior_upper_quantile(a, b, DELTA)
        assert 0.0 < qm < 1.0 and 0.0 < qo < 1.0
        assert qo > qm, f"a={a} b={b}: q_opt={qo} !> q_mean={qm}"


# ----------------------------------------------------------------------
# Test 2: quantile converges to the mean as evidence accumulates
# ----------------------------------------------------------------------
def test2_quantile_converges_to_mean_with_evidence():
    phat = 0.4
    prev_gap = np.inf
    for n in [2, 10, 50, 200, 2000, 20000]:
        a = 1 + n * phat
        b = 1 + n * (1 - phat)
        gap = posterior_upper_quantile(a, b, DELTA) - posterior_mean(a, b)
        assert gap > 0
        assert gap < prev_gap, "uncertainty gap must shrink as n grows"
        prev_gap = gap
    assert prev_gap < 0.01, f"gap at n=20000 should be tiny, got {prev_gap}"


# ----------------------------------------------------------------------
# Test 3: m*_opt <= m*_mean whenever the mean plan is finite; and the
#         optimistic plan can escape m*=inf (q_mean <= tau < q_opt)
# ----------------------------------------------------------------------
def test3_mstar_optimistic_never_exceeds_mean():
    states = [(0, 0), (1, 3), (2, 5), (3, 10), (9, 20), (16, 40), (5, 5)]
    for (s, n) in states:
        for (a, b) in [(1, 1), (2, 4), (6, 4), (11, 21), (3, 2)]:
            qm = posterior_mean(a, b)
            qo = posterior_upper_quantile(a, b, DELTA)
            mm = v6.est_remaining_execs(s, n, qm, TAU, 4000, DELTA)
            mo = v6.est_remaining_execs(s, n, qo, TAU, 4000, DELTA)
            if np.isfinite(mm):
                assert np.isfinite(mo) and mo <= mm, \
                    f"s={s} n={n} a={a} b={b}: m_opt={mo} > m_mean={mm}"
            else:
                # optimistic may still be finite (fallback escape) - allowed
                assert np.isfinite(mo) or np.isinf(mo)


def test3b_fallback_escape_example():
    # a=1,b=3 -> q_mean = 0.25 <= tau -> m_mean = inf,
    # but q_opt > tau -> m_opt finite (optimism escapes the dead zone)
    a, b = 1.0, 3.0
    qm = posterior_mean(a, b)
    qo = posterior_upper_quantile(a, b, DELTA)
    assert qm <= TAU < qo, f"need q_mean<=tau<q_opt, got {qm}, {qo}"
    assert np.isinf(v6.est_remaining_execs(0, 0, qm, TAU, 4000, DELTA))
    assert np.isfinite(v6.est_remaining_execs(0, 0, qo, TAU, 4000, DELTA))


# ----------------------------------------------------------------------
# Test 4: confirmation semantics are untouched by the planning rate
# ----------------------------------------------------------------------
def test4_confirmation_unchanged():
    # frozen rule on known states (20/20 confirms; 9/10 does NOT under this CS)
    assert v6.cs_lower(20, 20, DELTA) > TAU
    assert v6.confirmed_sn(20, 20, TAU, DELTA)
    assert not v6.confirmed_sn(9, 10, TAU, DELTA)
    assert not v6.confirmed_sn(0, 10, TAU, DELTA)
    # identical observed-count sequences -> identical confirmation verdicts,
    # independent of which selector scheduled them (confirmation is
    # data-driven; the planning rate never enters it).
    gs = E.GS(4.0, 4.0, 4.0)
    arms1 = E.Arms(2, np.array([0.5, 0.5]), gs, TAU)
    arms2 = E.Arms(2, np.array([0.5, 0.5]), gs, TAU)
    for (f1, f2) in [(1, 1), (1, 1), (1, 0), (1, 1), (1, 0), (1, 1), (1, 1)]:
        arms1.update(0, f1, 4.0)
        arms2.update(0, f2, 4.0)
        assert arms1.confirmed[0] == arms2.confirmed[0]
    assert arms1.confirmed[0] == (
        v6.cs_lower(int(arms1.s[0]), int(arms1.n[0]), DELTA) > TAU)
    # source guard: the optimistic selectors never call the confirmation path
    src_s = inspect.getsource(OS)
    src_a = inspect.getsource(OA)
    assert "cs_lower(" not in src_s and "confirmed_sn" not in src_s
    assert "cs_lower(" not in src_a and "confirmed_sn" not in src_a


# ----------------------------------------------------------------------
# Test 5: fallback ordering uses the POSTERIOR MEAN (not the optimistic rate)
# ----------------------------------------------------------------------
def test5_fallback_uses_posterior_mean():
    gs = E.GS(10.0, 2.0, 6.0)
    # four arms with BOTH rates <= tau (m* = inf for both modes -> forced
    # fallback) but distinct costs.  a=1, b=11 gives q_mean = 1/12 = 0.083
    # and q_opt = Beta.ppf(0.95;1,11) = 0.238, both <= tau = 0.3.
    qm = posterior_mean(1.0, 11.0)
    qo = posterior_upper_quantile(1.0, 11.0, DELTA)
    assert qm <= TAU and qo <= TAU, f"setup broken: qm={qm} qo={qo}"
    arms = E.Arms(4, np.full(4, 0.1), gs, TAU)
    for i in range(4):
        arms.n[i] = 2
        arms.post_a[i] = 1.0
        arms.post_b[i] = 11.0
        arms.fail_cost_sum[i] = 1.0 * (i + 1)
        arms.fail_cost_cnt[i] = 1
        arms.pass_cost_sum[i] = 1.0 * (i + 1)
        arms.pass_cost_cnt[i] = 1
    sel = OS.ERCCSelOptimistic(False, 1000, diagnostics=False)
    st = sel.init(arms, np.random.default_rng(0))
    pick = sel.select(arms, st, np.random.default_rng(0))
    # frozen fallback: max q_mean/C -> arm0 (cheapest); identical to frozen ERCC
    sel_f = E.ERCCSel(False, 1000)
    st_f = sel_f.init(arms, np.random.default_rng(0))
    pick_f = sel_f.select(arms, st_f, np.random.default_rng(0))
    assert pick == 0 and pick_f == 0
    assert sel.n_fallback == 1


# ----------------------------------------------------------------------
# Test 6: determinism - same seed reproduces the same pick sequence
# ----------------------------------------------------------------------
def test6_determinism():
    def run_once():
        gs = E.GS(4.0, 4.0, 4.0)
        true_p = np.array([0.55] * 5 + [0.05] * 5)
        K = 10

        def draw(i, rng):
            f = int(rng.random() < true_p[i])
            return f, 4.0

        rng = np.random.default_rng(42)
        arms = E.Arms(K, np.full(K, 0.5), gs, TAU)
        sel = OS.ERCCSelOptimistic(False, 500, diagnostics=False)
        st = sel.init(arms, rng)
        picks = []
        t = 0.0
        while t < 120.0:
            i = sel.select(arms, st, rng)
            if i < 0:
                break
            picks.append(i)
            f, c = draw(i, rng)
            arms.update(i, f, c)
            sel.update(arms, i, st)
            t += c
        return picks, tuple(arms.confirmed)

    p1, c1 = run_once()
    p2, c2 = run_once()
    assert p1 == p2 and c1 == c2


# ----------------------------------------------------------------------
# Test 7: planning_rate dispatch
# ----------------------------------------------------------------------
def test7_mode_dispatch():
    a, b = 3.0, 7.0
    assert planning_rate(a, b, MODE_MEAN, DELTA) == posterior_mean(a, b)
    assert planning_rate(a, b, MODE_OPTIMISTIC, DELTA) == \
        posterior_upper_quantile(a, b, DELTA)
    with pytest.raises(ValueError):
        planning_rate(a, b, "thompson", DELTA)
    with pytest.raises(ValueError):
        posterior_upper_quantile(-1, 2, DELTA)
    with pytest.raises(ValueError):
        posterior_upper_quantile(1, 2, 1.5)


# ----------------------------------------------------------------------
# Frozen anchors (guard rails)
# ----------------------------------------------------------------------
def test_anchor_cold_start_quantiles():
    # Beta(1,1): q_mean = 0.5, q_opt = 0.95 exactly
    assert posterior_mean(1, 1) == 0.5
    assert abs(posterior_upper_quantile(1, 1, DELTA) - 0.95) < 1e-12
    # frozen anchor: m*(q0 = 0.5) = 176 (Part A A3d)
    assert v6.est_remaining_execs(0, 0, 0.5, TAU, 4000, DELTA) == 176
    # optimistic cold start must need strictly fewer planned executions
    m_opt = v6.est_remaining_execs(0, 0, 0.95, TAU, 4000, DELTA)
    assert np.isfinite(m_opt) and m_opt < 176


def test_anchor_sensodat_kappa_prior():
    # kappa=5, prior_score=0.4 -> a=3, b=4 -> q_mean = 3/7
    a = 1 + 5 * 0.4
    b = 1 + 5 * 0.6
    assert abs(posterior_mean(a, b) - 3 / 7) < 1e-12
    assert posterior_upper_quantile(a, b, DELTA) > 3 / 7


def test_anchor_amini_selector_interface():
    # the optimistic Amini adapter honours the (arms, pools, rng, st) contract
    routes = A.load_routes()
    gs = A.build_gs(routes)
    K = len(routes)
    orders = A.make_orders(routes, 2000)
    pools = [A.Pool(r.outcomes, r.durations, o) for r, o in zip(routes, orders)]
    arms = A.Arms(K, gs)
    sel = OA.ReproAllocOptAdapter(False, 100, diagnostics=True)
    st = sel.init(arms, pools, np.random.default_rng(0))
    i = sel.select(arms, pools, np.random.default_rng(0), st)
    assert 0 <= i < K
    o, d = pools[i].draw()
    arms.update(i, o, d)
    sel.update(arms, i, st)
    assert len(sel.inner.last_diag) == 1
    rec = sel.inner.last_diag[0]
    # the diag is recorded AFTER the first update, so q_mean reflects the draw
    assert abs(rec["q_mean"] - rec["post_a"] / (rec["post_a"] + rec["post_b"])) < 1e-12
    assert rec["q_opt"] > rec["q_mean"]
    assert rec["m_opt"] > 0 and 0 < rec["m_opt"] < rec["m_mean"]
