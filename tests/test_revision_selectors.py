import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import experiment_amini_finite_trace as A
import optimistic_ercc_amini as OA
import optimistic_ercc_sensodat as OS
import v7_experiment as S
from revision_selectors import (
    AminiBayesUCBCost, AminiRerunK, AminiRevisionERCC,
    SensoBayesUCBCost, SensoRerunK, SensoRevisionERCC,
)
from revision_cap_diagnostics import SensoCapDiagnostic, AminiCapDiagnostic


def test_default_revision_orcc_matches_frozen_amini():
    routes = A.load_routes()
    gs = A.build_gs(routes)
    orders = A.make_orders(routes, 2000)
    ref = np.array([r.p_hat > A.TAU for r in routes])
    ceiling, _ = A.oracle_ceiling_for_order(routes, orders)
    budget = sum(sum(r.durations) for r in routes) * .10
    cap = A.m_cap_from_budget(budget, gs)
    frozen = OA.ReproAllocOptAdapter(False, cap, diagnostics=False)
    revised = AminiRevisionERCC(cap)
    a = A.run_one(routes, orders, ref, budget, frozen, A.method_rng(2000, 6, 1), gs, ceiling)
    b = A.run_one(routes, orders, ref, budget, revised, A.method_rng(2000, 6, 1), gs, ceiling)
    assert a["tp"] == b["tp"]
    assert a["time_used_s"] == b["time_used_s"]
    assert a["_alloc"] == b["_alloc"]


def test_default_revision_orcc_matches_frozen_sensodat_on_small_pool():
    gs = S.GS(1, 1, 1)
    score = np.array([.3, .6, .8, .9])
    fail = np.array([0, 1, 1, 1], dtype=bool)
    def draw(i, rng):
        return int(fail[i]), 1.0
    for selector in (OS.ERCCSelOptimistic(False, 40, diagnostics=False),
                     SensoRevisionERCC(40)):
        result = S.run_task(4, draw, fail, 35, score, gs, .3,
                            selector, np.random.default_rng(7))
        if type(selector) is OS.ERCCSelOptimistic:
            frozen = result
        else:
            assert result["tp"] == frozen["tp"]
            assert result["time_used"] == frozen["time_used"]


def test_rerunk_focuses_for_k_further_executions():
    gs = S.GS(1, 1, 1)
    arms = S.Arms(2, np.array([.9, .1]), gs, .3)
    sel = SensoRerunK(K=2)
    rng = np.random.default_rng(0)
    st = sel.init(arms, rng)
    i = sel.select(arms, st, rng)
    arms.update(i, 1, 1)
    sel.update(arms, i, st)
    assert st["focus"] == i and st["remaining"] == 2
    for remaining in (1, 0):
        assert sel.select(arms, st, rng) == i
        arms.update(i, 1, 1)
        sel.update(arms, i, st)
        assert st["remaining"] == remaining
    assert st["focus"] == -1


def test_amini_rerunk_and_bayes_follow_shared_interface():
    routes = A.load_routes()
    gs = A.build_gs(routes)
    orders = A.make_orders(routes, 2000)
    ref = np.array([r.p_hat > A.TAU for r in routes])
    ceiling, _ = A.oracle_ceiling_for_order(routes, orders)
    for selector in (AminiRerunK(10), AminiBayesUCBCost()):
        result = A.run_one(routes, orders, ref, 10000, selector,
                           A.method_rng(2000, 8, 0), gs, ceiling)
        assert result["executions"] > 0
        assert result["fp"] == 0


def test_senso_bayes_uses_current_execution_count():
    sel = SensoBayesUCBCost()
    arms = S.Arms(2, np.array([.5, .5]), S.GS(1, 1, 1), .3)
    rng = np.random.default_rng(0)
    st = sel.init(arms, rng)
    assert st["t"] == 1
    i = sel.select(arms, st, rng)
    arms.update(i, 1, 1)
    sel.update(arms, i, st)
    assert st["t"] == 2


def test_senso_bayes_optimized_selection_matches_full_quantiles():
    prior = np.array([.15, .8, .8, .4, .9, .3, .6, .9])
    arms = S.Arms(len(prior), prior, S.GS(2, 3, 2.5), .3)
    # Several visited arms have changed posteriors and observed costs.
    for i, fail, cost in ((0, 1, 2), (2, 0, 4), (4, 1, 1), (4, 0, 2)):
        arms.update(i, fail, cost)
    selector = SensoBayesUCBCost()
    for t_exec in (1, 2, 10, 100, 1000):
        st = {"t": t_exec}
        rng_fast = np.random.default_rng(43)
        rng_full = np.random.default_rng(43)
        got = selector.select(arms, st, rng_fast)
        cand = np.flatnonzero(~arms.confirmed)
        rho = 1 - 1 / (t_exec + 1)
        score = beta.ppf(rho, arms.post_a[cand], arms.post_b[cand]) / arms.mean_cost_arr()[cand]
        ties = cand[score >= score.max() - 1e-12]
        expected = int(ties[rng_full.integers(0, len(ties))])
        assert got == expected


def test_cap_diagnostics_do_not_change_amini_run():
    routes = A.load_routes()
    gs = A.build_gs(routes)
    orders = A.make_orders(routes, 2000)
    ref = np.array([r.p_hat > A.TAU for r in routes])
    ceiling, _ = A.oracle_ceiling_for_order(routes, orders)
    budget = sum(sum(r.durations) for r in routes) * .05
    cap = A.m_cap_from_budget(budget, gs)
    base = AminiRevisionERCC(cap)
    diag = AminiCapDiagnostic(cap)
    a = A.run_one(routes, orders, ref, budget, base, A.method_rng(2000, 6, 0), gs, ceiling)
    b = A.run_one(routes, orders, ref, budget, diag, A.method_rng(2000, 6, 0), gs, ceiling)
    assert a["_alloc"] == b["_alloc"]
    assert diag.steps == b["executions"]


def test_cap_diagnostics_do_not_change_sensodat_run():
    gs = S.GS(1, 1, 1)
    score = np.array([.3, .6, .8, .9])
    fail = np.array([0, 1, 1, 1], dtype=bool)
    def draw(i, rng):
        return int(fail[i]), 1.0
    base = SensoRevisionERCC(20)
    diag = SensoCapDiagnostic(20)
    a = S.run_task(4, draw, fail, 25, score, gs, .3, base, np.random.default_rng(7))
    b = S.run_task(4, draw, fail, 25, score, gs, .3, diag, np.random.default_rng(7))
    assert a["time_used"] == b["time_used"]
    assert a["tp"] == b["tp"]
    assert diag.steps == b["executions"]
