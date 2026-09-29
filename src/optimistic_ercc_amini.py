"""Amini-TransFuser-side Optimistic ERCC selector (finite-trace harness).

ReproAllocSelOptimistic is the frozen A.ReproAllocSel except that the rate
fed into est_remaining_execs is planning_rate(mode) instead of arms.q(i)
(posterior mean).  Default mode = "optimistic".

Unchanged, by construction:
  * utility: I = 1 / (m* * C), C = mean cost (use_oa=False) or OA cost (True);
  * m* = inf -> utility 0 -> the adapter's frozen ScoreCost fallback, which
    keeps using arms.q(i) (POSTERIOR MEAN);
  * finite-pool eligibility, sticky confirmation, tie-breaks.
"""
import numpy as np

import reproalloc_v6 as v6
import experiment_amini_finite_trace as A
from optimistic_ercc import planning_rate, posterior_mean, MODE_OPTIMISTIC


class ReproAllocSelOptimistic(A.ReproAllocSel):
    """ReproAlloc-Optimistic for the Amini finite-trace harness."""

    def __init__(self, use_oa, m_cap, mode=MODE_OPTIMISTIC, delta=0.05,
                 diagnostics=True):
        super().__init__(use_oa, m_cap)
        self.mode = mode
        self.delta = float(delta)
        self.diagnostics = diagnostics
        self.name = ("ReproAlloc-OA-Optimistic" if use_oa
                     else "ReproAlloc-Optimistic")

    def _rate(self, arms, i, mode):
        return planning_rate(arms.post_a[i], arms.post_b[i], mode, self.delta)

    def _util_mode(self, arms, cache, i, mode):
        if arms.confirmed[i]:
            return -np.inf, np.inf
        qi = self._rate(arms, i, mode)
        m_star = cache[i].get(int(arms.s[i]), int(arms.n[i]), qi,
                              A.TAU, self.m_cap, A.DELTA)
        if np.isinf(m_star):
            return 0.0, m_star
        c = arms.oa_cost(i) if self.use_oa else arms.mean_cost(i)
        return 1.0 / max(m_star * max(c, 1e-9), 1e-9), m_star

    def _util(self, arms, cache, i):
        u, _ = self._util_mode(arms, cache, i, self.mode)
        return u

    def init(self, arms, pools, rng):
        st = super().init(arms, pools, rng)
        if self.diagnostics:
            K = arms.K
            cache_mean = [v6.ERCCache() for _ in range(K)]
            util_mean = np.array([self._util_mode(arms, cache_mean, i, "mean")[0]
                                  for i in range(K)])
            st.update({"cache_mean": cache_mean, "util_mean": util_mean,
                       "_diag": [], "_pending_select": None})
            self.last_diag = st["_diag"]
        return st

    def update(self, arms, i, st):
        super().update(arms, i, st)
        if self.diagnostics:
            u_m, m_mean = self._util_mode(arms, st["cache_mean"], i, "mean")
            st["util_mean"][i] = u_m
            _, m_opt = self._util_mode(arms, st["cache"], i, self.mode)
            pend = st["_pending_select"]
            st["_diag"].append({
                "arm": int(i),
                "s": int(arms.s[i]), "n": int(arms.n[i]),
                "post_a": float(arms.post_a[i]), "post_b": float(arms.post_b[i]),
                "q_mean": posterior_mean(arms.post_a[i], arms.post_b[i]),
                "q_opt": self._rate(arms, i, self.mode),
                "m_mean": (float(m_mean) if np.isfinite(m_mean) else -1.0),
                "m_opt": (float(m_opt) if np.isfinite(m_opt) else -1.0),
                "confirmed": bool(arms.confirmed[i]),
                "sel_kind": (pend["kind"] if pend else ""),
                "would_differ": (bool(pend["would_differ"]) if pend else False),
            })
            st["_pending_select"] = None


class ReproAllocOptAdapter:
    """Mirror of A._ReproAllocAdapter wrapping the optimistic selector.

    The fallback below is the FROZEN one: posterior-mean ScoreCost ordering
    (arms.q(i) / mean_cost) - intentionally NOT optimistic.
    """

    def __init__(self, use_oa, m_cap, mode=MODE_OPTIMISTIC, diagnostics=True):
        self.inner = ReproAllocSelOptimistic(use_oa, m_cap, mode=mode,
                                             diagnostics=diagnostics)
        self.name = self.inner.name

    def init(self, arms, pools, rng):
        return self.inner.init(arms, pools, rng)

    def select(self, arms, pools, rng, st):
        elig = A._eligible(arms, pools)
        cand = np.flatnonzero(elig)
        if cand.size == 0:
            return -1
        util = st["util"]
        pos = cand[util[cand] > 0.0]
        if pos.size > 0:
            pick = A._pick_max(pos, util[pos], rng)
            if self.inner.diagnostics:
                um = st["util_mean"]
                pos_m = cand[um[cand] > 0.0]
                st["_pending_select"] = {
                    "kind": "main", "pick": pick,
                    "mean_pick": (A._pick_max(pos_m, um[pos_m],
                                              np.random.default_rng(0))
                                  if pos_m.size > 0 else -1),
                    "mean_kind": ("main" if pos_m.size > 0 else "fallback"),
                    "would_differ": bool(
                        pos_m.size == 0
                        or A._pick_max(pos_m, um[pos_m],
                                       np.random.default_rng(0)) != pick)}
            return pick
        # frozen fallback: posterior-MEAN ScoreCost ordering (unchanged)
        fb = np.array([arms.q(i) / max(arms.mean_cost(i), 1e-9) for i in cand])
        pick = A._pick_max(cand, fb, rng)
        if self.inner.diagnostics:
            um = st["util_mean"]
            pos_m = cand[um[cand] > 0.0]
            st["_pending_select"] = {
                "kind": "fallback", "pick": pick,
                "mean_pick": (A._pick_max(pos_m, um[pos_m],
                                          np.random.default_rng(0))
                              if pos_m.size > 0 else pick),
                "mean_kind": ("main" if pos_m.size > 0 else "fallback"),
                "would_differ": bool(pos_m.size > 0)}
        return pick

    def update(self, arms, i, st):
        self.inner.update(arms, i, st)


def build_selectors_7(m_cap, diagnostics=True):
    """Frozen six + ReproAlloc-Optimistic (mean cost)."""
    sels = A.build_selectors(m_cap)          # the frozen six, wrapped
    sels.append(ReproAllocOptAdapter(False, m_cap, diagnostics=diagnostics))
    return sels


class OptimisticScoreCostSel:
    """OptimisticScoreCost for the Amini finite-trace harness.

    Isomorphic to experiment_amini_finite_trace.ScoreCostSel; the ONLY method
    difference is the ranking scalar: q_O / mean_cost instead of q / mean_cost,
    with q_O = Beta.ppf(1-delta; post_a, post_b) (identical planning rate to
    ReproAlloc).  Same _eligible filtering (unconfirmed, pool not exhausted)
    and the same _pick_max tie-break.
    """
    name = "OptimisticScoreCost"

    def __init__(self, delta=0.05):
        self.delta = float(delta)

    def init(self, arms, pools, rng):
        return {}

    def select(self, arms, pools, rng):
        cand = np.flatnonzero(A._eligible(arms, pools))
        if cand.size == 0:
            return -1
        score = np.array([
            planning_rate(arms.post_a[i], arms.post_b[i],
                          MODE_OPTIMISTIC, self.delta)
            / max(arms.mean_cost(i), 1e-9) for i in cand])
        return A._pick_max(cand, score, rng)

    def update(self, arms, i, st):
        pass


def build_selectors_8(m_cap, diagnostics=True):
    """The seven of build_selectors_7 + OptimisticScoreCost (appended last,
    so existing method indices 0..6 keep their method_rng streams)."""
    sels = build_selectors_7(m_cap, diagnostics=diagnostics)
    sels.append(A._wrap(OptimisticScoreCostSel()))
    return sels
