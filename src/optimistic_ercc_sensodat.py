"""SensoDat-side Optimistic ERCC selector (v7_experiment harness).

ERCCSelOptimistic is byte-for-byte the frozen E.ERCCSel except that the rate
fed into est_remaining_execs comes from planning_rate(mode) instead of the
posterior mean _q().  Default mode = "optimistic".

Everything else is inherited/duplicated unchanged:
  * index: I = 1 / (m* * C),  C = mean cost (use_oc=False) or OA cost (True);
  * m* = inf  -> index 0 -> fixed ScoreCost fallback with the POSTERIOR MEAN
    (_scorecost_scalar, untouched);
  * eligible set / tie-breaks / confirmed exclusion.

Diagnostics (for logs/optimistic_ercc/):
  st["_diag"] accumulates one record per update with both rates and both m*
  values plus whether the optimistic pick would have differed under the mean
  index at the most recent select().  Overhead: one extra est_remaining_execs
  call per update on one arm (cached).
"""
import numpy as np

import reproalloc_v6 as v6
import v7_experiment as E
from optimistic_ercc import planning_rate, posterior_mean, MODE_OPTIMISTIC

EPS = 1e-9


class ERCCSelOptimistic(E.ERCCSel):
    """ReproAlloc-Optimistic (use_oc=False).  use_oc=True kept for parity."""

    def __init__(self, use_oc, m_cap, mode=MODE_OPTIMISTIC, delta=0.05,
                 diagnostics=True):
        super().__init__(use_oc, m_cap)
        self.mode = mode
        self.delta = float(delta)
        self.diagnostics = diagnostics

    def _rate(self, arms, i, mode):
        return planning_rate(arms.post_a[i], arms.post_b[i], mode, self.delta)

    def _ercc_at(self, arms, cache, i, mode):
        """ERCC index of arm i under the given planning-rate mode."""
        if arms.confirmed[i]:
            return -np.inf, np.inf
        q = self._rate(arms, i, mode)
        m_star = cache.get(int(arms.s[i]), int(arms.n[i]), q, arms.tau,
                           self.m_cap)
        if np.isinf(m_star):
            return 0.0, m_star
        cost = E._oa_cost(arms, i) if self.uoc else E._mean_cost(arms, i)
        return 1.0 / max(m_star * cost, EPS), m_star

    def init(self, arms, rng):
        K = arms.K
        cache = [v6.ERCCache() for _ in range(K)]
        cache_mean = [v6.ERCCache() for _ in range(K)] if self.diagnostics else None
        ercc = np.zeros(K)
        ercc_mean = np.zeros(K) if self.diagnostics else None
        for i in range(K):
            ercc[i], _ = self._ercc_at(arms, cache[i], i, self.mode)
            if self.diagnostics:
                ercc_mean[i], _ = self._ercc_at(arms, cache_mean[i], i, "mean")
        st = {"ercc": ercc, "cache": cache}
        if self.diagnostics:
            st.update({"cache_mean": cache_mean, "ercc_mean": ercc_mean,
                       "_diag": [], "_pending_select": None})
            self.last_diag = st["_diag"]
        return st

    def select(self, arms, st, rng):
        cand = np.where(~arms.confirmed)[0]
        if cand.size == 0:
            return -1
        ercc = st["ercc"]
        pos = cand[ercc[cand] > 0.0]
        if pos.size > 0:
            self.n_main_pos += 1
            mx = np.max(ercc[pos])
            ties = pos[ercc[pos] >= mx - 1e-12]
            pick = int(ties[rng.integers(0, ties.size)])
            if self.diagnostics:
                # would the frozen (mean) index have picked differently?
                em = st["ercc_mean"]
                pos_m = cand[em[cand] > 0.0]
                if pos_m.size > 0:
                    mxm = np.max(em[pos_m])
                    mean_pick = int(pos_m[em[pos_m] >= mxm - 1e-12][0])
                    mean_kind = "main"
                else:
                    mean_pick, mean_kind = -1, "fallback"
                st["_pending_select"] = {
                    "kind": "main", "pick": pick,
                    "mean_pick": mean_pick, "mean_kind": mean_kind,
                    "would_differ": bool(mean_pick != pick)}
            return pick
        # all-zero -> frozen fallback (posterior-mean ScoreCost, unchanged)
        self.n_fallback += 1
        fb = np.array([E._scorecost_scalar(arms, i, self.uoc) for i in cand])
        mx = np.max(fb)
        ties = cand[fb >= mx - 1e-12]
        pick = int(ties[rng.integers(0, ties.size)])
        if self.diagnostics:
            em = st["ercc_mean"]
            pos_m = cand[em[cand] > 0.0]
            st["_pending_select"] = {
                "kind": "fallback", "pick": pick,
                "mean_pick": (int(pos_m[np.argmax(em[pos_m])])
                              if pos_m.size > 0 else pick),
                "mean_kind": ("main" if pos_m.size > 0 else "fallback"),
                "would_differ": bool(pos_m.size > 0)}
        return pick

    def update(self, arms, i, st):
        ercc_new, m_opt = self._ercc_at(arms, st["cache"][i], i, self.mode)
        st["ercc"][i] = ercc_new
        if self.diagnostics:
            ercc_m, m_mean = self._ercc_at(arms, st["cache_mean"][i], i, "mean")
            st["ercc_mean"][i] = ercc_m
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


def make_sensodat_selectors(m_cap, diagnostics=True):
    """The frozen six + ReproAlloc-Optimistic (mean cost)."""
    sels = E.make_sels(m_cap) if hasattr(E, "make_sels") else None
    if sels is None:  # v7_experiment has no make_sels; build like the frozen final
        sels = {"uniform": E.FnSel("uniform"),
                "failrerun": E.FnSel("failrerun"),
                "apgai": E.FnSel("apgai"),
                "scorecost": E.ScoreCostSel(False),
                "ercc_mean": E.ERCCSel(False, m_cap),
                "ercc_outcome": E.ERCCSel(True, m_cap)}
    sels["ercc_optimistic"] = ERCCSelOptimistic(False, m_cap,
                                                diagnostics=diagnostics)
    return sels


METHODS_7 = [("Uniform", "uniform"),
             ("FailRerun", "failrerun"),
             ("APGAI", "apgai"),
             ("ScoreCost", "scorecost"),
             ("ReproAlloc", "ercc_mean"),
             ("ReproAlloc-OA", "ercc_outcome"),
             ("ReproAlloc-Optimistic", "ercc_optimistic")]


class OptimisticScoreCostSel:
    """OptimisticScoreCost: ScoreCost with the optimistic rate.

    Line-by-line isomorphic to v7_experiment.ScoreCostSel; the ONLY method
    difference is the ranking scalar: q_O / C_bar instead of q_mean / C_bar,
    where q_O = Beta.ppf(1-delta; post_a, post_b) (identical planning rate to
    ReproAlloc).  No m*, no fallback change: ScoreCostSel never had an m*
    fallback branch (its max score is always > 0 while candidates remain).
    """

    def __init__(self, use_oc, delta=0.05):
        self.uoc = use_oc
        self.delta = float(delta)
        self.n_main_pos = 0
        self.n_fallback = 0

    def _scalar(self, arms, i):
        if arms.confirmed[i]:
            return -np.inf
        q = planning_rate(arms.post_a[i], arms.post_b[i],
                          MODE_OPTIMISTIC, self.delta)
        denom = E._oa_cost(arms, i) if self.uoc else E._mean_cost(arms, i)
        return q / max(denom, EPS)

    def init(self, arms, rng):
        idx = np.array([self._scalar(arms, i) for i in range(arms.K)])
        idx = idx + rng.random(arms.K) * 1e-12
        return {"idx": idx}

    def select(self, arms, st, rng):
        idx = st["idx"]
        elig = np.flatnonzero(idx > -np.inf)
        if elig.size == 0:
            return -1
        mx = np.max(idx[elig])
        if mx > 0.0:
            self.n_main_pos += 1
        else:
            self.n_fallback += 1
        ties = elig[idx[elig] >= mx - 1e-12]
        return int(ties[rng.integers(0, ties.size)])

    def update(self, arms, i, st):
        st["idx"][i] = self._scalar(arms, i)


def make_sensodat_selectors_8(m_cap, diagnostics=True):
    """The seven of make_sensodat_selectors + OptimisticScoreCost (appended)."""
    sels = make_sensodat_selectors(m_cap, diagnostics=diagnostics)
    sels["optimistic_scorecost"] = OptimisticScoreCostSel(False)
    return sels


METHODS_8 = METHODS_7 + [("OptimisticScoreCost", "optimistic_scorecost")]
