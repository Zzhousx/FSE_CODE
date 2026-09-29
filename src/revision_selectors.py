"""Revision variants through the two existing scheduler interfaces."""

from functools import lru_cache

import numpy as np
from scipy.stats import beta

import reproalloc_v6 as V
import v7_experiment as S
import experiment_amini_finite_trace as A
import optimistic_ercc_sensodat as OS
import optimistic_ercc_amini as OA


@lru_cache(maxsize=300_000)
def _mstar(s, n, q, tau, delta, cap):
    return V.compute_mstar(s, n, q, tau, delta, cap=cap)


class SensoRevisionERCC(OS.ERCCSelOptimistic):
    """ORCC with configurable planning quantile and optional uncapped search."""

    def __init__(self, m_cap, planning_quantile=0.95, planning_cap="budget",
                 mode="optimistic"):
        super().__init__(False, m_cap, mode=mode, diagnostics=False)
        if planning_cap not in ("budget", "none"):
            raise ValueError(planning_cap)
        if not 0 < planning_quantile < 1:
            raise ValueError(planning_quantile)
        self.rho = planning_quantile
        self.planning_cap = planning_cap

    def _rate(self, arms, i, mode):
        a, b = arms.post_a[i], arms.post_b[i]
        return float(a / (a + b)) if mode == "mean" else float(beta.ppf(self.rho, a, b))

    def _ercc_at(self, arms, cache, i, mode):
        if arms.confirmed[i]:
            return -np.inf, np.inf
        q = self._rate(arms, i, mode)
        cap = self.m_cap if self.planning_cap == "budget" else None
        m = _mstar(int(arms.s[i]), int(arms.n[i]), q,
                   float(arms.tau), V.DELTA, cap)
        if np.isinf(m):
            return 0.0, m
        return 1.0 / max(m * S._mean_cost(arms, i), S.EPS), m


class AminiRevisionInner(OA.ReproAllocSelOptimistic):
    def __init__(self, m_cap, planning_quantile=0.95, planning_cap="budget",
                 mode="optimistic"):
        super().__init__(False, m_cap, mode=mode, diagnostics=False)
        if planning_cap not in ("budget", "none"):
            raise ValueError(planning_cap)
        if not 0 < planning_quantile < 1:
            raise ValueError(planning_quantile)
        self.rho = planning_quantile
        self.planning_cap = planning_cap

    def _rate(self, arms, i, mode):
        a, b = arms.post_a[i], arms.post_b[i]
        return float(a / (a + b)) if mode == "mean" else float(beta.ppf(self.rho, a, b))

    def _util_mode(self, arms, cache, i, mode):
        if arms.confirmed[i]:
            return -np.inf, np.inf
        q = self._rate(arms, i, mode)
        cap = self.m_cap if self.planning_cap == "budget" else None
        m = _mstar(int(arms.s[i]), int(arms.n[i]), q,
                   A.TAU, A.DELTA, cap)
        if np.isinf(m):
            return 0.0, m
        return 1.0 / max(m * max(arms.mean_cost(i), 1e-9), 1e-9), m


class AminiRevisionERCC(OA.ReproAllocOptAdapter):
    def __init__(self, m_cap, planning_quantile=0.95, planning_cap="budget",
                 mode="optimistic"):
        self.inner = AminiRevisionInner(m_cap, planning_quantile, planning_cap, mode)
        self.name = self.inner.name


class SensoRerunK(S.ScoreCostSel):
    def __init__(self, K=10):
        super().__init__(False)
        self.K = int(K)

    def init(self, arms, rng):
        st = super().init(arms, rng)
        st.update({"focused_ever": np.zeros(arms.K, dtype=bool),
                   "focus": -1, "remaining": 0})
        return st

    def select(self, arms, st, rng):
        i = st["focus"]
        if i >= 0 and st["remaining"] > 0 and not arms.confirmed[i]:
            return i
        st["focus"] = -1
        return super().select(arms, st, rng)

    def update(self, arms, i, st):
        super().update(arms, i, st)
        if st["focus"] == i:
            st["remaining"] -= 1
            if st["remaining"] <= 0 or arms.confirmed[i]:
                st["focus"] = -1
        elif arms.s[i] > 0 and not st["focused_ever"][i] and not arms.confirmed[i]:
            st["focused_ever"][i] = True
            st["focus"] = i
            st["remaining"] = self.K


class AminiRerunK:
    name = "RerunK"

    def __init__(self, K=10):
        self.K = int(K)

    def init(self, arms, pools, rng):
        return {"focused_ever": np.zeros(arms.K, dtype=bool),
                "focus": -1, "remaining": 0}

    def select(self, arms, pools, rng, st):
        i = st["focus"]
        if i >= 0 and st["remaining"] > 0 and not arms.confirmed[i] and not pools[i].exhausted():
            return i
        st["focus"] = -1
        cand = np.flatnonzero(A._eligible(arms, pools))
        if not len(cand):
            return -1
        score = np.array([arms.q(i) / max(arms.mean_cost(i), 1e-9) for i in cand])
        return A._pick_max(cand, score, rng)

    def update(self, arms, i, st):
        if st["focus"] == i:
            st["remaining"] -= 1
            if st["remaining"] <= 0 or arms.confirmed[i]:
                st["focus"] = -1
        elif arms.s[i] > 0 and not st["focused_ever"][i] and not arms.confirmed[i]:
            st["focused_ever"][i] = True
            st["focus"] = i
            st["remaining"] = self.K


class SensoBayesUCBCost:
    n_main_pos = 0
    n_fallback = 0

    def init(self, arms, rng):
        return {"t": 1}

    def select(self, arms, st, rng):
        cand = np.flatnonzero(~arms.confirmed)
        if not len(cand):
            return -1
        rho = 1 - 1 / (st["t"] + 1)
        # Every untouched arm has the same Beta total strength and pool-level
        # cost. Beta quantiles are then ordered by the prior failure score, for
        # every rho. Only the highest-prior untouched group can win. Retain a
        # conservative near-tie group so the original 1e-12 tie rule is exact.
        unseen = cand[arms.n[cand] == 0]
        seen = cand[arms.n[cand] > 0]
        if len(unseen):
            top_a = arms.post_a[unseen].max()
            unseen = unseen[arms.post_a[unseen] >= top_a - 1e-9]
        considered = np.sort(np.r_[seen, unseen])
        score = beta.ppf(rho, arms.post_a[considered], arms.post_b[considered]) / np.maximum(
            arms.mean_cost_arr()[considered], S.EPS)
        mx = score.max()
        ties = considered[score >= mx - 1e-12]
        return int(ties[rng.integers(0, len(ties))])

    def update(self, arms, i, st):
        st["t"] += 1


class AminiBayesUCBCost:
    name = "BayesUCB-Cost"

    def init(self, arms, pools, rng):
        return {"t": 1}

    def select(self, arms, pools, rng, st):
        cand = np.flatnonzero(A._eligible(arms, pools))
        if not len(cand):
            return -1
        rho = 1 - 1 / (st["t"] + 1)
        score = beta.ppf(rho, arms.post_a[cand], arms.post_b[cand]) / np.array(
            [max(arms.mean_cost(i), 1e-9) for i in cand])
        return A._pick_max(cand, score, rng)

    def update(self, arms, i, st):
        st["t"] += 1
