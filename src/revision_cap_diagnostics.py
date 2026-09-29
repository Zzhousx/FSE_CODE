"""Exact paired cap diagnostics without changing the frozen selection streams."""

import copy
import numpy as np

import v7_experiment as S
import experiment_amini_finite_trace as A
from revision_selectors import SensoRevisionERCC, AminiRevisionERCC, _mstar


def _clone_rng(rng):
    clone = np.random.default_rng()
    clone.bit_generator.state = copy.deepcopy(rng.bit_generator.state)
    return clone


class SensoCapDiagnostic(SensoRevisionERCC):
    def __init__(self, m_cap, mode="optimistic"):
        super().__init__(m_cap, planning_cap="budget", mode=mode)
        self.steps = self.cap_bound_steps = self.changed_steps = 0

    def _uncapped(self, arms, i):
        if arms.confirmed[i]:
            return -np.inf, False
        capped = self._ercc_at(arms, None, i, self.mode)[0]
        if capped > 0:
            return capped, False
        q = self._rate(arms, i, self.mode)
        m = _mstar(int(arms.s[i]), int(arms.n[i]), q, float(arms.tau), .05, None)
        if np.isinf(m):
            return 0.0, False
        return 1 / max(m * S._mean_cost(arms, i), S.EPS), True

    def init(self, arms, rng):
        st = super().init(arms, rng)
        values = [self._uncapped(arms, i) for i in range(arms.K)]
        st["uncapped"] = np.array([v for v, _ in values])
        st["cap_bound"] = np.array([b for _, b in values])
        return st

    def select(self, arms, st, rng):
        clone = _clone_rng(rng)
        pick = super().select(arms, st, rng)
        if pick < 0:
            return pick
        self.steps += 1
        self.cap_bound_steps += bool(st["cap_bound"].any())
        cand = np.flatnonzero(~arms.confirmed)
        util = st["uncapped"]
        pos = cand[util[cand] > 0]
        if len(pos):
            mx = util[pos].max()
            ties = pos[util[pos] >= mx - 1e-12]
            other = int(ties[clone.integers(0, len(ties))])
        else:
            fb = np.array([S._scorecost_scalar(arms, i, False) for i in cand])
            ties = cand[fb >= fb.max() - 1e-12]
            other = int(ties[clone.integers(0, len(ties))])
        self.changed_steps += other != pick
        return pick

    def update(self, arms, i, st):
        super().update(arms, i, st)
        st["uncapped"][i], st["cap_bound"][i] = self._uncapped(arms, i)


class AminiCapDiagnostic(AminiRevisionERCC):
    def __init__(self, m_cap, mode="optimistic"):
        super().__init__(m_cap, planning_cap="budget", mode=mode)
        self.steps = self.cap_bound_steps = self.changed_steps = 0

    def _uncapped(self, arms, i):
        if arms.confirmed[i]:
            return -np.inf, False
        capped = self.inner._util_mode(arms, None, i, self.inner.mode)[0]
        if capped > 0:
            return capped, False
        q = self.inner._rate(arms, i, self.inner.mode)
        m = _mstar(int(arms.s[i]), int(arms.n[i]), q, A.TAU, A.DELTA, None)
        if np.isinf(m):
            return 0.0, False
        return 1 / max(m * max(arms.mean_cost(i), 1e-9), 1e-9), True

    def init(self, arms, pools, rng):
        st = super().init(arms, pools, rng)
        values = [self._uncapped(arms, i) for i in range(arms.K)]
        st["uncapped"] = np.array([v for v, _ in values])
        st["cap_bound"] = np.array([b for _, b in values])
        return st

    def select(self, arms, pools, rng, st):
        clone = _clone_rng(rng)
        pick = super().select(arms, pools, rng, st)
        if pick < 0:
            return pick
        self.steps += 1
        cand = np.flatnonzero(A._eligible(arms, pools))
        self.cap_bound_steps += bool(st["cap_bound"][cand].any())
        util = st["uncapped"]
        pos = cand[util[cand] > 0]
        if len(pos):
            other = A._pick_max(pos, util[pos], clone)
        else:
            fb = np.array([arms.q(i) / max(arms.mean_cost(i), 1e-9) for i in cand])
            other = A._pick_max(cand, fb, clone)
        self.changed_steps += other != pick
        return pick

    def update(self, arms, i, st):
        super().update(arms, i, st)
        st["uncapped"][i], st["cap_bound"][i] = self._uncapped(arms, i)
