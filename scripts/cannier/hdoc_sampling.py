"""HDoC sampling/elimination selector shared by SensoDat, Amini, CANNIER."""
from __future__ import annotations

import numpy as np

from reproalloc_v6 import cs_lower


class HDoCSampling:
    """Execute one randomized initialization pass, then HDoC UCB/elimination."""
    name = "HDoC-Sampling"

    def __init__(self, K: int, tau: float = .3, delta: float = .05):
        self.K = int(K)
        self.tau = float(tau)
        self.delta = float(delta)

    def run(self, arms, env, budget: float, rng, finite_pool: bool):
        import time
        sched = 0.0
        used = 0.0
        init_used = 0.0
        executions = 0
        eliminated = np.zeros(self.K, dtype=bool)
        active = np.ones(self.K, dtype=bool)
        t0 = time.perf_counter()
        init_order = rng.permutation(self.K)
        sched += time.perf_counter() - t0
        init_completed = True

        def eligible(i):
            return (not arms.confirmed[i] and not eliminated[i]
                    and (env.has_remaining(i) if finite_pool else True))

        for i0 in init_order:
            i = int(i0)
            if used >= budget:
                init_completed = False
                break
            if not env.has_remaining(i):
                active[i] = False
                continue
            y, c = env.execute(i)
            t0 = time.perf_counter()
            arms.update(i, int(y), float(c))
            sched += time.perf_counter() - t0
            executions += 1
            used += float(c)
            init_used += float(c)
            if arms.confirmed[i]:
                active[i] = False
            elif finite_pool and not env.has_remaining(i):
                active[i] = False
        else:
            init_completed = True

        # HDoC's sampling phase begins only after the complete initialization.
        t = executions
        while init_completed and used < budget:
            t0 = time.perf_counter()
            cand = np.flatnonzero(active & ~arms.confirmed & ~eliminated)
            if cand.size == 0:
                break
            if t <= 0:
                raise AssertionError("HDoC adaptive phase cannot start with t=0")
            means = arms.s[cand] / np.maximum(arms.n[cand], 1)
            ucb = means + np.sqrt(np.log(t) / (2.0 * np.maximum(arms.n[cand], 1)))
            mx = float(np.max(ucb))
            ties = cand[ucb >= mx - 1e-12]
            i = int(ties[rng.integers(0, len(ties))])
            sched += time.perf_counter() - t0
            y, c = env.execute(i)
            t0 = time.perf_counter()
            arms.update(i, int(y), float(c))
            sched += time.perf_counter() - t0
            executions += 1
            used += float(c)
            t += 1
            n = int(arms.n[i]); s = int(arms.s[i])
            if arms.confirmed[i]:
                active[i] = False
            elif finite_pool and not env.has_remaining(i):
                active[i] = False
            elif s / n + np.sqrt(np.log(4.0 * self.K * n * n / self.delta)
                                 / (2.0 * n)) < self.tau:
                eliminated[i] = True
                active[i] = False
            sched += time.perf_counter() - t0

        return {"executions": executions, "B_used": used,
                "overtime": max(0.0, used - budget),
                "sched_cpu_s": sched,
                "n_eliminated": int(eliminated.sum()),
                "init_completed": bool(init_completed),
                "init_exec": int(min(executions, len(init_order))),
                "init_used": float(init_used)}

