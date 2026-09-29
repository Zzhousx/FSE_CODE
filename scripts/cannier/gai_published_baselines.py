"""Published GAI sampling rules adapted to the shared ReproAlloc replay protocol.

LUCB-G follows Kano et al. (2019), Algorithm 1, including one initial pull per
arm and its UCB/elimination rule. Murphy Sampling follows the Bernoulli rule in
the APGAI authors' public Julia implementation: draw independent Beta(1+s,
1+n-s) samples conditional on their maximum exceeding the threshold, then
pull the arm with the largest draw. Confirmations in both methods use the
benchmark's common anytime confidence sequence, not the original stopping
rule. Once confirmed, an arm leaves the candidate set so more arms can be
confirmed under the remaining time budget.

Sources:
https://link.springer.com/article/10.1007/s10994-019-05784-4
https://github.com/clreda/apgai/blob/main/fixed-conf-anytime-code/gai/samplingrules.jl
"""
from __future__ import annotations

import math
import time

import numpy as np


class PublishedGAISampling:
    def __init__(self, method: str, tau: float = .3, delta: float = .05):
        if method not in {"LUCB-G", "Murphy Sampling"}:
            raise ValueError(method)
        self.method, self.tau, self.delta = method, float(tau), float(delta)

    def run(self, arms, env, budget: float, rng, finite_pool: bool):
        k = arms.K
        used = 0.0
        executed = 0
        init_used = 0.0
        init_exec = 0
        eliminated = np.zeros(k, dtype=bool)
        exhausted = np.zeros(k, dtype=bool)
        sched_cpu = 0.0

        def draw(i):
            nonlocal used, executed, sched_cpu
            y, cost = env.execute(int(i))
            t0 = time.perf_counter()
            arms.update(int(i), int(y), float(cost))
            used += float(cost)
            executed += 1
            if finite_pool and not env.has_remaining(int(i)):
                exhausted[int(i)] = True
            sched_cpu += time.perf_counter() - t0
            return float(cost)

        # Kano et al. explicitly initialize LUCB-G by pulling every arm once.
        # Murphy's Beta posterior is defined even for n=0, so it can start
        # without a mandatory all-arm pass.
        init_completed = True
        if self.method == "LUCB-G":
            t0 = time.perf_counter()
            order = rng.permutation(k)
            sched_cpu += time.perf_counter() - t0
            for i in order:
                if used >= budget:
                    init_completed = False
                    break
                cost = draw(int(i))
                init_used += cost
                init_exec += 1

        while init_completed and used < budget:
            t0 = time.perf_counter()
            cand = np.flatnonzero(~arms.confirmed & ~eliminated & ~exhausted)
            if not cand.size:
                sched_cpu += time.perf_counter() - t0
                break
            if self.method == "LUCB-G":
                n = arms.n[cand].astype(float)
                mu = arms.s[cand] / n
                width = np.sqrt(np.log(4.0 * k * n * n / self.delta) / (2.0 * n))
                scores = mu + width
                max_score = float(scores.max())
                tied = cand[scores >= max_score - 1e-12]
                i = int(tied[rng.integers(len(tied))])
            else:
                # Exact rejection sampling from the independent Beta posterior
                # conditioned on max(sample) > tau, as in the authors' code.
                while True:
                    samples = rng.beta(1.0 + arms.s[cand],
                                       1.0 + arms.n[cand] - arms.s[cand])
                    if float(samples.max()) > self.tau:
                        break
                i = int(cand[int(np.argmax(samples))])
            sched_cpu += time.perf_counter() - t0
            draw(i)
            if self.method == "LUCB-G" and not arms.confirmed[i] and not exhausted[i]:
                t0 = time.perf_counter()
                n = float(arms.n[i])
                upper = arms.s[i] / n + math.sqrt(
                    math.log(4.0 * k * n * n / self.delta) / (2.0 * n))
                if upper < self.tau:
                    eliminated[i] = True
                sched_cpu += time.perf_counter() - t0

        return {"executions": executed, "B_used": used,
                "overtime": max(0.0, used - budget), "sched_cpu_s": sched_cpu,
                "n_eliminated": int(eliminated.sum()),
                "init_completed": bool(init_completed),
                "init_exec": init_exec, "init_used": init_used}
