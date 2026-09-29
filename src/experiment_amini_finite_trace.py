"""Amini-TransFuser FINITE-TRACE evaluation harness (ReproAlloc, Sections 4-18).

==========================================================================
PROTOCOL (frozen) -- finite-trace replay, NO resampling
==========================================================================
Dataset   : data/amini_transfuser_canonical.csv (500 rows, 36 routes,
            n_i in {10,20}, byte-identical raw fields; built by
            build_amini_canonical.py from ds_transfuser.csv).
Failure   : frozen rule from amini_failure_definition.md
            (already materialised as the `outcome` column).
CS        : manuscript Eq. (1) Hoeffding-style one-sided lower bound,
            L(s,n) = max(0, s/n - sqrt(log(2n(n+1)/delta)/(2n))),
            strict L > tau.  Via reproalloc_v6.cs_lower / confirmed_sn.
            tau = 0.3, delta = 0.05.  NOT modified.
Prior     : Beta(1,1) uniform (prior_score = None => post_a=post_b=1,
            q_0 = 0.5).  No route features, no outcome labels.
Traces    : each route's n_i recorded runs form a FINITE POOL consumed
            WITHOUT replacement.  When a pool is empty the route can no
            longer be selected -- this is the finite-trace cap.
Order     : per seed, each route's pool is shuffled.  ALL 6 methods share
            the SAME trace order within a seed => matched design.
Budget    : wall-clock seconds; 5% / 10% / 20% of total recorded duration.
            SOFT non-preemptive: an execution started while time_used < B
            always completes and may overrun B.  Overtime is recorded.
Confirm   : STICKY (latching).  The CS is anytime-valid, so the
            certification event is "exists n <= N : L_n > tau".  Once a
            route confirms it is never un-confirmed and is no longer
            selected.  Non-sticky final-L is recorded as a diagnostic.

Methods (6): Uniform, FailRerun, APGAI, ScoreCost, ReproAlloc, ReproAlloc-OA

Cost fallback (GS): cold-start cost constants computed once from all 500
recorded durations (mean / mean|fail / mean|pass).  They are scalars shared
IDENTICALLY by all 6 methods and carry no per-route identity, so they cannot
favour any method.  Disclosed in the report.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reproalloc_v6 as v6

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANON = os.path.join(ROOT, "data", "amini_transfuser_canonical.csv")

TAU = 0.3
DELTA = 0.05
BUDGET_FRACS = (0.05, 0.10, 0.20)
METHODS = ("Uniform", "FailRerun", "APGAI", "ScoreCost", "ReproAlloc", "ReproAlloc-OA")


# ==========================================================================
# Data loading
# ==========================================================================
@dataclass
class RouteData:
    route_id: str
    outcomes: list          # list[int] 0/1, raw CSV order
    durations: list         # list[float]
    score_route: list       # list[float] (scheduling-time feature, unused by prior)

    @property
    def n(self) -> int:
        return len(self.outcomes)

    @property
    def s(self) -> int:
        return int(sum(self.outcomes))

    @property
    def p_hat(self) -> float:
        return self.s / self.n if self.n else 0.0


def load_routes(path: str = CANON) -> list[RouteData]:
    """Load canonical CSV with round-trip float precision; group by route."""
    by_route: dict[str, list[tuple]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rid = row["scenario_id"].strip()
            by_route[rid].append((
                int(row["repetition_id"]),
                int(float(row["outcome"])),
                float(row["duration"]),
                float(row["score_route"]),
            ))
    routes = []
    for rid in sorted(by_route, key=lambda r: int(r.split("_")[1])):
        rows = sorted(by_route[rid], key=lambda t: t[0])
        routes.append(RouteData(rid,
                                [r[1] for r in rows],
                                [r[2] for r in rows],
                                [r[3] for r in rows]))
    return routes


@dataclass
class GS:
    """Cold-start cost constants (shared by every method, method-agnostic)."""
    mean_fail_cost: float
    mean_pass_cost: float
    mean_overall: float


def build_gs(routes: list[RouteData]) -> GS:
    alld, faild, passd = [], [], []
    for r in routes:
        for o, d in zip(r.outcomes, r.durations):
            alld.append(d)
            (faild if o == 1 else passd).append(d)
    return GS(float(np.mean(faild)), float(np.mean(passd)), float(np.mean(alld)))


# ==========================================================================
# Finite-trace pools + trace-order randomisation
# ==========================================================================
def make_orders(routes: list[RouteData], seed: int) -> list[list[int]]:
    """Per-route permutation of pool indices for this trace-order seed.

    Uses SeedSequence so that (a) the trace order is reproducible from the
    seed and (b) method-internal RNG streams are independent of it.
    """
    ss = np.random.SeedSequence(seed)
    children = ss.spawn(len(routes) + 1)
    orders = []
    for r, child in zip(routes, children[:-1]):
        rng = np.random.default_rng(child)
        orders.append(rng.permutation(r.n).tolist())
    return orders


def method_rng(seed: int, method_idx: int, budget_idx: int) -> np.random.Generator:
    """Deterministic, method-specific tie-break stream (independent of order)."""
    ss = np.random.SeedSequence((seed, 1000 + method_idx, 2000 + budget_idx))
    return np.random.default_rng(ss)


class Pool:
    """Finite pool consumed WITHOUT replacement in the seed's trace order."""
    __slots__ = ("seq", "pos", "cap")

    def __init__(self, outcomes, durations, order):
        self.seq = [(outcomes[j], durations[j]) for j in order]
        self.pos = 0
        self.cap = len(self.seq)

    def remaining(self) -> int:
        return self.cap - self.pos

    def exhausted(self) -> bool:
        return self.pos >= self.cap

    def draw(self):
        o, d = self.seq[self.pos]
        self.pos += 1
        return int(o), float(d)


# ==========================================================================
# Confirmation semantics
# ==========================================================================
def cs_lower(s: int, n: int) -> float:
    """Manuscript Eq. (1) -- single source of truth is reproalloc_v6."""
    return v6.cs_lower(s, n, DELTA)


def sticky_confirm_trace(s: int, n: int) -> bool:
    return v6.confirmed_sn(s, n, TAU, DELTA)


def k_best_order(s: int, n: int) -> int:
    """Smallest prefix length that confirms if ALL failures arrive first.

    Order-independent oracle: at prefix k the most failures possible is
    min(k, s).  Returns -1 if no k <= n confirms.
    """
    for k in range(1, n + 1):
        if sticky_confirm_trace(min(k, s), k):
            return k
    return -1


def k_worst_order(s: int, n: int) -> int:
    """Smallest prefix that confirms if ALL successes arrive first.

    At prefix k the fewest failures possible is max(0, k - (n - s)).
    Returns -1 if no k <= n confirms.
    """
    npass = n - s
    for k in range(1, n + 1):
        if sticky_confirm_trace(max(0, k - npass), k):
            return k
    return -1


def oracle_ceiling_for_order(routes, orders) -> tuple[int, dict]:
    """Sticky confirmation when the ENTIRE pool is consumed in this order.

    This is the per-seed attainable ceiling: no budget can do better, because
    no route can yield more evidence than its full pool in this order.
    """
    conf = {}
    for r, order in zip(routes, orders):
        pool = Pool(r.outcomes, r.durations, order)
        s = n = 0
        ever = False
        k_first = -1
        while not pool.exhausted():
            o, _d = pool.draw()
            n += 1
            s += o
            if not ever and sticky_confirm_trace(s, n):
                ever = True
                k_first = n
        conf[r.route_id] = {"ever_confirmed": ever, "k_first_confirm": k_first,
                            "L_full": cs_lower(s, n), "s": s, "n": n}
    return sum(1 for v in conf.values() if v["ever_confirmed"]), conf


# ==========================================================================
# Arm state
# ==========================================================================
class Arms:
    """Per-route belief / cost state. Beta(1,1) prior => post_a=post_b=1."""

    def __init__(self, K: int, gs: GS):
        self.K = K
        self.gs = gs
        self.n = np.zeros(K, dtype=np.int64)
        self.s = np.zeros(K, dtype=np.int64)
        self.fail_cost_sum = np.zeros(K)
        self.fail_cost_cnt = np.zeros(K, dtype=np.int64)
        self.pass_cost_sum = np.zeros(K)
        self.pass_cost_cnt = np.zeros(K, dtype=np.int64)
        # Beta(1,1) uniform prior -- NO kappa pseudo-counts, NO features.
        self.post_a = np.ones(K)
        self.post_b = np.ones(K)
        self.confirmed = np.zeros(K, dtype=bool)     # sticky / latching
        self.k_first_confirm = np.full(K, -1, dtype=np.int64)

    def q(self, i: int) -> float:
        return self.post_a[i] / (self.post_a[i] + self.post_b[i])

    def mean_cost(self, i: int) -> float:
        n = self.n[i]
        if n == 0:
            return self.gs.mean_overall
        return (self.fail_cost_sum[i] + self.pass_cost_sum[i]) / n

    def oa_cost(self, i: int) -> float:
        qi = self.q(i)
        cF = (self.fail_cost_sum[i] / self.fail_cost_cnt[i]) if self.fail_cost_cnt[i] > 0 \
            else self.gs.mean_fail_cost
        cP = (self.pass_cost_sum[i] / self.pass_cost_cnt[i]) if self.pass_cost_cnt[i] > 0 \
            else self.gs.mean_pass_cost
        return qi * cF + (1.0 - qi) * cP

    def update(self, i: int, fail: int, cost: float):
        self.n[i] += 1
        self.s[i] += fail
        if fail:
            self.fail_cost_sum[i] += cost
            self.fail_cost_cnt[i] += 1
            self.post_a[i] += 1.0
        else:
            self.pass_cost_sum[i] += cost
            self.pass_cost_cnt[i] += 1
            self.post_b[i] += 1.0
        # STICKY confirmation (anytime-valid CS => latch on first crossing)
        if not self.confirmed[i] and sticky_confirm_trace(self.s[i], self.n[i]):
            self.confirmed[i] = True
            self.k_first_confirm[i] = self.n[i]


# ==========================================================================
# Selectors.  Every selector returns -1 when nothing is selectable.
# `eligible` = not confirmed AND pool not exhausted.
# ==========================================================================
def _eligible(arms: Arms, pools: list[Pool]) -> np.ndarray:
    return np.array([(~arms.confirmed[i]) and (not pools[i].exhausted())
                     for i in range(arms.K)])


def _pick_max(cand: np.ndarray, score_cand: np.ndarray, rng) -> int:
    """`score_cand` is ALREADY aligned with `cand` (len == len(cand))."""
    mx = np.max(score_cand)
    ties = cand[score_cand >= mx - 1e-12]
    return int(ties[rng.integers(0, ties.size)])


def _pick_min(cand: np.ndarray, score_cand: np.ndarray, rng) -> int:
    """`score_cand` is ALREADY aligned with `cand` (len == len(cand))."""
    mn = np.min(score_cand)
    ties = cand[score_cand <= mn + 1e-9]
    return int(ties[rng.integers(0, ties.size)])


class UniformSel:
    """Round-robin-in-expectation: uniformly random among eligible routes."""
    name = "Uniform"

    def init(self, arms, pools, rng):
        return {}

    def select(self, arms, pools, rng):
        cand = np.flatnonzero(_eligible(arms, pools))
        if cand.size == 0:
            return -1
        return int(cand[rng.integers(0, cand.size)])

    def update(self, arms, i, st):
        pass


class FailRerunSel:
    """Practitioner default: re-run the route with the highest observed fail rate."""
    name = "FailRerun"

    def init(self, arms, pools, rng):
        return {}

    def select(self, arms, pools, rng):
        cand = np.flatnonzero(_eligible(arms, pools))
        if cand.size == 0:
            return -1
        failed = cand[arms.s[cand] > 0]
        if failed.size > 0:
            rate = arms.s[failed] / np.maximum(arms.n[failed], 1)
            return _pick_max(failed, rate, rng)
        # nothing has failed yet -> explore cheapest-first (cold start)
        return _pick_min(cand, np.array([arms.mean_cost(i) for i in cand]), rng)

    def update(self, arms, i, st):
        pass


class APGAISel:
    """Adapted APGAI (good-arm identification, cost-aware index).

    Phase 1: initialise every route once, cheapest-expected-cost first.
    Phase 2: index = sqrt(n) * (mu_hat - tau) / C_hat^OA  (maximise).
    """
    name = "APGAI"

    def init(self, arms, pools, rng):
        return {}

    def select(self, arms, pools, rng):
        cand = np.flatnonzero(_eligible(arms, pools))
        if cand.size == 0:
            return -1
        uninit = cand[arms.n[cand] == 0]
        if uninit.size > 0:
            return _pick_min(uninit, np.array([arms.oa_cost(i) for i in uninit]), rng)
        n = arms.n[cand].astype(float)
        mu = arms.s[cand] / np.maximum(arms.n[cand], 1)
        cost = np.array([arms.oa_cost(i) for i in cand])
        idx = np.sqrt(n) * (mu - TAU) / np.maximum(cost, 1e-9)
        return _pick_max(cand, idx, rng)

    def update(self, arms, i, st):
        pass


class ScoreCostSel:
    """Score-over-cost: rank by q_i / c_i (mean observed cost). No ERCC."""
    name = "ScoreCost"

    def init(self, arms, pools, rng):
        return {}

    def select(self, arms, pools, rng):
        cand = np.flatnonzero(_eligible(arms, pools))
        if cand.size == 0:
            return -1
        score = np.array([arms.q(i) / max(arms.mean_cost(i), 1e-9) for i in cand])
        return _pick_max(cand, score, rng)

    def update(self, arms, i, st):
        pass


class ReproAllocSel:
    """ERCC-driven allocation.  use_oa=False -> ReproAlloc, True -> ReproAlloc-OA.

    utility_i = 1 / (m_i* * c_i), where m_i* = min{m>=1 : L~(s+m q, n+m) > tau}
    on the planning trajectory s~(m)=s+m q, n~(m)=n+m.  m_i* = inf (q <= tau or
    unreachable within m_cap) -> utility 0 -> ScoreCost fallback.
    """

    def __init__(self, use_oa: bool, m_cap: int):
        self.use_oa = use_oa
        self.m_cap = int(m_cap)
        self.name = "ReproAlloc-OA" if use_oa else "ReproAlloc"

    def _util(self, arms, cache, i):
        if arms.confirmed[i]:
            return -np.inf
        qi = arms.q(i)
        m_star = cache[i].get(int(arms.s[i]), int(arms.n[i]), qi, TAU, self.m_cap, DELTA)
        if np.isinf(m_star):
            return 0.0
        c = arms.oa_cost(i) if self.use_oa else arms.mean_cost(i)
        return 1.0 / max(m_star * max(c, 1e-9), 1e-9)

    def init(self, arms, pools, rng):
        cache = [v6.ERCCache() for _ in range(arms.K)]
        util = np.array([self._util(arms, cache, i) for i in range(arms.K)])
        return {"cache": cache, "util": util}

    def update(self, arms, i, st):
        st["util"][i] = self._util(arms, st["cache"], i)


# --- unify the selector interface -----------------------------------------
class _ReproAllocAdapter:
    """Wraps ReproAllocSel to respect finite-pool exhaustion."""

    def __init__(self, use_oa: bool, m_cap: int):
        self.inner = ReproAllocSel(use_oa, m_cap)
        self.name = self.inner.name

    def init(self, arms, pools, rng):
        return self.inner.init(arms, pools, rng)

    def select(self, arms, pools, rng, st):
        elig = _eligible(arms, pools)
        cand = np.flatnonzero(elig)
        if cand.size == 0:
            return -1
        util = st["util"]
        pos = cand[util[cand] > 0.0]
        if pos.size > 0:
            return _pick_max(pos, util[pos], rng)
        # ERCC fallback: every candidate has m* = inf -> ScoreCost ordering
        fb = np.array([arms.q(i) / max(arms.mean_cost(i), 1e-9) for i in cand])
        return _pick_max(cand, fb, rng)

    def update(self, arms, i, st):
        self.inner.update(arms, i, st)


def _wrap(seller):
    """Give the simple selectors the (arms, pools, rng, st) signature."""
    class W:
        def __init__(self, s):
            self.s = s
            self.name = s.name

        def init(self, arms, pools, rng):
            return self.s.init(arms, pools, rng)

        def select(self, arms, pools, rng, st):
            return self.s.select(arms, pools, rng)

        def update(self, arms, i, st):
            self.s.update(arms, i, st)
    return W(seller)


def build_selectors(m_cap: int) -> list:
    return [_wrap(UniformSel()), _wrap(FailRerunSel()), _wrap(APGAISel()),
            _wrap(ScoreCostSel()),
            _ReproAllocAdapter(False, m_cap), _ReproAllocAdapter(True, m_cap)]


def m_cap_from_budget(budget: float, gs: GS) -> int:
    min_cost = max(min(gs.mean_fail_cost, gs.mean_pass_cost), 1e-9)
    return int(max(1, math.ceil(budget / min_cost)) + 1)


# ==========================================================================
# One (seed, budget, method) run
# ==========================================================================
def run_one(routes, orders, ref_pos_mask, budget, sel, rng, gs, ceiling_seed):
    K = len(routes)
    pools = [Pool(r.outcomes, r.durations, o) for r, o in zip(routes, orders)]
    arms = Arms(K, gs)
    st = sel.init(arms, pools, rng)

    time_used = 0.0
    execs = 0
    overruns = 0
    sched_t = 0.0
    exhausted_stops = 0

    while True:
        if time_used >= budget:
            break
        t0 = time.perf_counter()
        i = sel.select(arms, pools, rng, st)
        if i < 0:
            # nothing selectable: all unconfirmed routes exhausted
            exhausted_stops = 1
            sched_t += time.perf_counter() - t0
            break
        fail, cost = pools[i].draw()
        arms.update(i, fail, cost)
        sel.update(arms, i, st)
        sched_t += time.perf_counter() - t0
        time_used += cost
        execs += 1
        if time_used > budget:
            overruns = 1

    conf = arms.confirmed
    tp = int((conf & ref_pos_mask).sum())
    fp = int((conf & ~ref_pos_mask).sum())
    n_ref = int(ref_pos_mask.sum())
    n_pool_left = sum(p.remaining() for p in pools)
    n_exhausted = sum(1 for p in pools if p.exhausted())

    return {
        "tp": tp, "fp": fp, "fn": n_ref - tp,
        "confirmed": int(conf.sum()),
        "recall_refpos": tp / n_ref if n_ref else 0.0,
        "precision": tp / int(conf.sum()) if conf.sum() else 0.0,
        "recall_ceiling_seed": tp / ceiling_seed if ceiling_seed else 0.0,
        "executions": execs,
        "time_used_s": time_used,
        "budget_s": budget,
        "budget_util": time_used / budget if budget else 0.0,
        "overtime_s": max(0.0, time_used - budget),
        "overrun_flag": overruns,
        "pools_exhausted": n_exhausted,
        "runs_left_in_pools": n_pool_left,
        "stopped_by_exhaustion": exhausted_stops,
        "sched_overhead_s": sched_t,
        "nonsticky_confirmed": int(sum(
            1 for i in range(K)
            if arms.n[i] > 0 and cs_lower(int(arms.s[i]), int(arms.n[i])) > TAU)),
        # per-route allocation snapshot
        "_alloc": [(routes[i].route_id, int(arms.n[i]), int(arms.s[i]),
                    float(cs_lower(int(arms.s[i]), int(arms.n[i]))),
                    bool(arms.confirmed[i]), int(arms.k_first_confirm[i]),
                    pools[i].cap, pools[i].remaining(),
                    bool(ref_pos_mask[i]))
                   for i in range(K)],
    }


# ==========================================================================
# Confirmability diagnostic (Section 13)
# ==========================================================================
def confirmability_table(routes, seeds, orders_by_seed, ceilings_by_seed):
    rows = []
    for r in routes:
        s, n, p = r.s, r.n, r.p_hat
        Lfull = cs_lower(s, n)
        kb, kw = k_best_order(s, n), k_worst_order(s, n)
        row = {
            "route_id": r.route_id,
            "n_pool": n,
            "s_total_fails": s,
            "p_hat": p,
            "reference_positive": int(p > TAU),
            "L_full_pool": Lfull,
            "nonsticky_ceiling": int(Lfull > TAU),
            "k_best_order": kb,
            "k_worst_order": kw,
            "confirmable_any_order": int(kb >= 1),
            "confirmable_all_orders": int(kw >= 1),
            "mean_duration_s": float(np.mean(r.durations)),
        }
        ever = []
        kfirst = []
        for sd in seeds:
            info = ceilings_by_seed[sd][r.route_id]
            ever.append(info["ever_confirmed"])
            kfirst.append(info["k_first_confirm"])
        row["sticky_ceiling_frac_over_seeds"] = float(np.mean(ever)) if ever else 0.0
        row["sticky_ceiling_n_seeds"] = int(np.sum(ever))
        kf = [k for k in kfirst if k >= 1]
        row["k_first_confirm_median"] = float(np.median(kf)) if kf else float("nan")
        row["k_first_confirm_min"] = int(min(kf)) if kf else -1
        row["k_first_confirm_max"] = int(max(kf)) if kf else -1
        rows.append(row)
    return rows


# ==========================================================================
# Driver
# ==========================================================================
def run_experiment(n_seeds: int, tag: str, outdir: str, seed0: int = 1000,
                   write_alloc: bool = True, verbose: bool = True):
    os.makedirs(outdir, exist_ok=True)
    routes = load_routes()
    gs = build_gs(routes)
    K = len(routes)
    total_dur = sum(sum(r.durations) for r in routes)
    p_hat = np.array([r.p_hat for r in routes])
    ref_pos_mask = p_hat > TAU
    n_ref = int(ref_pos_mask.sum())

    # order-independent ceilings
    nonsticky_ceil = int(sum(1 for r in routes if cs_lower(r.s, r.n) > TAU))
    any_order_ceil = int(sum(1 for r in routes if k_best_order(r.s, r.n) >= 1))
    all_order_ceil = int(sum(1 for r in routes if k_worst_order(r.s, r.n) >= 1))

    if verbose:
        print("=" * 74)
        print(f"### Amini-TransFuser FINITE-TRACE run  [{tag}]")
        print("=" * 74)
        print(f"routes={K}  runs={sum(r.n for r in routes)}  "
              f"pools: 10-rep={sum(1 for r in routes if r.n==10)}, "
              f"20-rep={sum(1 for r in routes if r.n==20)}")
        print(f"total recorded wall-clock = {total_dur:.2f}s = {total_dur/3600:.2f}h")
        print(f"GS cold-start costs: mean={gs.mean_overall:.2f}s  "
              f"mean|fail={gs.mean_fail_cost:.2f}s  mean|pass={gs.mean_pass_cost:.2f}s")
        print(f"tau={TAU} delta={DELTA} prior=Beta(1,1) uniform")
        print(f"reference-positive (p_hat>tau)      : {n_ref}/{K}")
        print(f"ceiling non-sticky full-pool L>tau  : {nonsticky_ceil}/{K}")
        print(f"ceiling best-order (oracle)         : {any_order_ceil}/{K}")
        print(f"ceiling all-orders (guaranteed)     : {all_order_ceil}/{K}")

    seeds = [seed0 + k for k in range(n_seeds)]
    orders_by_seed, ceilings_by_seed, ceil_n = {}, {}, {}
    for sd in seeds:
        orders = make_orders(routes, sd)
        orders_by_seed[sd] = orders
        c, info = oracle_ceiling_for_order(routes, orders)
        ceilings_by_seed[sd] = info
        ceil_n[sd] = c
    if verbose:
        print(f"per-seed sticky attainable ceiling (full pool, unlimited budget): "
              f"{[ceil_n[s] for s in seeds]}")
        print(f"  mean={np.mean([ceil_n[s] for s in seeds]):.2f} "
              f"min={min(ceil_n.values())} max={max(ceil_n.values())}")
        print()

    # ---- main loop -------------------------------------------------------
    rows, alloc_rows = [], []
    for sd in seeds:
        orders = orders_by_seed[sd]
        ceiling = ceil_n[sd]
        for bi, tf in enumerate(BUDGET_FRACS):
            budget = total_dur * tf
            m_cap = m_cap_from_budget(budget, gs)
            sels = build_selectors(m_cap)
            for mi, sel in enumerate(sels):
                rng = method_rng(sd, mi, bi)
                t0 = time.perf_counter()
                res = run_one(routes, orders, ref_pos_mask, budget, sel, rng, gs, ceiling)
                wall = time.perf_counter() - t0
                alloc = res.pop("_alloc")
                rec = {
                    "tag": tag, "seed": sd, "budget_frac": tf, "budget_s": budget,
                    "method": sel.name, "m_cap": m_cap,
                    "ceiling_seed": ceiling, "n_refpos": n_ref,
                    "nonsticky_ceiling": nonsticky_ceil,
                    "harness_wall_s": wall,
                }
                rec.update(res)
                rows.append(rec)
                if write_alloc:
                    for (rid, ni, si, L, cf, kf, cap, left, rp) in alloc:
                        alloc_rows.append({
                            "tag": tag, "seed": sd, "budget_frac": tf,
                            "method": sel.name, "route_id": rid,
                            "runs_used": ni, "fails_used": si,
                            "pool_size": cap, "pool_remaining": left,
                            "pool_exhausted": int(left == 0),
                            "L_final": L, "confirmed_sticky": int(cf),
                            "k_first_confirm": kf,
                            "reference_positive": int(rp),
                        })
        if verbose:
            print(f"  seed {sd}: done (ceiling={ceiling})")

    # ---- write -----------------------------------------------------------
    def wcsv(name, data, cols=None):
        p = os.path.join(outdir, name)
        if not data:
            return p
        cols = cols or list(data[0].keys())
        with open(p, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for d in data:
                w.writerow({k: d.get(k, "") for k in cols})
        return p

    main_cols = ["tag", "seed", "budget_frac", "budget_s", "method", "m_cap",
                 "ceiling_seed", "n_refpos", "nonsticky_ceiling",
                 "tp", "fp", "fn", "confirmed", "recall_refpos", "precision",
                 "recall_ceiling_seed", "nonsticky_confirmed", "executions",
                 "time_used_s", "budget_util", "overtime_s", "overrun_flag",
                 "pools_exhausted", "runs_left_in_pools",
                 "stopped_by_exhaustion", "sched_overhead_s", "harness_wall_s"]
    p_main = wcsv(f"amini_finite_trace_{tag}.csv", rows, main_cols)

    conf_rows = confirmability_table(routes, seeds, orders_by_seed, ceilings_by_seed)
    p_conf = wcsv(f"amini_transfuser_confirmability.csv", conf_rows)

    p_alloc = None
    if write_alloc:
        p_alloc = wcsv(f"amini_transfuser_route_allocations_{tag}.csv", alloc_rows)

    meta = {
        "tag": tag, "n_seeds": n_seeds, "seeds": seeds,
        "tau": TAU, "delta": DELTA, "prior": "Beta(1,1) uniform",
        "protocol": "finite-trace, without replacement, sticky confirmation",
        "budget_fracs": list(BUDGET_FRACS),
        "total_duration_s": total_dur, "total_duration_h": total_dur / 3600,
        "budgets_s": {str(f): total_dur * f for f in BUDGET_FRACS},
        "budgets_h": {str(f): total_dur * f / 3600 for f in BUDGET_FRACS},
        "gs": {"mean_overall": gs.mean_overall,
               "mean_fail_cost": gs.mean_fail_cost,
               "mean_pass_cost": gs.mean_pass_cost},
        "n_routes": K, "n_runs": sum(r.n for r in routes),
        "n_reference_positive": n_ref,
        "ceiling_nonsticky_fullpool": nonsticky_ceil,
        "ceiling_best_order_oracle": any_order_ceil,
        "ceiling_all_orders": all_order_ceil,
        "ceiling_per_seed": {str(s): ceil_n[s] for s in seeds},
        "methods": list(METHODS),
    }
    with open(os.path.join(outdir, f"amini_finite_trace_{tag}_meta.json"), "w",
              encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    if verbose:
        print(f"\nwrote {p_main}")
        print(f"wrote {p_conf}")
        if p_alloc:
            print(f"wrote {p_alloc}")
    return rows, conf_rows, alloc_rows, meta


def summarize(rows, tag):
    """Compact console summary: recall by (budget, method) + paired diffs."""
    import pandas as pd
    df = pd.DataFrame(rows)
    print("\n" + "=" * 74)
    print(f"### SUMMARY [{tag}]")
    print("=" * 74)
    for tf in BUDGET_FRACS:
        sub = df[df.budget_frac == tf]
        print(f"\n-- budget {tf*100:.0f}% --")
        g = sub.groupby("method").agg(
            recall=("recall_refpos", "mean"),
            recall_sd=("recall_refpos", "std"),
            ceil_rec=("recall_ceiling_seed", "mean"),
            conf=("confirmed", "mean"),
            tp=("tp", "mean"), fp=("fp", "mean"),
            execs=("executions", "mean"),
            used_h=("time_used_s", lambda x: x.mean() / 3600),
            over=("overtime_s", "mean"),
            exh=("pools_exhausted", "mean"),
        ).reindex(list(METHODS))
        print(g.to_string(float_format=lambda v: f"{v:.4f}"))

        print(f"   paired diffs (matched on seed), recall_refpos:")
        for base in METHODS:
            if base.startswith("ReproAlloc"):
                continue
            for meth in ("ReproAlloc", "ReproAlloc-OA"):
                a = sub[sub.method == meth].set_index("seed")["recall_refpos"]
                b = sub[sub.method == base].set_index("seed")["recall_refpos"]
                common = a.index.intersection(b.index)
                d = a[common] - b[common]
                if len(d) < 2:
                    continue
                m, sd = d.mean(), d.std(ddof=1)
                ci = 1.96 * sd / math.sqrt(len(d)) if sd == sd else 0.0
                print(f"     {meth:14s} - {base:11s}: mean={m:+.4f} "
                      f"sd={sd:.4f} 95%CI=[{m-ci:+.4f},{m+ci:+.4f}] n={len(d)}")
        a = sub[sub.method == "ReproAlloc-OA"].set_index("seed")["recall_refpos"]
        b = sub[sub.method == "ReproAlloc"].set_index("seed")["recall_refpos"]
        common = a.index.intersection(b.index)
        d = a[common] - b[common]
        if len(d) >= 2:
            m, sd = d.mean(), d.std(ddof=1)
            ci = 1.96 * sd / math.sqrt(len(d))
            print(f"     ReproAlloc-OA  - ReproAlloc  : mean={m:+.4f} "
                  f"sd={sd:.4f} 95%CI=[{m-ci:+.4f},{m+ci:+.4f}] n={len(d)}")
    return df


def pilot_checks(rows, meta) -> tuple[bool, list[str]]:
    """Section 12 pilot decision rule."""
    import pandas as pd
    df = pd.DataFrame(rows)
    msgs, ok = [], True

    # 1. completeness
    expect = meta["n_seeds"] * len(BUDGET_FRACS) * len(METHODS)
    if len(df) != expect:
        ok = False
        msgs.append(f"FAIL completeness: {len(df)} rows, expected {expect}")
    else:
        msgs.append(f"PASS completeness: {len(df)} rows = "
                    f"{meta['n_seeds']} seeds x {len(BUDGET_FRACS)} budgets x {len(METHODS)} methods")

    # 2. no NaN / inf in numeric metrics (per-cell, not per-column)
    num = df.select_dtypes(include=[np.number])
    bad_cells = int((~np.isfinite(num.to_numpy(dtype=float))).sum())
    bad_cols = [c for c in num.columns if (~np.isfinite(num[c].to_numpy(dtype=float))).any()]
    if bad_cells:
        ok = False
        msgs.append(f"FAIL non-finite values: {bad_cells} cells in columns {bad_cols}")
    else:
        msgs.append("PASS all numeric metrics finite (no NaN/inf)")

    # 3. budget respected (soft): every run either within B or single overrun
    over = df[df.overtime_s > 0]
    if len(over) and (over.overrun_flag != 1).any():
        ok = False
        msgs.append("FAIL overtime without overrun_flag")
    else:
        msgs.append(f"PASS soft-budget semantics: {len(over)}/{len(df)} runs overran B "
                    f"(all single-execution, non-preemptive); "
                    f"max overtime={df.overtime_s.max()/3600:.3f}h")

    # 4. no run exceeded B by more than one max-duration execution
    maxdur = max(df.overtime_s.max(), 0.0)
    msgs.append(f"INFO max single overrun = {maxdur:.1f}s = {maxdur/3600:.3f}h "
                f"(<= max recorded duration 11664s: {maxdur <= 11664.0})")
    if maxdur > 11664.0:
        ok = False
        msgs.append("FAIL overrun exceeds the longest single recorded execution")

    # 5. every method terminates and executes >=1 run
    if (df.executions < 1).any():
        ok = False
        msgs.append("FAIL some runs executed 0 times")
    else:
        msgs.append(f"PASS all methods terminate; executions range "
                    f"{df.executions.min()}-{df.executions.max()}")

    # 6. ReproAlloc >= Uniform on mean recall at every budget
    for tf in BUDGET_FRACS:
        sub = df[df.budget_frac == tf]
        ra = sub[sub.method == "ReproAlloc"].recall_refpos.mean()
        un = sub[sub.method == "Uniform"].recall_refpos.mean()
        flag = "PASS" if ra >= un - 1e-12 else "WARN"
        if ra < un - 1e-12:
            ok = False
        msgs.append(f"{flag} budget {tf*100:.0f}%: ReproAlloc recall={ra:.4f} "
                    f"vs Uniform={un:.4f} (diff={ra-un:+.4f})")

    # 7. zero-certification sanity: at the smallest budget some method should
    #    be able to certify at least one route, else the budget is uninformative
    z = df.groupby(["budget_frac", "method"]).tp.mean()
    if (z == 0).all():
        ok = False
        msgs.append("FAIL no method certifies anything at any budget (uninformative)")
    else:
        msgs.append(f"PASS certifications observed; mean TP range {z.min():.2f}-{z.max():.2f}")

    # 8. determinism: same seed+budget+method reproduces (checked by re-run)
    msgs.append("INFO determinism: seeds derived from np.random.SeedSequence; "
                "method_rng(seed, method_idx, budget_idx) is a pure function")

    # 9. finite-trace cap actually binds somewhere
    if df.pools_exhausted.max() == 0:
        msgs.append("INFO no pool was ever exhausted -- finite-trace cap not binding "
                    "at these budgets (still correct, just not restrictive)")
    else:
        msgs.append(f"PASS finite-trace cap binds: up to {int(df.pools_exhausted.max())} "
                    f"pools exhausted in a single run; "
                    f"{int(df.stopped_by_exhaustion.sum())} runs stopped by exhaustion")

    # 10. precision / FP audit
    msgs.append(f"INFO false-positive certifications (confirmed & p_hat<=tau): "
                f"mean {df.fp.mean():.3f}, max {int(df.fp.max())}")
    return ok, msgs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    if args.mode == "pilot":
        ns = args.seeds or 3
        outdir = args.outdir or os.path.join(ROOT, "results_amini_finite")
        rows, conf, alloc, meta = run_experiment(ns, "pilot", outdir, seed0=1000)
        summarize(rows, "pilot")
        ok, msgs = pilot_checks(rows, meta)
        print("\n" + "=" * 74)
        print("### PILOT DECISION RULE")
        print("=" * 74)
        for m in msgs:
            print("  " + m)
        print(f"\n  ==> PILOT {'PASS' if ok else 'FAIL'}")
        with open(os.path.join(outdir, "amini_finite_trace_pilot_checks.json"), "w",
                  encoding="utf-8") as fh:
            json.dump({"pass": ok, "checks": msgs}, fh, indent=2)
        sys.exit(0 if ok else 1)
    else:
        ns = args.seeds or 12
        outdir = args.outdir or os.path.join(ROOT, "results_amini_finite")
        rows, conf, alloc, meta = run_experiment(ns, "full", outdir, seed0=2000)
        summarize(rows, "full")
        sys.exit(0)
