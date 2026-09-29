"""Run all count-based CANNIER-Shuffle scheduler and oracle cells."""
from __future__ import annotations

import csv
import json
from functools import lru_cache
import heapq
import math
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import reproalloc_v6 as V
import experiment_amini_finite_trace as A
import optimistic_ercc_amini as OA
from revision_selectors import AminiRevisionERCC, AminiRerunK, AminiBayesUCBCost
from replay_env import (CANNIERReplay, RemainingView, SequenceBank, SchedulerEnv,
                        release_views, release_scheduler_env)
from hdoc_sampling import HDoCSampling

_UNIQUE_MSTAR = V.est_remaining_execs


@lru_cache(maxsize=250_000)
def _cached_unique_mstar(s, n, q, tau, m_cap, delta=0.05):
    return _UNIQUE_MSTAR(s, n, q, tau, m_cap, delta)


# All tests in one posterior state share m*, so caching avoids repeated searches.
V.est_remaining_execs = _cached_unique_mstar

TAU, DELTA = .3, .05
SEEDS = range(2000, 2012)
RATIOS = (.5, 1., 2., 4.)
METHOD_ORDER = ["Uniform", "FailRerun", "APGAI", "ScoreCost", "RerunK-10",
                "BayesUCB-Cost", "HDoC-Sampling", "Mean Planning (capped)",
                "ReproAlloc"]
EXTERNALS = ["Uniform", "FailRerun", "APGAI", "ScoreCost", "RerunK-10",
             "BayesUCB-Cost", "HDoC-Sampling"]


_VIEW_ENVS = {}


class PoolViews(list):
    """Arm views for legacy selectors; each view exposes only exhaustion."""
    __slots__ = ()
    def __init__(self, env):
        super().__init__(RemainingView(i, env) for i in range(len(env._pos)))
        _VIEW_ENVS[id(self)] = env


def _fast_eligible(arms, pools):
    if isinstance(pools, PoolViews):
        return (~arms.confirmed) & (_VIEW_ENVS[id(pools)]._pos < 2500)
    return _ORIGINAL_ELIGIBLE(arms, pools)


_ORIGINAL_ELIGIBLE = A._eligible
A._eligible = _fast_eligible


class FastBayesUCBCost:
    """Exact BayesUCB scores using min-cost buckets per posterior state."""
    name = "BayesUCB-Cost"

    def _rho(self, st):
        return 1.0 - 1.0 / (st["t"] + 1.0)

    def _qvalues(self, rho, alpha, bpar):
        return beta.ppf(rho, alpha, bpar)

    @staticmethod
    def _add(i, key, cost, st):
        group = st["groups"].setdefault(
            key, {"buckets": {}, "heap": [], "versions": {}})
        bucket = group["buckets"].get(cost)
        if bucket is None:
            bucket = []
            group["buckets"][cost] = bucket
            version = group["versions"].get(cost, 0) + 1
            group["versions"][cost] = version
            heapq.heappush(group["heap"], (cost, version))
        st["arm_key"][i] = key
        st["arm_cost"][i] = cost
        st["bucket_pos"][i] = len(bucket)
        bucket.append(i)

    @staticmethod
    def _remove(i, st):
        key = st["arm_key"][i]
        if key is None:
            return
        cost = float(st["arm_cost"][i])
        group = st["groups"][key]
        bucket = group["buckets"][cost]
        pos = int(st["bucket_pos"][i])
        last = bucket.pop()
        if pos < len(bucket):
            bucket[pos] = last
            st["bucket_pos"][last] = pos
        if not bucket:
            del group["buckets"][cost]
        st["arm_key"][i] = None

    def init(self, arms, pools, rng):
        st = {"t": 1, "groups": {}, "arm_key": [None] * arms.K,
              "arm_cost": np.zeros(arms.K),
              "bucket_pos": np.zeros(arms.K, dtype=np.int64)}
        cold_cost = max(float(arms.gs.mean_overall), 1e-9)
        for i in range(arms.K):
            self._add(i, (0, 0), cold_cost, st)
        return st

    def select(self, arms, pools, rng, st):
        rho = self._rho(st)
        groups = st["groups"]
        active = []
        for key, group in groups.items():
            while (group["heap"] and
                   (group["heap"][0][1] != group["versions"].get(group["heap"][0][0])
                    or group["heap"][0][0] not in group["buckets"])):
                heapq.heappop(group["heap"])
            if not group["heap"]:
                continue
            active.append((key, group))
        if not active:
            return -1
        alpha = np.asarray([1 + key[1] for key, _ in active], dtype=float)
        bpar = np.asarray([1 + key[0] - key[1] for key, _ in active], dtype=float)
        qvalues = self._qvalues(rho, alpha, bpar)
        best = []
        max_score = -1.0
        for (key, group), q0 in zip(active, qvalues):
            q = float(q0)
            score = q / group["heap"][0][0]
            best.append((key, group, q, score))
            if score > max_score:
                max_score = score
        tied_buckets = []
        tied_n = 0
        min_score = max_score - 1e-12
        for key, group, q, score in best:
            if score < min_score:
                continue
            cost_limit = q / min_score if min_score > 0 else math.inf
            cost_entries = []
            while group["heap"] and group["heap"][0][0] <= cost_limit:
                c, version = heapq.heappop(group["heap"])
                if (version == group["versions"].get(c) and c in group["buckets"]):
                    cost_entries.append((c, version))
                    tied_buckets.append(group["buckets"][c])
                    tied_n += len(group["buckets"][c])
            for entry in cost_entries:
                heapq.heappush(group["heap"], entry)
        if tied_n == 0:
            raise AssertionError("BayesUCB found no maximizing arm")
        pick = int(rng.integers(0, tied_n))
        for bucket in tied_buckets:
            if pick < len(bucket):
                return int(bucket[pick])
            pick -= len(bucket)
        raise AssertionError("BayesUCB tie selection fell through")

    def update(self, arms, i, st, available=True):
        i = int(i)
        self._remove(i, st)
        if available and not arms.confirmed[i]:
            n, s = int(arms.n[i]), int(arms.s[i])
            cost = max(float(arms.mean_cost(i)), 1e-9)
            self._add(i, (n, s), cost, st)
        st["t"] += 1


class FastScoreCost(FastBayesUCBCost):
    """Vectorized q / mean-cost baseline."""
    name = "ScoreCost"

    def init(self, arms, pools, rng):
        return {}

    def select(self, arms, pools, rng, st):
        cand = np.flatnonzero(A._eligible(arms, pools))
        if not cand.size:
            return -1
        n = arms.n[cand]
        q = arms.post_a[cand] / (arms.post_a[cand] + arms.post_b[cand])
        observed = (arms.fail_cost_sum[cand] + arms.pass_cost_sum[cand]) / np.maximum(n, 1)
        cost = np.maximum(np.where(n > 0, observed, arms.gs.mean_overall), 1e-9)
        return A._pick_max(cand, q / cost, rng)

    def update(self, arms, i, st):
        pass


class FastAPGAI:
    """Vectorized APGAI with its original one-sample-per-test initialization."""
    name = "APGAI"

    def init(self, arms, pools, rng):
        return {}

    def select(self, arms, pools, rng, st):
        cand = np.flatnonzero(A._eligible(arms, pools))
        if not cand.size:
            return -1
        uninit = cand[arms.n[cand] == 0]
        if uninit.size:
            return int(uninit[rng.integers(0, len(uninit))])
        n = arms.n[cand].astype(float)
        mu = arms.s[cand] / n
        q = arms.post_a[cand] / (arms.post_a[cand] + arms.post_b[cand])
        cf = np.where(arms.fail_cost_cnt[cand] > 0,
                      arms.fail_cost_sum[cand] / np.maximum(arms.fail_cost_cnt[cand], 1),
                      arms.gs.mean_fail_cost)
        cp = np.where(arms.pass_cost_cnt[cand] > 0,
                      arms.pass_cost_sum[cand] / np.maximum(arms.pass_cost_cnt[cand], 1),
                      arms.gs.mean_pass_cost)
        cost = np.maximum(q * cf + (1.0 - q) * cp, 1e-9)
        score = np.sqrt(n) * (mu - .3) / cost
        return A._pick_max(cand, score, rng)

    def update(self, arms, i, st):
        pass


class FastRerunK10:
    """RerunK-10 with vectorized score-cost choices and frozen focus logic."""
    name = "RerunK-10"

    def init(self, arms, pools, rng):
        return {"focused_ever": np.zeros(arms.K, dtype=bool), "focus": -1,
                "remaining": 0}

    def select(self, arms, pools, rng, st):
        i = int(st["focus"])
        if i >= 0 and st["remaining"] > 0 and not arms.confirmed[i] and not pools[i].exhausted():
            return i
        st["focus"] = -1
        cand = np.flatnonzero(A._eligible(arms, pools))
        if not cand.size:
            return -1
        n = arms.n[cand]
        q = arms.post_a[cand] / (arms.post_a[cand] + arms.post_b[cand])
        obs = (arms.fail_cost_sum[cand] + arms.pass_cost_sum[cand]) / np.maximum(n, 1)
        cost = np.maximum(np.where(n > 0, obs, arms.gs.mean_overall), 1e-9)
        return A._pick_max(cand, q / cost, rng)

    def update(self, arms, i, st):
        i = int(i)
        if st["focus"] == i:
            st["remaining"] -= 1
            if st["remaining"] <= 0 or arms.confirmed[i]:
                st["focus"] = -1
        elif arms.s[i] > 0 and not st["focused_ever"][i] and not arms.confirmed[i]:
            st["focused_ever"][i] = True
            st["focus"] = i
            st["remaining"] = 10


class ScoreBucketQueue:
    """Dynamic max-score queue with O(1) uniform draws from exact ties."""
    def __init__(self, n):
        self.buckets = {}
        self.heap = []
        self.versions = {}
        self.score = np.full(n, np.nan)
        self.position = np.full(n, -1, dtype=np.int64)
        self.size = 0

    def add(self, i, score):
        score = float(score)
        bucket = self.buckets.get(score)
        if bucket is None:
            bucket = []
            self.buckets[score] = bucket
            ver = self.versions.get(score, 0) + 1
            self.versions[score] = ver
            heapq.heappush(self.heap, (-score, ver))
        self.position[i] = len(bucket)
        bucket.append(int(i))
        self.score[i] = score
        self.size += 1

    def remove(self, i):
        score = float(self.score[i])
        if not np.isfinite(score):
            return
        bucket = self.buckets[score]
        pos = int(self.position[i])
        last = bucket.pop()
        if pos < len(bucket):
            bucket[pos] = last
            self.position[last] = pos
        if not bucket:
            del self.buckets[score]
        self.score[i] = np.nan
        self.position[i] = -1
        self.size -= 1

    def choose_max(self, rng, tolerance=1e-12):
        while self.heap and (self.heap[0][1] != self.versions.get(-self.heap[0][0])
                             or -self.heap[0][0] not in self.buckets):
            heapq.heappop(self.heap)
        if not self.heap:
            return -1
        maximum = -self.heap[0][0]
        threshold = maximum - tolerance
        choices = []
        selected_keys = []
        while self.heap and -self.heap[0][0] >= threshold:
            neg, ver = heapq.heappop(self.heap)
            score = -neg
            if ver == self.versions.get(score) and score in self.buckets:
                bucket = self.buckets[score]
                choices.append(bucket)
                selected_keys.append((neg, ver))
        for entry in selected_keys:
            heapq.heappush(self.heap, entry)
        total = sum(len(bucket) for bucket in choices)
        if total == 0:
            raise AssertionError("score queue lost every maximizing item")
        pick = int(rng.integers(0, total))
        for bucket in choices:
            if pick < len(bucket):
                return int(bucket[pick])
            pick -= len(bucket)
        raise AssertionError("score queue selection fell through")


class FastERCC:
    """Heap-backed ReproAlloc/Mean Planning with the frozen fallback rule."""
    def __init__(self, inner):
        self.inner = inner
        self.name = inner.name
        self.fallback_count = 0

    @staticmethod
    def _fallback_score(arms, i):
        return float(arms.q(i) / max(arms.mean_cost(i), 1e-9))

    def init(self, arms, pools, rng):
        st = self.inner.init(arms, pools, rng)
        st["main_queue"] = ScoreBucketQueue(arms.K)
        st["fallback_queue"] = ScoreBucketQueue(arms.K)
        fallbacks = self._fallback_scores_vectorized(arms)
        for i in range(arms.K):
            if st["util"][i] > 0:
                st["main_queue"].add(i, st["util"][i])
            st["fallback_queue"].add(i, fallbacks[i])
        st["n_fallback"] = 0
        self.fallback_count = 0
        return st

    @staticmethod
    def _fallback_scores_vectorized(arms):
        q = arms.post_a / (arms.post_a + arms.post_b)
        n = arms.n
        obs = (arms.fail_cost_sum + arms.pass_cost_sum) / np.maximum(n, 1)
        costs = np.maximum(np.where(n > 0, obs, arms.gs.mean_overall), 1e-9)
        return q / costs

    def select(self, arms, pools, rng, st):
        i = st["main_queue"].choose_max(rng)
        if i >= 0:
            return i
        st["n_fallback"] += 1
        self.fallback_count += 1
        return st["fallback_queue"].choose_max(rng)

    def update(self, arms, i, st, available=True):
        i = int(i)
        st["main_queue"].remove(i)
        st["fallback_queue"].remove(i)
        self.inner.update(arms, i, st)
        if available and not arms.confirmed[i]:
            main = float(st["util"][i])
            if main > 0:
                st["main_queue"].add(i, main)
            fallback = self._fallback_score(arms, i)
            st["fallback_queue"].add(i, fallback)


def pool_read(path):
    d = np.load(path)
    return {k: d[k] for k in ("item_id", "F", "cost", "B_pass")}


def sequence(seed, project, item_id, failures):
    if failures == 0:
        return np.zeros(2500, np.int8)
    if failures == 2500:
        return np.ones(2500, np.int8)
    ss = np.random.SeedSequence([int(seed), zlib.crc32(project.encode()), int(item_id)])
    seq = np.zeros(2500, np.int8)
    seq[:int(failures)] = 1
    np.random.default_rng(ss).shuffle(seq)
    return seq


def exact_oracle(pool, project, seed, budget, bank, first_k, never):
    cr, ell = [], []
    positives = pool["F"] / 2500 > TAU
    crossed = np.flatnonzero(first_k > 0)
    for i in crossed:
        c = float(pool["cost"][i])
        cr.append(int(first_k[i]) * c); ell.append(c)
    cr = np.asarray(cr, float); ell = np.asarray(ell, float)
    best = 0
    for j in range(len(cr)):
        threshold = budget - (cr[j] - ell[j])
        order = np.argsort(cr, kind="stable")
        total = 0.0; count = 1
        for i in order:
            if i == j:
                continue
            if total + cr[i] < threshold:
                total += cr[i]; count += 1
            else:
                break
        best = max(best, count)
    return best, never


def selectors(cap):
    return {
        "Uniform": A._wrap(A.UniformSel()),
        "FailRerun": A._wrap(A.FailRerunSel()),
        "APGAI": FastAPGAI(),
        "ScoreCost": FastScoreCost(),
        "RerunK-10": FastRerunK10(),
        "BayesUCB-Cost": FastBayesUCBCost(),
        "Mean Planning (capped)": FastERCC(AminiRevisionERCC(cap, mode="mean")),
        "ReproAlloc": FastERCC(AminiRevisionERCC(cap, planning_quantile=.5)),
    }


def run_one(pool, project, seed, ratio, method, selector, oracle_count, bank):
    item_id, failures, costs = pool["item_id"], pool["F"], pool["cost"]
    ntest = len(item_id)
    c0 = float(np.mean(costs))
    gs = A.GS(c0, c0, c0)
    budget = float(ratio * pool["B_pass"])
    mcap = int(math.ceil(budget / c0) + 1)
    rng_seed = np.random.SeedSequence([seed, zlib.crc32(project.encode()),
                                      RATIOS.index(ratio), 731])
    rng = np.random.default_rng(rng_seed)
    env = CANNIERReplay(project, seed, item_id, failures, costs, bank=bank)
    views = PoolViews(env)
    ref = failures / 2500 > TAU
    ref_int = ref & (failures < 2500)
    arms = A.Arms(ntest, gs)
    n_eliminated = 0
    if method == "HDoC-Sampling":
        scheduler_env = SchedulerEnv(env)
        hres = HDoCSampling(ntest, TAU, DELTA).run(
            arms, scheduler_env, budget, rng, finite_pool=True)
        release_scheduler_env(scheduler_env)
        n_eliminated = hres["n_eliminated"]
        init_completed = hres["init_completed"]
        sched_cpu = hres["sched_cpu_s"]
        n_exec = hres["executions"]
        used = hres["B_used"]
        overtime = hres["overtime"]
    else:
        st = selector.init(arms, views, rng)
        used = 0.0; n_exec = 0; cpu = 0.0
        while used < budget:
            t0 = time.perf_counter()
            i = selector.select(arms, views, rng, st)
            if i < 0:
                cpu += time.perf_counter() - t0
                break
            y, c = env.execute(i)
            arms.update(i, y, c)
            if method == "BayesUCB-Cost":
                selector.update(arms, i, st, available=env.has_remaining(i))
            elif method in {"Mean Planning (capped)", "ReproAlloc"}:
                selector.update(arms, i, st, available=env.has_remaining(i))
            else:
                selector.update(arms, i, st)
            cpu += time.perf_counter() - t0
            used += c; n_exec += 1
        sched_cpu = cpu
        overtime = max(0., used - budget)
        init_completed = bool(np.all(arms.n > 0))
    conf = arms.confirmed
    tp = int(np.sum(conf & ref)); tpi = int(np.sum(conf & ref_int))
    ntp, nint = int(ref.sum()), int(ref_int.sum())
    false_conf = int(np.sum(conf & ~ref))
    served = env.reset_log()
    release_views(views)
    _VIEW_ENVS.pop(id(views), None)
    return ({"benchmark": "CANNIER-Shuffle", "project": project,
             "method": method, "r": ratio, "seed": seed,
             "n_candidates": ntest, "n_pos": ntp, "n_pos_int": nint,
             "confirmed_total": int(conf.sum()), "confirmed_pos": tp,
             "confirmed_pos_int": tpi, "false_conf": false_conf,
             "recall": tp / ntp if ntp else 0.,
             "recall_int": tpi / nint if nint else 0., "n_exec": n_exec,
             "B": budget, "B_used": used, "overtime": overtime,
             "sched_cpu_s": sched_cpu, "n_eliminated": n_eliminated,
             "n_fallback": int(getattr(selector, "fallback_count", 0)) if selector else 0,
             "init_completed": init_completed,
             "oracle_count": int(oracle_count)}, served)


def run_project(project, path, logwriter):
    p = pool_read(path)
    rows, oracle_rows, conf_rows = [], [], []
    for seed in SEEDS:
        bank = SequenceBank(project, seed, p["item_id"], p["F"])
        bank.audit_all()
        seqs = np.stack([bank.sequence(i) for i in range(len(p["item_id"]))])
        prefix = seqs.cumsum(axis=1, dtype=np.int32)
        nn = np.arange(1, 2501, dtype=np.float64)
        width = np.sqrt(np.log(2.0 * nn * (nn + 1.0) / DELTA) / (2.0 * nn))
        lower = np.maximum(0.0, prefix / nn[None, :] - width[None, :])
        crossing = lower > TAU
        has_crossed = crossing.any(axis=1)
        first_k = np.where(has_crossed, crossing.argmax(axis=1) + 1, -1)
        never = int(np.sum((p["F"] / 2500 > TAU) & ~has_crossed))
        oracle_cache = {}
        for ratio in RATIOS:
            budget = ratio * float(p["B_pass"])
            oracle_count, _ = exact_oracle(p, project, seed, budget, bank, first_k, never)
            oracle_cache[ratio] = oracle_count
            positives = int(np.sum(p["F"] / 2500 > TAU))
            oracle_rows.append({"project": project, "seed": seed, "r": ratio,
                                "oracle_count": oracle_count,
                                "oracle_recall": oracle_count / positives if positives else 0.})
            conf_rows.append({"project": project, "seed": seed, "r": ratio,
                              "n_positive_never_cross": never})
        for ratio in RATIOS:
            sels = selectors(int(math.ceil((ratio * p["B_pass"]) / p["cost"].mean()) + 1))
            cell_served = {}
            for method in METHOD_ORDER:
                if method == "HDoC-Sampling":
                    sel = None
                else:
                    sel = sels[method]
                res, served = run_one(p, project, seed, ratio, method, sel,
                                      oracle_cache[ratio], bank)
                rows.append(res)
                cell_served[method] = dict(((iid, k), y) for iid, k, y in served)
                logwriter.writerows((project, seed, ratio, method, iid, k, y)
                                    for iid, k, y in served)
            # Identical realization audit on all common (test, execution-index) pairs.
            names = list(cell_served)
            for ai, a in enumerate(names):
                for b in names[ai + 1:]:
                    common = cell_served[a].keys() & cell_served[b].keys()
                    if any(cell_served[a][k] != cell_served[b][k] for k in common):
                        raise AssertionError(f"realization mismatch: {project}/{seed}/{ratio}/{a}/{b}")
    return rows, oracle_rows, conf_rows


def main():
    start = time.perf_counter()
    pool_dir = ROOT / "results/cannier/pools"
    outdir = ROOT / "results/cannier"
    config = json.loads((ROOT / "configs/cannier_eval.json").read_text(encoding="utf-8"))
    names = config["projects"]
    all_runs = []; all_oracles = []; all_conf = []
    log_path = outdir / "served.csv.gz"
    with __import__("gzip").open(log_path, "wt", newline="", encoding="utf-8") as logf:
        writer = csv.writer(logf)
        writer.writerow(["project", "seed", "r", "method", "item_id", "k", "y"])
        for project in names:
            t0 = time.perf_counter()
            r, o, c = run_project(project, pool_dir / f"{project}.npz", writer)
            all_runs.extend(r); all_oracles.extend(o); all_conf.extend(c)
            logf.flush()
            pd.DataFrame(all_runs).to_csv(outdir / "runs.csv", index=False)
            pd.DataFrame(all_oracles).to_csv(outdir / "oracle.csv", index=False)
            pd.DataFrame(all_conf).to_csv(outdir / "confirmability.csv", index=False)
            print(f"{project}: runs={len(r)} elapsed={time.perf_counter()-t0:.1f}s",
                  flush=True)
    df = pd.DataFrame(all_runs)
    df.to_csv(outdir / "runs.csv", index=False)
    df.groupby(["project", "method", "r"], as_index=False).agg(
        {"confirmed_total": "mean", "confirmed_pos": "mean", "confirmed_pos_int": "mean",
         "false_conf": "mean", "recall": "mean", "recall_int": "mean",
         "n_exec": "mean", "B": "mean", "B_used": "mean", "overtime": "mean",
         "sched_cpu_s": "mean", "n_eliminated": "mean", "init_completed": "mean",
         "n_fallback": "mean", "oracle_count": "mean"}).to_csv(
             outdir / "project_means.csv", index=False)
    print(f"completed all {len(names)} projects in {time.perf_counter()-start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
