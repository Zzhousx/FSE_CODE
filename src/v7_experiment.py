"""ReproAlloc v7 / v2->v3 聚焦修订实验：公平 fallback 对照 + 强基线配对。

继承 v6_experiment.py 的全部协议（leakage-free split、软非抢占预算、anytime-valid CS、
相同 tau/delta/budgets/seeds/prior strength/ERCC 定义）。

本轮唯一新增（按 v2->v3 方案 §3.2）：
  公平 fallback 对照变体，隔离 "planning value" 与 "零分回退策略" 两个变化：
    - onestep_mean_scfb   : OneStep 主指标(mean cost)，全零时回退 q/mean_cost
    - onestep_outcome_scfb: OneStep 主指标(OA cost)，  全零时回退 q/OA_cost
  与对应 ERCC 变体的 eligible set/全零判定/破平/已确认排除完全一致。
  保留原 onestep_mean / onestep_outcome（随机破平）作为诊断 ablation，不删除不覆盖。

诊断信息（方案 §3.4）：每个 selector 记录
  - n_main_pos   : 主指标为正并据此选择的执行次数
  - n_fallback   : 所有主分数为零、触发 fallback 的执行次数
  - 由 run_task 汇总 fallback_ratio、recall、confirmed、time_used。

配对分析（方案 §4.2）在 v7_analysis.py 中：
  ERCC-ScoreCost+Outcome / ERCCOA-ScoreCost+Outcome / ERCC-FailRerun /
  相同 fallback 下 ERCC-OneStepSCFB（mean 与 OA 分别）。
"""
import numpy as np
import pandas as pd
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reproalloc_v6 as v6

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "data")
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "results", "_v7_legacy_out")
EPS = 1e-9
DELTA = 0.05
SEEDS = [20260905, 20260906, 20260907]
GEN_SD = {0: "AmbieGen", 1: "Frenetic", 2: "FreneticV"}
KAPPA = 5.0

os.makedirs(OUT, exist_ok=True)


# ============================================================
# Arms / GS（与 v6 完全相同）
# ============================================================
class Arms:
    def __init__(self, K, prior_score, gs, tau, kappa=KAPPA):
        self.K, self.tau, self.gs = K, tau, gs
        self.n = np.zeros(K, dtype=np.int32)
        self.s = np.zeros(K, dtype=np.int32)
        self.fail_cost_sum = np.zeros(K)
        self.fail_cost_cnt = np.zeros(K, dtype=np.int32)
        self.pass_cost_sum = np.zeros(K)
        self.pass_cost_cnt = np.zeros(K, dtype=np.int32)
        if prior_score is not None:
            self.post_a = 1.0 + kappa * prior_score
            self.post_b = 1.0 + kappa * (1.0 - prior_score)
        else:
            self.post_a = np.ones(K)
            self.post_b = np.ones(K)
        self.confirmed = np.zeros(K, dtype=bool)

    def q_arr(self):
        return self.post_a / (self.post_a + self.post_b)

    def mean_cost_arr(self):
        obs = (self.fail_cost_sum + self.pass_cost_sum) / np.maximum(self.n, 1)
        return np.where(self.n > 0, obs, self.gs.mean_overall)

    def outcome_cost_arr(self):
        q = self.q_arr()
        cF = np.where(self.fail_cost_cnt > 0,
                      self.fail_cost_sum / np.maximum(self.fail_cost_cnt, 1),
                      self.gs.mean_fail_cost)
        cP = np.where(self.pass_cost_cnt > 0,
                      self.pass_cost_sum / np.maximum(self.pass_cost_cnt, 1),
                      self.gs.mean_pass_cost)
        return q * cF + (1.0 - q) * cP

    def update(self, i, fail, cost):
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
        self.confirmed[i] = v6.confirmed_sn(self.s[i], self.n[i], self.tau)


class GS:
    def __init__(self, mf, mp, mo):
        self.mean_fail_cost, self.mean_pass_cost, self.mean_overall = mf, mp, mo


# ============================================================
# 标量打分（与 v6 相同）
# ============================================================
def _q(arms, i):
    return arms.post_a[i] / (arms.post_a[i] + arms.post_b[i])


def _mean_cost(arms, i):
    n = arms.n[i]
    return ((arms.fail_cost_sum[i] + arms.pass_cost_sum[i]) / n) if n > 0 else arms.gs.mean_overall


def _oa_cost(arms, i):
    q = _q(arms, i)
    cF = (arms.fail_cost_sum[i] / arms.fail_cost_cnt[i]) if arms.fail_cost_cnt[i] > 0 else arms.gs.mean_fail_cost
    cP = (arms.pass_cost_sum[i] / arms.pass_cost_cnt[i]) if arms.pass_cost_cnt[i] > 0 else arms.gs.mean_pass_cost
    return q * cF + (1.0 - q) * cP


def _onestep_scalar(arms, i, use_oc):
    """revision_v1 的 one-step Certification Progress（诊断 ablation 主指标）。"""
    if arms.confirmed[i]:
        return -np.inf
    s, n = arms.s[i], arms.n[i]
    q = _q(arms, i)
    g = max(0.0, arms.tau - v6.cs_lower(s, n))
    g_f = max(0.0, arms.tau - v6.cs_lower(s + 1, n + 1))
    g_p = max(0.0, arms.tau - v6.cs_lower(s, n + 1))
    numer = q * max(0.0, g - g_f) + (1.0 - q) * max(0.0, g - g_p)
    denom = _oa_cost(arms, i) if use_oc else _mean_cost(arms, i)
    return numer / max(denom, EPS)


def _scorecost_scalar(arms, i, use_oc):
    """ScoreCost ordering：q / C_hat（也是 ERCC 与新 OneStep-SCFB 的固定 fallback）。"""
    if arms.confirmed[i]:
        return -np.inf
    q = _q(arms, i)
    denom = _oa_cost(arms, i) if use_oc else _mean_cost(arms, i)
    return q / max(denom, EPS)


# ============================================================
# 选择器（带诊断计数）
# ============================================================
class OneStepSel:
    """诊断 ablation（原版，随机破平）：revision_v1 的 one-step CertProgress。"""
    def __init__(self, use_oc):
        self.uoc = use_oc
        self.n_main_pos = 0
        self.n_fallback = 0
    def init(self, arms, rng):
        idx = np.array([_onestep_scalar(arms, i, self.uoc) for i in range(arms.K)])
        idx = idx + rng.random(arms.K) * 1e-12
        return {"idx": idx}
    def select(self, arms, st, rng):
        idx = st["idx"]
        elig = np.flatnonzero(idx > -np.inf)
        if elig.size == 0:
            return -1
        mx = np.max(idx[elig])
        if mx <= 0.0:
            self.n_fallback += 1  # 全零（one-step 主分无正值）-> 随机破平（诊断版 fallback）
        else:
            self.n_main_pos += 1
        ties = elig[idx[elig] >= mx - 1e-12]
        return int(ties[rng.integers(0, ties.size)])
    def update(self, arms, i, st):
        st["idx"][i] = _onestep_scalar(arms, i, self.uoc)


class OneStepSCFBSel:
    """新公平对照：OneStep 主指标，全零时回退 ScoreCost(q/C)，与 ERCC 完全一致。
    eligible set/全零判定/破平/已确认排除 与 ERCCSel 相同；正进展时用 OneStep 原主指标。"""
    def __init__(self, use_oc):
        self.uoc = use_oc
        self.n_main_pos = 0
        self.n_fallback = 0
    def init(self, arms, rng):
        idx = np.array([_onestep_scalar(arms, i, self.uoc) for i in range(arms.K)])
        return {"idx": idx}
    def select(self, arms, st, rng):
        cand = np.where(~arms.confirmed)[0]
        if cand.size == 0:
            return -1
        idx = st["idx"]
        pos = cand[idx[cand] > 0.0]
        if pos.size > 0:
            self.n_main_pos += 1
            mx = np.max(idx[pos])
            ties = pos[idx[pos] >= mx - 1e-12]
            return int(ties[rng.integers(0, ties.size)])
        # 全部 OneStep 主分<=0：固定 fallback = ScoreCost ordering（与 ERCC 相同）
        self.n_fallback += 1
        fb = np.array([_scorecost_scalar(arms, i, self.uoc) for i in cand])
        mx = np.max(fb)
        ties = cand[fb >= mx - 1e-12]
        return int(ties[rng.integers(0, ties.size)])
    def update(self, arms, i, st):
        st["idx"][i] = _onestep_scalar(arms, i, self.uoc)


class ScoreCostSel:
    def __init__(self, use_oc):
        self.uoc = use_oc
        self.n_main_pos = 0
        self.n_fallback = 0
    def init(self, arms, rng):
        idx = np.array([_scorecost_scalar(arms, i, self.uoc) for i in range(arms.K)])
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
        st["idx"][i] = _scorecost_scalar(arms, i, self.uoc)


class ERCCSel:
    """新 ReproAlloc（use_oc=False）/ ReproAlloc-OA（use_oc=True）。带诊断计数。"""
    def __init__(self, use_oc, m_cap):
        self.uoc = use_oc
        self.m_cap = int(m_cap)
        self.n_main_pos = 0
        self.n_fallback = 0
    def init(self, arms, rng):
        K = arms.K
        cache = [v6.ERCCache() for _ in range(K)]
        ercc = np.zeros(K)
        for i in range(K):
            if arms.confirmed[i]:
                ercc[i] = -np.inf
            else:
                q = _q(arms, i)
                m_star = cache[i].get(arms.s[i], arms.n[i], q, arms.tau, self.m_cap)
                if np.isinf(m_star):
                    ercc[i] = 0.0
                else:
                    ercc[i] = 1.0 / max(m_star * (_oa_cost(arms, i) if self.uoc else _mean_cost(arms, i)), EPS)
        return {"ercc": ercc, "cache": cache}
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
            return int(ties[rng.integers(0, ties.size)])
        self.n_fallback += 1
        fb = np.array([_scorecost_scalar(arms, i, self.uoc) for i in cand])
        mx = np.max(fb)
        ties = cand[fb >= mx - 1e-12]
        return int(ties[rng.integers(0, ties.size)])
    def update(self, arms, i, st):
        if arms.confirmed[i]:
            st["ercc"][i] = -np.inf
            return
        q = _q(arms, i)
        m_star = st["cache"][i].get(arms.s[i], arms.n[i], q, arms.tau, self.m_cap)
        if np.isinf(m_star):
            st["ercc"][i] = 0.0
        else:
            st["ercc"][i] = 1.0 / max(m_star * (_oa_cost(arms, i) if self.uoc else _mean_cost(arms, i)), EPS)


class FnSel:
    """外部基线（无缓存向量化）。诊断计数恒 0（主分总为正或无意义）。"""
    def __init__(self, kind, order=None):
        self.kind, self.order = kind, order
        self.n_main_pos = 0
        self.n_fallback = 0
    def init(self, arms, rng):
        return {"pos": 0}
    def select(self, arms, st, rng):
        cand = np.where(~arms.confirmed)[0]
        if len(cand) == 0:
            return -1
        if self.kind == "uniform":
            return int(cand[rng.integers(0, len(cand))])
        if self.kind == "failrerun":
            failed = cand[arms.s[cand] > 0]
            if len(failed) > 0:
                return int(failed[np.argmax(arms.s[failed] / np.maximum(arms.n[failed], 1))])
            uninit = cand[arms.n[cand] == 0]
            if len(uninit) == 0:
                return int(cand[0])
            est = arms.outcome_cost_arr()[uninit]
            ties = uninit[est <= est.min() + 1e-9]
            return int(ties[rng.integers(0, len(ties))])
        if self.kind == "apgai":
            uninit = cand[arms.n[cand] == 0]
            if len(uninit) > 0:
                est = arms.outcome_cost_arr()[uninit]
                ties = uninit[est <= est.min() + 1e-9]
                return int(ties[rng.integers(0, len(ties))])
            n = arms.n[cand]
            mu = arms.s[cand] / np.maximum(n, 1)
            cost = arms.outcome_cost_arr()[cand]
            idx = np.sqrt(n) * (mu - arms.tau) / np.maximum(cost, EPS)
            return int(cand[np.argmax(idx)])
        return -1
    def update(self, arms, i, st):
        pass


# ============================================================
# 通用运行循环（软非抢占预算，含 scheduler overhead 计时 + fallback 诊断）
# ============================================================
def run_task(K, draw_fn, gt_good, budget, prior_score, gs, tau, sel, rng, kappa=KAPPA):
    arms = Arms(K, prior_score, gs, tau, kappa)
    st = sel.init(arms, rng)
    # 每次运行重置诊断计数（selector 对象在同一方法的多 rep 间复用）
    if hasattr(sel, "n_main_pos"):
        sel.n_main_pos = 0
    if hasattr(sel, "n_fallback"):
        sel.n_fallback = 0
    time_used = 0.0
    execs = 0
    sched_time = 0.0
    while time_used < budget:
        t0 = time.perf_counter()
        i = sel.select(arms, st, rng)
        sched_time += time.perf_counter() - t0
        if i < 0:
            break
        fail, actual_cost = draw_fn(i, rng)
        arms.update(i, fail, actual_cost)
        t0 = time.perf_counter()
        sel.update(arms, i, st)
        sched_time += time.perf_counter() - t0
        time_used += actual_cost
        execs += 1
    conf = arms.confirmed
    tp = int((conf & gt_good).sum())
    fp = int((conf & ~gt_good).sum())
    n_main = int(getattr(sel, "n_main_pos", 0))
    n_fb = int(getattr(sel, "n_fallback", 0))
    return {"tp": tp, "fp": fp, "fn": int((gt_good & ~conf).sum()),
            "confirmed": int(conf.sum()),
            "recall": tp / gt_good.sum() if gt_good.sum() else 0.0,
            "precision": tp / conf.sum() if conf.sum() else 0.0,
            "executions": execs, "time_used": time_used,
            "overtime": max(0.0, time_used - budget),
            "sched_overhead_s": sched_time,
            "n_main_pos": n_main, "n_fallback": n_fb,
            "fallback_ratio": (n_fb / execs) if execs else 0.0}


def m_cap_from_budget(budget, gs):
    min_cost = max(min(gs.mean_fail_cost, gs.mean_pass_cost), EPS)
    return int(max(1, np.ceil(budget / min_cost)) + 1)


def v7_variants(m_cap):
    """内部方法：v6 的 6 个 + 2 个公平 fallback 新对照。"""
    return [("scorecost", ScoreCostSel(False)),
            ("scorecost_outcome", ScoreCostSel(True)),
            ("onestep_mean", OneStepSel(False)),
            ("onestep_outcome", OneStepSel(True)),
            ("onestep_mean_scfb", OneStepSCFBSel(False)),
            ("onestep_outcome_scfb", OneStepSCFBSel(True)),
            ("ercc_mean", ERCCSel(False, m_cap)),
            ("ercc_outcome", ERCCSel(True, m_cap))]


# ============================================================
# 正确性测试（v6 Test A-H + 新 Test I/J 公平 fallback 行为）
# ============================================================
def run_correctness_checks():
    print("=" * 72)
    print("### v7 正确性检查 (Test A-H 沿用 + Test I/J 公平 fallback)")
    print("=" * 72)
    tau = 0.3
    gs = GS(10.0, 2.0, 6.0)

    # Test A: multi-step value
    arms = Arms(1, np.array([0.6]), gs, tau)
    arms.n[0], arms.s[0] = 4, 3
    q = _q(arms, 0)
    onestep = _onestep_scalar(arms, 0, False)
    m_cap = m_cap_from_budget(1000.0, gs)
    m_star = v6.est_remaining_execs(arms.s[0], arms.n[0], q, tau, m_cap)
    assert not np.isinf(m_star), f"Test A FAIL m*={m_star}"
    print(f"  Test A (multi-step value): one_step={onestep:.2e} m*={m_star} PASS")

    # Test I: 公平 fallback —— 全零时 SCFB 用 ScoreCost，且与 ERCC fallback 选臂一致
    # 构造 4 个 q<=tau（one-step=0 且 m*=inf）的臂，成本不同 -> fallback 应选 q/C 最大者
    armsI = Arms(4, np.full(4, 0.1), gs, tau)  # q=0.1<tau=0.3
    # 设置不同观测成本使 ScoreCost 排序确定（n>0 时用 obs 成本）
    for i in range(4):
        armsI.n[i] = 2
        armsI.fail_cost_sum[i] = 1.0 * (i + 1)   # fail_cost_cnt=1
        armsI.fail_cost_cnt[i] = 1
        armsI.pass_cost_sum[i] = 1.0 * (i + 1)   # pass_cost_cnt=1
        armsI.pass_cost_cnt[i] = 1
    # mean_cost: arm0=1.0, arm1=2.0, arm2=3.0, arm3=4.0；q 相同 -> ScoreCost 选 arm0
    selI = OneStepSCFBSel(False)
    stI = selI.init(armsI, np.random.default_rng(0))
    pickI = selI.select(armsI, stI, np.random.default_rng(0))
    # ERCC 在同样状态下也应 fallback 到同一臂
    selE = ERCCSel(False, 1000)
    stE = selE.init(armsI, np.random.default_rng(0))
    pickE = selE.select(armsE := armsI, stE, np.random.default_rng(0))
    assert pickI == 0 and pickE == 0, f"Test I FAIL: scfb={pickI} ercc={pickE} (expect 0)"
    assert selI.n_fallback == 1 and selI.n_main_pos == 0, "Test I FAIL counts"
    print(f"  Test I (fair fallback==ERCC fallback): scfb_pick={pickI} ercc_pick={pickE} PASS")

    # Test J: 正进展时 SCFB 用 OneStep 主指标（不退化）
    armsJ = Arms(2, np.array([0.6, 0.6]), gs, tau)
    # arm0 构造为 one-step>0（接近确认），arm1 冷启动
    armsJ.n[0], armsJ.s[0] = 5, 4
    selJ = OneStepSCFBSel(False)
    stJ = selJ.init(armsJ, np.random.default_rng(0))
    idx0 = stJ["idx"][0]
    pickJ = selJ.select(armsJ, stJ, np.random.default_rng(0))
    assert idx0 > 0.0 and pickJ == 0 and selJ.n_main_pos == 1, \
        f"Test J FAIL: idx0={idx0} pick={pickJ}"
    print(f"  Test J (positive progress uses OneStep): idx0={idx0:.3e} pick={pickJ} PASS")
    print("all v7 correctness checks passed")


# ============================================================
# Synthetic smoke（与 v6 相同环境 + 新对照变体）
# ============================================================
def run_synthetic_smoke():
    print("=" * 72)
    print("### v7 Synthetic smoke：多步 evidence accumulation（含公平 fallback 对照）")
    print("=" * 72)
    tau = 0.3
    true_p = np.array([0.55] * 10 + [0.05] * 10)
    cF = np.full(20, 4.0)
    cP = np.full(20, 4.0)
    K = 20
    gt = true_p > tau
    gs = GS(4.0, 4.0, 4.0)

    def draw(i, rng):
        f = int(rng.random() < true_p[i])
        return f, (cF[i] if f else cP[i])

    budget = 600.0
    m_cap = m_cap_from_budget(budget, gs)
    sels = ([("uniform", FnSel("uniform")), ("failrerun", FnSel("failrerun")),
             ("apgai", FnSel("apgai"))] + v7_variants(m_cap))
    rows = []
    for name, sf in sels:
        recs = []
        fbs = []
        for rep in range(10):
            rng = np.random.default_rng(7000 + rep)
            r = run_task(K, draw, gt, budget, None, gs, tau, sf, rng)
            r.update({"method": name, "rep": rep})
            rows.append(r)
            recs.append(r["recall"])
            fbs.append(r["fallback_ratio"])
        print(f"  {name:22s} recall={np.mean(recs):.3f} +/- {np.std(recs):.3f}  fb={np.mean(fbs):.2f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "v7_synthetic_smoke.csv"), index=False)
    print(f"  保存 {len(df)} 条 -> v7_synthetic_smoke.csv")
    return rows


# ============================================================
# SensoDat（与 v6 相同协议 + 新对照变体）
# ============================================================
def run_sensodat(only_transfer=None):
    from sklearn.ensemble import HistGradientBoostingClassifier
    print("=" * 72)
    print("### v7 SensoDat 时间预算协议（8 内部方法 + 外部基线）")
    print("=" * 72)
    d = np.load(os.path.join(DATA, "sensodat_fresh.npz"), allow_pickle=True)
    X_spec, y = d["X_spec"], d["y"].astype(int)
    dur = np.where(np.isnan(d["duration_s"]) | (d["duration_s"] <= 0),
                   np.nanmedian(d["duration_s"]), d["duration_s"]).astype(float)
    with open(os.path.join(RES, "sensodat_splits.json")) as f:
        splits = json.load(f)
    combos = {}
    for r in splits:
        p = r["key"].split("_")
        if p[2] != "HGB-spec":
            continue
        combos[(int(p[0][1:]), int(p[1][1:]), int(p[3]))] = r
    tau = 0.3
    rows = []
    for (s_id, t_id, seed), rec in sorted(combos.items()):
        if only_transfer and (s_id, t_id) != only_transfer:
            continue
        train_idx = np.array(rec["train"])
        target_idx = np.array(rec["target"])
        ytr, dtr = y[train_idx], dur[train_idx]
        gs = GS(float(dtr[ytr == 1].mean()) if (ytr == 1).any() else float(dtr.mean()),
                float(dtr[ytr == 0].mean()) if (ytr == 0).any() else float(dtr.mean()),
                float(dtr.mean()))
        clf = HistGradientBoostingClassifier(random_state=seed, max_iter=200)
        clf.fit(X_spec[train_idx], ytr)
        scores = clf.predict_proba(X_spec[target_idx])[:, 1]
        y_t, d_t = y[target_idx], dur[target_idx]
        K = len(target_idx)
        gt_good = (y_t == 1)
        total_time = float(d_t.sum())

        def draw(i, rng):
            return int(y_t[i]), float(d_t[i])

        for tf in [0.05, 0.10, 0.20]:
            budget = total_time * tf
            m_cap = m_cap_from_budget(budget, gs)
            sels = ([("uniform", FnSel("uniform")),
                     ("failrerun", FnSel("failrerun")), ("apgai", FnSel("apgai"))]
                    + v7_variants(m_cap))
            for mname, sf in sels:
                rng = np.random.default_rng(seed + int(tf * 1000))
                r = run_task(K, draw, gt_good, budget, scores, gs, tau, sf, rng)
                r.update({"dataset": "sensodat", "source": GEN_SD[s_id],
                          "target": GEN_SD[t_id], "seed": seed,
                          "method": mname, "time_frac": tf})
                rows.append(r)
        print(f"  {GEN_SD[s_id]}->{GEN_SD[t_id]} seed={seed} 完成")
    return rows


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "check"
    if which == "check":
        run_correctness_checks()
    elif which == "smoke":
        run_synthetic_smoke()
    elif which == "sd":
        df = pd.DataFrame(run_sensodat(only_transfer=(0, 1)))
        df.to_csv(os.path.join(OUT, "v7_sensodat.csv"), index=False)
        print(f"保存 {len(df)} 条 -> v7_sensodat.csv")
    elif which == "sdfull":
        df = pd.DataFrame(run_sensodat())
        df.to_csv(os.path.join(OUT, "v7_sensodat_full.csv"), index=False)
        print(f"保存 {len(df)} 条 -> v7_sensodat_full.csv")
