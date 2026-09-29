"""ReproAlloc v6 (Route B): Estimated Remaining Certification Cost (ERCC).

本轮建立在 revision_v1 之上，**只修改 scheduling value**，不修改：
  - confirmation rule（仍用 revision_v1 的 anytime-valid CS 单侧下界）
  - reference labels / TUM history-eval split / budgets / tau / delta

revision_v1 已证明 one-step Certification Progress 是 myopic proxy：
它只看下一次执行的 gap reduction，给"需连续积累多次 evidence 才能确认"
的测试零值/极低值。ERCC 改为估计"若未来结果延续当前 scheduling belief，
还需多少执行/时间才能达到 confirmation boundary"。

核心定义（论文 §Method）：
  planning trajectory:  s~(m)=s+m*q,  n~(m)=n+m        (planning-only, 实数)
  planning lower bound: L~(m)=L_CS(s~(m), n~(m))       (同一公式, 直接收实数)
  m* = min{ m>=1 : L~(m) > tau }                       (不存在则 inf)
  Core:  R_hat = m* * C_bar          (mean cost)    I = 1/R_hat
  OA:    R_hat = m* * C_hat_OA       (q*cF+(1-q)cP) I = 1/R_hat
  m*=inf -> I=0 -> 固定 fallback = ScoreCost ordering (q / C_hat)。

这不是 expected stopping time 的严格期望，是 long-horizon planning proxy，
不声称全局最优。
"""
import numpy as np

EPS = 1e-9
DELTA = 0.05


# ============================================================
# confirmation（与 revision_v1 完全一致的真实确认语义）
# ============================================================
def cs_lower(s, n, delta=DELTA):
    """anytime-valid 单侧下界（真实/整数路径）；s=0 或 n=0 时 0。"""
    if n == 0 or s == 0:
        return 0.0
    phat = s / n
    width = np.sqrt(np.log(2.0 * n * (n + 1) / delta) / (2.0 * n))
    return max(0.0, phat - width)


def confirmed_sn(s, n, tau, delta=DELTA):
    return (n > 0 and s > 0 and cs_lower(s, n, delta) > tau)


# ============================================================
# planning-only lower bound（直接收实数 s/n；绝不写回真实 state）
# ============================================================
def cs_lower_plan(s_real, n_real, delta=DELTA):
    """planning-only 下界：与 cs_lower 同一公式，但接受实数计数。
    仅用于 ERCC 的 expected-evidence trajectory，不用于真实确认。"""
    if n_real <= 0 or s_real <= 0:
        return 0.0
    phat = s_real / n_real
    width = np.sqrt(np.log(2.0 * n_real * (n_real + 1.0) / delta) / (2.0 * n_real))
    return max(0.0, phat - width)


def est_remaining_execs(s, n, q, tau, m_cap, delta=DELTA):
    """Estimated remaining executions to certification.

    m* = min{ m>=1 : L_CS(s+m*q, n+m) > tau }；搜索范围 [1, m_cap]。
    返回 m*（int）或 np.inf（范围内不可确认）。

    实现说明：
    - 不假设 L~(m) 对所有状态严格单调：用粗扫(doubling/coarse grid)定位候选区间，
      再在候选区间做线性精扫，返回**最小**满足条件的 m。
    - m_cap 由 budget 推导（调用方给出），仅为计算安全 cap，不是调参对象。
    """
    if q <= 0.0:
        return np.inf  # 无失败期望，永远不会跨 threshold
    # 极限判断：m->inf 时 width->0, L~ -> q。若 q<=tau，无穷远处仍 <=tau，返回 inf。
    if q <= tau:
        return np.inf
    m_cap = int(max(1, m_cap))
    # coarse grid：指数步长覆盖 [1, m_cap]
    grid = np.unique(np.concatenate([
        np.arange(1, min(m_cap, 64) + 1),
        (np.exp2(np.arange(6, int(np.log2(m_cap)) + 1))).astype(int),
        np.array([m_cap]),
    ]))
    grid = grid[(grid >= 1) & (grid <= m_cap)]
    prev_m, prev_ok = 0, False
    lo = None
    for m in grid:
        val = cs_lower_plan(s + m * q, n + m, delta)
        ok = val > tau
        if ok:
            lo = prev_m + 1
            hi = m
            break
        prev_m, prev_ok = m, ok
    else:
        return np.inf  # 整个 cap 内都未跨 threshold
    # 在 [lo, hi] 内线性精扫，找最小满足 m
    for m in range(lo, hi + 1):
        if cs_lower_plan(s + m * q, n + m, delta) > tau:
            return m
    return int(hi)  # 理论上不会到这（hi 已确认 ok）


def compute_mstar(s, n, q, tau, delta=DELTA, cap=None):
    """First planned crossing, with the frozen capped implementation as the default gate.

    ``cap`` selects the original search exactly. Without a cap, double the
    horizon and then scan every integer up to the first successful horizon.
    Chunked NumPy arithmetic keeps that exhaustive scan practical.
    """
    if cap is not None:
        return est_remaining_execs(s, n, q, tau, cap, delta)
    if q <= tau:
        return np.inf
    limit = 10_000_000
    hi = 1
    while hi <= limit and cs_lower_plan(s + hi * q, n + hi, delta) <= tau:
        hi *= 2
    if hi > limit:
        hi = limit
        if cs_lower_plan(s + hi * q, n + hi, delta) <= tau:
            return np.inf
    for lo in range(1, hi + 1, 100_000):
        ms = np.arange(lo, min(lo + 100_000, hi + 1), dtype=np.float64)
        nn = n + ms
        ss = s + ms * q
        width = np.sqrt(np.log(2.0 * nn * (nn + 1.0) / delta) / (2.0 * nn))
        hits = np.flatnonzero(np.maximum(0.0, ss / nn - width) > tau)
        if hits.size:
            return int(ms[hits[0]])
    return np.inf


class ERCCache:
    """per-arm m* 缓存：只有该臂被执行后才重算（调用方负责 invalidate）。"""
    def __init__(self):
        self.val = None
        self.key = None

    def get(self, s, n, q, tau, m_cap, delta=DELTA):
        k = (s, n, round(q, 12), tau, m_cap)
        if k != self.key:
            self.val = est_remaining_execs(s, n, q, tau, m_cap, delta)
            self.key = k
        return self.val
