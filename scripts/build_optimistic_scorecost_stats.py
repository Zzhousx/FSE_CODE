# -*- coding: utf-8 -*-
"""OptimisticScoreCost check: regression verification + statistics + report.

Reads results/optimistic_scorecost_check/{sensodat_osc_check.csv,
amini_osc_check.csv, amini_osc_check_alloc.csv} and the official main CSVs in
results/01_optimistic_core/, then:

  1. REGRESSION: the seven pre-existing methods must match the official main
     results exactly (per-cell recall/executions/time/tp/confirmed).
  2. Writes optimistic_scorecost_raw.csv      (unified long rows, spec sec.11)
  3. Writes optimistic_scorecost_summary.csv  (per benchmark x budget x method)
  4. Writes optimistic_scorecost_paired.csv   (ReproAlloc - OptimisticScoreCost;
     CI = 1.96*sd/sqrt(n) on paired diffs, Wilcoxon, wins/losses/ties at 1e-12,
     identical to build_final_statistics.paired)
  5. Writes reports/optimistic_scorecost_check.md (8-section structure).

Units: Amini = seed (12); SensoDat = split (10, nested reps averaged first).
Naming: ReproAlloc-Optimistic -> ReproAlloc; ReproAlloc -> Mean Planning.
"""
import math
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OSC = os.path.join(RES, "optimistic_scorecost_check")
CORE = os.path.join(RES, "01_optimistic_core")
REPORT = os.path.join(ROOT, "reports", "optimistic_scorecost_check.md")

RENAME = {"ReproAlloc-Optimistic": "ReproAlloc", "ReproAlloc": "Mean Planning"}
BUDGETS = [0.05, 0.10, 0.20]
FOUR = ["ScoreCost", "OptimisticScoreCost", "Mean Planning", "ReproAlloc"]


# ----------------------------------------------------------------------
def paired(a, b):
    """Byte-equivalent to build_final_statistics.paired."""
    common = a.index.intersection(b.index)
    d = a[common] - b[common]
    n = len(d)
    m_ = float(d.mean())
    sd_ = float(d.std(ddof=1))
    ci = 1.96 * sd_ / math.sqrt(n) if sd_ == sd_ else 0.0
    try:
        p = float(wilcoxon(d).pvalue) if (d != 0).any() else 1.0
    except Exception:
        p = float("nan")
    wins = int((d > 1e-12).sum())
    losses = int((d < -1e-12).sum())
    return m_, m_ - ci, m_ + ci, p, wins, losses, n - wins - losses


def mean_ci(vals):
    vals = np.asarray(vals, dtype=float)
    n = len(vals)
    m_ = float(vals.mean())
    sd_ = float(vals.std(ddof=1))
    ci = 1.96 * sd_ / math.sqrt(n) if sd_ == sd_ else 0.0
    return m_, m_ - ci, m_ + ci, n


# ----------------------------------------------------------------------
def regression_check():
    msgs = []
    ok_all = True
    # --- SensoDat ---
    new = pd.read_csv(os.path.join(OSC, "sensodat_osc_check.csv"))
    old = pd.read_csv(os.path.join(CORE, "sensodat", "sensodat_optimistic_main.csv"))
    keys = ["split_id", "rep", "time_frac", "method"]
    cols = ["recall", "executions", "time_used", "tp", "confirmed", "precision"]
    m = new.merge(old[keys + cols], on=keys, suffixes=("_new", "_old"))
    worst = 0.0
    for c in cols:
        d = (m[f"{c}_new"].astype(float) - m[f"{c}_old"].astype(float)).abs().max()
        worst = max(worst, float(d))
    ok = (len(m) == len(old)) and worst == 0.0
    ok_all &= ok
    msgs.append(f"SensoDat: {len(m)}/{len(old)} overlap rows, "
                f"max|diff| over {cols} = {worst:.3e} -> "
                f"{'IDENTICAL' if ok else 'MISMATCH'}")
    # --- Amini ---
    new = pd.read_csv(os.path.join(OSC, "amini_osc_check.csv"))
    old = pd.read_csv(os.path.join(CORE, "amini", "amini_optimistic_main.csv"))
    keys = ["seed", "budget_frac", "method"]
    cols = ["recall_refpos", "executions", "time_used_s", "tp", "confirmed",
            "precision"]
    m = new.merge(old[keys + cols], on=keys, suffixes=("_new", "_old"))
    worst = 0.0
    for c in cols:
        d = (m[f"{c}_new"].astype(float) - m[f"{c}_old"].astype(float)).abs().max()
        worst = max(worst, float(d))
    ok = (len(m) == len(old)) and worst == 0.0
    ok_all &= ok
    msgs.append(f"Amini: {len(m)}/{len(old)} overlap rows, "
                f"max|diff| over {cols} = {worst:.3e} -> "
                f"{'IDENTICAL' if ok else 'MISMATCH'}")
    return ok_all, msgs


# ----------------------------------------------------------------------
def build_raw():
    """Unified long raw table (spec sec.11 fields + behavior fields)."""
    rows = []
    sd = pd.read_csv(os.path.join(OSC, "sensodat_osc_check.csv"))
    for _, r in sd.iterrows():
        rows.append({
            "benchmark": "sensodat", "budget": r.time_frac,
            "seed_or_unit": int(r.split_id), "rep": int(r.rep),
            "method": RENAME.get(r.method, r.method),
            "confirmed_count": int(r.confirmed),
            "confirmed_recall": float(r.recall),
            "budget_used": float(r.time_used),
            "num_executions": int(r.executions),
            "unique_tests_visited": int(r.unique_tests_visited),
            "mean_executions_per_visited_test": float(r.mean_exec_per_visited),
            "fallback_count": int(r.n_fallback),
            "scheduler_steps": int(r.executions),
            "exec_finally_confirmed": int(r.exec_finally_confirmed),
            "exec_finally_unconfirmed": int(r.exec_finally_unconfirmed)})
    am = pd.read_csv(os.path.join(OSC, "amini_osc_check.csv"))
    al = pd.read_csv(os.path.join(OSC, "amini_osc_check_alloc.csv"))
    beh = {}
    for (sd_, tf, meth), g in al.groupby(["seed", "budget_frac", "method"]):
        vis = g[g.runs_used > 0]
        nv = len(vis)
        exc = int(vis.runs_used.sum())
        e_conf = int(g.loc[g.confirmed_sticky == 1, "runs_used"].sum())
        beh[(sd_, tf, meth)] = (nv, exc / nv if nv else 0.0, e_conf, exc - e_conf)
    for _, r in am.iterrows():
        nv, mepv, e_conf, e_unc = beh[(r.seed, r.budget_frac, r.method)]
        rows.append({
            "benchmark": "amini", "budget": r.budget_frac,
            "seed_or_unit": int(r.seed), "rep": 0,
            "method": RENAME.get(r.method, r.method),
            "confirmed_count": int(r.confirmed),
            "confirmed_recall": float(r.recall_refpos),
            "budget_used": float(r.time_used_s),
            "num_executions": int(r.executions),
            "unique_tests_visited": nv,
            "mean_executions_per_visited_test": mepv,
            "fallback_count": "",
            "scheduler_steps": int(r.executions),
            "exec_finally_confirmed": e_conf,
            "exec_finally_unconfirmed": e_unc})
    raw = pd.DataFrame(rows)
    raw.to_csv(os.path.join(OSC, "optimistic_scorecost_raw.csv"), index=False)
    return raw


def unit_recalls(raw):
    """(benchmark, budget, method) -> pd.Series indexed by unit."""
    out = {}
    for (bk, bud, meth), g in raw.groupby(["benchmark", "budget", "method"]):
        u = g.groupby("seed_or_unit")["confirmed_recall"].mean().sort_index()
        out[(bk, float(bud), meth)] = u
    return out


def build_summary_paired(raw):
    units = unit_recalls(raw)
    srows, prows = [], []
    for bk in ["sensodat", "amini"]:
        for bud in BUDGETS:
            for meth in FOUR:
                u = units[(bk, bud, meth)]
                m_, lo, hi, n = mean_ci(u.values)
                srows.append({"benchmark": bk, "budget": bud, "method": meth,
                              "n_units": n, "mean_recall": m_,
                              "ci_low": lo, "ci_high": hi})
            ra = units[(bk, bud, "ReproAlloc")]
            osc = units[(bk, bud, "OptimisticScoreCost")]
            m_, lo, hi, p, w_, l_, t_ = paired(ra, osc)
            prows.append({"benchmark": bk, "budget": bud,
                          "reproalloc_mean": float(ra.mean()),
                          "osc_mean": float(osc.mean()),
                          "paired_delta": m_, "paired_ci_low": lo,
                          "paired_ci_high": hi, "wilcoxon_p": p,
                          "wins": w_, "losses": l_, "ties": t_})
    s = pd.DataFrame(srows)
    p = pd.DataFrame(prows)
    s.to_csv(os.path.join(OSC, "optimistic_scorecost_summary.csv"), index=False)
    p.to_csv(os.path.join(OSC, "optimistic_scorecost_paired.csv"), index=False)
    return s, p, units


# ----------------------------------------------------------------------
def fmt(x, nd=4):
    return f"{x:.{nd}f}"


def write_report(reg_ok, reg_msgs, summary, paired_df, raw):
    L = []
    L.append("# OptimisticScoreCost Check")
    L.append("")
    L.append("## 1. Implementation")
    L.append("")
    L.append("- 定义：`I_OSC = q_O / C_bar`，其中 "
             "`q_O = Beta.ppf(1 - 0.05; post_a, post_b)`——与 ReproAlloc 完全相同的 "
             "optimistic planning rate（`src/optimistic_ercc.py::planning_rate` 单一实现，两侧共用）。")
    L.append("- 不计算 `m*`，不使用 `R_i = m* C_bar`；ScoreCost 中的 posterior mean 被原样替换为 q_O，其余不变。")
    L.append("- 代码位置：SensoDat `src/optimistic_ercc_sensodat.py::OptimisticScoreCostSel`"
             "（与 `v7_experiment.ScoreCostSel` 逐行同构，注册追加 `METHODS_8`）；"
             "Amini `src/optimistic_ercc_amini.py::OptimisticScoreCostSel`"
             "（与 `A.ScoreCostSel` 同构，`build_selectors_8` 追加在 method_idx=7）。"
             "Runner：`scripts/run_optimistic_scorecost.py`。")
    L.append("- 信息边界：只用 post_a/post_b、delta、观测执行成本、合法初始成本标量（n=0 时 gs.mean_overall）、"
             "当前候选状态；不用未来结果/未来 trace 顺序/全池失败率/reference 标签。")
    L.append("- 与 ScoreCost 的唯一差异：排序标量 q_mean→q_O。与 ReproAlloc 的唯一差异：去掉 m* remaining-confirmation 建模。"
             "确认规则（整数 anytime-valid CS 下界 > tau=0.3）对所有方法完全相同，q_O 只用于调度。")
    L.append("")
    L.append("## 2. Regression Check")
    L.append("")
    for m in reg_msgs:
        L.append(f"- {m}")
    L.append(f"- 结论：{'已有 7 个方法在新增 OptimisticScoreCost 后逐单元格完全一致，无代码污染。' if reg_ok else '**存在不一致——停止并排查。**'}")
    L.append("")

    # per-benchmark sections
    for bk, title, sec in [("sensodat", "SensoDat", "3"), ("amini", "Amini", "4")]:
        L.append(f"## {sec}. {title} Results")
        L.append("")
        L.append("| budget | ScoreCost | OptimisticScoreCost | Mean Planning | ReproAlloc |")
        L.append("|---:|---:|---:|---:|---:|")
        for bud in BUDGETS:
            cells = []
            for meth in FOUR:
                r = summary[(summary.benchmark == bk) &
                            (summary.budget == bud) &
                            (summary.method == meth)].iloc[0]
                cells.append(f"{fmt(r.mean_recall)} [{fmt(r.ci_low)}, {fmt(r.ci_high)}]")
            L.append(f"| {int(bud*100)}% | " + " | ".join(cells) + " |")
        n_u = 10 if bk == "sensodat" else 12
        L.append("")
        L.append(f"单元均值 [95% CI]，统计单位 n={n_u}"
                 + ("（10 个 source-training fits，3 次 tie-break 重复先在单元内平均）。"
                    if bk == "sensodat" else
                    "（12 个 matched trace-order seeds）。"))
        L.append("")

    L.append("## 5. Paired ReproAlloc vs OptimisticScoreCost")
    L.append("")
    L.append("| benchmark | budget | ReproAlloc | OptimisticScoreCost | paired Δ | 95% paired CI | Wilcoxon p | wins | losses | ties |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in paired_df.iterrows():
        L.append(f"| {r.benchmark} | {int(r.budget*100)}% | {fmt(r.reproalloc_mean)} | "
                 f"{fmt(r.osc_mean)} | {fmt(r.paired_delta)} | "
                 f"[{fmt(r.paired_ci_low)}, {fmt(r.paired_ci_high)}] | "
                 f"{r.wilcoxon_p:.4f} | {r.wins} | {r.losses} | {r.ties} |")
    L.append("")
    L.append("配对差 = ReproAlloc − OptimisticScoreCost（单元匹配）；CI = 1.96·s/√n（样本标准差 ddof=1），"
             "与论文主分析 `build_final_statistics.paired` 同一实现；Wilcoxon signed-rank（双侧，scipy 默认）。"
             "wins/losses/ties 以 1e-12 为阈值。")
    L.append("")

    # 2x2 component table
    L.append("## 6. 2×2 Component Comparison")
    L.append("")
    L.append("均值 recall（每 benchmark × budget）：")
    L.append("")
    L.append("| benchmark | budget | ScoreCost (mean rate, immediate) | OptimisticScoreCost (optimistic, immediate) | Mean Planning (mean rate, remaining-cost) | ReproAlloc (optimistic, remaining-cost) |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for bk in ["sensodat", "amini"]:
        for bud in BUDGETS:
            cells = []
            for meth in FOUR:
                r = summary[(summary.benchmark == bk) &
                            (summary.budget == bud) &
                            (summary.method == meth)].iloc[0]
                cells.append(fmt(r.mean_recall))
            L.append(f"| {bk} | {int(bud*100)}% | " + " | ".join(cells) + " |")
    L.append("")

    # behavior summary
    L.append("## 7. Behavior Summary")
    L.append("")
    L.append("单元均值（ReproAlloc vs OptimisticScoreCost）：")
    L.append("")
    L.append("| benchmark | budget | method | unique visited | exec/visited | confirmed | exec→confirmed | exec→unconfirmed |")
    L.append("|---|---:|---|---:|---:|---:|---:|---:|")
    for bk in ["sensodat", "amini"]:
        for bud in BUDGETS:
            for meth in ["OptimisticScoreCost", "ReproAlloc"]:
                g = raw[(raw.benchmark == bk) & (raw.budget == bud) &
                        (raw.method == meth)]
                u = g.groupby("seed_or_unit").agg(
                    vis=("unique_tests_visited", "mean"),
                    mepv=("num_executions", "mean"),
                    conf=("confirmed_count", "mean"),
                    econf=("exec_finally_confirmed", "mean"),
                    eunc=("exec_finally_unconfirmed", "mean"))
                vis = float(u.vis.mean())
                exc = float(u.mepv.mean())
                L.append(f"| {bk} | {int(bud*100)}% | {meth} | {vis:.1f} | "
                         f"{(exc/vis if vis else 0):.2f} | {u.conf.mean():.1f} | "
                         f"{u.econf.mean():.1f} | {u.eunc.mean():.1f} |")
    L.append("")

    # factual conclusion
    L.append("## 8. Factual Conclusion")
    L.append("")
    sig = paired_df[(paired_df.paired_ci_low > 0) & (paired_df.wilcoxon_p < 0.05)]
    n_cells = len(paired_df)
    n_ra_higher = int((paired_df.paired_delta > 0).sum())
    n_ra_sig = len(sig)
    n_osc_higher = int((paired_df.paired_delta < 0).sum())
    w_tot = int(paired_df.wins.sum())
    l_tot = int(paired_df.losses.sum())
    t_tot = int(paired_df.ties.sum())
    if n_ra_sig >= 4 and n_osc_higher == 0:
        case = "A"
        case_txt = ("Case A：ReproAlloc 在两个 benchmark 的大多数主预算上明显高于 "
                    "OptimisticScoreCost——remaining confirmation modeling adds clear "
                    "empirical value beyond optimism alone。")
    elif n_ra_higher > 0 and n_osc_higher > 0:
        case = "C"
        case_txt = ("Case C：不同 benchmark/budget 各有优势，不强行总结单一 winner。")
    elif n_ra_sig == 0 and n_osc_higher == 0:
        case = "B"
        case_txt = ("Case B：两者基本接近——optimism explains a substantial share of "
                    "the gain, while ORCC does not show a large additional empirical "
                    "advantage in these settings。")
    else:
        case = "D"
        case_txt = "Case D：OptimisticScoreCost 明显优于 ReproAlloc。"
    L.append(f"- 6 个 benchmark-budget 单元格中：ReproAlloc 均值更高 {n_ra_higher} 格，"
             f"OptimisticScoreCost 均值更高 {n_osc_higher} 格，"
             f"均值持平 {n_cells - n_ra_higher - n_osc_higher} 格。")
    L.append(f"- paired CI 排除 0 的格数：{int(((paired_df.paired_ci_low > 0) | (paired_df.paired_ci_high < 0)).sum())}"
             f"（其中 ReproAlloc 方向 {n_ra_sig} 格且 Wilcoxon p<0.05）。")
    L.append(f"- 单元级 wins/losses/ties 合计：{w_tot}/{l_tot}/{t_tot}。")
    L.append(f"- 判定：**{case_txt}**")
    L.append("")
    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    return case


def main():
    reg_ok, reg_msgs = regression_check()
    for m in reg_msgs:
        print(m, flush=True)
    raw = build_raw()
    print(f"raw rows: {len(raw)}", flush=True)
    summary, paired_df, units = build_summary_paired(raw)
    print(paired_df.to_string(index=False), flush=True)
    case = write_report(reg_ok, reg_msgs, summary, paired_df, raw)
    print(f"case: {case}", flush=True)
    print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
