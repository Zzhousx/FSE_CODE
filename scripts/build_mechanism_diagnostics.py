# -*- coding: utf-8 -*-
"""Mechanism diagnostics + fallback re-verification (spec sections 5.4, 9).

Everything is computed FROM THE RAW per-execution decision logs
(logs/optimistic_ercc/{amini,sensodat}/main/*.csv), not from the old
markdown summaries. No experiment is re-run.

Log encoding (both harnesses, see optimistic_ercc_{amini,sensodat}.py):
  one row per execution of the optimistic selector, logged in update()
  AFTER the outcome is incorporated:
    arm, s, n, post_a, post_b, q_mean, q_opt, m_mean, m_opt,
    confirmed, sel_kind, would_differ
  m_mean / m_opt == -1.0  encodes  m* = infinity.

Outputs (results/mechanism_diagnostics/):
  amini_route_mechanism_table.csv    full per-(budget, route) mechanism table
  amini_route_mechanism_compact.csv  auto-selected representative routes
  mechanism_aggregate.csv            per-(dataset, budget) aggregate stats
  fallback_reverification.txt        spec-5.4 re-verification vs old values
"""
import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "logs", "optimistic_ercc")
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "mechanism_diagnostics")
os.makedirs(OUT, exist_ok=True)

ESC = lambda d: (d.m_mean < 0) & (d.m_opt > 0)  # noqa: E731  escape state


def load_logs(pattern, id_cols):
    frames = []
    for p in sorted(glob.glob(pattern)):
        d = pd.read_csv(p)
        base = os.path.basename(p)
        for col in id_cols:
            pass
        frames.append(d.assign(_file=base))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --------------------------------------------------------------------- AMINI
def amini():
    frames = []
    for p in sorted(glob.glob(os.path.join(LOGS, "amini", "main", "*.csv"))):
        d = pd.read_csv(p)
        base = os.path.basename(p)
        # decisions_seed2007_b0.10.csv
        seed = int(base.split("_")[1].replace("seed", ""))
        tf = float(base.split("_")[2].replace("b", "").replace(".csv", ""))
        d["seed"] = seed
        d["budget_frac"] = tf
        frames.append(d)
    diag = pd.concat(frames, ignore_index=True)
    diag["exec_order"] = diag.groupby(["seed", "budget_frac"]).cumcount()

    # arm index -> route_id: routes are loaded sorted by the numeric part of
    # scenario_id (see experiment_amini_finite_trace.load_routes)
    import csv as _csv
    canon = os.path.join(ROOT, "data", "amini_transfuser_canonical.csv")
    rids = set()
    with open(canon, newline="", encoding="utf-8") as fh:
        for row in _csv.DictReader(fh):
            rids.add(row["scenario_id"].strip())
    arm2rid = sorted(rids, key=lambda r: int(r.split("_")[1]))
    diag["route_id"] = diag.arm.map(lambda i: arm2rid[int(i)])

    alloc = pd.read_csv(os.path.join(
        RES, "01_optimistic_core", "amini", "amini_optimistic_alloc_main.csv"))
    route_static = pd.read_csv(os.path.join(
        RES, "diagnostics", "amini_route_level_optimistic.csv"))
    static_cols = route_static[["budget_frac", "route_id", "n_pool", "s_total",
                                "p_hat", "ref_pos", "k_best", "k_worst",
                                "mean_dur"]].drop_duplicates()

    # --- aggregate stats (optimistic selector, from raw logs) --------------
    agg_rows = []
    for tf, g in diag.groupby("budget_frac"):
        per_run = g.groupby("seed")
        last = per_run.tail(1)[["seed", "arm", "confirmed"]]
        final_conf = dict(zip(last.arm, last.confirmed))
        g2 = g.copy()
        g2["finally_confirmed"] = [final_conf[(s_, a_)] if False else 0
                                   for s_, a_ in zip(g.seed, g.arm)]
        # final confirmed status per (seed, arm) from each arm's LAST row
        last_arm = (g.sort_values("exec_order")
                      .groupby(["seed", "arm"]).tail(1))
        fc_map = {(r.seed, r.arm): bool(r.confirmed)
                  for r in last_arm.itertuples()}
        g2["finally_confirmed"] = [fc_map[(s_, a_)]
                                   for s_, a_ in zip(g.seed, g.arm)]
        uniq = per_run["arm"].nunique()
        esc = ESC(g)
        agg_rows.append({
            "dataset": "amini", "budget": tf,
            "executions": len(g),
            "escape_states(m_mean=inf,m_opt=finite)": int(esc.sum()),
            "escape_frac": float(esc.mean()),
            "median_qO_minus_qmean": float((g.q_opt - g.q_mean).median()),
            "would_differ_frac": float(g.would_differ.mean()),
            "fallback_steps": int((g.sel_kind == "fallback").sum()),
            "fallback_frac": float((g.sel_kind == "fallback").mean()),
            "unique_arms_visited_mean": float(uniq.mean()),
            "execs_per_visited_arm_mean": float(len(g) / uniq.sum()),
            "execs_on_finally_confirmed_frac":
                float(g2.finally_confirmed.mean()),
            "execs_on_finally_unconfirmed_frac":
                float(1.0 - g2.finally_confirmed.mean()),
        })

    # --- per-route mechanism table ------------------------------------------
    # first-visit state of each route within each run
    first = (diag.sort_values("exec_order")
                 .groupby(["budget_frac", "seed", "route_id"]).head(1))
    fv = (first.groupby(["budget_frac", "route_id"])
               .agg(s_first=("s", "median"), n_first=("n", "median"),
                    q_mean_first=("q_mean", "median"),
                    q_opt_first=("q_opt", "median"),
                    m_mean_first=("m_mean", "median"),
                    m_opt_first=("m_opt", "median"))
               .reset_index())
    esc_route = (diag.assign(esc=ESC(diag))
                     .groupby(["budget_frac", "route_id"])["esc"].sum()
                     .reset_index(name="escape_states"))
    visits_diag = (diag.groupby(["budget_frac", "route_id"]).size()
                       .reset_index(name="visits_orcc_total"))

    def alloc_stats(method):
        a = alloc[alloc.method == method]
        return (a.groupby(["budget_frac", "route_id"])
                 .agg(visits=("runs_used", "mean"),
                      confirmed_rate=("confirmed_sticky", "mean"))
                 .reset_index())

    am_mean = alloc_stats("ReproAlloc").rename(
        columns={"visits": "visits_mean", "confirmed_rate": "confirmed_mean"})
    am_opt = alloc_stats("ReproAlloc-Optimistic").rename(
        columns={"visits": "visits_orcc", "confirmed_rate": "confirmed_orcc"})

    tab = (static_cols
           .merge(fv, on=["budget_frac", "route_id"], how="left")
           .merge(esc_route, on=["budget_frac", "route_id"], how="left")
           .merge(visits_diag, on=["budget_frac", "route_id"], how="left")
           .merge(am_mean, on=["budget_frac", "route_id"], how="left")
           .merge(am_opt, on=["budget_frac", "route_id"], how="left"))
    tab["escape_states"] = tab.escape_states.fillna(0).astype(int)
    tab["d_conf"] = tab.confirmed_orcc - tab.confirmed_mean
    tab["m_mean_first"] = tab.m_mean_first.replace(-1.0, np.inf)
    tab["m_opt_first"] = tab.m_opt_first.replace(-1.0, np.inf)
    tab = tab.rename(columns={
        "s_first": "early_s", "n_first": "early_n",
        "q_mean_first": "q_mean", "q_opt_first": "q_O",
        "m_mean_first": "m_star_mean", "m_opt_first": "m_star_ORCC"})
    cols = ["budget_frac", "route_id", "p_hat", "n_pool", "s_total",
            "ref_pos", "k_best", "k_worst", "early_s", "early_n",
            "q_mean", "q_O", "m_star_mean", "m_star_ORCC",
            "escape_states", "visits_mean", "visits_orcc",
            "confirmed_mean", "confirmed_orcc", "d_conf"]
    tab = tab[cols].sort_values(["budget_frac", "d_conf"],
                                ascending=[True, False])
    tab.to_csv(os.path.join(OUT, "amini_route_mechanism_table.csv"),
               index=False, float_format="%.6g")

    # --- compact table: AUTOMATIC criterion (spec 9.2/9.3) -------------------
    # C1 (mechanism): at this budget the route EITHER hits at least one
    #     escape state (m_mean=inf while m_opt finite) OR its first-visit
    #     plan is at least twice as long under the mean rate as under the
    #     optimistic rate (m_star_mean / m_star_ORCC >= 2, inf included) -
    #     i.e. the mean plan makes the route unreachable or very expensive
    #     while the optimistic plan keeps it reachable.
    # C2 (outcome):   d_conf = confirmed_orcc - confirmed_mean > 0.
    # Selection: C1 AND C2, ranked by d_conf (desc), escape_states (desc),
    # top 5 per budget.
    ratio = (tab.m_star_mean.replace(np.inf, np.finfo(float).max)
             / tab.m_star_ORCC.replace(0, np.nan))
    tab_c = tab.assign(m_ratio=ratio)
    comp = tab_c[(tab_c.escape_states > 0) | (tab_c.m_ratio >= 2)]
    comp = comp[comp.d_conf > 0].copy()
    comp = (comp.sort_values(["budget_frac", "d_conf", "escape_states"],
                             ascending=[True, False, False])
                .groupby("budget_frac").head(5))
    comp = comp.drop(columns=["m_ratio"])
    comp.to_csv(os.path.join(OUT, "amini_route_mechanism_compact.csv"),
                index=False, float_format="%.6g")
    return pd.DataFrame(agg_rows), tab, comp


# ------------------------------------------------------------------ SENSODAT
def sensodat():
    frames = []
    for p in sorted(glob.glob(os.path.join(LOGS, "sensodat", "main", "*.csv"))):
        d = pd.read_csv(p)
        base = os.path.basename(p)
        # decisions_split3_b0.20_r1.csv
        tok = base.replace(".csv", "").split("_")
        d["split_id"] = int(tok[1].replace("split", ""))
        d["time_frac"] = float(tok[2].replace("b", ""))
        d["rep"] = int(tok[3].replace("r", ""))
        frames.append(d)
    diag = pd.concat(frames, ignore_index=True)
    diag["exec_order"] = diag.groupby(
        ["split_id", "rep", "time_frac"]).cumcount()

    agg_rows = []
    for tf, g in diag.groupby("time_frac"):
        per_run = g.groupby(["split_id", "rep"])
        last_arm = (g.sort_values("exec_order")
                      .groupby(["split_id", "rep", "arm"]).tail(1))
        fc_map = {(r.split_id, r.rep, r.arm): bool(r.confirmed)
                  for r in last_arm.itertuples()}
        fc = np.array([fc_map[(s_, r_, a_)] for s_, r_, a_
                       in zip(g.split_id, g.rep, g.arm)])
        uniq = per_run["arm"].nunique()
        esc = ESC(g)
        agg_rows.append({
            "dataset": "sensodat", "budget": tf,
            "executions": len(g),
            "escape_states(m_mean=inf,m_opt=finite)": int(esc.sum()),
            "escape_frac": float(esc.mean()),
            "median_qO_minus_qmean": float((g.q_opt - g.q_mean).median()),
            "would_differ_frac": float(g.would_differ.mean()),
            "fallback_steps": int((g.sel_kind == "fallback").sum()),
            "fallback_frac": float((g.sel_kind == "fallback").mean()),
            "unique_arms_visited_mean": float(uniq.mean()),
            "execs_per_visited_arm_mean": float(len(g) / uniq.sum()),
            "execs_on_finally_confirmed_frac": float(fc.mean()),
            "execs_on_finally_unconfirmed_frac": float(1.0 - fc.mean()),
        })
    return pd.DataFrame(agg_rows), diag


def main():
    agg_am, tab, comp = amini()
    agg_sd, _ = sensodat()
    agg = pd.concat([agg_am, agg_sd], ignore_index=True)
    agg.to_csv(os.path.join(OUT, "mechanism_aggregate.csv"), index=False,
               float_format="%.6g")

    # ---------------- fallback re-verification (spec 5.4) -------------------
    lines = ["FALLBACK RE-VERIFICATION FROM RAW LOGS (spec 5.4)",
             "source: logs/optimistic_ercc/{amini,sensodat}/main/*.csv",
             "escape := executed state with m_mean=inf (-1.0) and m_opt finite",
             "fallback := sel_kind == 'fallback'", ""]
    old = {("amini", 0.05): (0.404, 0.491), ("amini", 0.10): (0.207, 0.363),
           ("amini", 0.20): (0.231, 0.405), ("sensodat", 0.05): (0.004, 0.304),
           ("sensodat", 0.10): (0.004, 0.336), ("sensodat", 0.20): (0.004, 0.341)}
    ok = True
    for _, r in agg.iterrows():
        key = (r.dataset, float(r.budget))
        o_esc, o_diff = old[key]
        match_e = abs(r.escape_frac - o_esc) < 0.01
        match_d = abs(r.would_differ_frac - o_diff) < 0.01
        ok &= match_e and match_d and (r.fallback_frac == 0.0)
        lines.append(
            f"{r.dataset} @{r.budget:.2f}: executions={int(r.executions)}  "
            f"escape {int(r['escape_states(m_mean=inf,m_opt=finite)'])} "
            f"({r.escape_frac:.3f}; old {o_esc:.3f} "
            f"{'OK' if match_e else 'MISMATCH'})  "
            f"fallback {int(r.fallback_steps)} ({r.fallback_frac:.4f}; "
            f"old 0.000 {'OK' if r.fallback_frac == 0 else 'MISMATCH'})  "
            f"would_differ {r.would_differ_frac:.3f} (old {o_diff:.3f} "
            f"{'OK' if match_d else 'MISMATCH'})")
    lines.append("")
    verdict = ("PASS - raw-log values reproduce the reported diagnostics"
               if ok else "FAIL - see mismatches above")
    lines.append(f"RESULT: {verdict}")
    with open(os.path.join(OUT, "fallback_reverification.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines))

    pd.set_option("display.width", 240)
    print("\n=== mechanism_aggregate ===")
    print(agg.to_string(index=False))
    print("\n=== compact mechanism table (auto-selected) ===")
    print(comp.to_string(index=False))
    print("\n=== fallback reverification ===")
    print("\n".join(lines))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
