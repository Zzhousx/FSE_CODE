"""Engineering value offline check (Amini TransFuser).

Two offline re-analyses on EXISTING frozen traces only:
  (1) time-to-k confirmations (k=1,2,3) in actual cumulative wall-clock hours
  (2) two automatically selected route timeline cases (ReproAlloc vs
      OptimisticScoreCost) at the 20% budget

NO new CARLA execution, NO new traces, NO method/seed/tau/delta/budget changes.
The replay below consumes the SAME recorded pools in the SAME per-seed orders
through the SAME frozen selectors with the SAME method_rng streams, and is
anchored against stored artifacts by four exact regression checks:

  R1 replayed anytime traces == results/diagnostics/amini_anytime_traces.csv
     (ScoreCost / MeanPlanning / ReproAlloc / APGAI, all 12 seeds, every step)
  R2 replay final per-seed values @20% == 01_optimistic_core main CSV (7 methods)
  R3 replay final per-seed values @20% == Batch-4 osc_check CSV (8 methods)
  R4 ReproAlloc diagnostic stream == logs/.../main/decisions_seedXXXX_b0.20.csv

Paper-name mapping (harness -> paper):
  ReproAlloc-Optimistic -> ReproAlloc        (optimistic planning rate q_O)
  ReproAlloc            -> Mean Planning     (posterior-mean planning rate)
  ReproAlloc-OA         -> ReproAlloc-OA     (raw CSV only)

Outputs (results/engineering_value_check/):
  time_to_k_raw.csv  time_to_k_attainment.csv  time_to_k_summary.csv
  time_to_k_paired.csv  budget_hours_confirmed_routes.csv
  route_case_selection.csv  route_case_A_timeline.csv  route_case_B_timeline.csv
  regression_report.json

The merged paper figure (Figure 4, `fig:cases`) is rendered separately by
scripts/build_fig4_allocation_traces.py from the two timeline CSVs above.
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

import experiment_amini_finite_trace as A  # noqa: E402
import optimistic_ercc_amini as OA  # noqa: E402
from optimistic_ercc import planning_rate, MODE_OPTIMISTIC  # noqa: E402

OUT = os.path.join(ROOT, "results", "engineering_value_check")
os.makedirs(OUT, exist_ok=True)

SEEDS = list(range(2000, 2012))
BUDGET_FRAC = 0.20
BUDGET_IDX = 2          # canonical index of 0.20 (frozen rng streams)
DELTA = 0.05
TAU = A.TAU

PAPER_NAME = {
    "ReproAlloc-Optimistic": "ReproAlloc",
    "ReproAlloc": "Mean Planning",
}
MAIN5 = ["ReproAlloc", "OptimisticScoreCost", "ScoreCost", "APGAI",
         "Mean Planning"]
COMPARATORS = ["OptimisticScoreCost", "ScoreCost", "APGAI", "Mean Planning"]

TRIGGERS = ["collisions_layout", "collisions_pedestrian",
            "collisions_vehicle", "red_light"]


def paper(name):
    return PAPER_NAME.get(name, name)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
routes = A.load_routes()
gs = A.build_gs(routes)
K = len(routes)
total_dur = sum(sum(r.durations) for r in routes)
p_hat = np.array([r.p_hat for r in routes])
ref_pos_mask = p_hat > TAU
budget = total_dur * BUDGET_FRAC
m_cap = A.m_cap_from_budget(budget, gs)

canon = pd.read_csv(A.CANON, float_precision="round_trip")
LABELS = {}
for r in canon.itertuples(index=False):
    LABELS[(r.scenario_id, int(r.repetition_id))] = {
        t: int(getattr(r, t)) for t in TRIGGERS}


def trigger_tags(route_id, rep_id):
    lab = LABELS[(route_id, rep_id)]
    return ";".join(f"{t}={lab[t]}" for t in TRIGGERS if lab[t] > 0)


# --------------------------------------------------------------------------
# full traced replay at 20% (all 8 methods x 12 seeds)
# --------------------------------------------------------------------------
visit_rows = []      # per-execution rows (route-level)
final_rows = []      # per-run finals
diag_rows = {}       # seed -> list of ReproAlloc-Optimistic _diag rows

for sd in SEEDS:
    orders = A.make_orders(routes, sd)
    sels = OA.build_selectors_8(m_cap, diagnostics=True)
    for mi, sel in enumerate(sels):
        rng = A.method_rng(sd, mi, BUDGET_IDX)
        pools = [A.Pool(r.outcomes, r.durations, o)
                 for r, o in zip(routes, orders)]
        arms = A.Arms(K, gs)
        st = sel.init(arms, pools, rng)
        time_used = 0.0
        execs = 0
        n_ref = int(ref_pos_mask.sum())
        visit_ctr = [0] * K
        is_ercc = isinstance(st, dict) and "cache" in st
        has_diag = isinstance(st, dict) and "_diag" in st
        if has_diag:
            diag_rows[sd] = st["_diag"]
        while time_used < budget:
            i = sel.select(arms, pools, rng, st)
            if i < 0:
                break
            # ---- pre-execution (selection-time) state --------------------
            s_pre, n_pre = int(arms.s[i]), int(arms.n[i])
            pa_pre, pb_pre = float(arms.post_a[i]), float(arms.post_b[i])
            q_mean_pre = pa_pre / (pa_pre + pb_pre)
            qO_pre = planning_rate(pa_pre, pb_pre, MODE_OPTIMISTIC, DELTA)
            c_mean_pre = arms.mean_cost(i)
            mstar_pre, orcc_pre, util_pre = None, None, None
            if is_ercc:
                qi_plan = (qO_pre if sel.name == "ReproAlloc-Optimistic"
                           else q_mean_pre)
                m = st["cache"][i].get(s_pre, n_pre, qi_plan, TAU, m_cap,
                                       A.DELTA)
                mstar_pre = m
                c_plan = (arms.oa_cost(i) if sel.name == "ReproAlloc-OA"
                          else c_mean_pre)
                orcc_pre = (m * c_plan) if np.isfinite(m) else np.inf
                util_pre = float(st["util"][i])
            rep_idx = orders[i][pools[i].pos]     # pool.pos pre-draw
            fail, cost = pools[i].draw()
            arms.update(i, fail, cost)
            sel.update(arms, i, st)
            time_used += cost
            execs += 1
            visit_ctr[i] += 1
            rid = routes[i].route_id
            sel_kind = ""
            if is_ercc:
                sel_kind = ("fallback" if not np.isfinite(mstar_pre)
                            else "main")
            if has_diag and st["_diag"]:
                d = st["_diag"][-1]
                if d["sel_kind"]:
                    assert d["sel_kind"] == sel_kind, (sd, mi, execs)
            visit_rows.append({
                "seed": sd, "method": paper(sel.name), "harness_method": sel.name,
                "exec_idx": execs, "route_idx": i, "route_id": rid,
                "repetition_id": rep_idx + 1,
                "route_visit_number": visit_ctr[i],
                "outcome": "fail" if fail else "pass",
                "failure_triggers": (trigger_tags(rid, rep_idx + 1)
                                     if fail else ""),
                "duration_s": cost,
                "cum_time_s": time_used, "cum_time_h": time_used / 3600.0,
                "s_i": int(arms.s[i]), "n_i": int(arms.n[i]),
                "post_a": float(arms.post_a[i]), "post_b": float(arms.post_b[i]),
                "q_mean": float(arms.q(i)),
                "q_O": planning_rate(float(arms.post_a[i]),
                                     float(arms.post_b[i]),
                                     MODE_OPTIMISTIC, DELTA),
                "m_star_at_select": ("" if mstar_pre is None else
                                     ("inf" if not np.isfinite(mstar_pre)
                                      else int(mstar_pre))),
                "orcc_at_select_s": ("" if orcc_pre is None else
                                     ("inf" if not np.isfinite(orcc_pre)
                                      else orcc_pre)),
                "utility_at_select": ("" if util_pre is None else util_pre),
                "mean_cost_at_select": c_mean_pre,
                "sel_kind": sel_kind,
                "confirmed_after": int(bool(arms.confirmed[i])),
                "confirmed_total": int(arms.confirmed.sum()),
                "tp_refpos": int((arms.confirmed & ref_pos_mask).sum()),
                "reference_positive": int(bool(ref_pos_mask[i])),
            })
        conf = arms.confirmed
        final_rows.append({
            "seed": sd, "method": paper(sel.name),
            "harness_method": sel.name,
            "executions": execs, "time_used_s": time_used,
            "confirmed": int(conf.sum()),
            "tp": int((conf & ref_pos_mask).sum()),
            "fp": int((conf & ~ref_pos_mask).sum()),
            "recall_refpos": (int((conf & ref_pos_mask).sum()) / n_ref
                              if n_ref else 0.0),
        })
    print(f"seed {sd} replayed", flush=True)

visits = pd.DataFrame(visit_rows)
finals = pd.DataFrame(final_rows)
visits.to_csv(os.path.join(OUT, "_replay_visits_20pct.csv"), index=False)

# --------------------------------------------------------------------------
# regression anchors
# --------------------------------------------------------------------------
reg = {}

# float columns are compared with an absolute tolerance (stored CSVs were
# written with 16-significant-digit pandas formatting -> 1-ULP round-trip
# artifacts of at most ~2e-12 on ~1e5 magnitudes); integer/event columns are
# compared EXACTLY.  max|diff| is always reported.
FTOL = 1e-9

def cmp_cols(a, b, exact):
    """return (ok, max_abs_diff, n_bad). exact=True -> integer compare."""
    if exact:
        neq = np.nonzero(a != b)[0]
        return len(neq) == 0, float(np.max(np.abs(a - b))) if len(a) else 0.0, \
            int(len(neq))
    d = np.abs(a - b)
    mx = float(d.max()) if len(d) else 0.0
    return mx <= FTOL, mx, int((d > FTOL).sum())

# R1: anytime traces (4 methods, every step)
tr_old = pd.read_csv(os.path.join(ROOT, "results", "diagnostics",
                                  "amini_anytime_traces.csv"))
r1_ok, r1_bad, r1_max = True, [], 0.0
for hm in ["ScoreCost", "ReproAlloc", "ReproAlloc-Optimistic", "APGAI"]:
    old = tr_old[tr_old.method == hm].sort_values(
        ["seed", "exec_idx"]).reset_index(drop=True)
    new = visits[visits.harness_method == hm].sort_values(
        ["seed", "exec_idx"]).reset_index(drop=True)
    if len(old) != len(new):
        r1_ok, r1_bad = False, r1_bad + [(hm, "len", len(old), len(new))]
        continue
    for col_old, col_new, exact in [("time_used_s", "cum_time_s", False),
                                    ("confirmed", "confirmed_total", True),
                                    ("tp", "tp_refpos", True)]:
        ok, mx, nb = cmp_cols(old[col_old].to_numpy(dtype=float),
                              new[col_new].to_numpy(dtype=float), exact)
        r1_max = max(r1_max, mx)
        if not ok:
            r1_ok = False
            r1_bad.append((hm, col_old, nb, mx))
reg["R1_anytime_traces"] = {"ok": r1_ok, "max_abs_diff": r1_max,
                            "mismatches": r1_bad}

# R2/R3: final per-seed values at 20%
def final_check(path, tag_methods, key):
    df = pd.read_csv(path)
    df = df[np.isclose(df.budget_frac, BUDGET_FRAC)]
    ok, bad, mx_all = True, [], 0.0
    for hm in tag_methods:
        old = df[df.method == hm].sort_values("seed").reset_index(drop=True)
        new = finals[finals.harness_method == hm].sort_values(
            "seed").reset_index(drop=True)
        if len(old) != len(new):
            ok = False
            bad.append((hm, "len", len(old), len(new)))
            continue
        for col, exact in [("confirmed", True), ("tp", True),
                           ("executions", True), ("recall_refpos", False),
                           ("time_used_s", False)]:
            c_ok, mx, nb = cmp_cols(old[col].to_numpy(dtype=float),
                                    new[col].to_numpy(dtype=float), exact)
            mx_all = max(mx_all, mx)
            if not c_ok:
                ok = False
                bad.append((hm, col, nb, mx))
    reg[key] = {"ok": ok, "max_abs_diff": mx_all, "mismatches": bad,
                "rows": int(len(df))}
    return df

main7 = ["Uniform", "FailRerun", "APGAI", "ScoreCost", "ReproAlloc",
         "ReproAlloc-OA", "ReproAlloc-Optimistic"]
df_main = final_check(os.path.join(
    ROOT, "results", "01_optimistic_core", "amini",
    "amini_optimistic_main.csv"), main7, "R2_official_main_7methods")

df_osc = final_check(os.path.join(
    ROOT, "results", "optimistic_scorecost_check",
    "amini_osc_check.csv"), main7 + ["OptimisticScoreCost"],
    "R3_batch4_osc_8methods")

# full (all-budget) official frames for the budget-hours table below
df_main_full = pd.read_csv(os.path.join(
    ROOT, "results", "01_optimistic_core", "amini",
    "amini_optimistic_main.csv"))
df_osc_full = pd.read_csv(os.path.join(
    ROOT, "results", "optimistic_scorecost_check", "amini_osc_check.csv"))

# R4: ReproAlloc-Optimistic diagnostic stream == stored decision logs
r4_ok, r4_bad, r4_max = True, [], 0.0
for sd in SEEDS:
    p = os.path.join(ROOT, "logs", "optimistic_ercc", "amini", "main",
                     f"decisions_seed{sd}_b0.20.csv")
    old = pd.read_csv(p)
    new = pd.DataFrame(diag_rows[sd])
    if len(old) != len(new):
        r4_ok = False
        r4_bad.append((sd, "len", len(old), len(new)))
        continue
    for col, exact in [("arm", True), ("s", True), ("n", True),
                       ("post_a", False), ("post_b", False),
                       ("q_mean", False), ("q_opt", False),
                       ("m_mean", False), ("m_opt", False),
                       ("confirmed", None), ("sel_kind", None),
                       ("would_differ", None)]:
        if exact is None:
            a = old[col].astype(str).to_numpy()
            b = new[col].astype(str).to_numpy()
            neq = int((a != b).sum())
            if neq:
                r4_ok = False
                r4_bad.append((sd, col, neq))
        else:
            c_ok, mx, nb = cmp_cols(old[col].to_numpy(dtype=float),
                                    new[col].to_numpy(dtype=float), exact)
            r4_max = max(r4_max, mx)
            if not c_ok:
                r4_ok = False
                r4_bad.append((sd, col, nb, mx))
reg["R4_decision_logs"] = {"ok": r4_ok, "max_abs_diff": r4_max,
                           "mismatches": r4_bad}

reg["all_ok"] = all(v.get("ok", True) for k, v in reg.items()
                    if isinstance(v, dict) and "ok" in v)
print("REGRESSION:", json.dumps(reg["all_ok"]), flush=True)
for k, v in reg.items():
    if isinstance(v, dict) and "ok" in v:
        print(f"  {k}: {'PASS' if v['ok'] else 'FAIL ' + str(v['mismatches'][:3])}",
              flush=True)

# --------------------------------------------------------------------------
# time to k confirmations
# --------------------------------------------------------------------------
def time_to_k(frame, count_col):
    """per (seed, method): hours at which count_col first reaches k (1..3)."""
    rows = []
    for (sd, meth), g in frame.groupby(["seed", "method"]):
        g = g.sort_values("exec_idx")
        c = g[count_col].to_numpy()
        t = g["cum_time_h"].to_numpy()
        rec = {"seed": sd, "method": meth}
        for k in (1, 2, 3):
            hit = np.nonzero(c >= k)[0]
            rec[f"reached{k}"] = int(len(hit) > 0)
            rec[f"T{k}_h"] = (float(t[hit[0]]) if len(hit) else np.nan)
        rows.append(rec)
    return pd.DataFrame(rows)


tk_ref = time_to_k(visits, "tp_refpos").assign(basis="reference_positive")
tk_tot = time_to_k(visits, "confirmed_total").assign(basis="all_confirmed")
tk = pd.concat([tk_ref, tk_tot], ignore_index=True)
tk.to_csv(os.path.join(OUT, "time_to_k_raw.csv"), index=False)

# attainment (feasibility audit first)
att = (tk.groupby(["basis", "method"])
       .agg(total_runs=("seed", "count"),
            reached1=("reached1", "sum"), reached2=("reached2", "sum"),
            reached3=("reached3", "sum")).reset_index())
att_rows = []
for r in att.itertuples(index=False):
    for k in (1, 2, 3):
        att_rows.append({"basis": r.basis, "method": r.method, "k": k,
                         "reached_count": int(getattr(r, f"reached{k}")),
                         "total_runs": int(r.total_runs),
                         "reached_fraction":
                         float(getattr(r, f"reached{k}")) / r.total_runs})
pd.DataFrame(att_rows).to_csv(
    os.path.join(OUT, "time_to_k_attainment.csv"), index=False)

# summary among reached runs (conditional)
sum_rows = []
for basis in tk.basis.unique():
    for meth in sorted(tk.method.unique()):
        g = tk[(tk.basis == basis) & (tk.method == meth)]
        for k in (1, 2, 3):
            v = g.loc[g[f"reached{k}"] == 1, f"T{k}_h"].to_numpy()
            row = {"basis": basis, "method": meth, "k": k,
                   "reached_n": int(len(v)), "total_runs": int(len(g))}
            if len(v):
                row.update({"mean_h": float(np.mean(v)),
                            "median_h": float(np.median(v)),
                            "q25_h": float(np.percentile(v, 25)),
                            "q75_h": float(np.percentile(v, 75)),
                            "min_h": float(np.min(v)),
                            "max_h": float(np.max(v))})
            sum_rows.append(row)
pd.DataFrame(sum_rows).to_csv(
    os.path.join(OUT, "time_to_k_summary.csv"), index=False)

# paired: comparator - ReproAlloc on jointly reached seeds
EPS = 1e-9
paired_rows = []
for basis in tk.basis.unique():
    base = tk[(tk.basis == basis) & (tk.method == "ReproAlloc")].set_index(
        "seed")
    for comp in COMPARATORS:
        oth = tk[(tk.basis == basis) & (tk.method == comp)].set_index(
            "seed")
        for k in (1, 2, 3):
            common = base.index.intersection(oth.index)
            both = [sd for sd in common
                    if base.loc[sd, f"reached{k}"] == 1
                    and oth.loc[sd, f"reached{k}"] == 1]
            d = np.array([oth.loc[sd, f"T{k}_h"] - base.loc[sd, f"T{k}_h"]
                          for sd in both])
            row = {"basis": basis, "comparator": comp, "k": k,
                   "jointly_reached_n": int(len(d))}
            if len(d):
                row.update({
                    "mean_paired_hours_saved": float(np.mean(d)),
                    "median_paired_hours_saved": float(np.median(d)),
                    "wins": int((d > EPS).sum()),
                    "losses": int((d < -EPS).sum()),
                    "ties": int((np.abs(d) <= EPS).sum())})
            if len(d) < 6:
                row["note"] = ("insufficient matched attainment for a "
                               "stable time comparison")
            paired_rows.append(row)
pd.DataFrame(paired_rows).to_csv(
    os.path.join(OUT, "time_to_k_paired.csv"), index=False)

# --------------------------------------------------------------------------
# budget in actual hours + confirmed routes (from OFFICIAL main results)
# --------------------------------------------------------------------------
bh_rows = []
for frac in (0.05, 0.10, 0.20):
    hours = total_dur * frac / 3600.0
    sub_m = df_main_full[np.isclose(df_main_full.budget_frac, frac)]
    sub_o = df_osc_full[np.isclose(df_osc_full.budget_frac, frac)]
    for hm in main7 + ["OptimisticScoreCost"]:
        src = sub_o[sub_o.method == hm] if hm == "OptimisticScoreCost" \
            else sub_m[sub_m.method == hm]
        tp = src.sort_values("seed")["tp"].to_numpy(dtype=float)
        bh_rows.append({"budget_frac": frac, "budget_hours": hours,
                        "method": paper(hm),
                        "mean_confirmed_routes": float(tp.mean()),
                        "sd_confirmed_routes": float(tp.std(ddof=1)),
                        "n_seeds": int(len(tp))})
pd.DataFrame(bh_rows).to_csv(
    os.path.join(OUT, "budget_hours_confirmed_routes.csv"), index=False)

# --------------------------------------------------------------------------
# route case selection (20%, paper ReproAlloc vs OptimisticScoreCost)
# --------------------------------------------------------------------------
# confirmation time per (method, seed, route) from replay
conf_time = {}
for meth in ["ReproAlloc", "OptimisticScoreCost"]:
    gv = visits[visits.method == meth]
    for (sd, rid), g in gv.groupby(["seed", "route_id"]):
        g = g.sort_values("exec_idx")
        hit = g[g.confirmed_after == 1]
        if len(hit):
            conf_time[(meth, sd, rid)] = float(hit.iloc[0]["cum_time_h"])
        else:
            conf_time[(meth, sd, rid)] = None

# cross-check confirmed status vs official alloc (confirmed_sticky)
alloc_off = pd.concat([
    pd.read_csv(os.path.join(ROOT, "results", "01_optimistic_core", "amini",
                             "amini_optimistic_alloc_main.csv")),
    pd.read_csv(os.path.join(ROOT, "results", "optimistic_scorecost_check",
                             "amini_osc_check_alloc.csv"))])
alloc_off = alloc_off[np.isclose(alloc_off.budget_frac, BUDGET_FRAC)]
alloc_check_bad = []
for hm, pm in [("ReproAlloc-Optimistic", "ReproAlloc"),
               ("OptimisticScoreCost", "OptimisticScoreCost")]:
    sub = alloc_off[alloc_off.method == hm]
    for r in sub.itertuples(index=False):
        replay_conf = conf_time.get((pm, int(r.seed), r.route_id)) is not None
        if bool(r.confirmed_sticky) != replay_conf:
            alloc_check_bad.append((hm, int(r.seed), r.route_id))
reg["R5_alloc_confirmed_status_match"] = {
    "ok": len(alloc_check_bad) == 0, "mismatches": alloc_check_bad[:10]}
reg["all_ok"] = all(v.get("ok", True) for k, v in reg.items()
                    if isinstance(v, dict) and "ok" in v)
print("R5 alloc confirmed-status match:",
      "PASS" if not alloc_check_bad else f"FAIL {alloc_check_bad[:5]}",
      flush=True)

ref_routes = [routes[i].route_id for i in range(K) if ref_pos_mask[i]]

caseA_stats = []
for rid in ref_routes:
    rescued, both_d, repro_t = [], [], []
    for sd in SEEDS:
        tr = conf_time.get(("ReproAlloc", sd, rid))
        to = conf_time.get(("OptimisticScoreCost", sd, rid))
        if tr is not None and to is None:
            rescued.append(sd)
            repro_t.append(tr)
        if tr is not None and to is not None:
            both_d.append(to - tr)
    lead = (float(np.median(both_d)) if both_d else float("inf"))
    caseA_stats.append({"route_id": rid, "rescued_n": len(rescued),
                        "rescued_seeds": rescued, "median_lead_h": lead,
                        "repro_times_rescued": repro_t})
# sort: rescued_n desc; +inf lead first; then finite lead desc; route_id asc
caseA_stats.sort(key=lambda r: (
    -r["rescued_n"],
    0 if not math.isfinite(r["median_lead_h"]) else 1,
    -r["median_lead_h"] if math.isfinite(r["median_lead_h"]) else 0.0,
    r["route_id"]))
caseA = caseA_stats[0]
# representative seed: ReproAlloc time closest to median over rescued seeds
med_t = float(np.median(caseA["repro_times_rescued"]))
caseA_seed = min(caseA["rescued_seeds"],
                 key=lambda sd: (abs(conf_time[("ReproAlloc", sd,
                                               caseA["route_id"])] - med_t),
                                 sd))

caseB_stats = []
for rid in ref_routes:
    diffs = []
    for sd in SEEDS:
        tr = conf_time.get(("ReproAlloc", sd, rid))
        to = conf_time.get(("OptimisticScoreCost", sd, rid))
        if tr is not None and to is not None:
            diffs.append((sd, to - tr))
    if len(diffs) >= 6:
        med = float(np.median([d for _, d in diffs]))
        if med > 0:
            caseB_stats.append({"route_id": rid, "n_both": len(diffs),
                                "median_diff_h": med, "diffs": diffs})
overall_med = float(np.median([r["median_diff_h"] for r in caseB_stats]))
caseB_stats.sort(key=lambda r: (abs(r["median_diff_h"] - overall_med),
                                r["route_id"]))
caseB = caseB_stats[0]
caseB_seed = min((sd for sd, _ in caseB["diffs"]),
                 key=lambda sd: (abs(dict(caseB["diffs"])[sd]
                                     - caseB["median_diff_h"]), sd))

sel_rows = [
    {"case": "A", "rule": "rescued confirmation",
     "selected_route": caseA["route_id"], "representative_seed": caseA_seed,
     "rescued_n": caseA["rescued_n"],
     "median_lead_h": caseA["median_lead_h"],
     "repro_median_time_h": med_t,
     "runner_ups": json.dumps([
         {"route_id": r["route_id"], "rescued_n": r["rescued_n"],
          "median_lead_h": (r["median_lead_h"]
                            if math.isfinite(r["median_lead_h"]) else "inf")}
         for r in caseA_stats[1:6]])},
    {"case": "B", "rule": "typical shared confirmation",
     "selected_route": caseB["route_id"], "representative_seed": caseB_seed,
     "n_both_confirmed": caseB["n_both"],
     "route_median_diff_h": caseB["median_diff_h"],
     "overall_median_diff_h": overall_med,
     "n_eligible_routes": len(caseB_stats),
     "runner_ups": json.dumps([
         {"route_id": r["route_id"], "median_diff_h": r["median_diff_h"],
          "n_both": r["n_both"]} for r in caseB_stats[1:6]])},
]
pd.DataFrame(sel_rows).to_csv(
    os.path.join(OUT, "route_case_selection.csv"), index=False)

# --------------------------------------------------------------------------
# timeline CSVs (the merged paper Figure 4 is rendered separately by
# scripts/build_fig4_allocation_traces.py from these two CSVs)
# --------------------------------------------------------------------------
TL_COLS = ["method", "exec_idx", "cum_time_h", "route_visit_number",
           "outcome", "failure_triggers", "duration_s", "s_i", "n_i",
           "post_a", "post_b", "q_mean", "q_O", "m_star_at_select",
           "orcc_at_select_s", "utility_at_select", "sel_kind",
           "confirmed_after", "repetition_id"]

fig_info = {}
for tag, rid, sdx in [("A", caseA["route_id"], caseA_seed),
                      ("B", caseB["route_id"], caseB_seed)]:
    sub = visits[(visits.route_id == rid) & (visits.seed == sdx)
                 & (visits.method.isin(["ReproAlloc",
                                        "OptimisticScoreCost"]))]
    sub = sub.sort_values(["method", "exec_idx"])
    out_csv = os.path.join(OUT, f"route_case_{tag}_timeline.csv")
    sub[["seed", "route_id"] + TL_COLS].to_csv(out_csv, index=False)
    fig_info[tag] = {"route": rid, "seed": sdx,
                     "repro_visits": int((sub.method == "ReproAlloc").sum()),
                     "osc_visits":
                     int((sub.method == "OptimisticScoreCost").sum())}

with open(os.path.join(OUT, "regression_report.json"), "w",
          encoding="utf-8") as fh:
    json.dump(reg, fh, indent=2, default=str)

# --------------------------------------------------------------------------
# compact stdout summary for the report
# --------------------------------------------------------------------------
summary = {
    "total_duration_s": total_dur,
    "total_duration_h": total_dur / 3600.0,
    "budget_hours": {str(f): total_dur * f / 3600.0 for f in (0.05, 0.1, 0.2)},
    "regression_all_ok": reg["all_ok"],
    "caseA": {"route": caseA["route_id"], "seed": caseA_seed,
              "rescued_n": caseA["rescued_n"],
              "median_lead_h": caseA["median_lead_h"]},
    "caseB": {"route": caseB["route_id"], "seed": caseB_seed,
              "n_both": caseB["n_both"],
              "median_diff_h": caseB["median_diff_h"],
              "overall_median_h": overall_med,
              "eligible": len(caseB_stats)},
    "fig_info": fig_info,
}
print("SUMMARY_JSON_BEGIN")
print(json.dumps(summary, indent=2, default=str))
print("SUMMARY_JSON_END")
