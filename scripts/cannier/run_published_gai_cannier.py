"""Run published GAI sampling rules on the CANNIER evaluation projects."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(Path(__file__).resolve().parent)]

import experiment_amini_finite_trace as A
from replay_env import CANNIERReplay, SequenceBank
from gai_published_baselines import PublishedGAISampling

METHODS = ("LUCB-G", "Murphy Sampling")
RATIOS = (.5, 1., 2., 4.)
FIELDNAMES = ("project", "seed", "r", "method", "n_candidates",
              "n_reference", "count", "recall", "false_conf", "B",
              "B_used", "overtime", "executions", "sched_cpu_s",
              "n_eliminated", "init_completed", "init_exec", "init_used")


def run_one(project, seed, ratio, method):
    with np.load(ROOT / "results" / "cannier" / "pools" / f"{project}.npz") as pool:
        item_id, failures, costs = (pool[key].copy() for key in
                                    ("item_id", "F", "cost"))
        budget = float(ratio * pool["B_pass"])
    k = len(item_id)
    ref = failures / 2500 > .3
    c0 = float(np.mean(costs))
    arms = A.Arms(k, A.GS(c0, c0, c0))
    bank = SequenceBank(project, seed, item_id, failures)
    env = CANNIERReplay(project, seed, item_id, failures, costs, bank=bank)
    rng = np.random.default_rng(np.random.SeedSequence(
        [seed, __import__("zlib").crc32(project.encode()),
         RATIOS.index(ratio), 947, METHODS.index(method)]))
    result = PublishedGAISampling(method).run(arms, env, budget, rng, True)
    tp = int(np.sum(arms.confirmed & ref))
    return {"project": project, "seed": seed, "r": ratio, "method": method,
            "n_candidates": k, "n_reference": int(np.sum(ref)),
            "count": tp, "recall": tp / int(np.sum(ref)),
            "false_conf": int(np.sum(arms.confirmed & ~ref)),
            "B": budget, **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projects", default="")
    parser.add_argument("--seeds", default=",".join(map(str, range(2000, 2012))))
    parser.add_argument("--ratios", default="0.5,1,2,4")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--output", default="cannier_test_runs.csv")
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/cannier_eval.json").read_text())
    projects = args.projects.split(",") if args.projects else config["projects"]
    seeds = list(map(int, args.seeds.split(",")))
    ratios = list(map(float, args.ratios.split(",")))
    methods = args.methods.split(",")
    outdir = ROOT / "results" / "published_gai"
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / args.output
    done = set()
    if out.exists():
        with out.open(newline="", encoding="utf-8") as f:
            done = {(r["project"], int(r["seed"]), float(r["r"]), r["method"])
                    for r in csv.DictReader(f)}
    with out.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not done:
            writer.writeheader()
        for project in projects:
            for seed in seeds:
                for ratio in ratios:
                    for method in methods:
                        key = (project, seed, ratio, method)
                        if key in done:
                            continue
                        t0 = time.perf_counter()
                        result = run_one(*key)
                        writer.writerow(result)
                        f.flush()
                        print(f"{project} {seed} {ratio:g} {method}: "
                              f"{result['count']} confirmed, "
                              f"{time.perf_counter()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
