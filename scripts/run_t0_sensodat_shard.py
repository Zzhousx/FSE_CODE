"""Run one unchanged SensoDat split for the T0 regression, in an isolated output dir."""

import sys
from pathlib import Path

import run_optimistic_scorecost as runner
import sensodat_final_experiment as senso


def main(split_id):
    repo = Path(__file__).resolve().parents[1]
    runner.OUTDIR = str(repo / "results" / "revision" / "t0_sensodat" / f"shard_{split_id}")
    runner.LOGDIR = str(repo / "results" / "revision" / "t0_sensodat" / f"shard_{split_id}" / "logs")
    senso.DATA = str(repo.parent / "data")
    original = senso.make_splits

    def one_split():
        data = original()
        data["splits"] = [data["splits"][split_id]]
        return data

    senso.make_splits = one_split
    runner.run_sensodat(n_splits=1, diag=False)


if __name__ == "__main__":
    main(int(sys.argv[1]))
