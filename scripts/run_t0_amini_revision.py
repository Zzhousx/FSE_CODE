"""Rerun the frozen eight-method Amini primary configuration in revision output."""

from pathlib import Path
import run_optimistic_scorecost as runner

ROOT = Path(__file__).resolve().parents[1]
runner.OUTDIR = str(ROOT / "results" / "revision" / "t0_amini_8")
runner.LOGDIR = str(ROOT / "results" / "revision" / "t0_amini_8" / "logs")
runner.run_amini(diag=False)
