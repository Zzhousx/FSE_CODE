"""Guard matched-unit counts and engineering-time basis in revision tables."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_revision_statistics import mean_halfwidth


def test_interval_uses_student_t_for_unit_means():
    values = [0, 1, 2, 3]
    expected = t.ppf(.975, 3) * np.std(values, ddof=1) / 2
    assert np.isclose(mean_halfwidth(values)[1], expected)


def test_primary_statistics_have_independent_unit_counts():
    data = pd.read_csv(ROOT / "results/revision/statistics_rev.csv")
    primary = data[data.budget.isin((.05, .10, .20)) & (data.tau == .3)]
    assert set(primary[primary.benchmark == "SensoDat"].n_units) == {10}
    assert set(primary[primary.benchmark == "Amini"].n_units) == {12}


def test_engineering_time_excludes_duplicate_basis():
    table = ROOT.parent / "paper_reproalloc_orcc/tables/revision"
    main = pd.read_csv(table / "Table4_rev.csv")
    paired = pd.read_csv(table / "Table4_rev_paired.csv")
    assert set(main.n_reached) == {10, 12}
    assert (paired.n_joint == paired.wins + paired.losses + paired.ties).all()
    assert set(paired.n_joint) == {10, 12}
