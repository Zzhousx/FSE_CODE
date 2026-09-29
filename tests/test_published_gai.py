"""Protocol checks for the two published GAI sampling rules."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts" / "cannier")]

import experiment_amini_finite_trace as A
from gai_published_baselines import PublishedGAISampling


class RepeatEnv:
    def __init__(self, values):
        self.values = values
        self.calls = np.zeros(len(values), dtype=int)

    def has_remaining(self, i):
        return True

    def execute(self, i):
        self.calls[i] += 1
        return self.values[i], 1.0


def test_lucbg_requires_full_initial_pass():
    arms = A.Arms(5, A.GS(1., 1., 1.))
    env = RepeatEnv([1] * 5)
    got = PublishedGAISampling("LUCB-G").run(
        arms, env, 2.5, np.random.default_rng(7), False)
    assert got["init_exec"] == 3
    assert not got["init_completed"]
    assert int(arms.n.sum()) == 3
    assert int(arms.confirmed.sum()) == 0


def test_murphy_can_start_without_full_pass_and_confirm_multiple():
    arms = A.Arms(2, A.GS(1., 1., 1.))
    env = RepeatEnv([1, 1])
    got = PublishedGAISampling("Murphy Sampling").run(
        arms, env, 1000., np.random.default_rng(3), False)
    assert got["init_exec"] == 0
    assert int(arms.confirmed.sum()) == 2
    assert got["executions"] < 1000
    assert np.all(env.calls > 0)
