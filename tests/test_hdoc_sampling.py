"""Protocol checks for the HDoC-Sampling initialization and elimination phases."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "cannier"))

import experiment_amini_finite_trace as A
from hdoc_sampling import HDoCSampling


class RepeatedEnv:
    def __init__(self, outcomes):
        self.outcomes = outcomes
        self.counts = np.zeros(len(outcomes), dtype=int)

    def has_remaining(self, i):
        return True

    def execute(self, i):
        self.counts[i] += 1
        return self.outcomes[i], 1.0


def test_incomplete_initialization_stops_without_confirmation():
    arms = A.Arms(5, A.GS(1., 1., 1.))
    env = RepeatedEnv([1, 0, 1, 0, 1])
    result = HDoCSampling(5).run(arms, env, 2.5, np.random.default_rng(7), False)
    assert not result["init_completed"]
    assert result["init_exec"] == 3
    assert int(arms.confirmed.sum()) == 0
    assert result["n_eliminated"] == 0
    assert int(arms.n.sum()) == 3


def test_adaptive_phase_confirms_and_permanently_eliminates():
    arms = A.Arms(2, A.GS(1., 1., 1.))
    env = RepeatedEnv([1, 0])
    result = HDoCSampling(2).run(arms, env, 1500., np.random.default_rng(11), False)
    assert result["init_completed"]
    assert result["init_exec"] == 2
    assert result["n_eliminated"] == 1
    assert bool(arms.confirmed[0])
    assert not bool(arms.confirmed[1])
    # The eliminated all-pass arm receives no executions after its discard point.
    assert env.counts[1] == arms.n[1]
    assert env.counts[1] < 1500
