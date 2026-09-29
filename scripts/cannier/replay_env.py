"""Count-based finite-population replay; outcome truth stays in this module."""
from __future__ import annotations

import zlib
import numpy as np


class SequenceBank:
    """One cached outcome permutation per test for a project and seed."""
    def __init__(self, project, seed, item_id, failures):
        self.project = str(project)
        self.seed = int(seed)
        self.item_id = np.asarray(item_id, dtype=np.int64)
        self.failures = np.asarray(failures, dtype=np.int16)
        self.cache = {}

    def sequence(self, i):
        seq = self.cache.get(int(i))
        if seq is None:
            i = int(i); f = int(self.failures[i])
            if f == 0:
                seq = np.zeros(2500, dtype=np.int8)
            elif f == 2500:
                seq = np.ones(2500, dtype=np.int8)
            else:
                ss = np.random.SeedSequence([
                    self.seed, zlib.crc32(self.project.encode()), int(self.item_id[i])])
                seq = np.zeros(2500, dtype=np.int8)
                seq[:f] = 1
                np.random.default_rng(ss).shuffle(seq)
            self.cache[i] = seq
        return seq

    def audit_all(self):
        for i in range(len(self.item_id)):
            if int(self.sequence(i).sum()) != int(self.failures[i]):
                raise AssertionError(f"sequence total mismatch for item {self.item_id[i]}")


class CANNIERReplay:
    def __init__(self, project, seed, item_id, failures, costs, bank=None):
        self.project = str(project)
        self.seed = int(seed)
        self._item_id = np.asarray(item_id, dtype=np.int64)
        self._failures = np.asarray(failures, dtype=np.int16)
        self._costs = np.asarray(costs, dtype=np.float64)
        self._pos = np.zeros(len(self._item_id), dtype=np.int32)
        self._bank = bank or SequenceBank(project, seed, item_id, failures)
        self.served = []

    def _sequence(self, i):
        return self._bank.sequence(i)

    def has_remaining(self, i):
        return int(self._pos[int(i)]) < 2500

    def execute(self, i):
        i = int(i)
        k = int(self._pos[i])
        if k >= 2500:
            raise RuntimeError("attempted execution after the finite pool was exhausted")
        y = int(self._sequence(i)[k])
        self._pos[i] += 1
        self.served.append((int(self._item_id[i]), k + 1, y))
        return y, float(self._costs[i])

    def reset_log(self):
        out = self.served
        self.served = []
        return out


_REMAINING_TARGETS = {}
_SCHEDULER_TARGETS = {}


class RemainingView:
    """Compatibility view exposing only whether one pool is exhausted."""
    __slots__ = ()

    def __init__(self, i, env):
        _REMAINING_TARGETS[id(self)] = (env, int(i))

    def exhausted(self):
        env, i = _REMAINING_TARGETS[id(self)]
        return not env.has_remaining(i)


class SchedulerEnv:
    """Restricted HDoC environment with exactly execute/has_remaining methods."""
    __slots__ = ()

    def __init__(self, env):
        _SCHEDULER_TARGETS[id(self)] = env

    def execute(self, i):
        return _SCHEDULER_TARGETS[id(self)].execute(i)

    def has_remaining(self, i):
        return _SCHEDULER_TARGETS[id(self)].has_remaining(i)


def release_views(views):
    for view in views:
        _REMAINING_TARGETS.pop(id(view), None)


def release_scheduler_env(env):
    _SCHEDULER_TARGETS.pop(id(env), None)

