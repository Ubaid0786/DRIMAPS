#!/usr/bin/env python3
"""Validation tests for DRIMAPSim's wait-for-graph deadlock instrumentation.

These verify that the environment's built-in deadlock detection (a) fires on
genuine wait cycles, (b) does NOT fire on transient contention or free-flowing
traffic (no false positives), (c) assigns a consistent structural taxonomy, and
(d) is fully reproducible from a seed.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim import drimapsim_v0
from sim.env_config import EnvConfig
from src.config import DeadlockType


def _greedy_actions(env):
    """One greedy step toward each agent's goal (Manhattan-descending)."""
    pos = env.get_agents_xy()
    tgt = env.get_targets_xy()
    acts = []
    for (r, c), (tr, tc) in zip(pos, tgt):
        if r == tr and c == tc:
            acts.append(0)
        elif abs(tr - r) >= abs(tc - c):
            acts.append(2 if tr > r else 1)
        else:
            acts.append(4 if tc > c else 3)
    return acts


def _run(env, actions_fn, steps):
    env.reset(seed=0)
    for t in range(steps):
        env.step(actions_fn(t))
    return env.get_deadlock_stats()


class TestDetection:
    def test_corridor_head_on_is_one_deadlock(self):
        """Two agents facing off in a 1-wide corridor = exactly one deadlock."""
        cfg = EnvConfig(
            map="#####\n#...#\n#####",
            agents_xy=[(1, 1), (1, 3)], targets_xy=[(1, 3), (1, 1)],
            collision_system="block_both", deadlock_tracking=True,
            on_target="nothing",
        )
        env = drimapsim_v0(cfg)
        stats = _run(env, lambda t: [4, 3], 12)  # push toward each other
        assert stats["deadlock_count"] == 1
        assert stats["active_deadlocks"] == 1
        # Exactly one structural label was assigned, and it is a valid type.
        total_typed = sum(stats["type_counts"].values())
        assert total_typed == stats["deadlock_count"]
        valid = {t.value for t in DeadlockType}
        assert all(k in valid for k in stats["type_counts"])

    def test_congestion_deadlock_detected_with_greedy_play(self):
        """Greedy-toward-goal agents converging on a bottleneck jam, and the
        monitor confirms at least one WFG deadlock."""
        cfg = EnvConfig(
            size=32, num_agents=30, map_type="bottleneck", density=0.2,
            collision_system="block_both", deadlock_tracking=True,
            on_target="nothing", seed=7,
        )
        env = drimapsim_v0(cfg)
        env.reset(seed=7)
        for _ in range(80):
            env.step(_greedy_actions(env))
        assert env.get_deadlock_stats()["deadlock_count"] >= 1

    def test_no_false_positive_on_open_map(self):
        """Random play on an empty grid should produce essentially no
        WFG-confirmed deadlocks (transient contention is filtered)."""
        cfg = EnvConfig(
            size=20, num_agents=10, density=0.0,
            collision_system="block_both", deadlock_tracking=True,
            on_target="nothing", seed=3,
        )
        env = drimapsim_v0(cfg)
        env.reset(seed=3)
        for _ in range(60):
            env.step(env.sample_actions())
        assert env.get_deadlock_stats()["deadlock_count"] <= 1

    def test_single_agent_never_deadlocks(self):
        cfg = EnvConfig(
            size=10, num_agents=1, density=0.0,
            collision_system="block_both", deadlock_tracking=True,
            on_target="nothing", seed=1,
        )
        env = drimapsim_v0(cfg)
        env.reset(seed=1)
        for _ in range(30):
            env.step(env.sample_actions())
        assert env.get_deadlock_stats()["deadlock_count"] == 0


class TestCongestion:
    def test_heatmap_accumulates_at_contended_cells(self):
        cfg = EnvConfig(
            map="#####\n#...#\n#####",
            agents_xy=[(1, 1), (1, 3)], targets_xy=[(1, 3), (1, 1)],
            collision_system="block_both", deadlock_tracking=True,
            on_target="nothing",
        )
        env = drimapsim_v0(cfg)
        _run(env, lambda t: [4, 3], 10)
        heat = env.get_congestion_heatmap()
        assert heat is not None
        assert heat.shape == (3, 5)
        assert heat.sum() > 0  # the central corridor cell was contended


class TestReproducibility:
    def test_same_seed_identical_rollout(self):
        def rollout():
            cfg = EnvConfig(
                size=16, num_agents=12, density=0.15,
                collision_system="block_both", deadlock_tracking=True,
                on_target="nothing", seed=7,
            )
            env = drimapsim_v0(cfg)
            env.reset(seed=7)
            traj = []
            for _ in range(40):
                env.step(env.sample_actions())
                traj.append(tuple(env.get_agents_xy()))
            return traj, env.get_deadlock_stats()

        t1, s1 = rollout()
        t2, s2 = rollout()
        assert t1 == t2                 # identical trajectories
        assert s1 == s2                 # identical deadlock statistics
