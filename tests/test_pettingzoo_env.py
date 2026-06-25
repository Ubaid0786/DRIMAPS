#!/usr/bin/env python3
"""Contract tests for the PettingZoo ParallelEnv adapter.

These validate the dict-keyed parallel API (reset/step return dicts keyed by
agent, agents are dropped on termination, rollouts are deterministic) without
requiring the optional ``pettingzoo`` package to be installed.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim import drimapsim_parallel_v0, EnvConfig


def _cfg(**kw):
    base = dict(size=12, num_agents=6, density=0.1,
                collision_system="block_both", on_target="finish", seed=0)
    base.update(kw)
    return EnvConfig(**base)


def test_reset_returns_agent_keyed_dicts():
    env = drimapsim_parallel_v0(_cfg())
    obs, infos = env.reset(seed=0)
    assert set(obs.keys()) == set(env.possible_agents)
    assert set(infos.keys()) == set(env.possible_agents)
    a0 = env.possible_agents[0]
    assert obs[a0].shape == env.observation_space(a0).shape


def test_step_returns_five_dicts():
    env = drimapsim_parallel_v0(_cfg())
    env.reset(seed=0)
    actions = {a: 0 for a in env.agents}
    obs, rew, term, trunc, infos = env.step(actions)
    for d in (obs, rew, term, trunc, infos):
        assert isinstance(d, dict)
        assert set(d.keys()) <= set(env.possible_agents)
    assert all(isinstance(v, float) for v in rew.values())
    assert all(isinstance(v, bool) for v in term.values())


def test_agents_dropped_on_finish():
    """In finish mode an agent that starts on its goal terminates and is
    removed from the live agent set on the next step."""
    cfg = EnvConfig(
        map="....\n....\n....\n....",
        agents_xy=[(0, 0), (3, 3)], targets_xy=[(0, 0), (0, 1)],
        collision_system="block_both", on_target="finish",
    )
    env = drimapsim_parallel_v0(cfg)
    env.reset(seed=0)
    n_before = len(env.agents)
    # agent_0 is already on its goal -> terminates this step.
    obs, rew, term, trunc, infos = env.step({a: 0 for a in env.agents})
    assert term["agent_0"] is True
    assert "agent_0" not in env.agents
    assert len(env.agents) < n_before


def test_parallel_rollout_is_deterministic():
    def run():
        env = drimapsim_parallel_v0(_cfg(num_agents=8, on_target="nothing"))
        env.reset(seed=4)
        rng = np.random.RandomState(4)
        traj = []
        for _ in range(20):
            acts = {a: int(rng.randint(0, 5)) for a in env.agents}
            env.step(acts)
            traj.append(tuple(env._env.get_agents_xy()))
        return traj

    assert run() == run()


def test_num_agents_property():
    env = drimapsim_parallel_v0(_cfg(num_agents=6))
    env.reset(seed=0)
    assert env.max_num_agents == 6
    assert env.num_agents == len(env.agents) == 6
