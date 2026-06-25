#!/usr/bin/env python3
"""
Collision-correctness tests for the DRIMAPSim simulation environment.

These tests pin down the ``block_both`` / ``strict`` movement model so it
matches the reference semantics in ``src/execution.py``:

* No two active agents ever occupy the same cell (no vertex collision).
* No two agents ever swap cells in a single step (no edge collision).
* A genuine circular wait stays stuck (a real deadlock) while remaining
  collision free.
* A following train of agents advances together (a follower may legally
  enter a cell its leader vacates in the same step).

Everything is driven through the public env API: ``drimapsim_v0``,
``EnvConfig``, ``reset`` and ``step``.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim import drimapsim_v0, EnvConfig


def _base_env(env):
    """Unwrap the metric/time-limit wrappers to reach the core env."""
    base = env
    while hasattr(base, "env"):
        base = base.env
    return base


def _active_positions(env):
    """Positions of the currently-active agents."""
    base = _base_env(env)
    grid = base.grid
    return [
        grid.get_agents_xy()[i]
        for i in range(base.config.num_agents)
        if grid.is_active[i]
    ]


def _assert_no_vertex_collision(env, context=""):
    positions = _active_positions(env)
    assert len(positions) == len(set(positions)), (
        f"vertex collision detected {context}: {positions}"
    )


def _assert_no_edge_swap(prev, curr, context=""):
    """Assert no pair of agents swapped cells between two snapshots.

    ``prev`` and ``curr`` map agent index -> position for the same set of
    agents. A swap is two agents exchanging cells in a single step.
    """
    agents = [i for i in prev if i in curr]
    for ai in range(len(agents)):
        for aj in range(ai + 1, len(agents)):
            i, j = agents[ai], agents[aj]
            if prev[i] == curr[j] and prev[j] == curr[i] and prev[i] != curr[i]:
                raise AssertionError(
                    f"edge swap detected {context}: agents {i},{j} "
                    f"swapped {prev[i]}<->{prev[j]}"
                )


def _agent_pos_map(env):
    base = _base_env(env)
    grid = base.grid
    return {
        i: grid.get_agents_xy()[i]
        for i in range(base.config.num_agents)
        if grid.is_active[i]
    }


# ---------------------------------------------------------------------------
# (a) Random rollouts never produce vertex collisions or edge swaps.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("collision_system", ["block_both", "strict"])
@pytest.mark.parametrize("seed", [0, 1, 7, 42, 123])
def test_random_rollout_is_collision_free(collision_system, seed):
    cfg = EnvConfig(
        size=14,
        num_agents=14,
        seed=seed,
        density=0.15,
        collision_system=collision_system,
        on_target="nothing",  # agents persist so collisions can occur
        max_episode_steps=400,
        deadlock_tracking=False,
    )
    env = drimapsim_v0(cfg)
    env.reset(seed=seed)
    rng = np.random.RandomState(seed)

    _assert_no_vertex_collision(env, context="at reset")
    prev = _agent_pos_map(env)

    for t in range(300):
        actions = list(rng.randint(0, 5, size=cfg.num_agents))
        env.step(actions)
        _assert_no_vertex_collision(env, context=f"at step {t}")
        curr = _agent_pos_map(env)
        _assert_no_edge_swap(prev, curr, context=f"at step {t}")
        prev = curr


# ---------------------------------------------------------------------------
# (b) A 3-agent circular wait stays a genuine, collision-free deadlock.
# ---------------------------------------------------------------------------

def test_circular_wait_stays_stuck_and_collision_free():
    """Three agents whose goals lie on the far side of a single shared
    junction form a genuine deadlock under ``block_both``.

    The map is a plus/junction: the only way through the centre is the cell
    ``(2, 2)``. Three agents sit on three of the four spokes and must all
    cross the centre to reach a goal on the opposite spoke. Each step every
    contender targets the same centre cell, so the vertex-conflict rule
    blocks all of them; they can never reroute (the spokes are one wide), so
    the configuration stalls forever while never sharing a cell.

    Map (``#`` = obstacle, ``.`` = free)::

        # # . # #
        # # . # #
        . . . . .
        # # . # #
        # # . # #
    """
    grid = [
        "##.##",
        "##.##",
        ".....",
        "##.##",
        "##.##",
    ]
    # Spoke heads, each wanting the opposite spoke through centre (2,2).
    #  agent 0 @ (2,0) -> goal (2,4)  (move right)
    #  agent 1 @ (0,2) -> goal (4,2)  (move down)
    #  agent 2 @ (2,4) -> goal (2,0)  (move left)
    cfg = EnvConfig(
        size=5,
        seed=0,
        collision_system="block_both",
        map="\n".join(grid),
        agents_xy=[(2, 1), (1, 2), (2, 3)],
        targets_xy=[(2, 4), (4, 2), (2, 0)],
        on_target="nothing",
        max_episode_steps=60,
        deadlock_tracking=True,
    )
    env = drimapsim_v0(cfg)
    env.reset()

    start = _active_positions(env)
    assert len(start) == len(set(start))  # sane start

    # All three drive toward the shared centre cell (2,2):
    #  agent 0 @ (2,1) right=4 ; agent 1 @ (1,2) down=2 ; agent 2 @ (2,3) left=3
    actions = [4, 2, 3]
    prev = _agent_pos_map(env)
    for t in range(40):
        env.step(actions)
        _assert_no_vertex_collision(env, context=f"deadlock step {t}")
        curr = _agent_pos_map(env)
        _assert_no_edge_swap(prev, curr, context=f"deadlock step {t}")
        prev = curr

    # Genuine deadlock: nobody ever moved.
    end = _active_positions(env)
    assert end == start, f"circular wait should be stuck, moved to {end}"

    # And the deadlock-event log should have fired (>=2 agents stagnated).
    base = _base_env(env)
    assert base.grid.deadlock_event_count() > 0, (
        "stagnation deadlock should have been recorded"
    )


# ---------------------------------------------------------------------------
# (c) A following train of agents advances.
# ---------------------------------------------------------------------------

def test_following_train_advances():
    """A line of agents all moving the same direction must advance together:
    the follower may legally enter the cell its leader vacates this step.
    """
    cfg = EnvConfig(
        size=6,
        seed=0,
        collision_system="block_both",
        map="\n".join(["......"] * 6),
        agents_xy=[(0, 0), (0, 1), (0, 2)],
        targets_xy=[(0, 3), (0, 4), (0, 5)],
        on_target="nothing",
        max_episode_steps=20,
        deadlock_tracking=False,
    )
    env = drimapsim_v0(cfg)
    env.reset()

    # All move right; the whole train should shift by one each step.
    env.step([4, 4, 4])
    assert _active_positions(env) == [(0, 1), (0, 2), (0, 3)]

    env.step([4, 4, 4])
    assert _active_positions(env) == [(0, 2), (0, 3), (0, 4)]

    # No collisions throughout.
    _assert_no_vertex_collision(env)


def test_cannot_enter_stationary_agent():
    """An agent must NOT move into a cell held by an agent that stays put."""
    cfg = EnvConfig(
        size=4,
        seed=0,
        collision_system="block_both",
        map="\n".join(["...."] * 4),
        agents_xy=[(0, 0), (0, 1)],
        targets_xy=[(0, 1), (3, 3)],  # agent 1's goal is far; it will idle
        on_target="nothing",
        max_episode_steps=10,
        deadlock_tracking=False,
    )
    env = drimapsim_v0(cfg)
    env.reset()

    # Agent 0 tries to move right into (0,1); agent 1 idles (action 0).
    env.step([4, 0])
    positions = _active_positions(env)
    assert positions[0] == (0, 0), "agent 0 must not enter a stationary agent"
    assert positions[1] == (0, 1)
    _assert_no_vertex_collision(env)
