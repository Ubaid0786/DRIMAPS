#!/usr/bin/env python3
"""
Load and edit MovingAI benchmark maps inside DRIMAPSim.
=======================================================

Bridges the vendored MovingAI benchmark (``benchmarks/``) into the DRIMAPSim
environment so the standard MAPF maps can be simulated *and edited* — add or
remove obstacles, move starts/goals, resize the team — before rolling out.

Example::

    from sim.benchmark import BenchmarkScenario

    scn = BenchmarkScenario.load("maze-32-32-2", num_agents=40)
    scn.set_obstacle(5, 5, blocked=False)     # carve an opening
    scn.add_agent(start=(0, 0), goal=(31, 31))
    env = scn.to_env(collision_system="block_both")
    obs, info = env.reset()

The resulting object is an ordinary :class:`sim.environment.DRIMAPSimEnv`, so
every wrapper, renderer, and the wait-for-graph deadlock monitor work unchanged.
"""

from typing import List, Optional, Tuple

import numpy as np

from src.movingai import load_instance, load_map
from src.utils import Position
from sim.env_config import EnvConfig
from sim.environment import DRIMAPSimEnv


class BenchmarkScenario:
    """An editable MovingAI instance (grid + start/goal pairs)."""

    def __init__(
        self,
        grid: np.ndarray,
        starts: List[Position],
        goals: List[Position],
        name: str = "custom",
    ) -> None:
        self.grid = np.array(grid, dtype=int)
        self.starts = [tuple(s) for s in starts]
        self.goals = [tuple(g) for g in goals]
        self.name = name

    # -- construction ---------------------------------------------------
    @classmethod
    def load(
        cls, map_name: str, num_agents: int, scen: str = "even-1",
    ) -> "BenchmarkScenario":
        """Load a vendored benchmark map + the first ``num_agents`` agents."""
        grid, starts, goals = load_instance(map_name, num_agents, scen)
        return cls(grid, starts, goals, name=map_name)

    @classmethod
    def from_map(cls, map_name: str) -> "BenchmarkScenario":
        """Load just the grid (no agents) for free-form editing."""
        return cls(load_map(map_name), [], [], name=map_name)

    # -- editing --------------------------------------------------------
    def set_obstacle(self, r: int, c: int, blocked: bool = True) -> "BenchmarkScenario":
        """Block or clear a single cell (raises if out of bounds)."""
        self.grid[r, c] = 1 if blocked else 0
        return self

    def fill_rect(self, r0: int, c0: int, r1: int, c1: int,
                  blocked: bool = True) -> "BenchmarkScenario":
        """Block or clear a rectangular region (inclusive)."""
        self.grid[r0:r1 + 1, c0:c1 + 1] = 1 if blocked else 0
        return self

    def add_agent(self, start: Position, goal: Position) -> "BenchmarkScenario":
        """Append an agent with explicit start/goal (both must be free cells)."""
        if self.grid[start[0], start[1]] == 1 or self.grid[goal[0], goal[1]] == 1:
            raise ValueError("start/goal must be on free cells")
        self.starts.append(tuple(start))
        self.goals.append(tuple(goal))
        return self

    def set_team_size(self, num_agents: int) -> "BenchmarkScenario":
        """Truncate the team to the first ``num_agents`` agents."""
        self.starts = self.starts[:num_agents]
        self.goals = self.goals[:num_agents]
        return self

    # -- export ---------------------------------------------------------
    @property
    def num_agents(self) -> int:
        return len(self.starts)

    def to_config(self, **overrides) -> EnvConfig:
        """Build an :class:`EnvConfig` for this (edited) instance.

        Extra keyword arguments override config fields (e.g.
        ``collision_system="block_both"``, ``max_episode_steps=512``).
        """
        cfg = EnvConfig(
            size=max(self.grid.shape),
            num_agents=self.num_agents,
            map=self.grid.tolist(),
            agents_xy=self.starts,
            targets_xy=self.goals,
            on_target=overrides.pop("on_target", "nothing"),
            collision_system=overrides.pop("collision_system", "block_both"),
            max_episode_steps=overrides.pop("max_episode_steps", 512),
        )
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

    def to_env(self, **overrides) -> DRIMAPSimEnv:
        """Build a ready-to-``reset`` DRIMAPSim environment for this instance."""
        return DRIMAPSimEnv(self.to_config(**overrides))


def load_benchmark_env(
    map_name: str, num_agents: int, scen: str = "even-1", **overrides
) -> DRIMAPSimEnv:
    """One-liner: vendored benchmark instance straight to a DRIMAPSim env."""
    return BenchmarkScenario.load(map_name, num_agents, scen).to_env(**overrides)
