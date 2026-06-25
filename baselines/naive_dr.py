#!/usr/bin/env python3
"""Naive detect-and-resolve baseline.

Detects deadlocks purely by stagnation (an agent that has not moved for a
threshold number of steps) and resolves them by random replanning: the stalled
agent is given a random sidestep followed by a fresh shortest path to its goal.
There is no wait-for graph, no cycle detection, and no structural
classification -- this isolates the value of DRIMAPS's structured resolution.
"""

import random
from typing import Dict, List, Optional, Set

import numpy as np

from src.config import DRIMAPSConfig
from src.utils import (
    Position, Path, bfs_shortest_path, get_neighbors,
)
from src.execution import ReactiveSimulator, reactive_execute
from baselines.common import independent_plans, make_result


class _NaiveController:
    """Per-step stagnation-triggered random replanning controller."""

    def __init__(self, grid: np.ndarray, threshold: int, rng: random.Random):
        self.grid = grid
        self.threshold = threshold
        self.rng = rng
        self.stag: Dict[int, int] = {}
        self.last: Dict[int, Position] = {}
        self.replans = 0

    def __call__(self, sim: ReactiveSimulator, t: int) -> None:
        cur = sim.positions
        finished = sim.finished
        for i in range(sim.n):
            if i in finished:
                self.stag[i] = 0
            elif self.last.get(i) == cur[i]:
                self.stag[i] = self.stag.get(i, 0) + 1
            else:
                self.stag[i] = 0
        self.last = dict(cur)

        # Keep exhausted agents progressing toward their goal.
        for i in range(sim.n):
            if i in finished:
                continue
            if len(sim.plans[i]) < 2 and cur[i] != sim.goals[i]:
                route = bfs_shortest_path(self.grid, cur[i], sim.goals[i])
                if route:
                    sim.set_plan(i, route)

        stalled = [
            i for i in range(sim.n)
            if i not in finished and self.stag.get(i, 0) >= self.threshold
        ]
        if len(stalled) < 2:
            return

        # Resolve one stalled agent at random with a random sidestep + replan.
        a = self.rng.choice(stalled)
        cur_a = cur[a]
        nbrs = [
            nb for nb in get_neighbors(cur_a, self.grid, include_wait=False)
            if nb not in {cur[j] for j in range(sim.n) if j != a}
        ]
        if not nbrs:
            return
        side = self.rng.choice(nbrs)
        tail = bfs_shortest_path(self.grid, side, sim.goals[a])
        route = [cur_a, side] + (tail[1:] if tail else [])
        sim.set_plan(a, route)
        self.replans += 1
        # Reset so the same agent is not immediately re-triggered.
        self.stag[a] = 0


class NaiveDRSolver:
    """Independent planning with stagnation-triggered random replanning."""

    def __init__(self, grid: np.ndarray, timeout: float = 30.0) -> None:
        self.grid = grid
        self.timeout = timeout
        self.config = DRIMAPSConfig(timeout=timeout)
        self.last_result = None

    def solve(
        self,
        starts: List[Position],
        goals: List[Position],
        seed: int = 42,
    ) -> Optional[List[Path]]:
        """Execute independent plans, randomly replanning stalled agents.

        Args:
            starts: Start cells.
            goals: Goal cells.
            seed: Seed for the random replanning decisions.

        Returns:
            Executed trajectories.
        """
        import time

        plans = independent_plans(self.grid, starts, goals)
        ctrl = _NaiveController(
            self.grid, self.config.stagnation_threshold, random.Random(seed)
        )
        deadline = time.time() + self.timeout * 0.9
        sim = reactive_execute(
            self.grid, starts, goals, plans,
            controller=ctrl, deadline=deadline,
        )
        paths = sim.paths()
        self.last_result = make_result(
            paths, goals,
            deadlocks_detected=ctrl.replans,
            deadlocks_resolved=ctrl.replans,
            agents_replanned=ctrl.replans,
        )
        return paths
