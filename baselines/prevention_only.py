#!/usr/bin/env python3
"""Prevention-only baseline: cooperative A* with no runtime repair.

Represents the prevention end of the spectrum -- conflicts are avoided at
planning time via a shared reservation table, and execution simply replays the
plan with no online deadlock handling. When cooperative planning succeeds the
plan is (near) conflict-free; in dense instances the sequential reservation
degrades (later agents are boxed in) and those agents fail to reach their goals.
"""

from typing import List, Optional

import numpy as np

from src.utils import Position, Path
from src.execution import reactive_execute
from baselines.common import cooperative_plans, make_result


class PreventionOnlySolver:
    """Cooperative-A* planner executed without runtime resolution."""

    def __init__(self, grid: np.ndarray, timeout: float = 30.0) -> None:
        self.grid = grid
        self.timeout = timeout
        self.last_result = None

    def solve(
        self,
        starts: List[Position],
        goals: List[Position],
        seed: int = 42,
    ) -> Optional[List[Path]]:
        """Plan cooperatively, then execute with no controller.

        Args:
            starts: Start cells.
            goals: Goal cells.
            seed: Unused (planning is deterministic); kept for interface parity.

        Returns:
            Executed trajectories.
        """
        import time

        plans = cooperative_plans(self.grid, starts, goals, self.timeout)
        deadline = time.time() + self.timeout * 0.9
        sim = reactive_execute(
            self.grid, starts, goals, plans,
            controller=None, deadline=deadline,
        )
        paths = sim.paths()
        self.last_result = make_result(paths, goals)
        return paths
