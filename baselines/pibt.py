#!/usr/bin/env python3
"""PIBT -- Priority Inheritance with Backtracking (Okumura et al., 2022).

A fast, decentralized one-step MAPF planner. At every timestep each agent, in
priority order, greedily claims the reachable cell closest to its goal; when an
agent wants a cell occupied by a lower-priority agent, that agent inherits the
priority and is recursively forced to move out of the way, with backtracking if
no valid move exists. PIBT produces collision-free joint moves and avoids
deadlocks by construction, but as an incomplete method it can leave some agents
short of their goals under heavy congestion.

This is a faithful, self-contained reimplementation used as the strong
decentralized reference point in the DRIMAPS evaluation.
"""

import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.utils import Position, Path, get_neighbors
from baselines.common import make_result


class PIBTSolver:
    """Priority Inheritance with Backtracking solver."""

    def __init__(self, grid: np.ndarray, timeout: float = 30.0) -> None:
        self.grid = grid
        self.timeout = timeout
        self.last_result = None

    # ------------------------------------------------------------------
    def _dist_field(self, goal: Position) -> Dict[Position, int]:
        """BFS distance from every free cell to ``goal`` (heuristic)."""
        dist: Dict[Position, int] = {goal: 0}
        q = deque([goal])
        while q:
            cur = q.popleft()
            for nb in get_neighbors(cur, self.grid, include_wait=False):
                if nb not in dist:
                    dist[nb] = dist[cur] + 1
                    q.append(nb)
        return dist

    def solve(
        self,
        starts: List[Position],
        goals: List[Position],
        seed: int = 42,
    ) -> Optional[List[Path]]:
        """Run PIBT until all agents reach their goals or limits are hit.

        Args:
            starts: Start cells.
            goals: Goal cells.
            seed: Unused; PIBT here is deterministic given priorities.

        Returns:
            Collision-free trajectories (one per agent).
        """
        n = len(starts)
        dist = [self._dist_field(goals[i]) for i in range(n)]
        BIG = self.grid.size + 1

        def h(agent: int, cell: Position) -> int:
            return dist[agent].get(cell, BIG)

        pos: List[Position] = list(starts)
        trajectory: List[List[Position]] = [[s] for s in starts]
        # Dynamic priority: steps spent away from goal (reset on arrival).
        elapsed = [0] * n

        longest = max((max(d.get(starts[i], BIG), 0) for i, d in enumerate(dist)), default=0)
        max_steps = min(1000, longest * 4 + 64)
        deadline = time.time() + self.timeout * 0.9

        for _ in range(max_steps):
            if all(pos[i] == goals[i] for i in range(n)):
                break
            if time.time() > deadline:
                break

            # Priority order: higher elapsed first, tie-break by index.
            order = sorted(range(n), key=lambda i: (-elapsed[i], i))
            next_pos: Dict[int, Position] = {}
            occupied_now = {pos[i]: i for i in range(n)}

            def pibt(ai: int, blocker_cell: Optional[Position]) -> bool:
                # Candidate cells: stay or step to a neighbour, closest-to-goal
                # first. Never move into the cell the pusher just vacated.
                cands = [pos[ai]] + list(
                    get_neighbors(pos[ai], self.grid, include_wait=False)
                )
                cands.sort(key=lambda c: h(ai, c))
                for v in cands:
                    if blocker_cell is not None and v == blocker_cell:
                        continue
                    if v in next_pos.values():
                        continue  # vertex conflict with an already-decided move
                    # Swap conflict: someone moving into our current cell from v.
                    swap = any(
                        next_pos.get(aj) == pos[ai] and pos[aj] == v
                        for aj in next_pos
                    )
                    if swap:
                        continue
                    next_pos[ai] = v
                    # If another (undecided) agent currently sits on v, it must move.
                    ak = occupied_now.get(v)
                    if ak is not None and ak != ai and ak not in next_pos:
                        if not pibt(ak, pos[ai]):
                            del next_pos[ai]
                            continue  # backtrack
                    return True
                next_pos[ai] = pos[ai]  # forced to wait
                return False

            for ai in order:
                if ai not in next_pos:
                    pibt(ai, None)

            pos = [next_pos[i] for i in range(n)]
            for i in range(n):
                trajectory[i].append(pos[i])
                elapsed[i] = 0 if pos[i] == goals[i] else elapsed[i] + 1

        self.last_result = make_result(trajectory, goals)
        return trajectory
