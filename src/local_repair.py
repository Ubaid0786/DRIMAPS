#!/usr/bin/env python3
"""
Bounded Space-Time A* for Local Repair

Provides a search-horizon-bounded Space-Time A* that replans a single
agent's path from its current position to its goal, respecting a
reservation table built from other agents' paths.

The search horizon H limits the temporal depth of the search, making
resolution tractable even on large grids. If the bounded search fails,
the caller can escalate by doubling H.
"""

import heapq
import numpy as np
from typing import Dict, List, Optional, Set, Tuple

from src.config import DRIMAPSConfig
from src.utils import (
    Position,
    Path,
    TimedPosition,
    get_neighbors,
    manhattan_distance,
    build_reservation_table,
)


class LocalRepairPlanner:
    """Bounded Space-Time A* for local path repair.

    Replans a single agent's path segment starting from a given position
    and timestep, subject to a reservation table from other agents.

    The search is bounded by `search_horizon` timesteps to keep
    resolution latency low. If no path is found within the horizon,
    returns None to signal escalation.

    Attributes:
        grid: Map grid.
        config: Algorithm configuration.
    """

    def __init__(self, grid: np.ndarray, config: DRIMAPSConfig) -> None:
        """Initialize the planner.

        Args:
            grid: Map grid (1=obstacle, 0=free).
            config: DRIMAPS configuration.
        """
        self.grid = grid
        self.config = config
        self.h, self.w = grid.shape

    def replan(
        self,
        agent: int,
        start: Position,
        goal: Position,
        start_time: int,
        reserved: Set[TimedPosition],
        search_horizon: Optional[int] = None,
    ) -> Optional[Path]:
        """Replan a single agent's path via bounded Space-Time A*.

        Args:
            agent: Agent index (for logging/diagnostics).
            start: Current position of the agent.
            goal: Goal position.
            start_time: Current timestep.
            reserved: Set of reserved (row, col, time) cells.
            search_horizon: Max timesteps to search (overrides config).

        Returns:
            New path segment from start to goal, or None if no path
            found within the search horizon.
        """
        horizon = search_horizon or self.config.search_horizon
        max_time = start_time + horizon

        def heuristic(pos: Position) -> int:
            return manhattan_distance(pos, goal)

        # Priority queue: (f_score, g_score, row, col, time)
        open_set: List[Tuple[int, int, int, int, int]] = [
            (heuristic(start), 0, start[0], start[1], start_time)
        ]
        came_from: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
        g_score: Dict[Tuple[int, int, int], int] = {
            (start[0], start[1], start_time): 0
        }

        expanded = 0
        max_nodes = self.config.max_expanded_nodes

        while open_set:
            f, g, r, c, t = heapq.heappop(open_set)
            expanded += 1

            # Node expansion cap
            if expanded > max_nodes:
                return None

            # Goal reached
            if (r, c) == goal:
                return self._reconstruct_path(came_from, (r, c, t))

            # Horizon exceeded
            if t >= max_time:
                continue

            # Check if this state was already processed with a better g
            state = (r, c, t)
            if g > g_score.get(state, float("inf")):
                continue

            # Expand neighbors (4 cardinal + wait)
            for neighbor in get_neighbors((r, c), self.grid, include_wait=True):
                nr, nc = neighbor
                nt = t + 1

                # Check reservation
                if (nr, nc, nt) in reserved:
                    continue

                # Edge conflict: prevent swapping
                if (nr, nc, t) in reserved and (r, c, nt) in reserved:
                    continue

                new_g = g + 1
                neighbor_state = (nr, nc, nt)

                if new_g < g_score.get(neighbor_state, float("inf")):
                    g_score[neighbor_state] = new_g
                    f_val = new_g + heuristic((nr, nc))
                    heapq.heappush(
                        open_set, (f_val, new_g, nr, nc, nt)
                    )
                    came_from[neighbor_state] = state

        return None  # No path within horizon

    def replan_to_bypass(
        self,
        agent: int,
        start: Position,
        bypass: Position,
        goal: Position,
        start_time: int,
        reserved: Set[TimedPosition],
        search_horizon: Optional[int] = None,
    ) -> Optional[Path]:
        """Replan via a bypass cell (for corridor resolution).

        Plans a path from start → bypass → goal, ensuring the agent
        steps aside to let others pass.

        Args:
            agent: Agent index.
            start: Current position.
            bypass: Intermediate bypass cell to route through.
            goal: Final goal position.
            start_time: Current timestep.
            reserved: Reservation table.
            search_horizon: Max timesteps.

        Returns:
            Combined path segment, or None if not found.
        """
        horizon = search_horizon or self.config.search_horizon

        # Phase 1: start → bypass
        path_to_bypass = self.replan(
            agent, start, bypass, start_time, reserved, horizon // 2
        )
        if path_to_bypass is None:
            return None

        # Update reserved with phase 1
        phase1_reserved = set(reserved)
        for dt, pos in enumerate(path_to_bypass):
            phase1_reserved.add((pos[0], pos[1], start_time + dt))

        # Phase 2: bypass → goal
        bypass_time = start_time + len(path_to_bypass) - 1
        path_to_goal = self.replan(
            agent, bypass, goal, bypass_time, phase1_reserved,
            horizon - len(path_to_bypass)
        )
        if path_to_goal is None:
            return None

        # Combine (avoid duplicating the bypass position)
        return path_to_bypass + path_to_goal[1:]

    def replan_with_temp_goal(
        self,
        agent: int,
        start: Position,
        temp_goal: Position,
        start_time: int,
        reserved: Set[TimedPosition],
        wait_steps: int = 2,
        search_horizon: Optional[int] = None,
    ) -> Optional[Path]:
        """Replan to a temporary position, wait, then return.

        Used for goal-blocking resolution: the blocking agent moves
        to an adjacent cell, waits, then returns.

        Args:
            agent: Agent index.
            start: Current position (which is on someone's goal).
            temp_goal: Adjacent free cell to move to.
            start_time: Current timestep.
            reserved: Reservation table.
            wait_steps: How many timesteps to wait at temp position.
            search_horizon: Max timesteps.

        Returns:
            Path segment: move to temp, wait, move back.
        """
        horizon = search_horizon or self.config.search_horizon

        # Step 1: move to temp
        path_to_temp = self.replan(
            agent, start, temp_goal, start_time, reserved, horizon // 3
        )
        if path_to_temp is None:
            return None

        # Step 2: wait at temp
        wait_path = [temp_goal] * wait_steps

        # Step 3: move back to start (original position = agent's own goal)
        temp_time = start_time + len(path_to_temp) - 1 + wait_steps
        temp_reserved = set(reserved)
        for dt, pos in enumerate(path_to_temp):
            temp_reserved.add((pos[0], pos[1], start_time + dt))
        for dt in range(wait_steps):
            temp_reserved.add(
                (temp_goal[0], temp_goal[1], start_time + len(path_to_temp) - 1 + dt)
            )

        path_back = self.replan(
            agent, temp_goal, start, temp_time, temp_reserved,
            horizon // 3
        )

        if path_back is None:
            # Can't return — just move out and stay
            return path_to_temp + wait_path

        return path_to_temp + wait_path + path_back[1:]

    def _reconstruct_path(
        self,
        came_from: Dict[Tuple[int, int, int], Tuple[int, int, int]],
        end_state: Tuple[int, int, int],
    ) -> Path:
        """Reconstruct the path from A* came_from map.

        Args:
            came_from: Parent map from A* search.
            end_state: Final (row, col, time) state.

        Returns:
            Path as list of (row, col) positions.
        """
        path = [(end_state[0], end_state[1])]
        state = end_state
        while state in came_from:
            state = came_from[state]
            path.append((state[0], state[1]))
        path.reverse()
        return path
