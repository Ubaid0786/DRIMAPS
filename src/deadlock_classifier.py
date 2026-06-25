#!/usr/bin/env python3
"""
Deadlock Classifier

Classifies detected deadlock cycles into structural types:
  - CORRIDOR: 2 agents facing each other in a narrow passage.
  - CYCLIC:   k agents (k ≥ 2) in a circular wait chain.
  - CONGESTION: multiple agents converging on a bottleneck.
  - GOAL_BLOCKING: an agent sitting on another agent's goal.

Classification determines which resolution strategy is applied, enabling
type-specific minimal-disruption resolution.
"""

import numpy as np
from typing import Dict, List, Set, Tuple

from src.config import DeadlockType, DRIMAPSConfig
from src.utils import Position, cell_degree, manhattan_distance


class DeadlockClassifier:
    """Classifies deadlock cycles by structural type.

    The classifier examines the map topology at the positions of
    deadlocked agents and the dependency structure to determine the
    most specific applicable type. The priority order is:

        1. Goal-blocking (most specific, easy to resolve)
        2. Corridor (topology-dependent, 2 agents)
        3. Congestion (density-dependent, multiple agents in small area)
        4. Cyclic (general case, any circular wait chain)

    Attributes:
        grid: Map grid (1=obstacle, 0=free).
        config: Algorithm configuration.
    """

    def __init__(self, grid: np.ndarray, config: DRIMAPSConfig) -> None:
        """Initialize the classifier.

        Args:
            grid: Map grid.
            config: DRIMAPS configuration.
        """
        self.grid = grid
        self.config = config

    def classify(
        self,
        cycle: List[int],
        positions: Dict[int, Position],
        goals: List[Position],
        finished: Set[int],
    ) -> DeadlockType:
        """Classify a deadlock cycle into a structural type.

        Checks types in priority order (most specific first). If
        classification is disabled in config, returns CYCLIC as default.

        Args:
            cycle: Ordered list of agent indices in the deadlock.
            positions: Current position of each agent.
            goals: Goal position of each agent.
            finished: Agents already at their goals.

        Returns:
            The deadlock's structural type.
        """
        if not self.config.enable_classification:
            return DeadlockType.CYCLIC

        # 1. Check goal-blocking
        if self._is_goal_blocking(cycle, positions, goals, finished):
            return DeadlockType.GOAL_BLOCKING

        # 2. Check corridor
        if self._is_corridor(cycle, positions):
            return DeadlockType.CORRIDOR

        # 3. Check congestion
        if self._is_congestion(cycle, positions):
            return DeadlockType.CONGESTION

        # 4. Default: cyclic
        return DeadlockType.CYCLIC

    def _is_goal_blocking(
        self,
        cycle: List[int],
        positions: Dict[int, Position],
        goals: List[Position],
        finished: Set[int],
    ) -> bool:
        """Check if the deadlock involves goal-blocking.

        Goal-blocking occurs when at least one agent in the cycle sits
        on another agent's goal.

        Args:
            cycle: Agent indices in the deadlock.
            positions: Current positions.
            goals: Goal positions.
            finished: Finished agents.

        Returns:
            True if goal-blocking is detected.
        """
        cycle_set = set(cycle)
        for agent in cycle:
            if agent >= len(goals):
                continue
            agent_goal = goals[agent]
            for other in cycle:
                if other == agent:
                    continue
                if other in positions and positions[other] == agent_goal:
                    return True
        return False

    def _is_corridor(
        self,
        cycle: List[int],
        positions: Dict[int, Position],
    ) -> bool:
        """Check if the deadlock is a corridor deadlock.

        Corridor deadlock: exactly 2 agents, both in narrow cells
        (degree ≤ corridor_degree_threshold), on the same row or column.

        Args:
            cycle: Agent indices.
            positions: Current positions.

        Returns:
            True if corridor deadlock.
        """
        if len(cycle) != 2:
            return False

        a, b = cycle
        if a not in positions or b not in positions:
            return False

        pos_a = positions[a]
        pos_b = positions[b]

        deg_a = cell_degree(pos_a, self.grid)
        deg_b = cell_degree(pos_b, self.grid)

        threshold = self.config.corridor_degree_threshold

        if deg_a > threshold or deg_b > threshold:
            return False

        # Check same row or same column (corridor alignment)
        if pos_a[0] == pos_b[0] or pos_a[1] == pos_b[1]:
            return True

        # Adjacent cells in a narrow passage
        if manhattan_distance(pos_a, pos_b) <= 2:
            return True

        return False

    def _is_congestion(
        self,
        cycle: List[int],
        positions: Dict[int, Position],
    ) -> bool:
        """Check if the deadlock is a congestion deadlock.

        Congestion deadlock: multiple agents clustered within a small
        radius, indicating a bottleneck region.

        Args:
            cycle: Agent indices.
            positions: Current positions.

        Returns:
            True if congestion deadlock.
        """
        if len(cycle) < self.config.congestion_agent_threshold:
            return False

        # Compute pairwise distances
        cycle_positions = [
            positions[a] for a in cycle if a in positions
        ]

        if len(cycle_positions) < 2:
            return False

        # Check if all agents are within congestion_radius of each other
        max_dist = 0
        for i in range(len(cycle_positions)):
            for j in range(i + 1, len(cycle_positions)):
                d = manhattan_distance(cycle_positions[i], cycle_positions[j])
                max_dist = max(max_dist, d)

        # If all agents fit within a tight region, it's congestion
        return max_dist <= self.config.congestion_radius * 2

    def classify_all(
        self,
        cycles: List[List[int]],
        positions: Dict[int, Position],
        goals: List[Position],
        finished: Set[int],
    ) -> List[Tuple[List[int], DeadlockType]]:
        """Classify multiple deadlock cycles.

        Args:
            cycles: List of cycles from the cycle detector.
            positions: Current agent positions.
            goals: Agent goals.
            finished: Finished agents.

        Returns:
            List of (cycle, type) pairs.
        """
        return [
            (cycle, self.classify(cycle, positions, goals, finished))
            for cycle in cycles
        ]
