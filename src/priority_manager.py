#!/usr/bin/env python3
"""
Priority Manager

Computes agent priorities for deadlock resolution ordering.

Priority determines which agent yields (lower priority yields first)
during resolution. The priority function balances distance-to-goal
against remaining path length, so agents close to finishing get
higher priority (less likely to be disrupted).
"""

import numpy as np
from typing import Dict, List, Set, Tuple

from src.config import DRIMAPSConfig
from src.utils import Position, Path, manhattan_distance


class PriorityManager:
    """Computes and manages agent priorities.

    Priority score for agent i:
        priority(i) = α · d_goal(i) + β · |remaining_path(i)|

    Lower score = closer to goal = HIGHER priority (less disruption).
    During resolution, the agent with the HIGHEST score (lowest priority)
    is selected for rerouting.

    Attributes:
        config: Algorithm configuration.
    """

    def __init__(self, config: DRIMAPSConfig) -> None:
        """Initialize with configuration.

        Args:
            config: DRIMAPS configuration.
        """
        self.config = config

    def compute_priority(
        self,
        agent: int,
        position: Position,
        goal: Position,
        remaining_path_length: int,
    ) -> float:
        """Compute priority score for a single agent.

        Lower score means higher priority (closer to completion).

        Args:
            agent: Agent index.
            position: Current position.
            goal: Goal position.
            remaining_path_length: Number of steps remaining in current path.

        Returns:
            Priority score (lower = more important to protect).
        """
        d_goal = manhattan_distance(position, goal)
        score = (
            self.config.priority_alpha * d_goal
            + self.config.priority_beta * remaining_path_length
        )
        return score

    def rank_agents_for_resolution(
        self,
        cycle: List[int],
        positions: Dict[int, Position],
        goals: List[Position],
        paths: List[Path],
        timestep: int,
    ) -> List[int]:
        """Rank agents in a deadlock cycle by resolution priority.

        Returns agents ordered from LOWEST priority (should yield first)
        to HIGHEST priority (should be protected).

        Args:
            cycle: Agent indices in the deadlock.
            positions: Current positions.
            goals: Goal positions.
            paths: Current paths.
            timestep: Current simulation timestep.

        Returns:
            Agent indices sorted by descending score (first = yield first).
        """
        scored: List[Tuple[float, int]] = []
        for agent in cycle:
            if agent >= len(goals) or agent >= len(paths):
                scored.append((float("inf"), agent))
                continue

            pos = positions.get(agent, paths[agent][0])
            goal = goals[agent]
            remaining = max(0, len(paths[agent]) - 1 - timestep)
            score = self.compute_priority(agent, pos, goal, remaining)
            scored.append((score, agent))

        # Sort by score descending: highest score = yield first
        scored.sort(key=lambda x: -x[0])
        return [agent for _, agent in scored]

    def select_yield_agent(
        self,
        cycle: List[int],
        positions: Dict[int, Position],
        goals: List[Position],
        paths: List[Path],
        timestep: int,
    ) -> int:
        """Select the single agent that should yield in a deadlock.

        Returns the agent with the LOWEST priority (highest score).

        Args:
            cycle: Agent indices in the deadlock.
            positions: Current positions.
            goals: Goal positions.
            paths: Current paths.
            timestep: Current simulation timestep.

        Returns:
            Agent index that should yield.
        """
        ranked = self.rank_agents_for_resolution(
            cycle, positions, goals, paths, timestep
        )
        return ranked[0]  # First element has lowest priority
