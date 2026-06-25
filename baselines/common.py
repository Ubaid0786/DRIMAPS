#!/usr/bin/env python3
"""Shared infrastructure for baseline solvers.

Provides the conflict-oblivious independent planner, a cooperative-A*
reservation planner (for the prevention-only baseline), and a result container
compatible with the experiment runner's metric extraction.
"""

from typing import List, Optional

import numpy as np

from src.config import DRIMAPSConfig, DRIMAPSResult
from src.utils import Position, Path, bfs_shortest_path
from src.local_repair import LocalRepairPlanner


def independent_plans(
    grid: np.ndarray,
    starts: List[Position],
    goals: List[Position],
) -> List[Path]:
    """Per-agent shortest paths ignoring all other agents (conflict-oblivious).

    Args:
        grid: Map grid.
        starts: Start cells.
        goals: Goal cells.

    Returns:
        One route per agent; an unreachable agent waits at its start.
    """
    return [
        bfs_shortest_path(grid, starts[i], goals[i]) or [starts[i]]
        for i in range(len(starts))
    ]


def cooperative_plans(
    grid: np.ndarray,
    starts: List[Position],
    goals: List[Position],
    timeout: float,
) -> List[Path]:
    """Cooperative A*: plan agents sequentially with a shared reservation table.

    Each agent is planned with bounded space-time A*, treating already-planned
    paths (and their goal occupations) as dynamic obstacles. The result is
    (near) conflict-free by construction -- the prevention-only strategy.

    Args:
        grid: Map grid.
        starts: Start cells.
        goals: Goal cells.
        timeout: Wall-clock budget for planning (seconds).

    Returns:
        One route per agent; agents that cannot be planned wait at their start.
    """
    import time

    config = DRIMAPSConfig(timeout=timeout)
    repair = LocalRepairPlanner(grid, config)
    n = len(starts)
    paths: List[Path] = []
    reserved = set()
    max_t = min(grid.shape[0] * grid.shape[1] * 2, 2000)
    deadline = time.time() + timeout * 0.9

    for i in range(n):
        path = None
        if time.time() <= deadline:
            path = repair.replan(
                i, starts[i], goals[i], 0, reserved, search_horizon=max_t
            )
        if path is None:
            path = [starts[i]]
        for t, pos in enumerate(path):
            reserved.add((pos[0], pos[1], t))
        final = path[-1]
        for t in range(len(path), max_t):
            reserved.add((final[0], final[1], t))
        paths.append(path)

    return paths


def make_result(
    paths: List[Path],
    goals: List[Position],
    deadlocks_detected: int = 0,
    deadlocks_resolved: int = 0,
    agents_replanned: int = 0,
) -> DRIMAPSResult:
    """Build a DRIMAPSResult so baselines report through the same channel.

    Args:
        paths: Executed trajectories.
        goals: Goal cells.
        deadlocks_detected: Intervention count (0 for non-resolving baselines).
        deadlocks_resolved: Successful interventions.
        agents_replanned: Total replanning events.

    Returns:
        Populated result (success = every agent on its goal).
    """
    from src.utils import compute_makespan, compute_sum_of_costs

    res = DRIMAPSResult()
    res.paths = paths
    res.makespan = compute_makespan(paths)
    res.sum_of_costs = compute_sum_of_costs(paths)
    res.deadlocks_detected = deadlocks_detected
    res.deadlocks_resolved = deadlocks_resolved
    res.agents_replanned = agents_replanned
    res.success = all(paths[i][-1] == goals[i] for i in range(len(goals)))
    return res
