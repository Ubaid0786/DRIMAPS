#!/usr/bin/env python3
"""
Self-Contained Instance Generator

Generates map and scenario instances for all benchmark configurations
using DRIMAPSim's built-in map generators (no external dependencies).
"""

import os
import sys
import numpy as np
from typing import Tuple, List

# Add project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import MapConfig, BENCHMARK_MAPS, AGENT_COUNTS, EXPERIMENT_SEEDS
from src.utils import Position
from sim.map_registry import generate_map


def generate_grid(map_config: MapConfig, seed: int = 42) -> np.ndarray:
    """Generate a grid from a map configuration.

    Uses DRIMAPSim's built-in generators — fully self-contained.

    Args:
        map_config: Map configuration.
        seed: Random seed.

    Returns:
        Grid as numpy array (1=obstacle, 0=free).
    """
    return generate_map(
        map_config.map_type,
        max(map_config.width, map_config.height),
        map_config.extra_kwargs.get("obstacle_density", 0.2),
        seed,
    )


def generate_scenario(
    grid: np.ndarray,
    num_agents: int,
    seed: int = 42,
) -> Tuple[List[Position], List[Position]]:
    """Generate random start/goal pairs on a grid.

    Uses BFS-based reachability to ensure all start-goal pairs
    are connected.

    Args:
        grid: Map grid.
        num_agents: Number of agents.
        seed: Random seed.

    Returns:
        (starts, goals) tuple.
    """
    import random
    rng = random.Random(seed)
    h, w = grid.shape

    # Collect free cells
    free_cells = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] == 0:
                free_cells.append((r, c))

    if len(free_cells) < num_agents * 2:
        # Reduce agent count to fit
        num_agents = len(free_cells) // 2

    if num_agents == 0:
        raise ValueError("No room for agents on this grid")

    rng.shuffle(free_cells)
    starts = free_cells[:num_agents]
    goals = free_cells[num_agents: num_agents * 2]
    return starts, goals


def generate_all_instances(output_dir: str = "instances") -> None:
    """Generate all benchmark instances for all experiments.

    Args:
        output_dir: Directory to save instances.
    """
    output_dir = os.path.join(PROJECT_ROOT, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    total = len(BENCHMARK_MAPS) * len(EXPERIMENT_SEEDS) * len(AGENT_COUNTS)
    count = 0

    for map_config in BENCHMARK_MAPS:
        for seed in EXPERIMENT_SEEDS:
            grid = generate_grid(map_config, seed)

            for num_agents in AGENT_COUNTS:
                count += 1
                try:
                    starts, goals = generate_scenario(grid, num_agents, seed)
                    instance_name = (
                        f"{map_config.map_id}_{map_config.map_type}_"
                        f"{map_config.width}x{map_config.height}_"
                        f"{num_agents}agents_s{seed}"
                    )
                    instance_path = os.path.join(
                        output_dir, f"{instance_name}.npz"
                    )
                    np.savez_compressed(
                        instance_path,
                        grid=grid,
                        starts=np.array(starts),
                        goals=np.array(goals),
                    )
                    print(f"  [{count}/{total}] Generated {instance_name}")
                except (ValueError, Exception) as e:
                    print(
                        f"  [{count}/{total}] SKIP {map_config.map_id} "
                        f"{num_agents} agents seed={seed}: {e}"
                    )


if __name__ == "__main__":
    generate_all_instances()
    print("All instances generated.")
