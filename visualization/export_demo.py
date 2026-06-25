#!/usr/bin/env python3
"""
Export DRIMAPS demo data as JSON for the interactive web visualizer.

Runs DRIMAPS on several map types and exports per-step agent positions,
goals, deadlock events, and grid data as a single JSON file.
"""

import os
import sys
import json
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import DRIMAPSConfig
from src.drimaps import DRIMAPS
from src.reactive_core import ReactiveCore
from experiments.generate_instances import generate_grid, generate_scenario
from src.config import MapConfig


def run_demo_scenario(map_type, size, num_agents, seed, density=0.2):
    """Run one scenario and capture per-step trajectory."""
    mc = MapConfig(map_type, map_type, size, size, map_type,
                   {"obstacle_density": density})
    grid = generate_grid(mc, seed)
    starts, goals = generate_scenario(grid, num_agents, seed)

    config = DRIMAPSConfig()
    config.verbose = False
    core = ReactiveCore(config)
    paths, stats = core.run(grid, starts, goals, seed=seed,
                            deadline=None)

    # Build per-step positions from paths
    max_t = max(len(p) for p in paths) if paths else 1
    trajectory = []
    for t in range(max_t):
        step_positions = []
        for i, p in enumerate(paths):
            idx = min(t, len(p) - 1)
            step_positions.append(list(p[idx]))
        trajectory.append(step_positions)

    # Check which agents reached goal
    reached = []
    for i, p in enumerate(paths):
        reached.append(list(p[-1]) == list(goals[i]) if p else False)

    return {
        "map_type": map_type,
        "size": size,
        "num_agents": num_agents,
        "seed": seed,
        "grid": grid.tolist(),
        "starts": [list(s) for s in starts],
        "goals": [list(g) for g in goals],
        "trajectory": trajectory,
        "total_steps": max_t,
        "reached_goal": reached,
        "deadlocks_detected": stats.deadlocks_detected,
        "deadlocks_resolved": stats.deadlocks_resolved,
        "agents_replanned": stats.agents_replanned,
    }


def main():
    scenarios = [
        # (map_type, size, agents, seed, density)
        ("bottleneck", 16, 10, 42, 0.15),
        ("maze", 16, 8, 42, 0.25),
        ("warehouse", 20, 12, 42, 0.15),
        ("corridor", 16, 8, 42, 0.1),
        ("random", 14, 10, 42, 0.2),
        ("room", 16, 10, 42, 0.15),
    ]

    demos = []
    for map_type, size, agents, seed, density in scenarios:
        print(f"Running {map_type} ({size}x{size}, {agents} agents)...")
        try:
            data = run_demo_scenario(map_type, size, agents, seed, density)
            demos.append(data)
            r = sum(data["reached_goal"])
            print(f"  Done: {r}/{agents} reached, "
                  f"{data['deadlocks_detected']} deadlocks, "
                  f"{data['total_steps']} steps")
        except Exception as e:
            print(f"  SKIP: {e}")

    out_path = os.path.join(PROJECT_ROOT, "visualization", "demo_data.json")
    with open(out_path, "w") as f:
        json.dump(demos, f)
    print(f"\nExported {len(demos)} scenarios to {out_path}")


if __name__ == "__main__":
    main()
