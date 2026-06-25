#!/usr/bin/env python3
"""
MovingAI MAPF benchmark loader.
===============================

Reads the standard Sturtevant ``.map`` grids and the Stern et al. ``.scen``
scenario files (the de-facto MAPF benchmark) into the grid / start / goal
representation used throughout DRIMAPS.

Map format::

    type octile
    height H
    width  W
    map
    <H rows of W characters>

Any character in ``.`` ``G`` ``S`` is traversable; everything else
(``@`` ``O`` ``T`` ``W`` ...) is an obstacle.

Scenario format (tab-separated, one agent per line after a ``version`` header)::

    bucket  map  width  height  start_x  start_y  goal_x  goal_y  optimal_length

MovingAI uses ``(x=col, y=row)``; we return ``(row, col)`` positions to match
the rest of the codebase.

The vendored benchmark lives in ``benchmarks/`` (33 maps, curated scenarios);
:func:`benchmarks_dir` resolves it regardless of the working directory.
"""

import os
from typing import Dict, List, Tuple

import numpy as np

from src.utils import Position

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def benchmarks_dir() -> str:
    """Absolute path to the vendored ``benchmarks/`` directory."""
    return os.path.join(PROJECT_ROOT, "benchmarks")


def load_map(path: str) -> np.ndarray:
    """Load a MovingAI ``.map`` file into a 0/1 grid (1 = obstacle).

    Args:
        path: Path to the ``.map`` file (absolute, or a bare map name resolved
            against the vendored ``benchmarks/mapf-map`` directory).

    Returns:
        ``np.ndarray`` of shape (height, width), dtype int, 0 free / 1 obstacle.
    """
    path = _resolve(path, "mapf-map", ".map")
    with open(path) as f:
        lines = f.read().splitlines()
    height = int(lines[1].split()[1])
    width = int(lines[2].split()[1])
    grid = np.zeros((height, width), dtype=int)
    body = lines[4:4 + height]
    for r, row in enumerate(body):
        for c in range(min(width, len(row))):
            if row[c] not in (".", "G", "S"):
                grid[r, c] = 1
    return grid


def load_scenario(
    path: str,
) -> List[Tuple[Position, Position, float]]:
    """Load a MovingAI ``.scen`` file.

    Args:
        path: Path to the ``.scen`` file (absolute, or a bare scenario name
            resolved against ``benchmarks/scen-even``).

    Returns:
        List of ``(start, goal, optimal_length)`` with start/goal as
        ``(row, col)``, in file order.
    """
    path = _resolve(path, "scen-even", ".scen")
    out: List[Tuple[Position, Position, float]] = []
    with open(path) as f:
        for line in f.read().splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) < 9:
                parts = line.split()
            if len(parts) < 9:
                continue
            sx, sy, gx, gy = (int(parts[4]), int(parts[5]),
                              int(parts[6]), int(parts[7]))
            opt = float(parts[8])
            out.append(((sy, sx), (gy, gx), opt))
    return out


def load_instance(
    map_name: str,
    num_agents: int,
    scen: str = "even-1",
) -> Tuple[np.ndarray, List[Position], List[Position]]:
    """Load a complete instance: grid + first ``num_agents`` start/goal pairs.

    Args:
        map_name: Map name without extension, e.g. ``"maze-32-32-2"``.
        num_agents: Number of agents (takes the first N scenario entries).
        scen: Scenario suffix, e.g. ``"even-1"`` or ``"random-3"``.

    Returns:
        ``(grid, starts, goals)``.
    """
    grid = load_map(map_name)
    bucket = "scen-random" if scen.startswith("random") else "scen-even"
    scen_path = os.path.join(benchmarks_dir(), bucket, f"{map_name}-{scen}.scen")
    entries = load_scenario(scen_path)
    entries = entries[:num_agents]
    starts = [s for s, g, _ in entries]
    goals = [g for s, g, _ in entries]
    return grid, starts, goals


def available_maps() -> List[str]:
    """Sorted list of vendored map names (without extension)."""
    d = os.path.join(benchmarks_dir(), "mapf-map")
    if not os.path.isdir(d):
        return []
    return sorted(f[:-4] for f in os.listdir(d) if f.endswith(".map"))


# Category labels for the 33-map benchmark (for per-map reporting).
MAP_CATEGORY: Dict[str, str] = {
    "empty-8-8": "empty", "empty-16-16": "empty", "empty-32-32": "empty",
    "empty-48-48": "empty",
    "random-32-32-10": "random", "random-32-32-20": "random",
    "random-64-64-10": "random", "random-64-64-20": "random",
    "room-32-32-4": "room", "room-64-64-8": "room", "room-64-64-16": "room",
    "maze-32-32-2": "maze", "maze-32-32-4": "maze", "maze-128-128-1": "maze",
    "maze-128-128-2": "maze", "maze-128-128-10": "maze",
    "warehouse-10-20-10-2-1": "warehouse", "warehouse-10-20-10-2-2": "warehouse",
    "warehouse-20-40-10-2-1": "warehouse", "warehouse-20-40-10-2-2": "warehouse",
    "den312d": "game", "den520d": "game", "ost003d": "game", "lak303d": "game",
    "brc202d": "game", "orz900d": "game", "ht_chantry": "game",
    "ht_mansion_n": "game", "lt_gallowstemplar_n": "game",
    "w_woundedcoast": "game",
    "Berlin_1_256": "city", "Boston_0_256": "city", "Paris_1_256": "city",
}


def _resolve(path: str, subdir: str, ext: str) -> str:
    """Resolve a bare name against the vendored benchmark dir, else return as-is."""
    if os.path.isabs(path) or os.path.exists(path):
        return path
    name = path if path.endswith(ext) else path + ext
    cand = os.path.join(benchmarks_dir(), subdir, name)
    return cand if os.path.exists(cand) else path
