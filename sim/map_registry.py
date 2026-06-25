"""
DRIMAPSim Map Registry

Built-in benchmark map generators (8 types) plus MovingAI .map import.
Each generator returns a numpy array (1=obstacle, 0=free).
"""

import numpy as np
from typing import Optional, Tuple


def generate_map(map_type: str, size: int, density: float = 0.2,
                 seed: Optional[int] = None) -> np.ndarray:
    """Generate a map by type name.

    Args:
        map_type: One of: random, warehouse, corridor, bottleneck,
                  maze, room, dense_random, open.
        size: Grid dimension (size x size).
        density: Obstacle density (for random-based maps).
        seed: Random seed.

    Returns:
        Grid as numpy array (1=obstacle, 0=free).
    """
    generators = {
        "random": generate_random,
        "warehouse": generate_warehouse,
        "corridor": generate_corridor,
        "bottleneck": generate_bottleneck,
        "maze": generate_maze,
        "room": generate_room,
        "dense_random": lambda s, d, sd: generate_random(s, 0.35, sd),
        "open": generate_open,
    }
    gen = generators.get(map_type, generate_random)
    return gen(size, density, seed)


def generate_random(size: int, density: float = 0.2,
                    seed: Optional[int] = None) -> np.ndarray:
    """Random obstacles with given density."""
    rng = np.random.RandomState(seed)
    grid = np.zeros((size, size), dtype=np.int32)
    n_obs = int(size * size * density)
    positions = rng.choice(size * size, size=n_obs, replace=False)
    for pos in positions:
        r, c = divmod(pos, size)
        grid[r, c] = 1
    # Clear borders for agent placement
    grid[0, :] = 0
    grid[-1, :] = 0
    grid[:, 0] = 0
    grid[:, -1] = 0
    return grid


def generate_warehouse(size: int, density: float = 0.2,
                       seed: Optional[int] = None) -> np.ndarray:
    """Warehouse layout with parallel aisles and shelf blocks."""
    grid = np.zeros((size, size), dtype=np.int32)
    aisle_width = 3
    shelf_width = 2
    pattern = aisle_width + shelf_width

    for c in range(size):
        col_in_pattern = c % pattern
        if col_in_pattern >= aisle_width:
            # Shelf column — add obstacles with gaps
            for r in range(2, size - 2):
                if r % 6 != 0:  # Leave cross-aisle gaps
                    grid[r, c] = 1

    # Clear border rows for movement
    grid[0, :] = 0
    grid[1, :] = 0
    grid[-1, :] = 0
    grid[-2, :] = 0
    return grid


def generate_corridor(size: int, density: float = 0.2,
                      seed: Optional[int] = None) -> np.ndarray:
    """Narrow corridors with bottleneck passages."""
    grid = np.ones((size, size), dtype=np.int32)

    # Main horizontal corridors
    for r in range(2, size - 2, 4):
        grid[r, :] = 0
        if r + 1 < size:
            grid[r + 1, :] = 0

    # Vertical connectors (limited — creates bottlenecks)
    rng = np.random.RandomState(seed)
    for c in range(3, size - 3, 6):
        grid[:, c] = 0
        # Add some random blockages to create bottlenecks
        for r in range(size):
            if grid[r, c] == 0 and rng.random() < 0.15:
                grid[r, c] = 1

    # Clear borders
    grid[0, :] = 0
    grid[-1, :] = 0
    grid[:, 0] = 0
    grid[:, -1] = 0
    return grid


def generate_bottleneck(size: int, density: float = 0.2,
                        seed: Optional[int] = None) -> np.ndarray:
    """Central wall with a single narrow passage — extreme deadlock potential."""
    grid = np.zeros((size, size), dtype=np.int32)

    # Central vertical wall
    mid_c = size // 2
    for r in range(size):
        grid[r, mid_c] = 1

    # Single gap in the middle
    gap_r = size // 2
    grid[gap_r, mid_c] = 0
    if gap_r + 1 < size:
        grid[gap_r + 1, mid_c] = 0

    return grid


def generate_maze(size: int, density: float = 0.2,
                  seed: Optional[int] = None) -> np.ndarray:
    """Recursive backtracking maze — complex topology."""
    rng = np.random.RandomState(seed)
    # Start with all walls
    grid = np.ones((size, size), dtype=np.int32)

    def carve(r, c):
        grid[r, c] = 0
        directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]
        rng.shuffle(directions)
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size and grid[nr, nc] == 1:
                grid[r + dr // 2, c + dc // 2] = 0
                carve(nr, nc)

    # Start from (1, 1)
    import sys
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old_limit, size * size + 100))
    carve(1, 1)
    sys.setrecursionlimit(old_limit)

    # Clear borders
    grid[0, :] = 0
    grid[-1, :] = 0
    grid[:, 0] = 0
    grid[:, -1] = 0
    return grid


def generate_room(size: int, density: float = 0.2,
                  seed: Optional[int] = None) -> np.ndarray:
    """Room-based layout with doorways — tests door congestion."""
    grid = np.zeros((size, size), dtype=np.int32)
    rng = np.random.RandomState(seed)
    room_size = max(6, size // 4)

    # Create room walls
    for r in range(room_size, size - 1, room_size):
        grid[r, :] = 1
        # Add doorways
        n_doors = max(1, size // room_size)
        door_positions = rng.choice(range(1, size - 1), size=n_doors, replace=False)
        for dc in door_positions:
            grid[r, dc] = 0
            if dc + 1 < size:
                grid[r, dc + 1] = 0

    for c in range(room_size, size - 1, room_size):
        grid[:, c] = 1
        n_doors = max(1, size // room_size)
        door_positions = rng.choice(range(1, size - 1), size=n_doors, replace=False)
        for dr in door_positions:
            grid[dr, c] = 0
            if dr + 1 < size:
                grid[dr + 1, c] = 0

    # Clear borders
    grid[0, :] = 0
    grid[-1, :] = 0
    grid[:, 0] = 0
    grid[:, -1] = 0
    return grid


def generate_open(size: int, density: float = 0.0,
                  seed: Optional[int] = None) -> np.ndarray:
    """Open grid with no obstacles."""
    return np.zeros((size, size), dtype=np.int32)


def parse_movingai_map(map_str: str) -> np.ndarray:
    """Parse a MovingAI .map format string into a grid.

    MovingAI format: header lines followed by grid where
    '.' and 'G' are free, '@' and 'T' and 'O' are obstacles.

    Args:
        map_str: Raw .map file content.

    Returns:
        Grid as numpy array.
    """
    lines = map_str.strip().split('\n')
    grid_lines = []
    reading_grid = False

    for line in lines:
        if line.startswith('map'):
            reading_grid = True
            continue
        if reading_grid and line.strip():
            row = []
            for ch in line.strip():
                if ch in ('.', 'G', 'S'):
                    row.append(0)
                else:
                    row.append(1)
            grid_lines.append(row)

    if not grid_lines:
        # Fallback: treat entire content as grid
        for line in lines:
            if line.strip() and not line.startswith(('type', 'height', 'width')):
                row = [0 if ch in ('.', 'G', 'S') else 1 for ch in line.strip()]
                grid_lines.append(row)

    max_w = max(len(row) for row in grid_lines) if grid_lines else 0
    for row in grid_lines:
        while len(row) < max_w:
            row.append(0)

    return np.array(grid_lines, dtype=np.int32)


def load_movingai_map(filepath: str) -> np.ndarray:
    """Load a MovingAI .map file.

    Args:
        filepath: Path to .map file.

    Returns:
        Grid as numpy array.
    """
    with open(filepath, 'r') as f:
        return parse_movingai_map(f.read())


# ── Map difficulty scoring ──────────────────────────────────────────

def compute_map_difficulty(grid: np.ndarray) -> dict:
    """Compute difficulty metrics for a map.

    Returns:
        Dict with keys: density, avg_degree, chokepoint_count,
        largest_component_ratio, difficulty_score.
    """
    h, w = grid.shape
    total_cells = h * w
    obstacle_count = int(grid.sum())
    free_count = total_cells - obstacle_count
    density = obstacle_count / total_cells if total_cells > 0 else 0

    # Average degree of free cells
    total_degree = 0
    min_degree = 4
    chokepoints = 0
    for r in range(h):
        for c in range(w):
            if grid[r, c] == 0:
                deg = 0
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == 0:
                        deg += 1
                total_degree += deg
                min_degree = min(min_degree, deg)
                if deg <= 1:
                    chokepoints += 1

    avg_degree = total_degree / free_count if free_count > 0 else 0

    # Simple difficulty score (higher = harder)
    difficulty_score = density * 30 + (4 - avg_degree) * 10 + chokepoints * 0.5

    return {
        "density": round(density, 3),
        "free_cells": free_count,
        "avg_degree": round(avg_degree, 2),
        "chokepoint_count": chokepoints,
        "difficulty_score": round(difficulty_score, 1),
    }
