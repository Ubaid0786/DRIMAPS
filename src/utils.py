#!/usr/bin/env python3
"""
DRIMAPS Shared Utilities

Common data structures, heuristics, and helper functions used
across all DRIMAPS modules.
"""

import heapq
import numpy as np
from typing import List, Tuple, Dict, Set, Optional

# Type aliases for readability
Position = Tuple[int, int]
Path = List[Position]
TimedPosition = Tuple[int, int, int]  # (row, col, timestep)


def manhattan_distance(a: Position, b: Position) -> int:
    """Compute Manhattan distance between two grid positions.

    Args:
        a: First position (row, col).
        b: Second position (row, col).

    Returns:
        Manhattan distance.
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def get_neighbors(
    pos: Position,
    grid: np.ndarray,
    include_wait: bool = True,
) -> List[Position]:
    """Get valid neighboring positions on the grid.

    Args:
        pos: Current position (row, col).
        grid: Map grid (1=obstacle, 0=free).
        include_wait: If True, include staying in place.

    Returns:
        List of valid neighbor positions.
    """
    h, w = grid.shape
    r, c = pos
    neighbors = []
    if include_wait:
        neighbors.append((r, c))
    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == 0:
            neighbors.append((nr, nc))
    return neighbors


def cell_degree(pos: Position, grid: np.ndarray) -> int:
    """Compute the degree of a cell (number of free neighbors).

    Args:
        pos: Position (row, col).
        grid: Map grid.

    Returns:
        Number of free orthogonal neighbors.
    """
    return len(get_neighbors(pos, grid, include_wait=False))


def compute_path_cost(path: Path) -> int:
    """Compute the cost of a path (number of moves, excluding initial position).

    Args:
        path: List of positions.

    Returns:
        Path cost (length - 1, or 0 if empty/single).
    """
    if not path:
        return 0
    return len(path) - 1


def compute_makespan(paths: List[Path]) -> int:
    """Compute makespan (max path length) of a set of paths.

    Args:
        paths: List of agent paths.

    Returns:
        Maximum path length.
    """
    if not paths:
        return 0
    return max(len(p) for p in paths)


def compute_sum_of_costs(paths: List[Path]) -> int:
    """Compute sum of costs across all agent paths.

    Args:
        paths: List of agent paths.

    Returns:
        Total path cost.
    """
    return sum(len(p) for p in paths)


def detect_vertex_conflicts(paths: List[Path]) -> List[Tuple[int, int, int, Position]]:
    """Detect vertex conflicts in a set of paths.

    A vertex conflict occurs when two agents occupy the same cell at the
    same timestep.

    Args:
        paths: List of agent paths.

    Returns:
        List of (agent_i, agent_j, timestep, position) conflicts.
    """
    if not paths:
        return []

    max_t = max(len(p) for p in paths)
    conflicts = []

    for t in range(max_t):
        pos_agents: Dict[Position, List[int]] = {}
        for i, path in enumerate(paths):
            idx = min(t, len(path) - 1)
            pos = path[idx]
            if pos not in pos_agents:
                pos_agents[pos] = []
            pos_agents[pos].append(i)

        for pos, agents in pos_agents.items():
            if len(agents) > 1:
                for ai in range(len(agents)):
                    for aj in range(ai + 1, len(agents)):
                        conflicts.append((agents[ai], agents[aj], t, pos))
    return conflicts


def detect_edge_conflicts(
    paths: List[Path],
) -> List[Tuple[int, int, int, Position, Position]]:
    """Detect edge (swap) conflicts in a set of paths.

    An edge conflict occurs when two agents swap positions in one timestep.

    Args:
        paths: List of agent paths.

    Returns:
        List of (agent_i, agent_j, timestep, pos_i, pos_j) conflicts.
    """
    if not paths:
        return []

    max_t = max(len(p) for p in paths)
    conflicts = []

    for t in range(max_t - 1):
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                idx_i_t = min(t, len(paths[i]) - 1)
                idx_i_t1 = min(t + 1, len(paths[i]) - 1)
                idx_j_t = min(t, len(paths[j]) - 1)
                idx_j_t1 = min(t + 1, len(paths[j]) - 1)

                if (paths[i][idx_i_t] == paths[j][idx_j_t1] and
                        paths[j][idx_j_t] == paths[i][idx_i_t1] and
                        paths[i][idx_i_t] != paths[i][idx_i_t1]):
                    conflicts.append((
                        i, j, t,
                        paths[i][idx_i_t], paths[j][idx_j_t]
                    ))
    return conflicts


def has_any_conflict(paths: List[Path]) -> bool:
    """Quick check whether any conflict exists.

    Args:
        paths: List of agent paths.

    Returns:
        True if any vertex or edge conflict exists.
    """
    return bool(detect_vertex_conflicts(paths)) or bool(detect_edge_conflicts(paths))


def get_position_at_time(path: Path, t: int) -> Position:
    """Get the position of an agent at a given timestep.

    If timestep exceeds path length, the agent stays at its final position.

    Args:
        path: Agent's path.
        t: Timestep.

    Returns:
        Position at timestep t.
    """
    if not path:
        raise ValueError("Empty path")
    idx = min(t, len(path) - 1)
    return path[idx]


def build_reservation_table(
    paths: List[Path],
    exclude_agents: Optional[Set[int]] = None,
    max_time: Optional[int] = None,
) -> Set[TimedPosition]:
    """Build a space-time reservation table from agent paths.

    Args:
        paths: List of agent paths.
        exclude_agents: Agent indices to exclude from the table.
        max_time: Maximum timestep to reserve (extends final positions).

    Returns:
        Set of (row, col, timestep) reserved cells.
    """
    exclude = exclude_agents or set()
    reserved: Set[TimedPosition] = set()

    if not paths:
        return reserved

    actual_max_t = max(len(p) for p in paths) if paths else 0
    if max_time is None:
        max_time = actual_max_t + 50  # Buffer

    for i, path in enumerate(paths):
        if i in exclude:
            continue
        for t, pos in enumerate(path):
            reserved.add((pos[0], pos[1], t))
        # Reserve final position for remaining time
        if path:
            final = path[-1]
            for t in range(len(path), max_time):
                reserved.add((final[0], final[1], t))

    return reserved


def bfs_shortest_path(
    grid: np.ndarray,
    start: Position,
    goal: Position,
) -> Optional[Path]:
    """Compute shortest path using BFS (ignoring other agents).

    Args:
        grid: Map grid.
        start: Start position.
        goal: Goal position.

    Returns:
        Shortest path or None if unreachable.
    """
    if start == goal:
        return [start]

    h, w = grid.shape
    visited = {start}
    queue = [(start, [start])]
    head = 0

    while head < len(queue):
        pos, path = queue[head]
        head += 1

        for nr, nc in get_neighbors(pos, grid, include_wait=False):
            if (nr, nc) == goal:
                return path + [(nr, nc)]
            if (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append(((nr, nc), path + [(nr, nc)]))

    return None


def bfs_avoiding(
    grid: np.ndarray,
    start: Position,
    goal: Position,
    blocked: Set[Position],
) -> Optional[Path]:
    """Shortest path from start to goal treating `blocked` cells as obstacles.

    Used by runtime deadlock resolution to route an agent *around* the other
    agents currently participating in a deadlock, forcing a detour. The goal
    cell itself is never treated as blocked (an agent may need to reach a goal
    that momentarily sits next to a blocker).

    Args:
        grid: Map grid.
        start: Start position.
        goal: Goal position.
        blocked: Cells to avoid (typically the current cells of other
            deadlocked agents).

    Returns:
        Shortest detour path, or None if no such path exists.
    """
    if start == goal:
        return [start]
    block = set(blocked)
    block.discard(start)
    block.discard(goal)

    visited = {start}
    queue = [(start, [start])]
    head = 0
    while head < len(queue):
        pos, path = queue[head]
        head += 1
        for nb in get_neighbors(pos, grid, include_wait=False):
            if nb in block or nb in visited:
                continue
            if nb == goal:
                return path + [nb]
            visited.add(nb)
            queue.append((nb, path + [nb]))
    return None


def find_nearest_bypass(
    pos: Position,
    grid: np.ndarray,
    min_degree: int = 3,
    max_search: int = 50,
) -> Optional[Position]:
    """Find the nearest bypass cell (cell with degree >= min_degree).

    Uses BFS from the given position to find the closest cell with
    sufficient connectivity for an agent to step aside.

    Args:
        pos: Starting position.
        grid: Map grid.
        min_degree: Minimum number of free neighbors for bypass.
        max_search: Maximum cells to search.

    Returns:
        Nearest bypass position, or None if not found.
    """
    if cell_degree(pos, grid) >= min_degree:
        return pos

    visited = {pos}
    queue = [pos]
    head = 0
    searched = 0

    while head < len(queue) and searched < max_search:
        curr = queue[head]
        head += 1
        searched += 1

        for nb in get_neighbors(curr, grid, include_wait=False):
            if nb not in visited:
                visited.add(nb)
                if cell_degree(nb, grid) >= min_degree:
                    return nb
                queue.append(nb)

    return None
