"""
DRIMAPSim Grid World Engine

Core simulation grid with O(1) collision checking, edge-conflict
detection, and support for all four collision systems.
"""

import numpy as np
from copy import deepcopy
from typing import Dict, List, Optional, Set, Tuple

from sim.env_config import EnvConfig, MOVES
from sim.map_registry import generate_map


class GridWorld:
    """Core grid world engine.

    Manages the obstacle grid, agent positions, targets, and movement
    with proper collision detection.

    Attributes:
        config: Environment configuration.
        obstacles: Obstacle grid (1=obstacle, 0=free).
        positions_xy: Agent positions as [(r, c), ...].
        targets_xy: Target positions as [(r, c), ...].
        occupancy: 2D array tracking agent occupancy.
        is_active: Per-agent active flags.
    """

    def __init__(self, config: EnvConfig):
        """Initialize the grid world.

        Args:
            config: Environment configuration.
        """
        self.config = config
        self.rng = np.random.RandomState(config.seed)
        self._step_count = 0

        # Build obstacle grid
        if config.map is not None:
            self.obstacles = self._parse_map(config.map)
        elif config.map_type is not None:
            self.obstacles = generate_map(
                config.map_type, config.size, config.density, config.seed
            )
        else:
            self.obstacles = generate_map(
                "random", config.size, config.density, config.seed
            )

        self.h, self.w = self.obstacles.shape

        # Place agents and targets
        if config.agents_xy and config.targets_xy:
            self.positions_xy = [tuple(p) for p in config.agents_xy]
            self.targets_xy = [tuple(t) for t in config.targets_xy]
            self.config.num_agents = len(self.positions_xy)
        else:
            self.positions_xy, self.targets_xy = self._generate_positions()

        # Build occupancy map
        self.occupancy = np.zeros_like(self.obstacles, dtype=np.int32)
        for r, c in self.positions_xy:
            self.occupancy[r, c] = 1

        # Active tracking
        self.is_active: Dict[int, bool] = {
            i: True for i in range(self.config.num_agents)
        }

        # History recording
        self.history: List[List[Tuple[int, int]]] = [] if config.record_history else None

        # Deadlock tracking
        self._stagnation_counter: Dict[int, int] = {
            i: 0 for i in range(self.config.num_agents)
        }
        # Seed last-known positions with the start cells so stagnation is
        # measured relative to the true initial state. Leaving this empty
        # offsets detection by one step (the first call would always treat
        # every agent as having "moved").
        self._last_positions: Dict[int, Tuple[int, int]] = {
            i: self.positions_xy[i] for i in range(self.config.num_agents)
        }
        self.deadlock_events: List[dict] = []

    def _parse_map(self, map_data) -> np.ndarray:
        """Parse a map from string or list format."""
        if isinstance(map_data, str):
            rows = []
            for line in map_data.strip().split('\n'):
                line = line.strip()
                if not line:
                    continue
                row = []
                for ch in line:
                    row.append(0 if ch == '.' else 1)
                rows.append(row)
            return np.array(rows, dtype=np.int32)
        return np.array(map_data, dtype=np.int32)

    def _generate_positions(self) -> Tuple[
        List[Tuple[int, int]], List[Tuple[int, int]]
    ]:
        """Generate random start/goal positions on free cells."""
        free_cells = []
        for r in range(self.h):
            for c in range(self.w):
                if self.obstacles[r, c] == 0:
                    free_cells.append((r, c))

        n = self.config.num_agents
        if len(free_cells) < n * 2:
            n = len(free_cells) // 2
            self.config.num_agents = n

        if n == 0:
            return [], []

        indices = self.rng.choice(len(free_cells), size=n * 2, replace=False)
        starts = [free_cells[i] for i in indices[:n]]
        goals = [free_cells[i] for i in indices[n:]]
        return starts, goals

    def step(self, actions: List[int]) -> Dict[str, any]:
        """Execute one timestep of agent movements.

        Args:
            actions: List of action indices for each agent.
                0=idle, 1=up, 2=down, 3=left, 4=right.

        Returns:
            Dict with step results: on_goal, collisions, deadlock_agents.
        """
        assert len(actions) == self.config.num_agents
        self._step_count += 1

        # Capture pre-step positions and each agent's *intended* next cell
        # (the cell it tried to enter, regardless of whether the move is later
        # blocked). These drive the wait-for-graph deadlock monitor.
        current: Dict[int, Tuple[int, int]] = {
            i: self.positions_xy[i] for i in range(self.config.num_agents)
        }
        desired: Dict[int, Tuple[int, int]] = {}
        for i in range(self.config.num_agents):
            if not self.is_active[i]:
                desired[i] = self.positions_xy[i]
                continue
            dr, dc = MOVES[actions[i]]
            r, c = self.positions_xy[i]
            nr, nc = r + dr, c + dc
            desired[i] = (nr, nc) if self._is_valid_target(nr, nc) else (r, c)

        if self.config.collision_system == "priority":
            self._move_priority(actions)
        elif self.config.collision_system == "block_both":
            self._move_block_both(actions)
        elif self.config.collision_system == "soft":
            self._move_soft(actions)
        elif self.config.collision_system == "strict":
            self._move_strict(actions)

        # Update stagnation tracking
        deadlock_agents = set()
        if self.config.deadlock_tracking:
            deadlock_agents = self._update_stagnation()

        # Record history
        if self.history is not None:
            self.history.append(list(self.positions_xy))

        return {
            "on_goal": [
                self.positions_xy[i] == self.targets_xy[i]
                for i in range(self.config.num_agents)
            ],
            "deadlock_agents": deadlock_agents,
            "current": current,
            "desired": desired,
        }

    def _move_priority(self, actions):
        """Priority collision system — first-come-first-served."""
        for i in range(self.config.num_agents):
            if not self.is_active[i]:
                continue
            self._try_move(i, actions[i])

    def _move_block_both(self, actions):
        """Block-both collision system — colliding agents both stay.

        Implements the standard MAPF ``block_both`` movement model,
        mirroring ``ReactiveSimulator.arbitrate`` in ``src/execution.py`` so
        the simulation and offline tracks share identical execution
        semantics:

        * An agent that proposes to stay does not move.
        * If two or more active agents target the same cell, none of them
          take it (vertex conflict -> block both).
        * If two agents propose to swap cells, neither moves (edge conflict).
        * A move into a cell currently held by an agent is committed only if
          that occupant is *itself* vacating the cell this step. This is
          enforced by a fixpoint that repeatedly cancels any move whose
          target is held by a stationary agent; cancellations cascade so a
          following train advances together but stalls entirely if its head
          is blocked. The result NEVER places two agents on one cell and
          NEVER lets an agent move onto a stationary agent.
        """
        # Build the desired-next cell for every active agent. A cell that is
        # off-grid, an obstacle, or statically occupied by an *inactive*
        # agent is not even a candidate, so the agent proposes to wait.
        proposed: Dict[int, Tuple[int, int]] = {}
        active_ids = [
            i for i in range(self.config.num_agents) if self.is_active[i]
        ]
        for i in active_ids:
            dr, dc = MOVES[actions[i]]
            r, c = self.positions_xy[i]
            nr, nc = r + dr, c + dc
            if self._is_valid_target(nr, nc):
                proposed[i] = (nr, nc)
            else:
                proposed[i] = (r, c)

        current = {i: self.positions_xy[i] for i in active_ids}

        # Count how many active agents target each cell (excluding waiters).
        target_count: Dict[Tuple[int, int], int] = {}
        for i in active_ids:
            if proposed[i] != current[i]:
                target_count[proposed[i]] = target_count.get(proposed[i], 0) + 1

        # Provisional movers: clear vertex (no two agents target one cell) and
        # edge (no head-on swap) conflicts.
        move: Dict[int, Tuple[int, int]] = {}
        for i in active_ids:
            tgt = proposed[i]
            cur = current[i]
            if tgt == cur:
                continue  # waiting
            if target_count.get(tgt, 0) > 1:
                continue  # vertex conflict -> all contenders wait
            swap = any(
                j != i and current[j] == tgt and proposed[j] == cur
                for j in active_ids
            )
            if swap:
                continue
            move[i] = tgt

        # Fixpoint: an agent may enter a cell only if its current occupant is
        # itself vacating this step. Repeatedly cancel any move whose target
        # is held by an active agent that is staying put; cancellations
        # cascade (a train advances together, but stalls entirely if its head
        # is blocked).
        changed = True
        while changed:
            changed = False
            stayer_cells = {current[j] for j in active_ids if j not in move}
            for i in list(move.keys()):
                if move[i] in stayer_cells:
                    del move[i]
                    changed = True

        # Commit moves. Because a train advances simultaneously, vacate all
        # source cells before claiming targets to keep the occupancy grid
        # consistent (a follower's target is the leader's just-freed cell).
        for i in move:
            r, c = current[i]
            self.occupancy[r, c] = 0
        for i, (nr, nc) in move.items():
            self.positions_xy[i] = (nr, nc)
            self.occupancy[nr, nc] = 1

    def _move_soft(self, actions):
        """Soft collision system — collisions are permitted, not prevented.

        Agents move independently into any free (non-obstacle, in-bounds)
        cell regardless of other agents. As a result, a single cell may end
        up holding more than one agent: ``occupancy`` is an integer count and
        may exceed 1. These overlaps are *tracked* via the occupancy grid but
        are NOT prevented here and are NOT penalised in the reward — the
        reward logic in ``DRIMAPSimEnv`` only rewards goal arrival and does
        not deduct anything for overlaps unless a caller explicitly adds such
        a penalty. Use ``block_both`` or ``strict`` for collision-free
        execution.
        """
        for i in range(self.config.num_agents):
            if not self.is_active[i]:
                continue
            dr, dc = MOVES[actions[i]]
            r, c = self.positions_xy[i]
            nr, nc = r + dr, c + dc
            if self._is_valid_soft(nr, nc):
                self.occupancy[r, c] = max(0, self.occupancy[r, c] - 1)
                self.positions_xy[i] = (nr, nc)
                self.occupancy[nr, nc] += 1

    def _move_strict(self, actions):
        """Strict collision system — collisions are forbidden."""
        self._move_block_both(actions)

    def _try_move(self, agent_id: int, action: int):
        """Try to move an agent, respecting obstacles and occupancy."""
        dr, dc = MOVES[action]
        r, c = self.positions_xy[agent_id]
        nr, nc = r + dr, c + dc

        if (0 <= nr < self.h and 0 <= nc < self.w and
                self.obstacles[nr, nc] == 0 and
                self.occupancy[nr, nc] == 0):
            self._execute_move(agent_id, (nr, nc))

    def _execute_move(self, agent_id: int, new_pos: Tuple[int, int]):
        """Execute a validated move."""
        old_r, old_c = self.positions_xy[agent_id]
        nr, nc = new_pos
        self.occupancy[old_r, old_c] = 0
        self.positions_xy[agent_id] = (nr, nc)
        self.occupancy[nr, nc] = 1

    def _is_valid(self, r: int, c: int) -> bool:
        """Check if a cell is valid for movement (free and unoccupied)."""
        return (0 <= r < self.h and 0 <= c < self.w and
                self.obstacles[r, c] == 0 and
                self.occupancy[r, c] == 0)

    def _is_valid_target(self, r: int, c: int) -> bool:
        """Check if a cell is a valid movement *target* under block_both.

        Unlike :meth:`_is_valid`, this does NOT reject currently-occupied
        cells: under the block_both model an agent may follow another into a
        cell that the occupant vacates the same step. Cell occupancy is
        instead resolved by the arbitration fixpoint in
        :meth:`_move_block_both`. Only the static grid (bounds + obstacles)
        is checked here. Inactive (finished) agents leave no occupancy mark,
        so their cells are correctly treated as free.
        """
        return (0 <= r < self.h and 0 <= c < self.w and
                self.obstacles[r, c] == 0)

    def _is_valid_soft(self, r: int, c: int) -> bool:
        """Check if a cell is valid in soft collision mode."""
        return (0 <= r < self.h and 0 <= c < self.w and
                self.obstacles[r, c] == 0)

    def _update_stagnation(self) -> Set[int]:
        """Update stagnation counters and detect stagnation-based deadlocks."""
        deadlock_agents = set()
        for i in range(self.config.num_agents):
            if not self.is_active[i]:
                continue
            pos = self.positions_xy[i]
            if self._last_positions.get(i) == pos:
                self._stagnation_counter[i] += 1
            else:
                self._stagnation_counter[i] = 0

            if self._stagnation_counter[i] >= 5:
                deadlock_agents.add(i)

            self._last_positions[i] = pos

        if len(deadlock_agents) >= 2:
            self.deadlock_events.append({
                "step": self._step_count,
                "agents": list(deadlock_agents),
            })

        return deadlock_agents

    def get_deadlock_events(self) -> List[dict]:
        """Return the recorded deadlock events.

        Each event is a dict ``{"step": int, "agents": List[int]}`` logged
        whenever two or more agents are simultaneously detected as
        stagnation-deadlocked. Exposed so callers (the environment ``info``
        dict, metrics, cross-validation scripts) can actually consume the
        signal rather than it being recorded and discarded.

        Returns:
            A shallow copy of the deadlock-event list.
        """
        return list(self.deadlock_events)

    def deadlock_event_count(self) -> int:
        """Number of deadlock events recorded so far."""
        return len(self.deadlock_events)

    def on_goal(self, agent_id: int) -> bool:
        """Check if an agent is on its goal."""
        return self.positions_xy[agent_id] == self.targets_xy[agent_id]

    def hide_agent(self, agent_id: int):
        """Remove an agent from the grid (reached goal in finish mode)."""
        if not self.is_active[agent_id]:
            return
        self.is_active[agent_id] = False
        r, c = self.positions_xy[agent_id]
        self.occupancy[r, c] = 0

    def get_obstacles_for_agent(self, agent_id: int) -> np.ndarray:
        """Get the obstacle observation window for an agent."""
        r, c = self.positions_xy[agent_id]
        rad = self.config.obs_radius
        return self._get_window(self.obstacles, r, c, rad)

    def get_agents_for_agent(self, agent_id: int) -> np.ndarray:
        """Get the agent occupancy observation window."""
        r, c = self.positions_xy[agent_id]
        rad = self.config.obs_radius
        return self._get_window(self.occupancy, r, c, rad)

    def get_target_channel(self, agent_id: int) -> np.ndarray:
        """Get the target observation channel — marks the relative target."""
        rad = self.config.obs_radius
        full = rad * 2 + 1
        target = np.zeros((full, full), dtype=np.float32)

        ar, ac = self.positions_xy[agent_id]
        tr, tc = self.targets_xy[agent_id]
        dr, dc = tr - ar, tc - ac

        dr = max(-rad, min(rad, dr))
        dc = max(-rad, min(rad, dc))
        target[rad + dr, rad + dc] = 1.0
        return target

    def _get_window(self, grid: np.ndarray, r: int, c: int,
                    radius: int) -> np.ndarray:
        """Extract a square window from the grid, padding with 1s."""
        full = radius * 2 + 1
        window = np.ones((full, full), dtype=np.float32)  # Pad with obstacles

        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                gr, gc = r + dr, c + dc
                if 0 <= gr < self.h and 0 <= gc < self.w:
                    window[radius + dr, radius + dc] = grid[gr, gc]

        return window

    def get_obstacles(self) -> np.ndarray:
        """Return a copy of the obstacle grid."""
        return self.obstacles.copy()

    def get_agents_xy(self) -> List[Tuple[int, int]]:
        """Return current agent positions."""
        return list(self.positions_xy)

    def get_targets_xy(self) -> List[Tuple[int, int]]:
        """Return target positions."""
        return list(self.targets_xy)

    def generate_new_target(self, agent_id: int):
        """Generate a new random target for an agent (lifelong mode)."""
        free_cells = []
        for r in range(self.h):
            for c in range(self.w):
                if (self.obstacles[r, c] == 0 and
                        (r, c) != self.positions_xy[agent_id]):
                    free_cells.append((r, c))
        if free_cells:
            idx = self.rng.randint(len(free_cells))
            self.targets_xy[agent_id] = free_cells[idx]

    def render_ascii(self) -> str:
        """Render the grid as ASCII art."""
        lines = []
        agent_positions = {pos: i for i, pos in enumerate(self.positions_xy)
                          if self.is_active.get(i, False)}
        target_positions = {pos: i for i, pos in enumerate(self.targets_xy)}

        for r in range(self.h):
            row = []
            for c in range(self.w):
                if (r, c) in agent_positions:
                    aid = agent_positions[(r, c)]
                    if (r, c) == self.targets_xy[aid]:
                        row.append('★')  # On goal
                    else:
                        row.append(str(aid % 10))
                elif (r, c) in target_positions:
                    row.append('◎')
                elif self.obstacles[r, c] == 1:
                    row.append('█')
                else:
                    row.append('·')
            lines.append(' '.join(row))
        return '\n'.join(lines)
