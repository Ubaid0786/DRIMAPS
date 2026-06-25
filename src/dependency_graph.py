#!/usr/bin/env python3
"""
Wait-For Graph (WFG) — Runtime Dependency Graph

Maintains a directed graph where edge (a_i → a_j) means agent i is blocked
by agent j. Supports O(1) edge insertion/deletion via adjacency lists with
hash-set backed neighbor storage, and incremental updates (not rebuilt
from scratch each timestep).
"""

from typing import Dict, List, Set, Tuple, Optional
from src.utils import Position


class WaitForGraph:
    """Incremental Wait-For Graph for runtime dependency tracking.

    The WFG is a directed graph over agent indices. An edge i → j means
    agent i cannot make its planned move because agent j currently occupies
    or is moving to the cell agent i needs.

    Design decisions:
        - Adjacency list with set-valued neighbors gives O(1) amortized
          insertion and deletion.
        - Reverse adjacency (predecessors) maintained in parallel for
          efficient queries (e.g., "who is waiting for agent j?").
        - Edge metadata tracks the blocking position for diagnostics.

    Attributes:
        _adj: Forward adjacency: agent → set of agents it waits for.
        _pred: Reverse adjacency: agent → set of agents waiting for it.
        _edge_meta: Metadata per edge (blocker, position info).
        _num_edges: Current edge count.
    """

    def __init__(self, num_agents: int) -> None:
        """Initialize an empty WFG.

        Args:
            num_agents: Total number of agents (nodes 0..n-1).
        """
        self.num_agents = num_agents
        self._adj: Dict[int, Set[int]] = {i: set() for i in range(num_agents)}
        self._pred: Dict[int, Set[int]] = {i: set() for i in range(num_agents)}
        self._edge_meta: Dict[Tuple[int, int], dict] = {}
        self._num_edges: int = 0

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def add_edge(
        self,
        waiter: int,
        blocker: int,
        position: Optional[Position] = None,
    ) -> bool:
        """Add a dependency edge: waiter is blocked by blocker.

        Args:
            waiter: Agent index that is waiting.
            blocker: Agent index that causes the block.
            position: The contested cell (for diagnostics).

        Returns:
            True if the edge was newly added, False if it already existed.
        """
        if blocker in self._adj[waiter]:
            return False  # Edge already present
        self._adj[waiter].add(blocker)
        self._pred[blocker].add(waiter)
        self._edge_meta[(waiter, blocker)] = {"position": position}
        self._num_edges += 1
        return True

    def remove_edge(self, waiter: int, blocker: int) -> bool:
        """Remove a dependency edge.

        Args:
            waiter: Agent that was waiting.
            blocker: Agent that was blocking.

        Returns:
            True if the edge existed and was removed.
        """
        if blocker not in self._adj[waiter]:
            return False
        self._adj[waiter].discard(blocker)
        self._pred[blocker].discard(waiter)
        self._edge_meta.pop((waiter, blocker), None)
        self._num_edges -= 1
        return True

    def clear_agent(self, agent: int) -> None:
        """Remove all edges involving the given agent (both directions).

        Useful when an agent reaches its goal and is no longer relevant.

        Args:
            agent: Agent index to clear.
        """
        # Remove outgoing edges
        for blocker in list(self._adj[agent]):
            self.remove_edge(agent, blocker)
        # Remove incoming edges
        for waiter in list(self._pred[agent]):
            self.remove_edge(waiter, agent)

    def clear_outgoing(self, agent: int) -> None:
        """Remove all outgoing edges from an agent.

        Called at each timestep before recomputing who this agent waits for.

        Args:
            agent: Agent index.
        """
        for blocker in list(self._adj[agent]):
            self.remove_edge(agent, blocker)

    def clear_all(self) -> None:
        """Remove all edges from the graph."""
        for i in range(self.num_agents):
            self._adj[i].clear()
            self._pred[i].clear()
        self._edge_meta.clear()
        self._num_edges = 0

    # ------------------------------------------------------------------
    # Incremental Update
    # ------------------------------------------------------------------

    def update_for_agent(
        self,
        agent: int,
        current_positions: Dict[int, Position],
        next_positions: Dict[int, Position],
        goals: List[Position],
        finished: Set[int],
    ) -> None:
        """Incrementally update edges for a single agent.

        Clears old outgoing edges for this agent, then recomputes who
        it is currently blocked by. An agent i is blocked by agent j if:
          1. Agent i's next planned cell == agent j's current cell, OR
          2. Agent i's next planned cell == agent j's next cell (vertex
             conflict at next step), OR
          3. Agent j is at agent i's goal position (goal-blocking).

        Agents in `finished` (already at goal) are skipped as waiters
        but can still act as blockers.

        Args:
            agent: Agent to update.
            current_positions: Current position of every agent.
            next_positions: Planned next position of every agent.
            goals: Goal position for every agent.
            finished: Set of agents already at their goals.
        """
        if agent in finished:
            self.clear_outgoing(agent)
            return

        self.clear_outgoing(agent)

        my_curr = current_positions[agent]
        my_next = next_positions[agent]

        # Only check for blocks if the agent is trying to move or is stuck
        # (my_next != goal means it hasn't arrived yet)
        if agent < len(goals) and my_curr == goals[agent]:
            return  # Already at goal, no dependencies

        for other in range(self.num_agents):
            if other == agent or other in finished:
                continue

            other_curr = current_positions[other]
            other_next = next_positions[other]

            blocked = False
            contested_pos = None

            # Case 1: Next cell occupied by other's current position
            if my_next == other_curr and my_next != my_curr:
                blocked = True
                contested_pos = my_next

            # Case 2: Vertex conflict at next step
            if my_next == other_next and my_next != my_curr:
                blocked = True
                contested_pos = my_next

            # Case 3: Agent is stuck (wants to move but can't) and other
            # is at a position on agent's path toward goal
            if my_curr == my_next and my_curr != goals[agent]:
                if other_curr == my_curr:
                    pass  # Same cell — already a conflict
                # Check if other blocks the immediate desired direction
                if agent < len(goals):
                    goal = goals[agent]
                    # Simple heuristic: other is adjacent and toward goal
                    if other_curr == goal:
                        blocked = True
                        contested_pos = other_curr

            # Case 4: Goal-blocking — other sits on agent's goal
            if agent < len(goals) and other_curr == goals[agent]:
                if other not in finished or other_curr != goals[other]:
                    blocked = True
                    contested_pos = goals[agent]

            if blocked:
                self.add_edge(agent, other, position=contested_pos)

    def update_all(
        self,
        current_positions: Dict[int, Position],
        next_positions: Dict[int, Position],
        goals: List[Position],
        finished: Set[int],
    ) -> None:
        """Rebuild the entire WFG from current state.

        This is a full rebuild, used at initialization. After the first
        call, prefer incremental updates via `update_for_agent`.

        Args:
            current_positions: Current position of every agent.
            next_positions: Planned next position of every agent.
            goals: Goal positions.
            finished: Agents already at their goals.
        """
        self.clear_all()
        for agent in range(self.num_agents):
            self.update_for_agent(
                agent, current_positions, next_positions, goals, finished
            )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def successors(self, agent: int) -> Set[int]:
        """Agents that `agent` is waiting for (outgoing edges).

        Args:
            agent: Agent index.

        Returns:
            Set of blocker agent indices.
        """
        return self._adj[agent]

    def predecessors(self, agent: int) -> Set[int]:
        """Agents that are waiting for `agent` (incoming edges).

        Args:
            agent: Agent index.

        Returns:
            Set of waiter agent indices.
        """
        return self._pred[agent]

    def has_edge(self, waiter: int, blocker: int) -> bool:
        """Check if a specific dependency edge exists.

        Args:
            waiter: Potential waiter.
            blocker: Potential blocker.

        Returns:
            True if waiter → blocker edge exists.
        """
        return blocker in self._adj[waiter]

    def edge_count(self) -> int:
        """Total number of edges in the WFG.

        Returns:
            Edge count.
        """
        return self._num_edges

    def node_count(self) -> int:
        """Total number of nodes (agents).

        Returns:
            Number of agents.
        """
        return self.num_agents

    def active_nodes(self, finished: Set[int]) -> Set[int]:
        """Nodes that have outgoing or incoming edges and are not finished.

        Args:
            finished: Agents at their goals.

        Returns:
            Set of active (participating) agent indices.
        """
        active = set()
        for i in range(self.num_agents):
            if i in finished:
                continue
            if self._adj[i] or self._pred[i]:
                active.add(i)
        return active

    def get_adjacency_list(self) -> Dict[int, Set[int]]:
        """Return a copy of the forward adjacency list.

        Returns:
            Dict mapping each agent to its set of blockers.
        """
        return {k: set(v) for k, v in self._adj.items()}

    def __repr__(self) -> str:
        edges = []
        for i in range(self.num_agents):
            for j in self._adj[i]:
                edges.append(f"{i}→{j}")
        return f"WFG({self.num_agents} agents, {self._num_edges} edges: [{', '.join(edges)}])"
