#!/usr/bin/env python3
"""
Incremental Cycle Detector — Tarjan-based SCC Detection

Detects strongly connected components (SCCs) of size ≥ 2 in the Wait-For
Graph. Each such SCC constitutes a deadlock: a set of agents in a mutual
blocking relationship from which no agent can make progress without
external intervention.

Runs in O(V + E) via Tarjan's algorithm applied to the current snapshot
of the WFG.
"""

from typing import Dict, List, Set, Tuple

from src.dependency_graph import WaitForGraph


class CycleDetector:
    """Tarjan-based cycle (SCC) detector for the Wait-For Graph.

    At each detection call, runs Tarjan's SCC algorithm on the WFG
    and returns all SCCs of size ≥ 2 (these are the deadlock cycles).

    For small cycles (size 2), the SCC is exactly the cycle. For larger
    SCCs, we also extract a specific cycle path via DFS for use in
    resolution ordering.

    Complexity:
        Time: O(V + E) per call, where V = number of agents and
              E = number of dependency edges.
        Space: O(V) for the Tarjan state arrays.
    """

    def __init__(self) -> None:
        """Initialize the detector (stateless between calls)."""
        pass

    def detect_cycles(
        self,
        wfg: WaitForGraph,
        finished: Set[int] = None,
    ) -> List[List[int]]:
        """Detect all deadlock cycles in the WFG.

        Runs Tarjan's SCC algorithm and returns SCCs with ≥ 2 nodes.
        Finished agents are excluded from consideration.

        Args:
            wfg: Current Wait-For Graph.
            finished: Set of agents that have reached their goals.

        Returns:
            List of cycles, each cycle being a list of agent indices.
            For size-2 SCCs, returns the pair. For larger SCCs, returns
            a specific directed cycle extracted via DFS.
        """
        if finished is None:
            finished = set()

        adj = wfg.get_adjacency_list()

        # Filter out finished agents
        active_agents = [
            i for i in range(wfg.num_agents) if i not in finished
        ]

        # Tarjan's SCC algorithm
        sccs = self._tarjan_scc(adj, active_agents)

        # Filter to SCCs of size >= 2 (deadlocks)
        deadlock_sccs = [scc for scc in sccs if len(scc) >= 2]

        # For each SCC, extract a specific directed cycle
        cycles = []
        for scc in deadlock_sccs:
            cycle = self._extract_cycle_from_scc(adj, scc)
            if cycle and len(cycle) >= 2:
                cycles.append(cycle)

        return cycles

    def _tarjan_scc(
        self,
        adj: Dict[int, Set[int]],
        nodes: List[int],
    ) -> List[List[int]]:
        """Tarjan's SCC algorithm.

        Args:
            adj: Forward adjacency list.
            nodes: List of node indices to consider.

        Returns:
            List of SCCs, each as a list of node indices.
        """
        index_counter = [0]
        stack: List[int] = []
        on_stack: Set[int] = set()
        lowlink: Dict[int, int] = {}
        index: Dict[int, int] = {}
        sccs: List[List[int]] = []
        node_set = set(nodes)

        def strongconnect(v: int) -> None:
            index[v] = index_counter[0]
            lowlink[v] = index_counter[0]
            index_counter[0] += 1
            stack.append(v)
            on_stack.add(v)

            for w in adj.get(v, set()):
                if w not in node_set:
                    continue
                if w not in index:
                    strongconnect(w)
                    lowlink[v] = min(lowlink[v], lowlink[w])
                elif w in on_stack:
                    lowlink[v] = min(lowlink[v], index[w])

            # If v is a root node, pop the SCC
            if lowlink[v] == index[v]:
                scc = []
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    scc.append(w)
                    if w == v:
                        break
                sccs.append(scc)

        for v in nodes:
            if v not in index:
                strongconnect(v)

        return sccs

    def _extract_cycle_from_scc(
        self,
        adj: Dict[int, Set[int]],
        scc: List[int],
    ) -> List[int]:
        """Extract a specific directed cycle from an SCC via DFS.

        For a size-2 SCC {a, b}, the cycle is simply [a, b] if a→b
        and b→a.

        For larger SCCs, uses DFS to find a back-edge and extracts the
        cycle path.

        Args:
            adj: Forward adjacency list.
            scc: List of agent indices forming the SCC.

        Returns:
            Ordered list of agent indices forming a directed cycle.
        """
        scc_set = set(scc)

        if len(scc) == 2:
            a, b = scc
            if b in adj.get(a, set()) and a in adj.get(b, set()):
                return [a, b]
            # One-directional within SCC — shouldn't happen for a real SCC
            # of size 2, but handle gracefully
            if b in adj.get(a, set()):
                return [a, b]
            return [b, a]

        # DFS to find a cycle
        visited: Set[int] = set()
        rec_stack: Dict[int, int] = {}  # node → position in path
        path: List[int] = []

        def dfs(node: int) -> List[int]:
            visited.add(node)
            rec_stack[node] = len(path)
            path.append(node)

            for neighbor in adj.get(node, set()):
                if neighbor not in scc_set:
                    continue
                if neighbor not in visited:
                    result = dfs(neighbor)
                    if result:
                        return result
                elif neighbor in rec_stack:
                    # Found cycle: extract from rec_stack position
                    cycle_start = rec_stack[neighbor]
                    return path[cycle_start:]

            path.pop()
            del rec_stack[node]
            return []

        for start in scc:
            if start not in visited:
                cycle = dfs(start)
                if cycle:
                    return cycle

        # Fallback: return the SCC itself
        return scc

    def has_deadlock(
        self,
        wfg: WaitForGraph,
        finished: Set[int] = None,
    ) -> bool:
        """Quick check: does any deadlock exist?

        Args:
            wfg: Current Wait-For Graph.
            finished: Agents at their goals.

        Returns:
            True if at least one deadlock cycle exists.
        """
        return len(self.detect_cycles(wfg, finished)) > 0
