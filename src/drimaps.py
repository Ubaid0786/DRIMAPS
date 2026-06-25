#!/usr/bin/env python3
"""
DRIMAPS — Deadlock-Resilient Intelligent Multi-Agent Path-finding System

Main algorithm class implementing the Runtime Adaptive Dependency
Resolution (RADR) framework. The algorithm has seven phases:

  Phase 1: Initial Path Computation  (pluggable solver)
  Phase 2: Runtime WFG Maintenance   (incremental dependency graph)
  Phase 3: Incremental Cycle Detection (Tarjan-based SCC)
  Phase 4: Deadlock Classification    (type-specific categorization)
  Phase 5: Type-Specific Resolution   (minimal-disruption repair)
  Phase 6: Safety Verification        (post-resolution checks)
  Phase 7: Progress Guarantee         (termination assurance)

The algorithm separates initial planning from deadlock resilience,
allowing any fast initial solver to be used. The contribution is the
online deadlock detection + classification + resolution layer.
"""

import time
import traceback
import resource
import numpy as np
from typing import Dict, List, Optional

from src.config import (
    DRIMAPSConfig,
    DRIMAPSResult,
)
from src.utils import (
    Position,
    Path,
    compute_makespan,
    compute_sum_of_costs,
    detect_vertex_conflicts,
    detect_edge_conflicts,
    bfs_shortest_path,
)
from src.reactive_core import ReactiveCore


class DRIMAPS:
    """Deadlock-Resilient Inheritance for Multi-Agent Path-finding Systems.

    DRIMAPS executes a MAPF instance with a deadlock-free reactive core
    (priority inheritance with backtracking) and closes the residual
    completeness gap with a novel detection-guided layer: a wait-for graph
    is maintained each step, Tarjan's SCC extracts wait cycles, persistent
    cycles are classified by structural type, and the exact agents involved
    receive a short-lived scatter target (targeted minimal-disruption escape).
    The heavy lifting lives in :class:`src.reactive_core.ReactiveCore`; this
    class adapts it to the :class:`DRIMAPSResult` reporting interface.

    Usage:
        solver = DRIMAPS(DRIMAPSConfig())
        result = solver.solve(grid, starts, goals)
        if result.success:
            print(f"Makespan: {result.makespan}")

    Attributes:
        config: Algorithm configuration.
        grid: Map grid (set during solve).
    """

    def __init__(self, config: Optional[DRIMAPSConfig] = None) -> None:
        """Initialize DRIMAPS with configuration.

        Args:
            config: Algorithm configuration. Uses defaults if None.
        """
        self.config = config or DRIMAPSConfig()
        self.grid: Optional[np.ndarray] = None

    def solve(
        self,
        grid: np.ndarray,
        starts: List[Position],
        goals: List[Position],
    ) -> DRIMAPSResult:
        """Solve a MAPF instance with deadlock resilience.

        End-to-end pipeline:
            1. Compute initial paths with the pluggable solver.
            2. Simulate execution with runtime deadlock detection
               and resolution.
            3. Return the final conflict-free paths.

        Args:
            grid: Map grid (1=obstacle, 0=free). Shape (height, width).
            starts: List of (row, col) start positions, one per agent.
            goals: List of (row, col) goal positions, one per agent.

        Returns:
            DRIMAPSResult with paths, metrics, and diagnostics.
        """
        result = DRIMAPSResult()
        start_time = time.time()

        self.grid = grid
        n = len(starts)

        try:
            # --- Initial reference plan (conflict-oblivious shortest paths) ---
            # Used only for the initial sum-of-costs reference and detour
            # baselines; the reactive core does its own per-step planning.
            t0 = time.time()
            initial_paths = self._independent_plans(grid, starts, goals)
            result.initial_planning_time = time.time() - t0
            if initial_paths is not None and len(initial_paths) == n:
                result.initial_sum_of_costs = compute_sum_of_costs(initial_paths)

            # --- Reactive execution: deadlock-free PIBT core + WFG-guided
            #     targeted escape (Sections on detection/classification/escape) ---
            core = ReactiveCore(self.config)
            seed = self.config.seeds[0] if self.config.seeds else 42
            deadline = start_time + self.config.timeout * 0.95
            final_paths, st = core.run(
                grid, starts, goals, seed=seed, deadline=deadline
            )

            result.paths = final_paths
            result.makespan = compute_makespan(final_paths)
            result.sum_of_costs = compute_sum_of_costs(final_paths)
            result.deadlocks_detected = st.deadlocks_detected
            result.deadlocks_resolved = st.deadlocks_resolved
            result.agents_replanned = st.agents_replanned
            result.detection_time = st.detection_time
            result.resolution_time = st.resolution_time
            result.wfg_update_time = st.wfg_update_time

            # Check if all agents reached goals
            all_reached = True
            for i in range(n):
                final_pos = final_paths[i][-1] if final_paths[i] else starts[i]
                if final_pos != goals[i]:
                    all_reached = False
                    break
            result.success = all_reached

            # Count remaining conflicts (collision-free by construction).
            v_conf = detect_vertex_conflicts(final_paths)
            e_conf = detect_edge_conflicts(final_paths)
            result.collision_count = len(v_conf) + len(e_conf)

        except Exception as e:
            if self.config.verbose:
                traceback.print_exc()
            result.success = False

        result.runtime = time.time() - start_time

        # Peak memory
        try:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            result.peak_memory_mb = usage.ru_maxrss / 1024.0  # KB to MB
        except Exception:
            result.peak_memory_mb = 0.0

        return result

    # ==================================================================
    # Phase 1: Initial Path Computation
    # ==================================================================

    def _independent_plans(
        self,
        grid: np.ndarray,
        starts: List[Position],
        goals: List[Position],
    ) -> Optional[List[Path]]:
        """Plan each agent's shortest path independently (ignoring others).

        This is the fast, conflict-oblivious planner that produces the
        execution-time deadlocks DRIMAPS is designed to resolve. Each route is
        a single-source BFS shortest path on the static grid.

        Args:
            grid: Map grid.
            starts: Start positions.
            goals: Goal positions.

        Returns:
            List of routes (one per agent); an agent with no route waits at its
            start. Returns None only if the agent count is inconsistent.
        """
        from src.utils import bfs_shortest_path

        n = len(starts)
        paths: List[Path] = []
        for i in range(n):
            route = bfs_shortest_path(grid, starts[i], goals[i])
            paths.append(route if route else [starts[i]])
        return paths if len(paths) == n else None

