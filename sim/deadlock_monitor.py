"""
DRIMAPSim Deadlock Monitor
==========================

First-class, execution-time deadlock instrumentation for the DRIMAPSim
environment. Unlike the stagnation heuristic (an agent that simply has not
moved for *k* steps), this monitor detects *genuine* deadlocks: cyclic
wait-for dependencies among agents that are mutually blocking one another.

How it works each timestep
--------------------------
1. From the agents' *intended* next cells (the cell each agent tried to move
   into this step) and their current cells, it builds a **wait-for graph**
   (WFG): an edge ``i -> j`` means agent *i* wanted the cell agent *j* occupies
   or is also entering.
2. It runs **Tarjan's strongly-connected-component algorithm** over the WFG to
   find wait cycles of size >= 2.
3. A cycle is reported as a deadlock only once its agents have been **stalled**
   for ``stagnation_threshold`` steps, which filters transient contention that
   resolves on its own.
4. Each confirmed deadlock is **classified** into one of four structural
   categories (corridor / cyclic / congestion / goal-blocking).

This reuses the same validated WFG, cycle detector, and classifier that the
DRIMAPS *algorithm* uses, so the environment's deadlock labels are consistent
with the solver's. To our knowledge no existing MAPF environment (e.g. POGEMA)
ships WFG-based deadlock detection and a structural taxonomy as built-in
metrics; that instrumentation is what DRIMAPSim adds. Stagnation counts are
still exposed alongside, so the two signals can be compared directly.
"""

from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from src.config import DRIMAPSConfig, DeadlockType
from src.dependency_graph import WaitForGraph
from src.cycle_detector import CycleDetector
from src.deadlock_classifier import DeadlockClassifier

Cell = Tuple[int, int]


class DeadlockMonitor:
    """Runtime wait-for-graph deadlock detector and classifier for the env.

    Attributes:
        num_agents: Number of agents.
        stagnation_threshold: Steps of no movement before a wait cycle is
            confirmed as a deadlock (filters transient contention).
        deadlock_count: Cumulative count of distinct confirmed deadlock events.
        type_counts: Cumulative count of confirmed deadlocks per structural type.
        congestion: Float grid accumulating per-cell contention (how often a
            cell was simultaneously contended by two or more agents).
    """

    def __init__(
        self,
        num_agents: int,
        obstacles: np.ndarray,
        goals: List[Cell],
        stagnation_threshold: int = 5,
        config: Optional[DRIMAPSConfig] = None,
    ) -> None:
        self.num_agents = num_agents
        self.obstacles = obstacles
        self.goals = list(goals)
        self.stagnation_threshold = stagnation_threshold
        cfg = config or DRIMAPSConfig()
        cfg.stagnation_threshold = stagnation_threshold

        self._wfg = WaitForGraph(num_agents)
        self._detector = CycleDetector()
        self._classifier = DeadlockClassifier(obstacles, cfg)

        self._stag: Dict[int, int] = {i: 0 for i in range(num_agents)}
        self._last_pos: Dict[int, Cell] = {}
        self._active_sigs: Set[frozenset] = set()

        self.deadlock_count = 0
        self.type_counts: Dict[str, int] = {t.value: 0 for t in DeadlockType}
        self.congestion = np.zeros_like(obstacles, dtype=np.float64)
        self._per_step: List[int] = []  # distinct deadlocks confirmed per step

    def update(
        self,
        current: Dict[int, Cell],
        desired: Dict[int, Cell],
        finished: Set[int],
    ) -> dict:
        """Process one timestep and return the live deadlock state.

        Args:
            current: Each agent's cell *before* this step's movement.
            desired: Each agent's intended next cell this step (== current if
                it tried to stay or was blocked from proposing a move).
            finished: Agents that have reached their goals (excluded as waiters).

        Returns:
            Dict with keys:
                ``confirmed``    -> list of agent-lists, one per confirmed deadlock
                ``types``        -> {agent_tuple: type_str} for confirmed deadlocks
                ``agents``       -> set of agents currently in a confirmed deadlock
                ``new_count``    -> number of newly confirmed deadlocks this step
                ``wfg_edges``    -> current WFG edge count (diagnostic)
        """
        # --- stagnation bookkeeping (did each agent move?) ---
        for i in range(self.num_agents):
            if i in finished:
                self._stag[i] = 0
            elif self._last_pos.get(i) == current[i]:
                self._stag[i] = self._stag.get(i, 0) + 1
            else:
                self._stag[i] = 0
        self._last_pos = dict(current)

        # --- congestion: cells contended by >=2 agents this step ---
        contenders: Dict[Cell, int] = {}
        for i in range(self.num_agents):
            if i in finished:
                continue
            tgt = desired.get(i, current[i])
            if tgt != current[i]:
                contenders[tgt] = contenders.get(tgt, 0) + 1
        for cell, k in contenders.items():
            if k >= 2:
                self.congestion[cell] += k

        # --- build WFG from current + desired, detect cycles ---
        self._wfg.update_all(current, desired, self.goals, finished)
        cycles = self._detector.detect_cycles(self._wfg, finished)

        # --- confirm persistence, dedupe, classify ---
        confirmed: List[List[int]] = []
        types: Dict[tuple, str] = {}
        agents_in_dl: Set[int] = set()
        live_sigs: Set[frozenset] = set()
        new_count = 0

        for cycle in cycles:
            stuck = [
                a for a in cycle
                if a not in finished
                and self._stag.get(a, 0) >= self.stagnation_threshold
            ]
            if len(stuck) < 2:
                continue
            sig = frozenset(stuck)
            live_sigs.add(sig)
            confirmed.append(stuck)
            agents_in_dl.update(stuck)
            dl_type = self._classifier.classify(
                stuck, current, self.goals, finished
            )
            types[tuple(sorted(stuck))] = dl_type.value
            if sig not in self._active_sigs:
                # Newly formed deadlock (count each distinct episode once).
                self.deadlock_count += 1
                self.type_counts[dl_type.value] += 1
                new_count += 1

        self._active_sigs = live_sigs
        self._per_step.append(new_count)
        return {
            "confirmed": confirmed,
            "types": types,
            "agents": agents_in_dl,
            "new_count": new_count,
            "wfg_edges": self._wfg.edge_count(),
        }

    def stats(self) -> dict:
        """Aggregate deadlock statistics for the episode so far."""
        return {
            "deadlock_count": self.deadlock_count,
            "type_counts": dict(self.type_counts),
            "active_deadlocks": len(self._active_sigs),
            "total_new_events": int(sum(self._per_step)),
        }

    def congestion_heatmap(self) -> np.ndarray:
        """Per-cell contention heatmap accumulated over the episode."""
        return self.congestion.copy()
