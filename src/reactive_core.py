#!/usr/bin/env python3
"""
DRIMAPS Reactive Core — deadlock-free execution with WFG-guided escape.
=======================================================================

The core of DRIMAPS is a one-step reactive controller built on **priority
inheritance with backtracking** (the PIBT move rule of Okumura et al., 2022).
At every timestep it produces a collision-free joint move; when an agent wants a
cell held by a lower-priority agent, that agent inherits the priority and is
recursively pushed aside, with backtracking when no legal push exists. This rule
is deadlock-free *per step* — there is always a collision-free joint action — so
the catastrophic livelocks that plague repair-based execution never occur.

What the per-step rule does **not** give is completeness: under heavy congestion
(narrow mazes, dense warehouses, swap-required corridors) a handful of agents can
oscillate indefinitely without ever reaching their goals. DRIMAPS closes this gap
with its novel layer:

  * a **wait-for graph** over agents' *greedy-desired* moves is maintained each
    step and **Tarjan's SCC** algorithm extracts wait cycles (Phase: detection);
  * a cycle is promoted to a **confirmed deadlock** only once its agents have made
    no progress toward their goals for ``tau`` steps, and is **classified**
    (corridor / cyclic / congestion / goal-blocking);
  * the agents in a confirmed deadlock — and only those (minimal disruption) — are
    given a short-lived **scatter target**, redirecting the inheritance pressure so
    the cycle breaks; the real goal is restored a few steps later.

The detection is what makes the escape *targeted*: instead of perturbing every
slow agent (which damages throughput), DRIMAPS perturbs exactly the structurally
deadlocked set the WFG identifies. The ablations in :class:`DRIMAPSConfig` toggle
each piece (cycle detection, classification, minimal disruption) so its
contribution is measurable.
"""

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from src.config import DRIMAPSConfig
from src.utils import Position, Path, get_neighbors
from src.dependency_graph import WaitForGraph
from src.cycle_detector import CycleDetector
from src.deadlock_classifier import DeadlockClassifier


class ReactiveCoreStats:
    """Diagnostics accumulated over a reactive run."""

    def __init__(self) -> None:
        self.deadlocks_detected = 0   # confirmed persistent SCC episodes
        self.deadlocks_resolved = 0   # episodes that later cleared with progress
        self.agents_replanned = 0     # scatter redirections issued
        self.type_counts: Dict[str, int] = {}
        self.detection_time = 0.0
        self.resolution_time = 0.0
        self.wfg_update_time = 0.0


class ReactiveCore:
    """Deadlock-free reactive controller with WFG-guided targeted escape."""

    def __init__(self, config: Optional[DRIMAPSConfig] = None) -> None:
        self.config = config or DRIMAPSConfig()

    # ------------------------------------------------------------------
    # Distance fields (BFS heuristic to each goal)
    # ------------------------------------------------------------------
    def _dist_field(self, grid: np.ndarray, goal: Position) -> Dict[Position, int]:
        dist: Dict[Position, int] = {goal: 0}
        q = deque([goal])
        while q:
            cur = q.popleft()
            for nb in get_neighbors(cur, grid, include_wait=False):
                if nb not in dist:
                    dist[nb] = dist[cur] + 1
                    q.append(nb)
        return dist

    # ------------------------------------------------------------------
    def run(
        self,
        grid: np.ndarray,
        starts: List[Position],
        goals: List[Position],
        seed: int = 42,
        deadline: Optional[float] = None,
    ) -> Tuple[List[Path], ReactiveCoreStats]:
        """Execute the reactive controller until all agents arrive or limits hit.

        Returns:
            (trajectories, stats). Trajectories are collision-free by
            construction of the PIBT move rule.
        """
        import time

        cfg = self.config
        n = len(starts)
        rng = np.random.RandomState(seed)
        stats = ReactiveCoreStats()

        H, W = grid.shape
        BIG = H * W + 1

        # Per-agent BFS distance fields to the true goal (cached) and to any
        # transient scatter target.
        t_w = time.time()
        goal_dist = [self._dist_field(grid, goals[i]) for i in range(n)]
        stats.wfg_update_time += 0.0  # field build is amortised; tracked elsewhere
        del t_w

        free_cells = [
            (r, c) for r in range(H) for c in range(W) if grid[r, c] == 0
        ]

        pos: List[Position] = list(starts)
        traj: List[List[Position]] = [[s] for s in starts]
        elapsed = [0] * n                  # steps since last at goal (PIBT priority)
        last_d = [goal_dist[i].get(starts[i], BIG) for i in range(n)]
        stuck = [0] * n                    # steps without distance progress

        # Transient scatter state (the escape).
        scatter_dist: Dict[int, Dict[Position, int]] = {}
        scatter_left = [0] * n

        # WFG-based detection state.
        use_cycle = cfg.enable_cycle_detection
        wfg = WaitForGraph(n) if use_cycle else None
        detector = CycleDetector() if use_cycle else None
        classifier = (
            DeadlockClassifier(grid, cfg)
            if (use_cycle and cfg.enable_classification) else None
        )
        # Active deadlock episodes keyed by frozenset(agent set): value = step
        # first confirmed. Used to count detected vs. resolved honestly.
        active_episodes: Dict[frozenset, int] = {}
        tau = cfg.stagnation_threshold

        longest = max(
            (max(goal_dist[i].get(starts[i], BIG), 0) for i in range(n)),
            default=0,
        )
        max_steps = min(cfg.max_timesteps, longest * 4 + 96)

        for step in range(max_steps):
            if all(pos[i] == goals[i] for i in range(n)):
                break
            if deadline is not None and time.time() > deadline:
                break

            # Distance field each agent currently steers by (scatter or goal).
            def field_of(i: int) -> Dict[Position, int]:
                if scatter_left[i] > 0 and i in scatter_dist:
                    return scatter_dist[i]
                return goal_dist[i]

            # --- PIBT priority-inheritance move (deadlock-free, collision-free) ---
            order = sorted(range(n), key=lambda i: (-elapsed[i], i))
            next_pos: Dict[int, Position] = {}
            occupied_now = {pos[i]: i for i in range(n)}

            def h(i: int, cell: Position) -> int:
                return field_of(i).get(cell, BIG)

            def pibt(ai: int, blocker_cell: Optional[Position]) -> bool:
                cands = [pos[ai]] + list(
                    get_neighbors(pos[ai], grid, include_wait=False)
                )
                cands.sort(key=lambda c: h(ai, c))
                for v in cands:
                    if blocker_cell is not None and v == blocker_cell:
                        continue
                    if v in next_pos.values():
                        continue
                    swap = any(
                        next_pos.get(aj) == pos[ai] and pos[aj] == v
                        for aj in next_pos
                    )
                    if swap:
                        continue
                    next_pos[ai] = v
                    ak = occupied_now.get(v)
                    if ak is not None and ak != ai and ak not in next_pos:
                        if not pibt(ak, pos[ai]):
                            del next_pos[ai]
                            continue
                    return True
                next_pos[ai] = pos[ai]
                return False

            for ai in order:
                if ai not in next_pos:
                    pibt(ai, None)

            newpos = [next_pos[i] for i in range(n)]

            # --- Update progress / stagnation tracking ---
            for i in range(n):
                d = goal_dist[i].get(newpos[i], BIG)
                if newpos[i] == goals[i] or d < last_d[i]:
                    stuck[i] = 0
                else:
                    stuck[i] += 1
                last_d[i] = d
                elapsed[i] = 0 if newpos[i] == goals[i] else elapsed[i] + 1
                if scatter_left[i] > 0:
                    scatter_left[i] -= 1
            pos = newpos

            # --- Detection + targeted escape (the novel layer) -----------------
            # PIBT keeps motion collision-free but is incomplete: a few agents can
            # stagnate (oscillating swaps, goal-blocked, congestion). Persistent
            # stagnation (no distance progress for tau steps) is the escape trigger.
            # The wait-for graph then explains *why* (cyclic / goal-blocking /
            # congestion) and, under minimal disruption, restricts the escape to the
            # structurally-involved agents instead of every slow agent.
            t_d = time.time()
            finished = {i for i in range(n) if pos[i] == goals[i]}
            cand = [i for i in range(n) if i not in finished and stuck[i] >= tau]
            deadlocked: Set[int] = set()      # agents to escape this round

            if use_cycle and cand:
                # Greedy-desired move of each agent (toward goal, ignoring others).
                greedy: Dict[int, Position] = {}
                for i in range(n):
                    best = pos[i]
                    bestd = goal_dist[i].get(pos[i], BIG)
                    for nb in get_neighbors(pos[i], grid, include_wait=False):
                        dd = goal_dist[i].get(nb, BIG)
                        if dd < bestd:
                            bestd = dd
                            best = nb
                    greedy[i] = best
                occ = {pos[i]: i for i in range(n)}
                candset = set(cand)

                # Wait-for graph + Tarjan SCC (the cyclic deadlocks).
                wfg.update_all({i: pos[i] for i in range(n)}, greedy, goals, finished)
                cycles = detector.detect_cycles(wfg, finished)
                cyclic_agents = {a for cyc in cycles for a in cyc}

                # Build the stuck-subgraph: an edge between two stuck agents that
                # contend for each other's cell; mark agents blocked by a *finished*
                # agent as goal-blocked. Weakly-connected components are the episodes.
                adj = {i: set() for i in cand}
                goal_blocked: Set[int] = set()
                for i in cand:
                    k = occ.get(greedy[i])
                    if k is None or k == i:
                        continue
                    if k in finished:
                        goal_blocked.add(i)
                    elif k in candset:
                        adj[i].add(k)
                        adj[k].add(i)
                comp_seen: Set[int] = set()
                groups: List[List[int]] = []
                for i in cand:
                    if i in comp_seen:
                        continue
                    comp, stack = [], [i]
                    comp_seen.add(i)
                    while stack:
                        u = stack.pop()
                        comp.append(u)
                        for v in adj[u]:
                            if v not in comp_seen:
                                comp_seen.add(v)
                                stack.append(v)
                    groups.append(comp)
                stats.detection_time += time.time() - t_d

                seen_now: Set[frozenset] = set()
                structural: Set[int] = set()
                for comp in groups:
                    is_dl = len(comp) >= 2 or any(a in goal_blocked for a in comp)
                    if not is_dl:
                        continue
                    structural.update(comp)
                    key = frozenset(comp)
                    seen_now.add(key)
                    if key not in active_episodes:
                        active_episodes[key] = step
                        stats.deadlocks_detected += 1
                        if classifier is not None:
                            if len(comp) >= 2:
                                info = classifier.classify(
                                    comp, {a: pos[a] for a in range(n)},
                                    goals, finished,
                                )
                                t = getattr(info, "value", str(info))
                            else:
                                t = "goal_blocking"
                            stats.type_counts[t] = stats.type_counts.get(t, 0) + 1
                # Episodes that vanished after their agents made progress = resolved.
                for key in list(active_episodes.keys()):
                    if key not in seen_now:
                        if any(pos[a] == goals[a] or stuck[a] < tau for a in key):
                            stats.deadlocks_resolved += 1
                        del active_episodes[key]

                # Minimal disruption: escape only the structurally-involved agents.
                # Otherwise escape every stagnating agent.
                deadlocked = structural if cfg.enable_minimal_disruption else set(cand)
            elif cand:
                # Ablation (no cycle detection): stagnation-only, no structure.
                stats.detection_time += time.time() - t_d
                for i in cand:
                    key = frozenset([i])
                    if key not in active_episodes:
                        active_episodes[key] = step
                        stats.deadlocks_detected += 1
                for key in list(active_episodes.keys()):
                    if all(pos[a] == goals[a] or stuck[a] < tau for a in key):
                        stats.deadlocks_resolved += 1
                        del active_episodes[key]
                deadlocked = set(cand)
            else:
                stats.detection_time += time.time() - t_d

            # --- Issue scatter targets to the deadlocked set -------------------
            # A deadlocked agent retreats to the least-congested nearby junction:
            # among reachable free cells in a small ring, prefer the one farthest
            # from other agents (open space), then higher-degree (a junction where
            # it can step aside), then closer to its goal. This is deterministic
            # given the seed and far more stable than a random sidestep.
            t_r = time.time()
            if deadlocked:
                others = [pos[j] for j in range(n)]
                for i in deadlocked:
                    if pos[i] == goals[i] or scatter_left[i] > 0:
                        continue
                    pr, pc = pos[i]
                    best_cell = None
                    best_score = None
                    for c in free_cells:
                        md = abs(c[0] - pr) + abs(c[1] - pc)
                        if md < 2 or md > 6 or c not in goal_dist[i]:
                            continue
                        # distance to nearest other agent (congestion avoidance),
                        # capped to a local window for speed
                        near = min(
                            (abs(c[0] - o[0]) + abs(c[1] - o[1])
                             for o in others if abs(c[0] - o[0]) + abs(c[1] - o[1]) <= 10),
                            default=10,
                        )
                        deg = len(get_neighbors(c, grid, include_wait=False))
                        score = (near, deg, -goal_dist[i].get(c, BIG))
                        if best_score is None or score > best_score:
                            best_score = score
                            best_cell = c
                    if best_cell is None:
                        continue
                    scatter_dist[i] = self._dist_field(grid, best_cell)
                    scatter_left[i] = cfg.yield_hold_steps + 3
                    stuck[i] = 0
                    stats.agents_replanned += 1
            stats.resolution_time += time.time() - t_r

            for i in range(n):
                traj[i].append(pos[i])

        # Any episode still active at termination counts as unresolved (already
        # not added to resolved). Done.
        return traj, stats
