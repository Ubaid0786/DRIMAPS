#!/usr/bin/env python3
"""
Reactive Execution Engine
=========================

The shared, controller-agnostic simulator that *executes* MAPF plans one
timestep at a time under a realistic movement model. This is the component
that makes runtime deadlocks a genuine, reproducible phenomenon rather than a
replay of pre-computed conflict-free paths.

Why this module exists
----------------------
A reservation-based planner (cooperative A*) returns paths that are conflict
free *by construction*; replaying them never produces a deadlock, so a runtime
deadlock-resolution layer has nothing to do. Real multi-robot systems do not
work that way: agents commit to routes and then physically move, and when two
flows meet in a corridor (or pile into a doorway) the execution layer must
arbitrate who moves. That arbitration is where deadlocks are born.

``ReactiveSimulator`` models exactly this. Each agent owns a *plan* — an
ordered list of grid cells it intends to traverse. Every timestep:

1. Each unfinished agent proposes its next desired cell (the next waypoint in
   its plan).
2. A movement arbiter applies the ``block_both`` rule (the standard MAPF
   execution model, identical to ``sim/grid_world.py``): an agent advances only
   if its target cell is not simultaneously claimed by another agent (vertex
   conflict) and it is not swapping cells with a neighbour (edge conflict).
   Everyone else waits in place.
3. A *controller* (DRIMAPS, naive detect-and-resolve, or a no-op) may inspect
   the resulting wait structure and rewrite agent plans to break deadlocks.

A genuine deadlock is then a stalled wait cycle: e.g. two agents facing each
other in a one-wide corridor each want the cell the other occupies (a 2-cycle
swap), which ``block_both`` forbids, so without intervention they wait forever.

This same engine is used by ``DRIMAPS`` (offline track) and by the
``baselines`` package, guaranteeing every method is evaluated under an
identical, deterministic execution model — a prerequisite for a fair, honest
comparison.
"""

from collections import Counter
from typing import Callable, Dict, List, Optional, Set

import numpy as np

from src.utils import Position, Path, manhattan_distance


class ReactiveSimulator:
    """Step-by-step MAPF executor under the ``block_both`` movement model.

    Attributes:
        grid: Map grid (1=obstacle, 0=free), shape (height, width).
        goals: Goal cell for each agent.
        n: Number of agents.
        positions: Current cell of each agent.
        plans: Per-agent remaining route, ``plans[i][0]`` is the current cell
            and ``plans[i][1]`` (if present) is the next desired cell.
        finished: Set of agents currently resting on their goal cell.
        step_count: Timesteps elapsed.
        trajectory: Full position history, ``trajectory[i][t]`` = cell of agent
            ``i`` at timestep ``t`` (recorded for cost/makespan accounting).
    """

    def __init__(
        self,
        grid: np.ndarray,
        starts: List[Position],
        goals: List[Position],
        plans: List[Path],
    ) -> None:
        """Initialise the simulator.

        Args:
            grid: Map grid.
            starts: Start cell per agent.
            goals: Goal cell per agent.
            plans: Initial route per agent (each must begin at the agent's
                start cell). Agents follow these routes; a controller may
                rewrite them at runtime.
        """
        self.grid = grid
        self.goals = list(goals)
        self.n = len(starts)
        self.positions: Dict[int, Position] = {i: starts[i] for i in range(self.n)}
        # Defensive copy; ensure each plan starts at the agent's current cell.
        self.plans: List[List[Position]] = []
        for i in range(self.n):
            p = list(plans[i]) if plans[i] else [starts[i]]
            if p[0] != starts[i]:
                p = [starts[i]] + p
            self.plans.append(p)
        self.finished: Set[int] = set()
        self.step_count = 0
        self.trajectory: List[List[Position]] = [
            [starts[i]] for i in range(self.n)
        ]
        self._refresh_finished()

    # ------------------------------------------------------------------
    # Plan / state inspection
    # ------------------------------------------------------------------

    def _refresh_finished(self) -> None:
        for i in range(self.n):
            if self.positions[i] == self.goals[i]:
                self.finished.add(i)
            else:
                self.finished.discard(i)

    def desired_next(self) -> Dict[int, Position]:
        """Compute each agent's intended next cell for the current step.

        An agent at its goal (and with no remaining plan) stays put. An agent
        with a remaining waypoint proposes that waypoint. An agent that has
        exhausted its plan but is *not* at its goal proposes to stay (it is
        stuck and relies on the controller to give it a new plan).

        Returns:
            Map from agent index to desired next cell.
        """
        desired: Dict[int, Position] = {}
        for i in range(self.n):
            pos = self.positions[i]
            if i in self.finished:
                desired[i] = pos
                continue
            plan = self.plans[i]
            # plan[0] is the current cell; the next cell is plan[1].
            if len(plan) >= 2:
                desired[i] = plan[1]
            else:
                desired[i] = pos
        return desired

    def arbitrate(self, desired: Dict[int, Position]) -> Set[int]:
        """Apply the ``block_both`` movement rule and advance the world.

        Mirrors ``sim/grid_world.py`` exactly so the offline and simulation
        tracks share identical execution semantics:

        * An agent that proposes to stay does not move.
        * If two or more agents propose the same cell, none of them take it
          (vertex conflict -> block both).
        * If two agents propose to swap cells, neither moves (edge conflict).
        * Otherwise the move is committed. Following moves (entering a cell a
          neighbour vacates in the same step) are permitted, so trains of
          agents advance together.

        Args:
            desired: Per-agent desired next cell.

        Returns:
            Set of agents that actually moved this step.
        """
        pos = self.positions
        # Count how many agents target each cell (excluding waiters).
        target_count: Counter = Counter()
        for i, tgt in desired.items():
            if tgt != pos[i]:
                target_count[tgt] += 1

        # Provisional movers: clear vertex (no two agents target one cell) and
        # edge (no head-on swap) conflicts.
        move: Dict[int, Position] = {}
        for i in range(self.n):
            tgt = desired[i]
            cur = pos[i]
            if tgt == cur:
                continue  # waiting
            if target_count[tgt] > 1:
                continue  # vertex conflict -> all contenders wait
            swap = any(
                j != i and pos[j] == tgt and desired[j] == cur
                for j in range(self.n)
            )
            if swap:
                continue
            move[i] = tgt

        # Fixpoint: an agent may enter a cell only if its current occupant is
        # itself vacating this step. Repeatedly cancel any move whose target is
        # held by an agent that is staying put; cancellations cascade (a train
        # advances together, but stalls entirely if its head is blocked).
        changed = True
        while changed:
            changed = False
            stayer_cells = {pos[j] for j in range(self.n) if j not in move}
            for i in list(move.keys()):
                if move[i] in stayer_cells:
                    del move[i]
                    changed = True

        # Commit moves and advance plans for agents that moved.
        moved: Set[int] = set(move)
        for i, tgt in move.items():
            self.positions[i] = tgt
            # Pop the consumed waypoint so plan[0] stays at the current cell.
            if len(self.plans[i]) >= 2 and self.plans[i][1] == tgt:
                self.plans[i].pop(0)
        return moved

    def advance(self) -> Set[int]:
        """Run one full timestep (desire -> arbitrate -> record).

        Returns:
            Set of agents that moved this step.
        """
        desired = self.desired_next()
        moved = self.arbitrate(desired)
        self.step_count += 1
        for i in range(self.n):
            self.trajectory[i].append(self.positions[i])
        self._refresh_finished()
        return moved

    def all_finished(self) -> bool:
        """Whether every agent currently rests on its goal."""
        return len(self.finished) == self.n

    def set_plan(self, agent: int, route: Path) -> None:
        """Replace an agent's remaining plan with a new route.

        Args:
            agent: Agent index.
            route: New route; if it does not start at the agent's current
                cell, the current cell is prepended.
        """
        cur = self.positions[agent]
        route = list(route) if route else [cur]
        if route[0] != cur:
            route = [cur] + route
        self.plans[agent] = route

    def paths(self) -> List[Path]:
        """Return the recorded trajectories as MAPF paths."""
        return [list(tr) for tr in self.trajectory]


def reactive_execute(
    grid: np.ndarray,
    starts: List[Position],
    goals: List[Position],
    plans: List[Path],
    controller: Optional[Callable[["ReactiveSimulator", int], None]] = None,
    max_steps: int = 512,
    deadline: Optional[float] = None,
    extra_steps: int = 32,
) -> ReactiveSimulator:
    """Execute plans reactively until all agents finish or limits are hit.

    Args:
        grid: Map grid.
        starts: Start cells.
        goals: Goal cells.
        plans: Initial per-agent routes.
        controller: Optional callback ``controller(sim, t)`` invoked once per
            timestep *before* arbitration, allowed to rewrite agent plans. Used
            by DRIMAPS / naive detect-and-resolve; ``None`` means pure
            execution (the prevention-only baseline).
        max_steps: Hard timestep cap.
        deadline: Optional wall-clock deadline (``time.time()`` value).
        extra_steps: Slack added beyond the longest initial plan, bounding how
            long execution runs once plans are exhausted.

    Returns:
        The simulator after execution (inspect ``.paths()``, ``.finished``).
    """
    import time as _time

    sim = ReactiveSimulator(grid, starts, goals, plans)
    longest = max((len(p) for p in sim.plans), default=0)
    horizon = min(max_steps, longest + extra_steps)

    for t in range(horizon):
        if sim.all_finished():
            break
        if deadline is not None and _time.time() > deadline:
            break
        if controller is not None:
            controller(sim, t)
        sim.advance()
    return sim
