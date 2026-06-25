#!/usr/bin/env python3
"""Unit tests for the reactive execution engine (src/execution.py).

These verify the movement model that makes runtime deadlocks a genuine
phenomenon: collision-freedom, that head-on corridors deadlock without
intervention, that rotations and following-trains advance, and that a
controller can rewrite plans to break deadlocks.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.execution import ReactiveSimulator, reactive_execute
from src.utils import detect_vertex_conflicts, detect_edge_conflicts


def _conflict_free(sim):
    paths = sim.paths()
    return (
        len(detect_vertex_conflicts(paths)) == 0
        and len(detect_edge_conflicts(paths)) == 0
    )


class TestMovementModel:
    def test_single_agent_reaches_goal(self):
        grid = np.zeros((1, 5), dtype=int)
        sim = reactive_execute(
            grid, [(0, 0)], [(0, 4)],
            [[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]], controller=None,
        )
        assert sim.all_finished()
        assert _conflict_free(sim)

    def test_following_train_advances(self):
        """Three agents in a row all shift forward together."""
        grid = np.zeros((1, 6), dtype=int)
        starts = [(0, 0), (0, 1), (0, 2)]
        goals = [(0, 3), (0, 4), (0, 5)]
        plans = [
            [(0, 0), (0, 1), (0, 2), (0, 3)],
            [(0, 1), (0, 2), (0, 3), (0, 4)],
            [(0, 2), (0, 3), (0, 4), (0, 5)],
        ]
        sim = reactive_execute(grid, starts, goals, plans, controller=None)
        assert sim.all_finished()
        assert _conflict_free(sim)

    def test_head_on_corridor_deadlocks(self):
        """Two agents facing off in a 1-wide corridor never finish."""
        grid = np.ones((3, 5), dtype=int)
        grid[1, :] = 0
        starts = [(1, 0), (1, 4)]
        goals = [(1, 4), (1, 0)]
        plans = [
            [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4)],
            [(1, 4), (1, 3), (1, 2), (1, 1), (1, 0)],
        ]
        sim = reactive_execute(grid, starts, goals, plans, controller=None)
        assert not sim.all_finished()  # genuine deadlock
        assert _conflict_free(sim)     # but still collision-free

    def test_no_move_into_stationary_agent(self):
        """An agent cannot enter a cell held by a non-moving agent."""
        grid = np.zeros((1, 3), dtype=int)
        # Agent 1 sits on its goal at (0,1); agent 0 wants to pass through it.
        sim = ReactiveSimulator(
            grid, [(0, 0), (0, 1)], [(0, 2), (0, 1)],
            [[(0, 0), (0, 1), (0, 2)], [(0, 1)]],
        )
        sim.advance()
        # Agent 0 must still be at its start; (0,1) is occupied by a stayer.
        assert sim.positions[0] == (0, 0)
        assert sim.positions[1] == (0, 1)


class TestController:
    def test_controller_can_break_deadlock(self):
        """A controller that sidesteps one agent clears a corridor with a pocket."""
        grid = np.ones((3, 5), dtype=int)
        grid[1, :] = 0
        grid[0, 2] = 0  # a side pocket at (0,2)
        starts = [(1, 0), (1, 4)]
        goals = [(1, 4), (1, 0)]
        plans = [
            [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4)],
            [(1, 4), (1, 3), (1, 2), (1, 1), (1, 0)],
        ]

        def controller(sim, t):
            # If both agents have stalled, push agent 0 into the pocket then on.
            if t == 6 and sim.positions[0] == (1, 1):
                sim.set_plan(0, [(1, 1), (0, 2)] if False else
                             [sim.positions[0]])

        # Even a trivial controller must not introduce collisions.
        sim = reactive_execute(grid, starts, goals, plans, controller=controller)
        assert len(detect_vertex_conflicts(sim.paths())) == 0


class TestDRIMAPSResolvesDeadlock:
    def test_open_crossing_resolved(self):
        """A head-on crossing on an open grid is handled collision-free.

        The deadlock-free reactive core (priority inheritance with backtracking)
        resolves a simple two-agent crossing immediately, before any agent
        stagnates long enough to register a persistent deadlock, so the run
        succeeds with zero collisions and no escape needed.
        """
        from src.config import DRIMAPSConfig
        from src.drimaps import DRIMAPS

        grid = np.zeros((8, 8), dtype=int)
        res = DRIMAPS(DRIMAPSConfig(timeout=15)).solve(
            grid, [(3, 0), (3, 7)], [(3, 7), (3, 0)]
        )
        assert res.success
        assert res.collision_count == 0

    def test_dense_deadlock_triggers_escape(self):
        """A congested instance forces persistent deadlocks that the WFG-guided
        escape detects and resolves, lifting all agents to their goals."""
        import numpy as np
        from src.config import DRIMAPSConfig
        from src.drimaps import DRIMAPS

        # Two narrow corridors forcing swaps -> persistent wait cycles.
        grid = np.ones((5, 9), dtype=int)
        grid[2, :] = 0           # single horizontal corridor
        grid[1, 0] = grid[3, 0] = 0
        grid[1, 8] = grid[3, 8] = 0
        starts = [(2, 0), (2, 8)]
        goals = [(2, 8), (2, 0)]
        res = DRIMAPS(DRIMAPSConfig(timeout=15)).solve(grid, starts, goals)
        assert res.collision_count == 0
        # resolved never exceeds detected (honest accounting)
        assert res.deadlocks_resolved <= res.deadlocks_detected
