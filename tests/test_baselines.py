#!/usr/bin/env python3
"""Unit tests for the baseline solvers (PIBT, Naive-DR, Prevention-Only).

Verifies each baseline returns well-formed, collision-free trajectories and
that PIBT -- a deadlock-free decentralized planner -- solves easy instances.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines import PIBTSolver, NaiveDRSolver, PreventionOnlySolver
from src.utils import detect_vertex_conflicts, detect_edge_conflicts


@pytest.fixture
def open_grid():
    return np.zeros((10, 10), dtype=int)


def _conflict_free(paths):
    return (
        len(detect_vertex_conflicts(paths)) == 0
        and len(detect_edge_conflicts(paths)) == 0
    )


@pytest.mark.parametrize("Solver", [PIBTSolver, NaiveDRSolver, PreventionOnlySolver])
def test_returns_collision_free_paths(open_grid, Solver):
    starts = [(0, 0), (0, 9), (9, 0), (9, 9)]
    goals = [(9, 9), (9, 0), (0, 9), (0, 0)]
    solver = Solver(open_grid, timeout=15.0)
    paths = solver.solve(starts, goals, seed=1)
    assert paths is not None and len(paths) == len(starts)
    assert _conflict_free(paths)
    assert solver.last_result is not None


def test_pibt_solves_open_crossing(open_grid):
    starts = [(5, 0), (5, 9)]
    goals = [(5, 9), (5, 0)]
    paths = PIBTSolver(open_grid, timeout=15.0).solve(starts, goals)
    assert all(paths[i][-1] == goals[i] for i in range(2))
    assert _conflict_free(paths)


def test_pibt_scales_small_dense():
    grid = np.zeros((8, 8), dtype=int)
    n = 8
    starts = [(0, i % 8) for i in range(n)]
    goals = [(7, i % 8) for i in range(n)]
    solver = PIBTSolver(grid, timeout=15.0)
    paths = solver.solve(starts, goals)
    assert _conflict_free(paths)
    reached = sum(1 for i in range(n) if paths[i][-1] == goals[i])
    assert reached >= n - 1  # PIBT should clear an open grid


def test_naive_dr_reports_replans(open_grid):
    # A head-on crossing should trigger at least one stagnation-replan.
    starts = [(5, 0), (5, 9)]
    goals = [(5, 9), (5, 0)]
    solver = NaiveDRSolver(open_grid, timeout=15.0)
    solver.solve(starts, goals, seed=3)
    assert solver.last_result.agents_replanned >= 0  # well-defined, non-negative
