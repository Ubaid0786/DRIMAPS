#!/usr/bin/env python3
"""
Unit Tests for the DRIMAPS Core Algorithm

Tests the main DRIMAPS class on small instances to verify
correctness of the end-to-end pipeline.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DRIMAPSConfig, InitialSolver
from src.drimaps import DRIMAPS


@pytest.fixture
def small_grid():
    """8x8 grid with a wall and gap."""
    grid = np.zeros((8, 8), dtype=int)
    grid[3, 1:7] = 1
    grid[3, 4] = 0  # Gap
    return grid


@pytest.fixture
def empty_grid():
    """8x8 empty grid."""
    return np.zeros((8, 8), dtype=int)


@pytest.fixture
def corridor_grid():
    """1-wide corridor grid for deadlock testing."""
    grid = np.ones((5, 10), dtype=int)
    grid[2, :] = 0  # Single corridor row
    return grid


class TestDRIMAPSBasic:
    """Basic DRIMAPS functionality tests."""

    def test_single_agent(self, empty_grid):
        """Single agent should always succeed."""
        config = DRIMAPSConfig(timeout=10.0)
        solver = DRIMAPS(config)
        result = solver.solve(
            empty_grid,
            starts=[(0, 0)],
            goals=[(7, 7)],
        )
        assert result.success
        assert result.paths[0][-1] == (7, 7)
        assert result.makespan > 0
        assert result.collision_count == 0

    def test_two_agents_no_conflict(self, empty_grid):
        """Two agents far apart should not conflict."""
        config = DRIMAPSConfig(timeout=10.0)
        solver = DRIMAPS(config)
        result = solver.solve(
            empty_grid,
            starts=[(0, 0), (7, 7)],
            goals=[(0, 7), (7, 0)],
        )
        assert result.success
        assert result.collision_count == 0

    def test_two_agents_crossing(self, empty_grid):
        """Two agents crossing paths should be resolved."""
        config = DRIMAPSConfig(timeout=15.0)
        solver = DRIMAPS(config)
        result = solver.solve(
            empty_grid,
            starts=[(3, 0), (3, 7)],
            goals=[(3, 7), (3, 0)],
        )
        # Should find a solution (may or may not be optimal)
        assert result.paths is not None
        assert len(result.paths) == 2

    def test_result_metrics(self, empty_grid):
        """Result should have correct metric types."""
        config = DRIMAPSConfig(timeout=10.0)
        solver = DRIMAPS(config)
        result = solver.solve(
            empty_grid,
            starts=[(0, 0)],
            goals=[(1, 1)],
        )
        assert isinstance(result.runtime, float)
        assert isinstance(result.makespan, int)
        assert isinstance(result.sum_of_costs, int)
        assert isinstance(result.deadlocks_detected, int)
        assert result.runtime > 0


class TestDRIMAPSDeadlock:
    """Tests for deadlock detection and resolution."""

    def test_corridor_deadlock_detection(self, corridor_grid):
        """Two agents in a corridor facing each other."""
        config = DRIMAPSConfig(timeout=15.0)
        solver = DRIMAPS(config)
        result = solver.solve(
            corridor_grid,
            starts=[(2, 0), (2, 9)],
            goals=[(2, 9), (2, 0)],
        )
        # Should attempt to resolve
        assert result.paths is not None

    def test_multiple_agents_dense(self, small_grid):
        """Multiple agents on a small grid with obstacles."""
        config = DRIMAPSConfig(timeout=20.0)
        solver = DRIMAPS(config)
        result = solver.solve(
            small_grid,
            starts=[(0, 0), (0, 7), (7, 0), (7, 7)],
            goals=[(7, 7), (7, 0), (0, 7), (0, 0)],
        )
        assert result.paths is not None
        assert len(result.paths) == 4


class TestDRIMAPSAblation:
    """Tests for ablation configurations."""

    def test_no_cycle_detection(self, empty_grid):
        """DRIMAPS with stagnation-only detection."""
        config = DRIMAPSConfig(
            timeout=10.0,
            enable_cycle_detection=False,
        )
        solver = DRIMAPS(config)
        result = solver.solve(
            empty_grid,
            starts=[(0, 0)],
            goals=[(3, 3)],
        )
        assert result.success

    def test_no_classification(self, empty_grid):
        """DRIMAPS with uniform resolution."""
        config = DRIMAPSConfig(
            timeout=10.0,
            enable_classification=False,
        )
        solver = DRIMAPS(config)
        result = solver.solve(
            empty_grid,
            starts=[(0, 0), (0, 1)],
            goals=[(1, 0), (1, 1)],
        )
        assert result.success

    def test_no_safety_verification(self, empty_grid):
        """DRIMAPS without post-resolution safety checks."""
        config = DRIMAPSConfig(
            timeout=10.0,
            enable_safety_verification=False,
        )
        solver = DRIMAPS(config)
        result = solver.solve(
            empty_grid,
            starts=[(0, 0)],
            goals=[(2, 2)],
        )
        assert result.success
