#!/usr/bin/env python3
"""Tests for the MovingAI benchmark loader and the editable sim scenario."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.movingai import (
    load_map, load_scenario, load_instance, available_maps, MAP_CATEGORY,
)
from src.config import DRIMAPSConfig
from src.drimaps import DRIMAPS


def test_all_33_maps_vendored():
    maps = available_maps()
    assert len(maps) == 33
    assert "maze-32-32-2" in maps and "Berlin_1_256" in maps


def test_load_map_shape_and_obstacles():
    grid = load_map("maze-32-32-2")
    assert grid.shape == (32, 32)
    assert set(np.unique(grid)) <= {0, 1}
    assert grid.sum() > 0  # has obstacles


def test_load_scenario_positions_in_bounds():
    grid = load_map("room-32-32-4")
    entries = load_scenario("room-32-32-4-even-1")
    assert len(entries) > 0
    for (sr, sc), (gr, gc), opt in entries[:50]:
        assert 0 <= sr < 32 and 0 <= sc < 32
        assert grid[sr, sc] == 0 and grid[gr, gc] == 0  # on free cells
        assert opt >= 0


def test_load_instance_counts():
    grid, starts, goals = load_instance("maze-32-32-2", 25, "even-1")
    assert len(starts) == 25 and len(goals) == 25
    assert grid.shape == (32, 32)


def test_every_map_has_a_category():
    for m in available_maps():
        assert m in MAP_CATEGORY


def test_drimaps_solves_loaded_instance_collision_free():
    grid, starts, goals = load_instance("room-32-32-4", 30, "even-1")
    res = DRIMAPS(DRIMAPSConfig(timeout=20)).solve(grid, starts, goals)
    assert res.collision_count == 0
    assert res.deadlocks_resolved <= res.deadlocks_detected


class TestBenchmarkScenarioEditing:
    def test_load_and_edit(self):
        from sim.benchmark import BenchmarkScenario
        scn = BenchmarkScenario.load("room-32-32-4", num_agents=10)
        assert scn.num_agents == 10
        free = [tuple(int(x) for x in p) for p in zip(*np.where(scn.grid == 0))]
        scn.set_obstacle(*free[0], blocked=True)
        assert scn.grid[free[0]] == 1
        scn.set_obstacle(*free[0], blocked=False)
        assert scn.grid[free[0]] == 0
        scn.add_agent(free[1], free[-1])
        assert scn.num_agents == 11

    def test_add_agent_rejects_obstacle(self):
        from sim.benchmark import BenchmarkScenario
        scn = BenchmarkScenario.load("maze-32-32-2", num_agents=5)
        obstacle = tuple(int(x) for x in np.argwhere(scn.grid == 1)[0])
        with pytest.raises(ValueError):
            scn.add_agent(obstacle, obstacle)

    def test_to_env_rolls_out_collision_free(self):
        from sim.benchmark import load_benchmark_env
        env = load_benchmark_env("warehouse-10-20-10-2-1", 20)
        env.reset(seed=1)
        for _ in range(20):
            env.step(env.sample_actions())
        xy = env.get_agents_xy()
        active = [tuple(p) for i, p in enumerate(xy) if env.grid.is_active[i]]
        assert len(active) == len(set(active))  # no vertex collisions
