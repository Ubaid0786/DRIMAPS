#!/usr/bin/env python3
"""
Unit Tests for DRIMAPSim Simulation Environment

Tests all core features: creation, step/reset, collision systems,
MAPF modes, map generators, metrics, and rendering.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim import drimapsim_v0, EnvConfig
from sim.environment import DRIMAPSimEnv
from sim.grid_world import GridWorld
from sim.map_registry import (
    generate_map, generate_random, generate_warehouse,
    generate_corridor, generate_bottleneck, generate_maze,
    generate_room, generate_open, compute_map_difficulty,
    parse_movingai_map,
)
from sim.rendering import render_svg


class TestEnvCreation:
    """Test environment creation and reset."""

    def test_default_config(self):
        """Environment should work with default config."""
        env = drimapsim_v0()
        obs, info = env.reset()
        assert len(obs) == 8  # Default num_agents
        assert obs[0].shape == (3, 11, 11)  # 3 channels, obs_radius=5

    def test_custom_config(self):
        """Environment with custom config."""
        cfg = EnvConfig(size=16, num_agents=4, seed=42, max_episode_steps=32)
        env = drimapsim_v0(cfg)
        obs, info = env.reset()
        assert len(obs) == 4
        assert len(info) == 4

    def test_explicit_positions(self):
        """Environment with explicit agent/target positions."""
        cfg = EnvConfig(
            size=8, seed=42, max_episode_steps=32,
            agents_xy=[(0, 0), (1, 1)],
            targets_xy=[(7, 7), (6, 6)],
        )
        env = DRIMAPSimEnv(cfg)
        obs, info = env.reset()
        assert len(obs) == 2

    def test_seed_reproducibility(self):
        """Same seed should produce identical environments."""
        cfg1 = EnvConfig(size=16, num_agents=4, seed=42, max_episode_steps=32)
        cfg2 = EnvConfig(size=16, num_agents=4, seed=42, max_episode_steps=32)
        env1 = DRIMAPSimEnv(cfg1)
        env2 = DRIMAPSimEnv(cfg2)
        obs1, _ = env1.reset()
        obs2, _ = env2.reset()
        assert np.allclose(obs1[0], obs2[0])


class TestStepMechanics:
    """Test step execution and outcomes."""

    def test_step_returns_correct_shapes(self):
        """Step should return correct output shapes."""
        cfg = EnvConfig(size=8, num_agents=4, seed=42, max_episode_steps=32)
        env = drimapsim_v0(cfg)
        env.reset()
        obs, rewards, terminated, truncated, infos = env.step([0, 0, 0, 0])
        assert len(obs) == 4
        assert len(rewards) == 4
        assert len(terminated) == 4
        assert len(truncated) == 4
        assert len(infos) == 4

    def test_idle_action(self):
        """Idle action (0) should not move agents."""
        cfg = EnvConfig(size=8, num_agents=2, seed=42, max_episode_steps=32)
        env = DRIMAPSimEnv(cfg)
        env.reset()
        pos_before = env.get_agents_xy()
        env.step([0, 0])
        pos_after = env.get_agents_xy()
        assert pos_before == pos_after

    def test_truncation_after_max_steps(self):
        """Episode should truncate after max_episode_steps."""
        cfg = EnvConfig(size=8, num_agents=2, seed=42, max_episode_steps=5)
        env = drimapsim_v0(cfg)
        env.reset()
        for _ in range(4):
            _, _, terminated, truncated, _ = env.step([0, 0])
            assert not all(truncated)
        _, _, terminated, truncated, _ = env.step([0, 0])
        assert all(truncated)

    def test_sample_actions(self):
        """sample_actions should return valid actions."""
        cfg = EnvConfig(size=8, num_agents=6, seed=42, max_episode_steps=32)
        env = drimapsim_v0(cfg)
        env.reset()
        actions = env.sample_actions()
        assert len(actions) == 6
        assert all(0 <= a <= 4 for a in actions)


class TestCollisionSystems:
    """Test all four collision systems."""

    def test_priority_system(self):
        cfg = EnvConfig(size=8, num_agents=4, seed=42,
                       collision_system="priority", max_episode_steps=16)
        env = drimapsim_v0(cfg)
        env.reset()
        for _ in range(5):
            env.step(env.sample_actions())

    def test_block_both_system(self):
        cfg = EnvConfig(size=8, num_agents=4, seed=42,
                       collision_system="block_both", max_episode_steps=16)
        env = drimapsim_v0(cfg)
        env.reset()
        for _ in range(5):
            env.step(env.sample_actions())

    def test_soft_system(self):
        cfg = EnvConfig(size=8, num_agents=4, seed=42,
                       collision_system="soft", max_episode_steps=16)
        env = drimapsim_v0(cfg)
        env.reset()
        for _ in range(5):
            env.step(env.sample_actions())

    def test_strict_system(self):
        cfg = EnvConfig(size=8, num_agents=4, seed=42,
                       collision_system="strict", max_episode_steps=16)
        env = drimapsim_v0(cfg)
        env.reset()
        for _ in range(5):
            env.step(env.sample_actions())


class TestMAPFModes:
    """Test all three MAPF modes."""

    def test_finish_mode(self):
        """Agents should disappear on goal in finish mode."""
        cfg = EnvConfig(size=8, num_agents=2, seed=42,
                       on_target="finish", max_episode_steps=100)
        env = drimapsim_v0(cfg)
        env.reset()
        for _ in range(100):
            _, _, terminated, truncated, _ = env.step(env.sample_actions())
            if all(terminated) or all(truncated):
                break

    def test_nothing_mode(self):
        """Cooperative mode: all must reach."""
        cfg = EnvConfig(size=8, num_agents=2, seed=42,
                       on_target="nothing", max_episode_steps=50)
        env = drimapsim_v0(cfg)
        env.reset()
        for _ in range(50):
            _, _, terminated, truncated, _ = env.step(env.sample_actions())
            if all(terminated) or all(truncated):
                break

    def test_restart_mode(self):
        """Lifelong mode: agents get new targets."""
        cfg = EnvConfig(size=16, num_agents=4, seed=42,
                       on_target="restart", max_episode_steps=30)
        env = drimapsim_v0(cfg)
        env.reset()
        for _ in range(30):
            obs, rewards, terminated, truncated, _ = env.step(env.sample_actions())
            assert not any(terminated)  # Never terminates in lifelong


class TestMapGenerators:
    """Test all 8 map generators."""

    @pytest.mark.parametrize("map_type", [
        "random", "warehouse", "corridor", "bottleneck",
        "maze", "room", "dense_random", "open",
    ])
    def test_map_generation(self, map_type):
        grid = generate_map(map_type, 32, 0.2, 42)
        assert grid.shape == (32, 32)
        assert grid.dtype == np.int32
        assert np.all((grid == 0) | (grid == 1))

    def test_map_difficulty_scoring(self):
        grid = generate_map("corridor", 32, 0.2, 42)
        diff = compute_map_difficulty(grid)
        assert "density" in diff
        assert "avg_degree" in diff
        assert "chokepoint_count" in diff
        assert "difficulty_score" in diff
        assert diff["difficulty_score"] > 0

    def test_movingai_parse(self):
        map_str = """type octile
height 4
width 4
map
....
.##.
.##.
....
"""
        grid = parse_movingai_map(map_str)
        assert grid.shape == (4, 4)
        assert grid[1, 1] == 1  # Obstacle
        assert grid[0, 0] == 0  # Free

    def test_open_map_empty(self):
        grid = generate_open(16, 0.0, 42)
        assert np.sum(grid) == 0

    def test_env_with_map_type(self):
        """Environment should work with explicit map types."""
        for mt in ["warehouse", "corridor", "bottleneck"]:
            cfg = EnvConfig(size=16, num_agents=4, seed=42,
                           map_type=mt, max_episode_steps=16)
            env = drimapsim_v0(cfg)
            obs, _ = env.reset()
            assert len(obs) == 4


class TestMetrics:
    """Test metric wrappers."""

    def test_isr_metric(self):
        cfg = EnvConfig(size=8, num_agents=2, seed=42, max_episode_steps=50)
        env = drimapsim_v0(cfg)
        env.reset()
        for _ in range(50):
            _, _, terminated, truncated, infos = env.step(env.sample_actions())
            if all(terminated) or all(truncated):
                break
        assert "ISR" in infos[-1]
        assert 0.0 <= infos[-1]["ISR"] <= 1.0

    def test_makespan_metric(self):
        cfg = EnvConfig(size=8, num_agents=2, seed=42, max_episode_steps=50)
        env = drimapsim_v0(cfg)
        env.reset()
        for _ in range(10):
            _, _, _, _, infos = env.step(env.sample_actions())
        assert "makespan" in infos[-1]
        assert infos[-1]["makespan"] > 0

    def test_deadlock_metrics(self):
        cfg = EnvConfig(size=8, num_agents=4, seed=42,
                       deadlock_tracking=True, max_episode_steps=50)
        env = drimapsim_v0(cfg)
        env.reset()
        for _ in range(20):
            _, _, _, _, infos = env.step(env.sample_actions())
        assert "deadlock_count" in infos[-1]


class TestRendering:
    """Test rendering systems."""

    def test_ascii_render(self):
        cfg = EnvConfig(size=8, num_agents=4, seed=42, max_episode_steps=16)
        env = drimapsim_v0(cfg)
        env.reset()
        result = env.render(mode="ansi")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_svg_render(self):
        cfg = EnvConfig(size=8, num_agents=4, seed=42, max_episode_steps=16)
        env = DRIMAPSimEnv(cfg)
        env.reset()
        svg = render_svg(
            env.get_obstacles(),
            env.get_agents_xy(),
            env.get_targets_xy(),
        )
        assert "<svg" in svg
        assert "<circle" in svg
        assert "</svg>" in svg


class TestWrappers:
    """Wrappers must forward the DRIMAPSim helper API and export valid JSON.

    Gymnasium 1.0 dropped ``Wrapper.__getattr__``, so these guard against the
    regression where wrapped envs lost ``get_agents_xy``/``sample_actions``/
    ``render(mode=...)`` and trajectory export choked on numpy scalars.
    """

    def _make(self, wrapper_cls, **kw):
        from sim.environment import DRIMAPSimEnv
        cfg = EnvConfig(size=10, num_agents=5, seed=42, density=0.1,
                        max_episode_steps=20, on_target="nothing",
                        record_history=True)
        return wrapper_cls(DRIMAPSimEnv(cfg), **kw)

    def test_helper_api_passthrough(self):
        from sim.wrappers import MultiTimeLimit, RecordTrajectory
        for wrapper_cls, kw in [(MultiTimeLimit, {"max_steps": 20}),
                                (RecordTrajectory, {})]:
            env = self._make(wrapper_cls, **kw)
            env.reset(seed=42)
            # These all live on the base env and must forward through.
            assert len(env.get_agents_xy()) == 5
            assert len(env.get_targets_xy()) == 5
            assert env.get_obstacles() is not None
            actions = env.sample_actions()
            assert len(actions) == 5
            assert isinstance(env.get_deadlock_stats(), dict)
            env.step(actions)

    def test_render_mode_through_wrapper(self):
        from sim.wrappers import RecordTrajectory
        env = self._make(RecordTrajectory)
        env.reset(seed=42)
        out = env.render(mode="ansi")
        assert isinstance(out, str) and len(out) > 0

    def test_record_trajectory_json_export(self, tmp_path):
        import json
        from sim.wrappers import RecordTrajectory
        env = self._make(RecordTrajectory)
        env.reset(seed=42)
        for _ in range(10):
            env.step(env.sample_actions())  # numpy int64 actions
        out = tmp_path / "traj.json"
        env.export_json(str(out))  # must not raise on numpy scalars
        data = json.loads(out.read_text())
        assert data["num_agents"] == 5
        assert data["num_steps"] == 10
        # initial reset frame + one per step
        assert len(data["trajectories"]) == 11


class TestStressTests:
    """Stress tests with many agents."""

    def test_50_agents(self):
        """50 agents should work without crashing."""
        cfg = EnvConfig(size=32, num_agents=50, seed=42,
                       density=0.1, max_episode_steps=50)
        env = drimapsim_v0(cfg)
        obs, _ = env.reset()
        assert len(obs) == 50
        for _ in range(20):
            env.step(env.sample_actions())

    def test_100_agents(self):
        """100 agents should work without crashing."""
        cfg = EnvConfig(size=64, num_agents=100, seed=42,
                       density=0.1, max_episode_steps=30)
        env = drimapsim_v0(cfg)
        obs, _ = env.reset()
        assert len(obs) == 100
        for _ in range(10):
            env.step(env.sample_actions())

    def test_dense_environment(self):
        """Dense environment with many obstacles."""
        cfg = EnvConfig(size=32, num_agents=20, seed=42,
                       density=0.3, max_episode_steps=50)
        env = drimapsim_v0(cfg)
        obs, _ = env.reset()
        for _ in range(30):
            env.step(env.sample_actions())
