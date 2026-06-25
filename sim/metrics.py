"""
DRIMAPSim Metric Wrappers

Gymnasium wrappers that automatically compute and log MAPF metrics.
Includes both standard metrics (ISR, CSR, makespan) and DRIMAPSim-exclusive
deadlock and congestion metrics.
"""

import gymnasium
import numpy as np
from typing import Optional


class _MetricWrapper(gymnasium.Wrapper):
    """Base metric wrapper."""

    def __init__(self, env):
        super().__init__(env)
        self._metric_value = None

    def get_metric(self):
        return self._metric_value

    def sample_actions(self):
        """Propagate sample_actions through the wrapper chain."""
        return self.env.sample_actions()

    def get_agents_xy(self):
        return self.env.get_agents_xy()

    def get_targets_xy(self):
        return self.env.get_targets_xy()

    def get_obstacles(self):
        return self.env.get_obstacles()

    def get_deadlock_stats(self):
        """Propagate WFG deadlock stats through the wrapper chain."""
        return self.env.get_deadlock_stats()

    def get_congestion_heatmap(self):
        """Propagate the congestion heatmap through the wrapper chain."""
        return self.env.get_congestion_heatmap()

    def get_deadlock_events(self):
        return self.env.get_deadlock_events()

    def render(self, mode="human"):
        return self.env.render(mode=mode)


class ISRMetric(_MetricWrapper):
    """Individual Success Rate — fraction of agents that reached their goal."""

    def __init__(self, env):
        super().__init__(env)
        self._reached = set()

    def reset(self, **kwargs):
        self._reached = set()
        self._metric_value = 0.0
        return self.env.reset(**kwargs)

    def step(self, actions):
        obs, rewards, terminated, truncated, infos = self.env.step(actions)
        n = len(infos)
        for i in range(n):
            if infos[i].get("on_goal", False):
                self._reached.add(i)

        self._metric_value = len(self._reached) / n if n > 0 else 0.0

        # Inject metric into last agent's info
        if infos:
            infos[-1]["ISR"] = self._metric_value

        return obs, rewards, terminated, truncated, infos


class CSRMetric(_MetricWrapper):
    """Collective Success Rate — 1.0 if ALL agents reached goals, else 0.0."""

    def __init__(self, env):
        super().__init__(env)
        self._reached = set()

    def reset(self, **kwargs):
        self._reached = set()
        self._metric_value = 0.0
        return self.env.reset(**kwargs)

    def step(self, actions):
        obs, rewards, terminated, truncated, infos = self.env.step(actions)
        n = len(infos)
        for i in range(n):
            if infos[i].get("on_goal", False):
                self._reached.add(i)

        self._metric_value = 1.0 if len(self._reached) >= n else 0.0

        if infos:
            infos[-1]["CSR"] = self._metric_value

        return obs, rewards, terminated, truncated, infos


class EpLengthMetric(_MetricWrapper):
    """Episode length counter."""

    def __init__(self, env):
        super().__init__(env)
        self._steps = 0

    def reset(self, **kwargs):
        self._steps = 0
        self._metric_value = 0
        return self.env.reset(**kwargs)

    def step(self, actions):
        obs, rewards, terminated, truncated, infos = self.env.step(actions)
        self._steps += 1
        self._metric_value = self._steps

        if infos:
            infos[-1]["episode_length"] = self._steps

        return obs, rewards, terminated, truncated, infos


class MakespanMetric(_MetricWrapper):
    """Makespan — time until the last agent reaches its goal."""

    def __init__(self, env):
        super().__init__(env)
        self._arrival_times = {}

    def reset(self, **kwargs):
        self._arrival_times = {}
        self._metric_value = 0
        self._steps = 0
        return self.env.reset(**kwargs)

    def step(self, actions):
        obs, rewards, terminated, truncated, infos = self.env.step(actions)
        self._steps += 1
        n = len(infos)

        for i in range(n):
            if infos[i].get("on_goal", False) and i not in self._arrival_times:
                self._arrival_times[i] = self._steps

        if self._arrival_times:
            self._metric_value = max(self._arrival_times.values())
        else:
            self._metric_value = self._steps

        if infos:
            infos[-1]["makespan"] = self._metric_value

        return obs, rewards, terminated, truncated, infos


class SumOfCostsMetric(_MetricWrapper):
    """Sum of costs — total steps taken by all agents until goal arrival."""

    def __init__(self, env):
        super().__init__(env)
        self._arrival_times = {}

    def reset(self, **kwargs):
        self._arrival_times = {}
        self._metric_value = 0
        self._steps = 0
        return self.env.reset(**kwargs)

    def step(self, actions):
        obs, rewards, terminated, truncated, infos = self.env.step(actions)
        self._steps += 1
        n = len(infos)

        for i in range(n):
            if infos[i].get("on_goal", False) and i not in self._arrival_times:
                self._arrival_times[i] = self._steps

        self._metric_value = sum(self._arrival_times.values())

        if infos:
            infos[-1]["sum_of_costs"] = self._metric_value

        return obs, rewards, terminated, truncated, infos


class ThroughputMetric(_MetricWrapper):
    """Throughput — agents reaching goal per timestep."""

    def __init__(self, env):
        super().__init__(env)
        self._total_arrivals = 0

    def reset(self, **kwargs):
        self._total_arrivals = 0
        self._metric_value = 0.0
        self._steps = 0
        return self.env.reset(**kwargs)

    def step(self, actions):
        obs, rewards, terminated, truncated, infos = self.env.step(actions)
        self._steps += 1
        n = len(infos)

        for i in range(n):
            if rewards[i] > 0:
                self._total_arrivals += 1

        self._metric_value = (
            self._total_arrivals / self._steps if self._steps > 0 else 0.0
        )

        if infos:
            infos[-1]["throughput"] = round(self._metric_value, 4)

        return obs, rewards, terminated, truncated, infos


class DeadlockMetric(_MetricWrapper):
    """DRIMAPSim-exclusive: wait-for-graph deadlock metrics.

    Surfaces the genuine, WFG-confirmed deadlock signal computed by the
    environment's :class:`~sim.deadlock_monitor.DeadlockMonitor` (not the
    stagnation heuristic), together with the structural-type breakdown and the
    classic stagnation statistics for comparison.

    Reports:
        - deadlock_count: Distinct WFG-confirmed deadlock episodes.
        - deadlock_frequency: Confirmed deadlocks per 100 timesteps.
        - deadlock_types: Per-category counts (corridor/cyclic/congestion/
          goal_blocking).
        - avg_stagnation: Average stagnation duration (legacy heuristic).
    """

    def __init__(self, env):
        super().__init__(env)
        self._total_stagnation = 0
        self._stagnation_samples = 0

    def reset(self, **kwargs):
        self._total_stagnation = 0
        self._stagnation_samples = 0
        self._steps = 0
        self._metric_value = {}
        return self.env.reset(**kwargs)

    def step(self, actions):
        obs, rewards, terminated, truncated, infos = self.env.step(actions)
        self._steps += 1
        n = len(infos)

        for i in range(n):
            stag = infos[i].get("stagnation_count", 0)
            if stag > 0:
                self._total_stagnation += stag
                self._stagnation_samples += 1

        # WFG-confirmed deadlock state is identical across agents' info dicts.
        last = infos[-1] if infos else {}
        deadlock_count = last.get("deadlock_count", 0)

        self._metric_value = {
            "deadlock_count": deadlock_count,
            "deadlock_frequency": round(
                deadlock_count / self._steps * 100, 2
            ) if self._steps > 0 else 0,
            "deadlock_types": last.get("deadlock_types", {}),
            "active_deadlocks": last.get("active_deadlocks", 0),
            "avg_stagnation": round(
                self._total_stagnation / self._stagnation_samples, 2
            ) if self._stagnation_samples > 0 else 0,
        }

        if infos:
            infos[-1].update(self._metric_value)

        return obs, rewards, terminated, truncated, infos


class PathEfficiencyMetric(_MetricWrapper):
    """Path efficiency — actual path length vs. Manhattan distance optimal.

    Reports ratio: 1.0 = optimal, >1.0 = suboptimal.
    """

    def __init__(self, env):
        super().__init__(env)
        self._optimal_dists = {}

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self._optimal_dists = {}
        self._arrival_times = {}
        self._steps = 0
        self._metric_value = 0.0

        # Compute optimal distances
        base_env = self.env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        if hasattr(base_env, 'grid') and base_env.grid is not None:
            for i in range(base_env.config.num_agents):
                start = base_env.grid.positions_xy[i]
                goal = base_env.grid.targets_xy[i]
                self._optimal_dists[i] = abs(start[0] - goal[0]) + abs(start[1] - goal[1])

        return result

    def step(self, actions):
        obs, rewards, terminated, truncated, infos = self.env.step(actions)
        self._steps += 1
        n = len(infos)

        for i in range(n):
            if infos[i].get("on_goal", False) and i not in self._arrival_times:
                self._arrival_times[i] = self._steps

        if self._arrival_times and self._optimal_dists:
            ratios = []
            for agent_id, arrival in self._arrival_times.items():
                opt = self._optimal_dists.get(agent_id, 1)
                if opt > 0:
                    ratios.append(arrival / opt)
            self._metric_value = sum(ratios) / len(ratios) if ratios else 0.0

        if infos:
            infos[-1]["path_efficiency"] = round(self._metric_value, 3)

        return obs, rewards, terminated, truncated, infos
