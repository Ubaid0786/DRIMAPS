"""
DRIMAPSim Main Environment — Gymnasium-Compatible MAPF Environment

Full Gymnasium API with multi-agent support, three MAPF modes
(standard, cooperative, lifelong), and built-in deadlock tracking.
"""

import numpy as np
import gymnasium
from typing import Dict, List, Optional, Tuple

from sim.env_config import EnvConfig, MOVES
from sim.grid_world import GridWorld
from sim.deadlock_monitor import DeadlockMonitor


class DRIMAPSimEnv(gymnasium.Env):
    """Multi-agent path finding Gymnasium environment.

    Supports three MAPF modes:
        - ``finish``: Agents disappear on goal (standard MAPF).
        - ``nothing``: All agents must reach goals (cooperative).
        - ``restart``: Agents get new targets on goal (lifelong).

    Observations are per-agent 3-channel tensors:
        Channel 0: obstacles (partial or full)
        Channel 1: other agents
        Channel 2: target marker

    Actions: 0=idle, 1=up, 2=down, 3=left, 4=right.

    Attributes:
        config: Environment configuration.
        grid: Grid world engine.
    """

    metadata = {"render_modes": ["ansi", "human"]}

    def __init__(self, config: EnvConfig = None):
        """Initialize the environment.

        Args:
            config: Environment configuration.
        """
        super().__init__()
        self.config = config or EnvConfig()
        self.grid: Optional[GridWorld] = None
        self.deadlock_monitor: Optional[DeadlockMonitor] = None
        self._step_count = 0
        self._episode_done = False
        # Dedicated RNG for action sampling so rollouts are reproducible from
        # the configured seed (the global numpy RNG is not used).
        self._action_rng = np.random.RandomState(self.config.seed)

        # Action space: 5 discrete actions per agent
        self.action_space = gymnasium.spaces.Discrete(len(MOVES))

        # Observation space
        full_size = self.config.obs_radius * 2 + 1
        self.observation_space = gymnasium.spaces.Box(
            0.0, 1.0, shape=(3, full_size, full_size), dtype=np.float32
        )

        # Track per-agent arrival for metrics
        self._was_on_goal: List[bool] = []
        self._arrived_at: Dict[int, int] = {}  # agent → step when arrived

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment to initial state.

        Args:
            seed: Random seed override.
            options: Additional options.

        Returns:
            (observations, infos) tuple.
        """
        if seed is not None:
            self.config.seed = seed

        self.grid = GridWorld(self.config)
        self._action_rng = np.random.RandomState(self.config.seed)
        # Wait-for-graph deadlock monitor over the realised agent set.
        if self.config.deadlock_tracking:
            self.deadlock_monitor = DeadlockMonitor(
                self.config.num_agents,
                self.grid.obstacles,
                self.grid.get_targets_xy(),
                stagnation_threshold=5,
            )
        else:
            self.deadlock_monitor = None
        self._step_count = 0
        self._episode_done = False
        self._was_on_goal = [False] * self.config.num_agents
        self._arrived_at = {}

        obs = self._get_observations()
        infos = self._get_infos()
        return obs, infos

    def step(self, actions: list):
        """Execute one timestep.

        Args:
            actions: List of action indices, one per agent.

        Returns:
            (observations, rewards, terminated, truncated, infos).
        """
        if self.grid is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        assert len(actions) == self.config.num_agents
        self._step_count += 1

        # Execute movement
        step_result = self.grid.step(actions)

        # --- Wait-for-graph deadlock monitoring (this step's contention) ---
        dl_state = None
        if self.deadlock_monitor is not None:
            n = self.config.num_agents
            finished = {
                i for i in range(n)
                if self.grid.on_goal(i) or not self.grid.is_active[i]
            }
            # Targets may change in lifelong mode; keep the monitor in sync.
            self.deadlock_monitor.goals = self.grid.get_targets_xy()
            dl_state = self.deadlock_monitor.update(
                step_result["current"], step_result["desired"], finished
            )

        # Update arrival tracking
        self._was_on_goal = step_result["on_goal"]
        for i in range(self.config.num_agents):
            if step_result["on_goal"][i] and i not in self._arrived_at:
                self._arrived_at[i] = self._step_count

        # Compute rewards and termination
        rewards, terminated, truncated = self._compute_outcomes(step_result)

        # Handle on_target behavior
        if self.config.on_target == "finish":
            for i in range(self.config.num_agents):
                if self.grid.on_goal(i) and self.grid.is_active[i]:
                    self.grid.hide_agent(i)

        elif self.config.on_target == "restart":
            for i in range(self.config.num_agents):
                if self.grid.on_goal(i):
                    self.grid.generate_new_target(i)

        obs = self._get_observations()
        infos = self._get_infos()

        # Add deadlock info: both the legacy stagnation signal and the genuine
        # wait-for-graph deadlock state, so the two can be compared directly.
        if self.config.deadlock_tracking:
            dl_agents = dl_state["agents"] if dl_state else set()
            dl_stats = self.deadlock_monitor.stats() if dl_state else {}
            for i in range(self.config.num_agents):
                infos[i]["deadlock_involved"] = i in dl_agents
                infos[i]["stagnation_count"] = (
                    self.grid._stagnation_counter.get(i, 0)
                )
                # Cumulative count of distinct WFG-confirmed deadlock episodes,
                # and the per-type breakdown (the structural taxonomy).
                infos[i]["deadlock_count"] = dl_stats.get("deadlock_count", 0)
                infos[i]["deadlock_types"] = dl_stats.get("type_counts", {})
                infos[i]["active_deadlocks"] = dl_stats.get("active_deadlocks", 0)

        return obs, rewards, terminated, truncated, infos

    def _compute_outcomes(self, step_result):
        """Compute rewards, terminated, and truncated flags."""
        n = self.config.num_agents
        rewards = [0.0] * n
        terminated = [False] * n
        truncated = [False] * n

        if self.config.on_target == "finish":
            for i in range(n):
                on_goal = step_result["on_goal"][i]
                if on_goal and self.grid.is_active[i]:
                    rewards[i] = 1.0
                terminated[i] = on_goal

        elif self.config.on_target == "nothing":
            # Cooperative: all must be on goal simultaneously
            all_on = all(step_result["on_goal"])
            for i in range(n):
                terminated[i] = all_on
                rewards[i] = 1.0 if all_on else 0.0

        elif self.config.on_target == "restart":
            # Lifelong: reward for each goal reached
            for i in range(n):
                if step_result["on_goal"][i]:
                    rewards[i] = 1.0

            terminated = [False] * n  # Never terminates in lifelong

        return rewards, terminated, truncated

    def _get_observations(self):
        """Build observations for all agents."""
        obs = []
        for i in range(self.config.num_agents):
            if self.config.observation_type == "full":
                obs.append(self._get_full_obs(i))
            else:
                obs.append(self._get_local_obs(i))
        return obs

    def _get_local_obs(self, agent_id: int) -> np.ndarray:
        """Get 3-channel local observation for an agent."""
        obstacles = self.grid.get_obstacles_for_agent(agent_id)
        agents = self.grid.get_agents_for_agent(agent_id)
        target = self.grid.get_target_channel(agent_id)
        return np.stack([obstacles, agents, target], axis=0)

    def _get_full_obs(self, agent_id: int) -> np.ndarray:
        """Get full-grid observation (global state)."""
        local = self._get_local_obs(agent_id)
        return local  # Can be extended with global info

    def _get_infos(self):
        """Build info dict for each agent."""
        infos = []
        for i in range(self.config.num_agents):
            info = {
                "is_active": self.grid.is_active[i],
                "on_goal": self.grid.on_goal(i),
                "step": self._step_count,
            }
            infos.append(info)
        return infos

    def render(self, mode="human"):
        """Render the environment.

        Args:
            mode: Rendering mode ("human" for ASCII to stdout,
                  "ansi" for ASCII string).

        Returns:
            ASCII string in "ansi" mode, None in "human" mode.
        """
        if self.grid is None:
            return None

        ascii_str = self.grid.render_ascii()
        if mode == "human":
            print(ascii_str)
            print(f"Step: {self._step_count}")
            return None
        return ascii_str

    def sample_actions(self) -> List[int]:
        """Sample random actions for all agents from the env's seeded RNG.

        Uses a dedicated, seed-derived RNG (not the global numpy RNG) so that
        a full rollout is reproducible: same seed -> identical action stream
        and trajectory.

        Returns:
            List of random action indices.
        """
        return list(
            self._action_rng.randint(0, len(MOVES), size=self.config.num_agents)
        )

    def get_deadlock_stats(self) -> dict:
        """Aggregate wait-for-graph deadlock statistics for the episode.

        Returns a dict with ``deadlock_count`` (distinct confirmed deadlock
        episodes), ``type_counts`` (per structural category), ``active_deadlocks``
        and ``total_new_events``. Empty if deadlock tracking is disabled.
        """
        return self.deadlock_monitor.stats() if self.deadlock_monitor else {}

    def get_congestion_heatmap(self):
        """Per-cell contention heatmap accumulated over the episode (or None)."""
        return (
            self.deadlock_monitor.congestion_heatmap()
            if self.deadlock_monitor else None
        )

    def close(self):
        """Clean up resources."""
        pass

    # ── Utility properties ──────────────────────────────────────────

    @property
    def num_agents(self) -> int:
        """Number of agents."""
        return self.config.num_agents

    @property
    def step_count(self) -> int:
        """Current step count."""
        return self._step_count

    def get_agents_xy(self):
        """Get current agent positions."""
        return self.grid.get_agents_xy() if self.grid else []

    def get_targets_xy(self):
        """Get target positions."""
        return self.grid.get_targets_xy() if self.grid else []

    def get_obstacles(self):
        """Get obstacle grid."""
        return self.grid.get_obstacles() if self.grid else None

    def get_deadlock_events(self):
        """Get the list of deadlock events recorded by the grid world."""
        return self.grid.get_deadlock_events() if self.grid else []
