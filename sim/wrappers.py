"""
DRIMAPSim Utility Wrappers

Time limit, trajectory recording, and integration wrappers.
"""

import json
import gymnasium
import numpy as np
from typing import List, Optional


class DRIMAPSimWrapper(gymnasium.Wrapper):
    """Base wrapper that forwards DRIMAPSim's helper API to the wrapped env.

    Gymnasium 1.0 removed ``Wrapper.__getattr__``, so wrappers no longer
    transparently delegate attribute access to the env they wrap. That broke
    the DRIMAPSim helper methods (``get_agents_xy``, ``sample_actions``,
    ``get_deadlock_stats``, ...) and the ``mode`` argument on ``render`` for any
    wrapped environment. This base restores that delegation explicitly: any
    public attribute not defined on the wrapper is looked up on the inner env,
    and ``render`` forwards its arguments through.
    """

    def __getattr__(self, name):
        # __getattr__ only fires when normal lookup fails. Guard ``env`` and
        # dunder/private names to avoid recursing before the wrapper is fully
        # initialised (gymnasium sets ``self.env`` in ``Wrapper.__init__``).
        if name == "env" or name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.env, name)

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)


class MultiTimeLimit(DRIMAPSimWrapper):
    """Episode time limit wrapper — truncates after max_steps.

    Matches POGEMA's MultiTimeLimit behavior.
    """

    def __init__(self, env, max_steps: int = 256):
        super().__init__(env)
        self.max_steps = max_steps
        self._elapsed = 0

    def reset(self, **kwargs):
        self._elapsed = 0
        return self.env.reset(**kwargs)

    def step(self, actions):
        obs, rewards, terminated, truncated, infos = self.env.step(actions)
        self._elapsed += 1

        if self._elapsed >= self.max_steps:
            truncated = [True] * len(truncated)

        return obs, rewards, terminated, truncated, infos


class RecordTrajectory(DRIMAPSimWrapper):
    """Records full agent trajectories for replay and analysis.

    Stores positions at each timestep and can export to JSON.
    """

    def __init__(self, env):
        super().__init__(env)
        self._trajectories: List[List[List[int]]] = []
        self._actions_log: List[List[int]] = []
        self._rewards_log: List[List[float]] = []

    def _record_positions(self):
        """Append current agent positions as native-int pairs (JSON-safe)."""
        base_env = self.env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        if hasattr(base_env, 'grid') and base_env.grid is not None:
            positions = base_env.grid.get_agents_xy()
            self._trajectories.append([[int(c) for c in p] for p in positions])

    def reset(self, **kwargs):
        self._trajectories = []
        self._actions_log = []
        self._rewards_log = []
        result = self.env.reset(**kwargs)
        self._record_positions()
        return result

    def step(self, actions):
        obs, rewards, terminated, truncated, infos = self.env.step(actions)
        # Coerce to native Python scalars so the log is JSON-serialisable
        # (env actions/rewards may be numpy int64/float64).
        self._actions_log.append([int(a) for a in actions])
        self._rewards_log.append([float(r) for r in rewards])
        self._record_positions()
        return obs, rewards, terminated, truncated, infos

    def export_json(self, filepath: str):
        """Export recorded trajectory to JSON file.

        Args:
            filepath: Output file path.
        """
        data = {
            "trajectories": self._trajectories,
            "actions": self._actions_log,
            "rewards": self._rewards_log,
            "num_agents": len(self._trajectories[0]) if self._trajectories else 0,
            "num_steps": len(self._actions_log),
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def get_trajectories(self):
        """Return recorded trajectories."""
        return self._trajectories


class NormalizeObservation(DRIMAPSimWrapper):
    """Normalize observations to [0, 1] range."""

    def reset(self, **kwargs):
        obs, infos = self.env.reset(**kwargs)
        return [self._normalize(o) for o in obs], infos

    def step(self, actions):
        obs, rewards, terminated, truncated, infos = self.env.step(actions)
        return [self._normalize(o) for o in obs], rewards, terminated, truncated, infos

    @staticmethod
    def _normalize(obs):
        if isinstance(obs, np.ndarray):
            obs_min = obs.min()
            obs_max = obs.max()
            if obs_max - obs_min > 0:
                return (obs - obs_min) / (obs_max - obs_min)
        return obs


class FlattenActions(DRIMAPSimWrapper):
    """Convert multi-agent env to single flat action space.

    Useful for single-agent RL algorithms operating on the joint action.
    """

    def __init__(self, env, num_agents: int):
        super().__init__(env)
        self.num_agents = num_agents
        self.action_space = gymnasium.spaces.MultiDiscrete(
            [5] * num_agents
        )

    def step(self, actions):
        if isinstance(actions, np.ndarray):
            actions = actions.tolist()
        return self.env.step(actions)
