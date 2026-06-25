"""
DRIMAPSim PettingZoo ParallelEnv adapter
========================================

Exposes DRIMAPSim through the PettingZoo ``ParallelEnv`` API (dict-keyed
observations/rewards/terminations/truncations/infos), the standard interface
for cooperative multi-agent RL and the one POGEMA also supports. This broadens
DRIMAPSim's reach to the PettingZoo / MARL ecosystem.

The adapter subclasses ``pettingzoo.ParallelEnv`` when the package is
installed, and otherwise provides the identical API as a standalone class so it
works (and is testable) without the optional dependency.
"""

from typing import Dict, List, Optional

from sim.env_config import EnvConfig
from sim.environment import DRIMAPSimEnv

try:  # optional dependency
    from pettingzoo import ParallelEnv as _ParallelEnvBase
    HAS_PETTINGZOO = True
except Exception:  # pragma: no cover - exercised only when pettingzoo absent
    _ParallelEnvBase = object
    HAS_PETTINGZOO = False


class DRIMAPSimParallelEnv(_ParallelEnvBase):
    """PettingZoo ParallelEnv wrapper around :class:`DRIMAPSimEnv`.

    Agents are named ``"agent_0" ... "agent_{n-1}"``. In ``finish`` mode agents
    are removed from :attr:`agents` once they reach their goal, following the
    PettingZoo contract (an agent appears in the dicts on the step it
    terminates, then is dropped on subsequent steps).
    """

    metadata = {"name": "drimapsim_parallel_v0", "render_modes": ["ansi"]}

    def __init__(self, config: Optional[EnvConfig] = None) -> None:
        self._env = DRIMAPSimEnv(config or EnvConfig())
        # Materialise the grid once so the true agent count is known up front
        # (it can differ from the config when explicit start/goal lists or a
        # small map reduce the realisable number of agents).
        self._env.reset(seed=self._env.config.seed)
        n = self._env.config.num_agents
        self.possible_agents: List[str] = [f"agent_{i}" for i in range(n)]
        self.agents: List[str] = list(self.possible_agents)
        self._idx: Dict[str, int] = {
            a: i for i, a in enumerate(self.possible_agents)
        }

    # --- PettingZoo required API ---------------------------------------
    def observation_space(self, agent: str):
        return self._env.observation_space

    def action_space(self, agent: str):
        return self._env.action_space

    @property
    def num_agents(self) -> int:
        return len(self.agents)

    @property
    def max_num_agents(self) -> int:
        return len(self.possible_agents)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        obs_list, info_list = self._env.reset(seed=seed, options=options)
        self.agents = list(self.possible_agents)
        obs = {a: obs_list[self._idx[a]] for a in self.agents}
        infos = {a: info_list[self._idx[a]] for a in self.agents}
        return obs, infos

    def step(self, actions: Dict[str, int]):
        # Missing agents (already done) implicitly idle.
        act_list = [0] * len(self.possible_agents)
        for a, act in actions.items():
            act_list[self._idx[a]] = int(act)

        obs_l, rew_l, term_l, trunc_l, info_l = self._env.step(act_list)
        live = list(self.agents)
        obs = {a: obs_l[self._idx[a]] for a in live}
        rewards = {a: float(rew_l[self._idx[a]]) for a in live}
        terminations = {a: bool(term_l[self._idx[a]]) for a in live}
        truncations = {a: bool(trunc_l[self._idx[a]]) for a in live}
        infos = {a: info_l[self._idx[a]] for a in live}

        # Drop agents that finished this step (kept in this step's dicts).
        self.agents = [
            a for a in live if not (terminations[a] or truncations[a])
        ]
        return obs, rewards, terminations, truncations, infos

    def render(self):
        return self._env.render(mode="ansi")

    def close(self):
        self._env.close()

    def state(self):
        """Global obstacle/agent state (optional PettingZoo hook)."""
        return self._env.get_obstacles()


def drimapsim_parallel_v0(config: Optional[EnvConfig] = None) -> DRIMAPSimParallelEnv:
    """Factory for the PettingZoo ParallelEnv flavour of DRIMAPSim.

    Mirrors PettingZoo's ``*_v0`` factory convention (and POGEMA's parallel
    interface).
    """
    return DRIMAPSimParallelEnv(config or EnvConfig())
