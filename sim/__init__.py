"""
DRIMAPSim — a MAPF simulation environment with built-in deadlock analysis

A Gymnasium- and PettingZoo-compatible multi-agent path finding environment.
Its distinguishing feature is first-class, validated execution-time deadlock
instrumentation: a per-step wait-for-graph monitor that detects and classifies
genuine wait-cycles (not the stagnation heuristic), plus a provably
collision-free movement model, deterministic rollouts, diverse map generators,
and standard MAPF metrics.

Usage:
    from sim import drimapsim_v0, EnvConfig

    env = drimapsim_v0(EnvConfig(num_agents=16, size=32, density=0.2))
    obs, info = env.reset()
    while True:
        actions = env.sample_actions()
        obs, rewards, terminated, truncated, info = env.step(actions)
        if all(terminated) or all(truncated):
            break

    # PettingZoo ParallelEnv flavour:
    from sim import drimapsim_parallel_v0
    penv = drimapsim_parallel_v0(EnvConfig(num_agents=16))
    obs, infos = penv.reset(seed=0)
"""

from sim.env_config import EnvConfig
from sim.environment import DRIMAPSimEnv
from sim.metrics import (
    ISRMetric,
    CSRMetric,
    EpLengthMetric,
    MakespanMetric,
    SumOfCostsMetric,
    ThroughputMetric,
    DeadlockMetric,
)
from sim.wrappers import MultiTimeLimit, RecordTrajectory


def drimapsim_v0(config: EnvConfig = None) -> DRIMAPSimEnv:
    """Create a DRIMAPSim environment with standard wrappers.

    Mirrors POGEMA's ``pogema_v0`` factory function.

    Args:
        config: Environment configuration. Uses defaults if None.

    Returns:
        Wrapped Gymnasium environment.
    """
    config = config or EnvConfig()
    env = DRIMAPSimEnv(config)
    env = MultiTimeLimit(env, config.max_episode_steps)
    env = ISRMetric(env)
    env = CSRMetric(env)
    env = EpLengthMetric(env)
    env = MakespanMetric(env)
    env = SumOfCostsMetric(env)
    if config.deadlock_tracking:
        env = DeadlockMetric(env)
    return env


from sim.pettingzoo_env import (
    drimapsim_parallel_v0,
    DRIMAPSimParallelEnv,
    HAS_PETTINGZOO,
)
from sim.benchmark import BenchmarkScenario, load_benchmark_env

__all__ = [
    "drimapsim_v0",
    "drimapsim_parallel_v0",
    "EnvConfig",
    "DRIMAPSimEnv",
    "DRIMAPSimParallelEnv",
    "HAS_PETTINGZOO",
    "BenchmarkScenario",
    "load_benchmark_env",
]
