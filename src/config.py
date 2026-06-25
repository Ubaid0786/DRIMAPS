#!/usr/bin/env python3
"""
DRIMAPS Configuration Module

Centralizes all algorithm parameters, constants, and configuration
into typed dataclasses. No magic numbers anywhere in the codebase.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class DeadlockType(Enum):
    """Classification of deadlock structural types."""
    CORRIDOR = "corridor"
    CYCLIC = "cyclic"
    CONGESTION = "congestion"
    GOAL_BLOCKING = "goal_blocking"
    UNKNOWN = "unknown"


class ResolutionStrategy(Enum):
    """Available deadlock resolution strategies."""
    CORRIDOR_BACKTRACK = "corridor_backtrack"
    CYCLIC_REROUTE = "cyclic_reroute"
    CONGESTION_TOKEN = "congestion_token"
    GOAL_SWAP = "goal_swap"
    FALLBACK_REPLAN = "fallback_replan"


class InitialSolver(Enum):
    """Pluggable initial path computation solvers.

    INDEPENDENT plans each agent's shortest path while ignoring all other
    agents. The resulting routes are fast to compute but conflict-oblivious,
    so they deadlock during execution — which is precisely the regime the
    DRIMAPS runtime resolution layer is designed to repair. COOPERATIVE_ASTAR
    and PRIORITIZED produce (near) conflict-free reservations and are used by
    the prevention-only baseline.
    """
    INDEPENDENT = "independent"
    COOPERATIVE_ASTAR = "cooperative_astar"
    PRIORITIZED = "prioritized"


@dataclass
class DRIMAPSConfig:
    """Master configuration for the DRIMAPS algorithm.

    All algorithm parameters are centralized here. Defaults are
    tuned for 32x32 grids with up to 500 agents.

    Attributes:
        initial_solver: Which solver to use for initial path computation.
        search_horizon: Maximum timesteps for bounded Space-Time A* repair.
        max_escalation_depth: Maximum safety-check escalation levels before
            falling back to full replanning of involved agents.
        detection_interval: Run cycle detection every k timesteps.
            Set to 1 for every-timestep detection.
        corridor_degree_threshold: Max cell degree to classify as corridor.
        congestion_radius: Radius (cells) defining a congestion region.
        congestion_agent_threshold: Min agents in radius to flag congestion.
        priority_alpha: Weight for distance-to-goal in priority computation.
        priority_beta: Weight for remaining-path-length in priority.
        timeout: Wall-clock timeout in seconds.
        seeds: Random seeds for reproducibility.
        max_timesteps: Hard cap on simulation timesteps.
        bypass_min_degree: Minimum cell degree to qualify as a bypass cell.
        enable_safety_verification: Toggle post-resolution safety checks.
        enable_classification: Toggle type-specific resolution vs uniform.
        enable_cycle_detection: Toggle Tarjan vs stagnation-only detection.
        enable_minimal_disruption: Toggle minimal vs full replanning.
        verbose: Print progress information.
    """
    # --- Solver Selection ---
    # DRIMAPS pairs a fast, conflict-oblivious initial planner with runtime
    # deadlock resolution; INDEPENDENT planning is the setting under which
    # execution-time deadlocks actually arise and the framework is exercised.
    initial_solver: InitialSolver = InitialSolver.INDEPENDENT

    # --- Search Parameters ---
    search_horizon: int = 30
    max_escalation_depth: int = 3
    detection_interval: int = 1
    max_timesteps: int = 1000
    stagnation_threshold: int = 5
    max_expanded_nodes: int = 10000
    focal_weight: float = 1.5
    enable_random_perturbation: bool = True

    # --- Classification Thresholds ---
    corridor_degree_threshold: int = 2
    congestion_radius: int = 3
    congestion_agent_threshold: int = 4
    bypass_min_degree: int = 3
    # Steps a yielding agent holds at a bypass cell so higher-priority traffic
    # can clear before it resumes toward its goal.
    yield_hold_steps: int = 3
    # Steps before the same confirmed deadlock may be resolved again. A larger
    # cooldown avoids thrashing (repeatedly disrupting the same agents) and is
    # the main lever trading disruption against responsiveness.
    resolution_cooldown: int = 10
    # Maximum length of a priority-inheritance yield chain. When the agent
    # asked to yield can only step into a cell held by another agent, that
    # agent is recursively asked to yield as well; this bounds the recursion
    # and hence the number of agents one resolution may touch.
    max_chain_depth: int = 6

    # --- Priority Weights ---
    priority_alpha: float = 1.0
    priority_beta: float = 0.5

    # --- Runtime Limits ---
    timeout: float = 60.0

    # --- Ablation Toggles ---
    enable_safety_verification: bool = True
    enable_classification: bool = True
    enable_cycle_detection: bool = True
    enable_minimal_disruption: bool = True

    # --- Reproducibility ---
    seeds: List[int] = field(
        default_factory=lambda: [42, 123, 456, 789, 1024]
    )

    # --- Output ---
    verbose: bool = False


@dataclass
class DeadlockInfo:
    """Information about a single detected deadlock.

    Attributes:
        deadlock_type: Structural classification.
        agents: List of agent indices involved in the deadlock.
        cycle: Ordered cycle of agent indices (for cyclic deadlocks).
        timestep: Timestep at which the deadlock was detected.
        positions: Current positions of involved agents.
        confidence: Detection confidence score in [0, 1].
    """
    deadlock_type: DeadlockType
    agents: List[int]
    cycle: List[int]
    timestep: int
    positions: List[tuple]
    confidence: float = 1.0


@dataclass
class ResolutionResult:
    """Result of a deadlock resolution attempt.

    Attributes:
        success: Whether the deadlock was resolved.
        strategy_used: Which resolution strategy was applied.
        agents_modified: Indices of agents whose paths were changed.
        cost_increase: Total increase in path cost from resolution.
        resolution_time_ms: Time spent on resolution in milliseconds.
        escalation_level: How many escalation rounds were needed.
    """
    success: bool
    strategy_used: ResolutionStrategy
    agents_modified: List[int]
    cost_increase: int = 0
    resolution_time_ms: float = 0.0
    escalation_level: int = 0


@dataclass
class DRIMAPSResult:
    """Complete result from a DRIMAPS solve call.

    Attributes:
        success: Whether all agents reached their goals.
        paths: Final conflict-free paths for all agents.
        makespan: Maximum path length across all agents.
        sum_of_costs: Total path length across all agents.
        runtime: Total wall-clock time in seconds.
        deadlocks_detected: Number of deadlocks found during execution.
        deadlocks_resolved: Number of deadlocks successfully resolved.
        agents_replanned: Total count of agent replanning events.
        resolution_results: List of individual resolution outcomes.
        initial_sum_of_costs: Sum of costs from the initial solver.
        wfg_update_time: Cumulative WFG maintenance time in seconds.
        detection_time: Cumulative cycle detection time in seconds.
        resolution_time: Cumulative resolution time in seconds.
        initial_planning_time: Time for initial path computation.
        collision_count: Post-resolution conflicts (should be 0).
        peak_memory_mb: Peak memory usage in megabytes.
    """
    success: bool = False
    paths: List[List[tuple]] = field(default_factory=list)
    makespan: int = 0
    sum_of_costs: int = 0
    runtime: float = 0.0
    deadlocks_detected: int = 0
    deadlocks_resolved: int = 0
    agents_replanned: int = 0
    resolution_results: List[ResolutionResult] = field(default_factory=list)
    initial_sum_of_costs: int = 0
    wfg_update_time: float = 0.0
    detection_time: float = 0.0
    resolution_time: float = 0.0
    initial_planning_time: float = 0.0
    collision_count: int = 0
    peak_memory_mb: float = 0.0


# --- Map Configuration for Experiments ---
@dataclass
class MapConfig:
    """Configuration for a benchmark map instance.

    Attributes:
        map_id: Short identifier (e.g., "M1").
        map_type: Generator type name.
        width: Grid width.
        height: Grid height.
        description: Human-readable description.
        extra_kwargs: Additional generator parameters.
    """
    map_id: str
    map_type: str
    width: int
    height: int
    description: str = ""
    extra_kwargs: dict = field(default_factory=dict)


# Pre-defined benchmark maps from Section 6.1
BENCHMARK_MAPS: List[MapConfig] = [
    MapConfig("M1", "random", 32, 32, "20% random obstacles",
              {"obstacle_density": 0.20}),
    MapConfig("M2", "random", 64, 64, "20% random obstacles (large)",
              {"obstacle_density": 0.20}),
    MapConfig("M3", "warehouse", 32, 32, "Parallel aisles, shelves"),
    MapConfig("M4", "warehouse", 64, 64, "Large warehouse"),
    MapConfig("M5", "corridor", 32, 32, "Narrow passages, bottlenecks"),
    MapConfig("M6", "bottleneck", 32, 32, "Central wall, single passage"),
    MapConfig("M7", "maze", 32, 32, "Recursive backtracking maze"),
    MapConfig("M8", "random", 32, 32, "35% random obstacles (dense)",
              {"obstacle_density": 0.35}),
]

# Agent count sweep
AGENT_COUNTS: List[int] = [10, 20, 50, 100, 150, 200, 300, 400, 500]

# Experiment seeds
EXPERIMENT_SEEDS: List[int] = [42, 123, 456, 789, 1024]
