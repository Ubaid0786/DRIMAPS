"""
DRIMAPSim Environment Configuration

Superset of POGEMA's GridConfig — supports all POGEMA features plus
deadlock tracking, diverse map types, difficulty presets, trajectory
recording, and strict collision mode.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
from enum import Enum


class CollisionSystem(Enum):
    """How agent collisions are handled."""
    PRIORITY = "priority"       # First-come-first-served
    BLOCK_BOTH = "block_both"   # Both agents stay in place
    SOFT = "soft"               # Collisions permitted & tracked, not penalised by default
    STRICT = "strict"           # Collisions forbidden, instant fail


class OnTarget(Enum):
    """What happens when an agent reaches its target."""
    FINISH = "finish"           # Agent disappears (standard MAPF)
    NOTHING = "nothing"         # Agent stays, all must reach (cooperative)
    RESTART = "restart"         # New target assigned (lifelong MAPF)


class ObservationType(Enum):
    """Observation format returned to agents."""
    DEFAULT = "default"         # 3-channel tensor (obstacles, agents, target)
    FULL = "full"               # Global state + local obs
    VECTOR = "vector"           # Flat vector encoding


class Difficulty(Enum):
    """Preset difficulty levels that auto-configure density and agents."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXTREME = "extreme"
    CUSTOM = "custom"


# Actions: idle, up, down, left, right
MOVES = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_NAMES = ["idle", "up", "down", "left", "right"]


@dataclass
class EnvConfig:
    """Complete environment configuration.

    Attributes:
        size: Grid size (size x size).
        density: Obstacle density [0, 1].
        num_agents: Number of agents.
        obs_radius: Observation radius for partial observability.
        seed: Random seed (None for random).
        map_type: Built-in map type or None for random.
        map: Custom map as 2D list or string.
        collision_system: Collision handling mode.
        on_target: Behavior when agent reaches target.
        observation_type: Observation format.
        max_episode_steps: Episode step limit.
        difficulty: Difficulty preset (overrides density/agents if not CUSTOM).
        deadlock_tracking: Enable runtime deadlock detection and logging.
        record_history: Record full trajectory for replay.
        agents_xy: Explicit start positions (overrides random).
        targets_xy: Explicit target positions (overrides random).
    """
    # --- Grid ---
    size: int = 32
    density: float = 0.2
    num_agents: int = 8
    obs_radius: int = 5
    seed: Optional[int] = None

    # --- Map ---
    map_type: Optional[str] = None  # random, warehouse, corridor, bottleneck, maze, room, dense_random, open
    map: Optional[Union[list, str]] = None
    map_name: Optional[str] = None

    # --- Mechanics ---
    collision_system: str = "block_both"
    on_target: str = "finish"
    observation_type: str = "default"
    max_episode_steps: int = 256

    # --- Difficulty ---
    difficulty: str = "custom"

    # --- DRIMAPSim Unique ---
    deadlock_tracking: bool = True
    record_history: bool = False

    # --- Explicit positions ---
    agents_xy: Optional[List[Tuple[int, int]]] = None
    targets_xy: Optional[List[Tuple[int, int]]] = None

    # --- Constants ---
    FREE: int = 0
    OBSTACLE: int = 1

    def __post_init__(self):
        """Apply difficulty presets if not custom."""
        if self.difficulty == "easy":
            self.density = min(self.density, 0.1)
            self.num_agents = min(self.num_agents, 4)
        elif self.difficulty == "medium":
            self.density = max(self.density, 0.2)
            self.num_agents = max(self.num_agents, 16)
        elif self.difficulty == "hard":
            self.density = max(self.density, 0.3)
            self.num_agents = max(self.num_agents, 64)
        elif self.difficulty == "extreme":
            self.density = max(self.density, 0.35)
            self.num_agents = max(self.num_agents, 128)

    def validate(self):
        """Validate configuration parameters."""
        assert 2 <= self.size <= 1024, f"size must be in [2, 1024], got {self.size}"
        assert 0.0 <= self.density <= 1.0, f"density must be in [0, 1], got {self.density}"
        assert 1 <= self.num_agents <= 10000, f"num_agents must be in [1, 10000], got {self.num_agents}"
        assert 1 <= self.obs_radius <= 128, f"obs_radius must be in [1, 128], got {self.obs_radius}"
        assert self.collision_system in ("priority", "block_both", "soft", "strict"), \
            f"Unknown collision_system: {self.collision_system}"
        assert self.on_target in ("finish", "nothing", "restart"), \
            f"Unknown on_target: {self.on_target}"
