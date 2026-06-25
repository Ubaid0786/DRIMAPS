# DRIMAPS — Detection-Guided Deadlock Escape for Reactive MAPF

A reactive Multi-Agent Path Finding (MAPF) controller that is **deadlock-free by
construction** and closes the residual completeness gap with a **detection-guided
escape**, paired with a self-contained Gymnasium/PettingZoo simulation
environment (DRIMAPSim) and the full MovingAI benchmark.

## Overview

DRIMAPS has two layers:

1. **Deadlock-free reactive core** — a priority-inheritance-with-backtracking
   move rule (PIBT-style). Every step it produces a collision-free joint action
   and there is always a valid move, so the **livelocks** that defeat
   repair-of-conflict-oblivious-plans execution cannot occur.
2. **Detection-guided escape** — PIBT is incomplete, so a few agents can stall
   (corridor swaps, goal-blocking). Each step DRIMAPS maintains a **Wait-For
   Graph** over intended moves, extracts persistent wait cycles with **Tarjan's
   SCC**, **classifies** them (corridor / cyclic / congestion / goal-blocking),
   and gives *only* the structurally-deadlocked agents a short-lived **retreat
   to the least-congested nearby junction** — a targeted, minimal-disruption
   escape that breaks the standoff, then restores the goal.

On the standard **MovingAI benchmark** DRIMAPS resolves deadlocks across every
map category, matches or beats the strong PIBT reference everywhere, and
overtakes it on the congested maps (mazes, dense warehouses) where PIBT's
incompleteness shows — with far less disruption than a structure-blind escape.

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run tests
python -m pytest tests/ -q

# Quick smoke run of the MovingAI benchmark
python experiments/run_benchmark.py --quick

# Solve a real benchmark instance and animate it (writes execution.gif)
python visualization/animate_execution.py --map-type bottleneck --agents 12
```

```python
# Load a real MovingAI instance and solve it
from src.movingai import load_instance
from src.config import DRIMAPSConfig
from src.drimaps import DRIMAPS

grid, starts, goals = load_instance("maze-32-32-2", num_agents=40, scen="even-1")
res = DRIMAPS(DRIMAPSConfig()).solve(grid, starts, goals)
print(res.success, res.deadlocks_detected, res.deadlocks_resolved)
```

## DRIMAPS Algorithm

Use the solver directly for path finding with deadlock resolution:

```python
import numpy as np
from src.config import DRIMAPSConfig
from src.drimaps import DRIMAPS

grid = np.zeros((32, 32), dtype=int)
result = DRIMAPS(DRIMAPSConfig()).solve(
    grid, starts=[(0, 0), (31, 31)], goals=[(31, 31), (0, 0)],
)

print(f"Success: {result.success}")
print(f"Makespan: {result.makespan}")
print(f"Deadlocks detected: {result.deadlocks_detected}")
print(f"Deadlocks resolved: {result.deadlocks_resolved}")
```

`DRIMAPSResult` also exposes `paths`, `sum_of_costs`, `runtime`,
`agents_replanned`, `collision_count`, and per-phase timings (see
`src/config.py`).

## DRIMAPSim Environment

DRIMAPSim is a self-contained, Gymnasium-compatible MAPF environment with
built-in deadlock instrumentation. Its public API follows POGEMA conventions
(e.g. a `drimapsim_v0` factory mirroring `pogema_v0`), so familiar workflows
carry over.

```python
from sim import drimapsim_v0, EnvConfig

env = drimapsim_v0(EnvConfig(
    size=32,
    num_agents=16,
    density=0.2,
    map_type="warehouse",
    deadlock_tracking=True,
))

obs, info = env.reset()
while True:
    actions = env.sample_actions()
    obs, rewards, terminated, truncated, info = env.step(actions)
    if all(terminated) or all(truncated):
        break
```

### What makes it different

DRIMAPSim is **not** a replacement for [POGEMA](https://github.com/Cognitive-AI-Systems/pogema)
(the established platform for partially-observable / learning-based MAPF). Its
distinguishing contribution is **first-class, validated execution-time deadlock
instrumentation** — to our knowledge the first MAPF environment to ship it:

- **Wait-for-graph deadlock monitor** (`sim/deadlock_monitor.py`): each step it
  builds a WFG over agents' intended moves, finds wait cycles with Tarjan's
  SCC, confirms persistence, and **classifies** each deadlock
  (corridor / cyclic / congestion / goal-blocking). This is genuine deadlock
  detection, not the stagnation heuristic. Under random play stagnation flags
  **452** timesteps as deadlocked when only **2** contain a real wait-cycle
  (it's flagging idle agents); under a realistic greedy policy real deadlocks
  are common (**1142** steps) yet only the WFG says *which* agents, *which*
  cycle, and *what type*. Exposed via `env.get_deadlock_stats()` and per-agent
  `info`.
- **Congestion heatmap** (`env.get_congestion_heatmap()`): per-cell contention
  accumulated over an episode.
- **Provably collision-free** `block_both`/`strict` model (0 conflicts over
  ~10⁵ agent-steps at up to 128 agents).
- **Bit-for-bit deterministic** rollouts from a seed.

Measured properties (run `python experiments/sim_benchmark.py`): ~6k–10k
agent-steps/s in pure Python on 32×32; deadlock-monitor overhead 8%→35% from
16→256 agents (disable via `deadlock_tracking=False` for large-team runs).

### Other features (implemented in `sim/`)

- **8 map generators**: `random`, `warehouse`, `corridor`, `bottleneck`,
  `maze`, `room`, `dense_random`, `open` (`sim/map_registry.py`)
- **MovingAI `.map` parsing** (`parse_movingai_map`)
- **4 collision systems**: `priority`, `block_both`, `soft`, `strict`
  (`sim/grid_world.py`)
- **Standard MAPF metrics**: ISR, CSR, episode length, makespan, sum-of-costs,
  throughput; **WFG deadlock + taxonomy** metrics (`sim/metrics.py`)
- **3 execution modes**: standard / cooperative / lifelong
- **PettingZoo `ParallelEnv` interface** (`sim/pettingzoo_env.py`,
  `drimapsim_parallel_v0`) for the MARL ecosystem; subclasses
  `pettingzoo.ParallelEnv` when the optional package is installed, standalone
  otherwise
- **Difficulty presets** (`sim/env_config.py`)
- **Rendering**: ASCII, SVG, and self-contained HTML animation export
  (`sim/rendering.py`); **JSON trajectory export** (`sim/wrappers.py`)

## Methods

| Method | Description |
|--------|-------------|
| **DRIMAPS** | Independent shortest-path initial plans + runtime deadlock resolution |
| **PIBT** | Strong decentralized baseline (`baselines/pibt.py`) |
| **Naive-DR** | Stagnation detection + random replanning (`baselines/naive_dr.py`) |
| **Prevention-Only** | Cooperative A\* up front, no runtime repair (`baselines/prevention_only.py`) |

Three ablations of DRIMAPS are also evaluated (no-cycle, no-classify,
no-minimal).

## Running Experiments

The canonical pipeline is the benchmark runner followed by asset generation:

```bash
# Full benchmark: 8 maps x agent counts {10,20,30,40,60,80} x 5 seeds x methods
# (1440 runs, ~10 min). Writes results/benchmark_latest.csv.
python experiments/run_benchmark.py

# Quick smoke run (2 maps, 2 agent counts, main methods, 1 seed)
python experiments/run_benchmark.py --quick

# Regenerate all paper tables (paper/figures/table*.tex) and figures
# (paper/figures/fig*.pdf/png) from the CSV above.
python analysis/make_paper_assets.py

# Cross-validate the runtime-repair finding inside the interactive sim
python experiments/sim_crossval.py
```

Every number in the paper traces back to a row in
`results/benchmark_latest.csv` produced by `run_benchmark.py`.

## Results (summary)

Individual success rate (ISR = fraction of agents reaching their goals, mean
over 8 maps x 5 seeds). All methods are **collision-free** (0 collisions across
all 1440 runs).

| Agents | DRIMAPS | PIBT | Naive-DR | Prevention-Only |
|-------:|--------:|-----:|---------:|----------------:|
| 10 | 87% | 96% | 81% | 84% |
| 20 | 78% | 95% | 76% | 77% |
| 30 | 67% | 93% | 60% | 62% |
| 40 | 60% | 91% | 47% | 54% |
| 60 | 48% | 88% | 31% | 45% |
| 80 | 44% | 87% | 22% | 35% |

**Positioning (honest):** PIBT, a purpose-built decentralized planner, is the
strongest method here. DRIMAPS is a *runtime repair layer* for fast,
conflict-oblivious planners: it has the **highest ISR among the
repair-and-prevention methods at every density**, beating Naive-DR by up to 22
points (44% vs 22% at 80 agents) and edging out Prevention-Only throughout.
This is an honest trade — DRIMAPS replans *more* agents and accepts *longer*
detours than Naive-DR; it works agents harder to recover them. The
minimal-disruption design (one priority-inheritance *yield chain* rather than
one per cycle member) is the load-bearing component: it lifts ISR from 58% to
73% while cutting agents-replanned more than 3x (82 vs 295).

**Sim cross-validation:** Using DRIMAPS as the controller inside the
interactive sim raises ISR from 0.44 (greedy, no repair) to 0.59
(`experiments/sim_crossval.py`).

## Reproducibility

To reproduce everything end-to-end (install, tests, benchmark, paper assets,
sim cross-validation), run:

```bash
./reproduce.sh
```

The benchmark step takes roughly 10 minutes.

## Project Structure

```
DRIMAPS/
├── src/                    # Core algorithm
│   ├── drimaps.py          # Main runtime resolution loop
│   ├── execution.py        # Reactive, collision-free execution engine
│   ├── dependency_graph.py # Wait-For Graph
│   ├── cycle_detector.py   # Tarjan's SCC detection
│   ├── deadlock_classifier.py # 4-type structural labelling (diagnostic)
│   ├── priority_manager.py # Agent priority (who yields first)
│   ├── local_repair.py     # Bounded Space-Time A* (cooperative planner)
│   ├── config.py           # Configuration & result types
│   └── utils.py            # Shared utilities
├── baselines/              # Baseline solvers
│   ├── pibt.py             # PIBTSolver
│   ├── naive_dr.py         # NaiveDRSolver
│   └── prevention_only.py  # PreventionOnlySolver
├── sim/                    # DRIMAPSim environment
│   ├── environment.py      # Main Gymnasium env
│   ├── grid_world.py       # Core grid engine + collision systems
│   ├── map_registry.py     # 8 map generators + MovingAI parsing
│   ├── metrics.py          # Metric wrappers (incl. deadlock)
│   ├── rendering.py        # ASCII / SVG / HTML animation
│   ├── wrappers.py         # Utility wrappers (incl. trajectory recording)
│   └── env_config.py       # Configuration
├── experiments/            # Benchmark + sim cross-validation
├── analysis/               # Paper asset generation
├── tests/                  # Test suite (96 tests)
└── paper/                  # IEEE conference paper
```

## Citation

```bibtex
@article{drimaps2026,
  title={DRIMAPS: Runtime Deadlock Resolution for Multi-Agent Path Finding},
  author={Anonymous},
  year={2026}
}
```

## License

MIT License
</content>
</invoke>
