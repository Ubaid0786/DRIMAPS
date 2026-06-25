# MovingAI MAPF benchmark (vendored)

This directory vendors the standard **MovingAI** multi-agent path-finding
benchmark so DRIMAPS is reproducible from a clean checkout.

- `mapf-map/` — all 33 benchmark grids (Sturtevant 2012), `.map` (octile) format.
- `scen-even/` — *even* scenario files (agents bucketed by optimal path length).
- `scen-random/` — *random* scenario files.

To keep the repository small we vendor every map but only the `-even-1`,
`-even-2`, and `-random-1` scenario files for each map — the exact instances the
paper's tables and figures are computed from. The full scenario set is available
from the MovingAI benchmark.

## Attribution

Maps and scenarios are from the MovingAI 2D pathfinding / MAPF benchmark:

- N. Sturtevant, "Benchmarks for Grid-Based Pathfinding," *IEEE Transactions on
  Computational Intelligence and AI in Games*, 2012.
- R. Stern et al., "Multi-Agent Pathfinding: Definitions, Variants, and
  Benchmarks," *SoCS* 2019.

Source: https://movingai.com/benchmarks/mapf/index.html — free for research use
with attribution. The maps/scenarios here are unmodified.

## Loading

```python
from src.movingai import load_instance
grid, starts, goals = load_instance("maze-32-32-2", num_agents=40, scen="even-1")

from sim.benchmark import BenchmarkScenario
scn = BenchmarkScenario.load("warehouse-10-20-10-2-1", 60)  # editable in-sim
env = scn.to_env()
```
