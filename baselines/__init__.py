"""Baseline MAPF solvers for the DRIMAPS evaluation.

All baselines share the reactive execution model in ``src.execution`` (except
PIBT, which is itself a step-wise planner), so every method is compared under
identical movement semantics.

Exposed solvers:
    * ``PIBTSolver`` -- Priority Inheritance with Backtracking (Okumura et al.,
      2022), a fast decentralized planner that avoids deadlocks by construction.
    * ``NaiveDRSolver`` -- detect-and-resolve with stagnation detection and
      random replanning, with no structural reasoning.
    * ``PreventionOnlySolver`` -- cooperative A* reservation planning with no
      runtime repair (prevention-only end of the spectrum).
"""

from baselines.pibt import PIBTSolver
from baselines.naive_dr import NaiveDRSolver
from baselines.prevention_only import PreventionOnlySolver

__all__ = ["PIBTSolver", "NaiveDRSolver", "PreventionOnlySolver"]
