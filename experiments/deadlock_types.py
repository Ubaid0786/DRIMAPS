#!/usr/bin/env python3
"""
Per-deadlock-type analysis
==========================

Quantifies which structural deadlock types DRIMAPS actually encounters and
escapes on the MovingAI benchmark, broken down by map topology. This grounds
the taxonomy (cyclic / corridor / congestion / goal-blocking) empirically.

The reactive core classifies every confirmed persistent deadlock as it escapes
it; we aggregate those ``type_counts`` over the difficult maps and scenarios.

Run: ``python experiments/deadlock_types.py`` -> results/deadlock_types.json
"""

import json
import os
import sys
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import DRIMAPSConfig
from src.reactive_core import ReactiveCore
from src.movingai import load_instance
from experiments.run_benchmark import SCALING_MAPS, SCENARIOS


def main():
    counts = [100, 150]
    per_map = {}
    overall = Counter()
    core = ReactiveCore(DRIMAPSConfig(timeout=30))
    for map_name in SCALING_MAPS:
        tally = Counter()
        for scen in SCENARIOS:
            for na in counts:
                try:
                    grid, starts, goals = load_instance(map_name, na, scen)
                except Exception:
                    continue
                if len(starts) < na:
                    continue
                _, stats = core.run(grid, starts, goals, seed=42)
                for k, v in stats.type_counts.items():
                    tally[k] += v
                    overall[k] += v
        per_map[map_name] = dict(tally)
        total = sum(tally.values()) or 1
        dist = ", ".join(f"{k}={100*v/total:.0f}%" for k, v in
                         sorted(tally.items(), key=lambda x: -x[1]))
        print(f"  {map_name:22s} (n={sum(tally.values()):4d}): {dist}")

    grand = sum(overall.values()) or 1
    print("\n  OVERALL: " + ", ".join(
        f"{k}={100*v/grand:.0f}%" for k, v in
        sorted(overall.items(), key=lambda x: -x[1])))

    out = {"per_map": per_map, "overall": dict(overall),
           "agent_counts": counts, "scenarios": SCENARIOS}
    path = os.path.join(PROJECT_ROOT, "results", "deadlock_types.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {os.path.relpath(path, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
