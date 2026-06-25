#!/usr/bin/env python3
"""
DRIMAPS Benchmark Runner (MovingAI)
===================================

Runs every method on the standard MovingAI MAPF benchmark (Sturtevant maps +
Stern et al. scenarios, vendored under ``benchmarks/``) under one reactive
execution model and writes a tidy long-format CSV that every paper table and
figure is derived from. Every number in the paper traces back to a row in
``results/benchmark_latest.csv``: ``python experiments/run_benchmark.py``.

Two campaigns are run into the same CSV:

  * **scaling** — the difficult maps (mazes, rooms, dense random, warehouse,
    game) across an agent-count sweep, all methods + ablations, several
    scenarios; this is the statistical headline.
  * **per-map** — a broad set spanning every benchmark category (up to
    256x256) at a fixed count, DRIMAPS vs. the PIBT reference; this shows the
    method resolves deadlocks across the whole benchmark.

Primary metric is ISR (individual success rate -- the fraction of agents that
reach their goals). Binary instance success, arrival-based makespan/flowtime,
agent disruption, deadlock counts and per-phase timing are recorded alongside.
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import DRIMAPSConfig
from src.drimaps import DRIMAPS
from src.utils import (
    Position, Path, bfs_shortest_path,
    detect_vertex_conflicts, detect_edge_conflicts,
)
from src.movingai import load_instance, load_map, MAP_CATEGORY
from baselines import PIBTSolver, NaiveDRSolver, PreventionOnlySolver

# --- Scaling campaign: difficult maps that run fast, full method set ---------
SCALING_MAPS = [
    "maze-32-32-2", "maze-32-32-4", "room-32-32-4", "room-64-64-8",
    "random-32-32-20", "random-64-64-20", "warehouse-10-20-10-2-1", "den312d",
]
SCALING_COUNTS = [50, 100, 150, 200]
SCENARIOS = ["even-1", "even-2", "random-1"]   # instance/seed dimension

MAIN_ALGORITHMS = ["drimaps", "pibt", "naive_dr", "prevention_only"]
ABLATIONS = ["drimaps_no_cycle", "drimaps_no_classify", "drimaps_no_minimal"]

# --- Per-map coverage campaign: every category, up to 256x256 ----------------
PERMAP_MAPS = [
    "empty-32-32", "empty-48-48",
    "random-32-32-10", "random-32-32-20", "random-64-64-20",
    "room-32-32-4", "room-64-64-8", "room-64-64-16",
    "maze-32-32-2", "maze-32-32-4", "maze-128-128-10",
    "warehouse-10-20-10-2-1", "warehouse-10-20-10-2-2", "warehouse-20-40-10-2-1",
    "den312d", "den520d", "ost003d", "lak303d", "ht_chantry",
    "Berlin_1_256", "Paris_1_256",
]
PERMAP_COUNT = 100
PERMAP_SCENARIOS = ["even-1", "even-2"]
PERMAP_ALGORITHMS = ["drimaps", "pibt"]


def make_solver(algorithm: str, grid: np.ndarray, timeout: float):
    """Instantiate a solver (DRIMAPS variant or baseline) by name."""
    if algorithm.startswith("drimaps"):
        cfg = DRIMAPSConfig(timeout=timeout)
        cfg.enable_cycle_detection = algorithm != "drimaps_no_cycle"
        cfg.enable_classification = algorithm != "drimaps_no_classify"
        cfg.enable_minimal_disruption = algorithm != "drimaps_no_minimal"
        return DRIMAPS(cfg)
    if algorithm == "pibt":
        return PIBTSolver(grid, timeout)
    if algorithm == "naive_dr":
        return NaiveDRSolver(grid, timeout)
    if algorithm == "prevention_only":
        return PreventionOnlySolver(grid, timeout)
    raise ValueError(f"Unknown algorithm: {algorithm}")


def arrival_metrics(
    paths: List[Path], starts: List[Position], goals: List[Position], grid: np.ndarray,
) -> Tuple[int, int, int, int, float]:
    """Completion and cost metrics from executed trajectories.

    Returns (num_reached, makespan, flowtime, optimal_flowtime, detour_ratio).
    """
    reached = 0
    makespan = 0
    flowtime = 0
    opt_flow = 0
    ratios = []
    for i in range(len(goals)):
        p = paths[i]
        arrival = None
        for t in range(len(p)):
            if p[t] == goals[i] and all(p[u] == goals[i] for u in range(t, len(p))):
                arrival = t
                break
        if arrival is None:
            continue
        reached += 1
        makespan = max(makespan, arrival)
        flowtime += arrival
        sp = bfs_shortest_path(grid, starts[i], goals[i])
        opt = (len(sp) - 1) if sp else arrival
        opt_flow += opt
        if opt > 0:
            ratios.append(arrival / opt)
    detour = float(np.mean(ratios)) if ratios else 1.0
    return reached, makespan, flowtime, opt_flow, detour


def run_instance(
    algorithm: str, grid: np.ndarray,
    starts: List[Position], goals: List[Position],
    timeout: float, seed: int,
) -> Dict:
    """Run one (algorithm, instance) pair and return a metrics row."""
    n = len(starts)
    row = {
        "algorithm": algorithm, "num_agents": n, "seed": seed,
        "isr": 0.0, "success": False, "reached": 0,
        "makespan": -1, "flowtime": -1, "detour_ratio": 1.0,
        "runtime": 0.0, "collisions": 0,
        "deadlocks_detected": 0, "deadlocks_resolved": 0,
        "resolution_rate": 1.0, "agents_replanned": 0,
        "memory_mb": 0.0, "initial_planning_time": 0.0,
        "wfg_update_time": 0.0, "detection_time": 0.0, "resolution_time": 0.0,
    }
    try:
        solver = make_solver(algorithm, grid, timeout)
        t0 = time.time()
        if isinstance(solver, DRIMAPS):
            res = solver.solve(grid, starts, goals)
            paths = res.paths
        else:
            paths = solver.solve(starts, goals, seed=seed)
            res = solver.last_result
        row["runtime"] = round(time.time() - t0, 4)

        if not paths or len(paths) != n:
            return row

        reached, mk, ft, _, detour = arrival_metrics(paths, starts, goals, grid)
        row["reached"] = reached
        row["isr"] = round(reached / n, 4)
        row["success"] = reached == n
        row["makespan"] = mk
        row["flowtime"] = ft
        row["detour_ratio"] = round(detour, 4)
        row["collisions"] = (
            len(detect_vertex_conflicts(paths)) + len(detect_edge_conflicts(paths))
        )
        if res is not None:
            row["deadlocks_detected"] = res.deadlocks_detected
            row["deadlocks_resolved"] = res.deadlocks_resolved
            row["resolution_rate"] = round(
                res.deadlocks_resolved / res.deadlocks_detected, 4
            ) if res.deadlocks_detected else 1.0
            row["agents_replanned"] = res.agents_replanned
            row["memory_mb"] = round(res.peak_memory_mb, 2)
            row["initial_planning_time"] = round(res.initial_planning_time, 4)
            row["wfg_update_time"] = round(res.wfg_update_time, 4)
            row["detection_time"] = round(res.detection_time, 4)
            row["resolution_time"] = round(res.resolution_time, 4)
    except Exception:
        if "--debug" in sys.argv:
            import traceback
            traceback.print_exc()
        row["runtime"] = 0.0
    return row


def _free_cells(grid: np.ndarray) -> int:
    return int((grid == 0).sum())


def _emit(rows, row, map_name, scen, grid):
    row.update({
        "map_id": map_name, "map_type": MAP_CATEGORY.get(map_name, "other"),
        "map_name": map_name, "scenario": scen,
        "map_size": f"{grid.shape[0]}x{grid.shape[1]}",
    })
    rows.append(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="DRIMAPS MovingAI benchmark runner")
    parser.add_argument("--quick", action="store_true", help="Tiny smoke run.")
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--output", type=str, default="results")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.quick:
        scaling_maps, scaling_counts = ["maze-32-32-2", "room-32-32-4"], [50, 100]
        scenarios = ["even-1"]
        algos = MAIN_ALGORITHMS
        permap_maps, permap_scen = ["empty-32-32", "den312d"], ["even-1"]
    else:
        scaling_maps, scaling_counts = SCALING_MAPS, SCALING_COUNTS
        scenarios = SCENARIOS
        algos = MAIN_ALGORITHMS + ABLATIONS
        permap_maps, permap_scen = PERMAP_MAPS, PERMAP_SCENARIOS

    out_dir = os.path.join(PROJECT_ROOT, args.output)
    os.makedirs(out_dir, exist_ok=True)

    rows: List[Dict] = []
    t_start = time.time()
    done = 0

    # ---- Scaling campaign --------------------------------------------------
    for map_name in scaling_maps:
        grid = load_map(map_name)
        free = _free_cells(grid)
        for scen in scenarios:
            for na in scaling_counts:
                if na > free * 0.6:       # keep instances feasible
                    continue
                try:
                    g, starts, goals = load_instance(map_name, na, scen)
                except Exception:
                    continue
                if len(starts) < na:
                    continue
                for algo in algos:
                    row = run_instance(algo, grid, starts, goals, args.timeout,
                                       seed=SCENARIOS.index(scen) if scen in SCENARIOS else 0)
                    _emit(rows, row, map_name, scen, grid)
                    done += 1
                    if done % 20 == 0:
                        el = time.time() - t_start
                        print(f"  [scaling {done}] {el:5.0f}s {algo:18s} "
                              f"{map_name:20s} {na:3d}ag {scen} ISR={row['isr']:.2f}",
                              flush=True)

    # ---- Per-map coverage campaign ----------------------------------------
    for map_name in permap_maps:
        grid = load_map(map_name)
        free = _free_cells(grid)
        na = min(PERMAP_COUNT, int(free * 0.5))
        for scen in permap_scen:
            try:
                g, starts, goals = load_instance(map_name, na, scen)
            except Exception:
                continue
            if len(starts) < na:
                continue
            for algo in PERMAP_ALGORITHMS:
                row = run_instance(algo, grid, starts, goals, args.timeout,
                                   seed=permap_scen.index(scen))
                row["campaign"] = "permap"
                _emit(rows, row, map_name, scen, grid)
                done += 1
                el = time.time() - t_start
                print(f"  [permap {done}] {el:5.0f}s {algo:10s} "
                      f"{map_name:20s} {na:3d}ag {scen} ISR={row['isr']:.2f}",
                      flush=True)

    # Tag scaling rows that never got a campaign field.
    for r in rows:
        r.setdefault("campaign", "scaling")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fields = list(rows[0].keys())
    for path in [
        os.path.join(out_dir, f"benchmark_{ts}.csv"),
        os.path.join(out_dir, "benchmark_latest.csv"),
    ]:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to results/benchmark_latest.csv "
          f"({time.time() - t_start:.0f}s total)")


if __name__ == "__main__":
    main()
