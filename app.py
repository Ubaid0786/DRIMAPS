#!/usr/bin/env python3
"""
DRIMAPS Research Simulator — Web Server
========================================

Lightweight Flask server that exposes the existing DRIMAPS framework
over HTTP/JSON.  No existing code is modified; this module imports and
wraps the existing classes.

Usage:
    python app.py                    # default port 8050
    python app.py --port 9000        # custom port
    python app.py --debug            # auto-reload during development
"""

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, fields
from typing import Any, Dict

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

# ── Ensure project root is importable ──────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    AGENT_COUNTS,
    BENCHMARK_MAPS,
    EXPERIMENT_SEEDS,
    DRIMAPSConfig,
    MapConfig,
)
from src.drimaps import DRIMAPS
from src.reactive_core import ReactiveCore
from src.utils import (
    bfs_shortest_path,
    compute_makespan,
    compute_sum_of_costs,
    detect_edge_conflicts,
    detect_vertex_conflicts,
)
from sim.env_config import EnvConfig
from sim.environment import DRIMAPSimEnv
from sim.map_registry import (
    compute_map_difficulty,
    generate_map,
    parse_movingai_map,
)
from experiments.generate_instances import generate_grid, generate_scenario

# ── Flask app ──────────────────────────────────────────────────────
app = Flask(
    __name__,
    static_folder="static",
    static_url_path="/static",
)


# ═══════════════════════════════════════════════════════════════════
#  Serve the UI
# ═══════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


# ═══════════════════════════════════════════════════════════════════
#  Map endpoints
# ═══════════════════════════════════════════════════════════════════

@app.route("/api/maps/types", methods=["GET"])
def map_types():
    """List built-in map generators."""
    types = [
        {"id": "random",       "label": "Random",        "description": "Random obstacles with configurable density"},
        {"id": "warehouse",    "label": "Warehouse",      "description": "Parallel aisles with shelf blocks and cross-aisle gaps"},
        {"id": "corridor",     "label": "Corridor",       "description": "Narrow passages with bottleneck connections"},
        {"id": "bottleneck",   "label": "Bottleneck",     "description": "Central wall with a single narrow passage"},
        {"id": "maze",         "label": "Maze",           "description": "Recursive backtracking maze — complex topology"},
        {"id": "room",         "label": "Room",           "description": "Room-based layout with doorway congestion"},
        {"id": "dense_random", "label": "Dense Random",   "description": "35% obstacle density random map"},
        {"id": "open",         "label": "Open",           "description": "Empty grid with no obstacles"},
    ]
    return jsonify(types)


@app.route("/api/maps/generate", methods=["POST"])
def map_generate():
    """Generate a map grid from parameters."""
    data = request.get_json(force=True)
    map_type = data.get("map_type", "random")
    size = int(data.get("size", 32))
    density = float(data.get("density", 0.2))
    seed = data.get("seed", 42)
    seed = int(seed) if seed is not None else None

    size = max(4, min(size, 256))  # clamp

    try:
        grid = generate_map(map_type, size, density, seed)
        difficulty = compute_map_difficulty(grid)
        return jsonify({
            "grid": grid.tolist(),
            "height": grid.shape[0],
            "width": grid.shape[1],
            "difficulty": difficulty,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/maps/difficulty", methods=["POST"])
def map_difficulty():
    """Compute difficulty metrics for a given grid."""
    data = request.get_json(force=True)
    grid = np.array(data["grid"], dtype=np.int32)
    return jsonify(compute_map_difficulty(grid))


@app.route("/api/maps/movingai", methods=["POST"])
def map_movingai():
    """Parse a MovingAI .map file (uploaded as text)."""
    data = request.get_json(force=True)
    map_text = data.get("map_text", "")
    try:
        grid = parse_movingai_map(map_text)
        difficulty = compute_map_difficulty(grid)
        return jsonify({
            "grid": grid.tolist(),
            "height": grid.shape[0],
            "width": grid.shape[1],
            "difficulty": difficulty,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ═══════════════════════════════════════════════════════════════════
#  Scenario endpoints
# ═══════════════════════════════════════════════════════════════════

@app.route("/api/scenario/generate", methods=["POST"])
def scenario_generate():
    """Generate random start/goal positions for a given grid."""
    data = request.get_json(force=True)
    grid = np.array(data["grid"], dtype=np.int32)
    num_agents = int(data.get("num_agents", 8))
    seed = int(data.get("seed", 42))

    try:
        starts, goals = generate_scenario(grid, num_agents, seed)
        return jsonify({
            "starts": [list(s) for s in starts],
            "goals": [list(g) for g in goals],
            "num_agents": len(starts),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ═══════════════════════════════════════════════════════════════════
#  Simulation endpoints
# ═══════════════════════════════════════════════════════════════════

@app.route("/api/simulate", methods=["POST"])
def simulate():
    """Run DRIMAPS (or a baseline) on a MAPF instance.

    Expects JSON:
        grid:       2D list (1=obstacle, 0=free)
        starts:     [[r,c], ...]
        goals:      [[r,c], ...]
        algorithm:  "drimaps" | "pibt" | "naive_dr" | "prevention"
        config:     optional DRIMAPSConfig overrides
    Returns the full trajectory + metrics.
    """
    data = request.get_json(force=True)
    grid = np.array(data["grid"], dtype=np.int32)
    starts = [tuple(s) for s in data["starts"]]
    goals = [tuple(g) for g in data["goals"]]
    algorithm = data.get("algorithm", "drimaps")
    config_overrides = data.get("config", {})

    # Build config
    config = DRIMAPSConfig()
    for key, val in config_overrides.items():
        if hasattr(config, key):
            setattr(config, key, val)

    try:
        t0 = time.time()

        if algorithm == "drimaps":
            core = ReactiveCore(config)
            seed = config.seeds[0] if config.seeds else 42
            paths, stats = core.run(grid, starts, goals, seed=seed)
        elif algorithm == "pibt":
            from baselines.pibt import PIBTSolver
            solver = PIBTSolver(grid, timeout=config.timeout)
            seed = config.seeds[0] if config.seeds else 42
            paths = solver.solve(starts, goals, seed=seed)
            stats = solver.last_result
        elif algorithm == "naive_dr":
            from baselines.naive_dr import NaiveDRSolver
            solver = NaiveDRSolver(grid, timeout=config.timeout)
            seed = config.seeds[0] if config.seeds else 42
            paths = solver.solve(starts, goals, seed=seed)
            stats = solver.last_result
        elif algorithm == "prevention":
            from baselines.prevention_only import PreventionOnlySolver
            solver = PreventionOnlySolver(grid, timeout=config.timeout)
            seed = config.seeds[0] if config.seeds else 42
            paths = solver.solve(starts, goals, seed=seed)
            stats = solver.last_result
        else:
            return jsonify({"error": f"Unknown algorithm: {algorithm}"}), 400

        runtime = time.time() - t0

        # Build per-step trajectory from paths
        max_t = max(len(p) for p in paths) if paths else 1
        trajectory = []
        for t in range(max_t):
            step_pos = []
            for p in paths:
                idx = min(t, len(p) - 1)
                step_pos.append(list(p[idx]))
            trajectory.append(step_pos)

        # Metrics
        reached = [
            (list(p[-1]) == list(goals[i]) if p else False)
            for i, p in enumerate(paths)
        ]
        n = len(starts)
        v_conf = detect_vertex_conflicts(paths)
        e_conf = detect_edge_conflicts(paths)

        result = {
            "algorithm": algorithm,
            "trajectory": trajectory,
            "total_steps": max_t,
            "num_agents": n,
            "starts": [list(s) for s in starts],
            "goals": [list(g) for g in goals],
            "reached_goal": reached,
            "metrics": {
                "isr": sum(reached) / n if n > 0 else 0,
                "makespan": compute_makespan(paths),
                "sum_of_costs": compute_sum_of_costs(paths),
                "runtime_s": round(runtime, 4),
                "deadlocks_detected": getattr(stats, "deadlocks_detected", 0),
                "deadlocks_resolved": getattr(stats, "deadlocks_resolved", 0),
                "agents_replanned": getattr(stats, "agents_replanned", 0),
                "vertex_conflicts": len(v_conf),
                "edge_conflicts": len(e_conf),
                "collision_free": len(v_conf) == 0 and len(e_conf) == 0,
            },
        }
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════
#  Algorithm + config endpoints
# ═══════════════════════════════════════════════════════════════════

@app.route("/api/algorithms", methods=["GET"])
def algorithms():
    """List available algorithms."""
    return jsonify([
        {"id": "drimaps",    "label": "DRIMAPS",     "description": "Detection-guided deadlock escape (this work)"},
        {"id": "pibt",       "label": "PIBT",        "description": "Priority Inheritance with Backtracking"},
        {"id": "naive_dr",   "label": "Naive-DR",    "description": "Stagnation detection + random sidestep"},
        {"id": "prevention", "label": "Prevention",  "description": "Cooperative reservation table (no repair)"},
    ])


@app.route("/api/config/defaults", methods=["GET"])
def config_defaults():
    """Return default DRIMAPSConfig as JSON."""
    cfg = DRIMAPSConfig()
    result = {}
    for f in fields(cfg):
        val = getattr(cfg, f.name)
        # Convert enums to string
        if hasattr(val, "value"):
            val = val.value
        result[f.name] = val
    return jsonify(result)


@app.route("/api/benchmarks/maps", methods=["GET"])
def benchmark_maps():
    """Return the pre-defined benchmark map configurations."""
    maps = []
    for mc in BENCHMARK_MAPS:
        maps.append({
            "map_id": mc.map_id,
            "map_type": mc.map_type,
            "width": mc.width,
            "height": mc.height,
            "description": mc.description,
        })
    return jsonify({
        "maps": maps,
        "agent_counts": AGENT_COUNTS,
        "seeds": EXPERIMENT_SEEDS,
    })


# ═══════════════════════════════════════════════════════════════════
#  Batch / comparison endpoints
# ═══════════════════════════════════════════════════════════════════

@app.route("/api/simulate/compare", methods=["POST"])
def simulate_compare():
    """Run two algorithms on the same instance for side-by-side comparison."""
    data = request.get_json(force=True)
    grid = np.array(data["grid"], dtype=np.int32)
    starts = [tuple(s) for s in data["starts"]]
    goals = [tuple(g) for g in data["goals"]]
    algorithms = data.get("algorithms", ["drimaps", "pibt"])

    results = {}
    for alg in algorithms[:4]:  # cap at 4
        sim_data = {
            "grid": data["grid"],
            "starts": data["starts"],
            "goals": data["goals"],
            "algorithm": alg,
            "config": data.get("config", {}),
        }
        with app.test_request_context(
            "/api/simulate", method="POST",
            json=sim_data, content_type="application/json",
        ):
            resp = simulate()
            if isinstance(resp, tuple):
                results[alg] = {"error": "failed"}
            else:
                results[alg] = resp.get_json()

    return jsonify(results)


# ═══════════════════════════════════════════════════════════════════
#  Export endpoints
# ═══════════════════════════════════════════════════════════════════

@app.route("/api/export/svg", methods=["POST"])
def export_svg():
    """Render a simulation frame as SVG using the existing renderer."""
    from sim.rendering import render_svg
    data = request.get_json(force=True)
    grid = np.array(data["grid"], dtype=np.int32)
    agents = [tuple(a) for a in data["agents_xy"]]
    targets = [tuple(t) for t in data["targets_xy"]]
    deadlocked = set(data.get("deadlock_agents", []))
    title = data.get("title", "")

    svg = render_svg(grid, agents, targets,
                     deadlock_agents=deadlocked, title=title)
    return svg, 200, {"Content-Type": "image/svg+xml"}


# ═══════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DRIMAPS Research Simulator")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print(f"\n  DRIMAPS Research Simulator")
    print(f"  http://localhost:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=args.debug)
