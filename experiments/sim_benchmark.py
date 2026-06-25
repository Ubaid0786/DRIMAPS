#!/usr/bin/env python3
"""
DRIMAPSim Environment Benchmark
===============================

Measures the environment's engineering properties with reproducible numbers
used in the paper's DRIMAPSim section:

  1. Throughput      -- agent-steps per second vs. team size and map size.
  2. Determinism     -- identical rollouts from the same seed.
  3. Collision-free  -- zero vertex/edge conflicts over long random rollouts
                        under the block_both model, at scale.
  4. WFG vs.\ stagnation -- how much the stagnation heuristic over-reports
                        "deadlocks" relative to genuine WFG-confirmed cycles.

Run: ``python experiments/sim_benchmark.py``  (writes results/sim_benchmark.json)
"""

import json
import os
import sys
import time
from typing import Dict, List

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sim import drimapsim_v0
from sim.env_config import EnvConfig


def _active_positions(env) -> List:
    grid = env.unwrapped.grid
    return [
        grid.positions_xy[i]
        for i in range(grid.config.num_agents) if grid.is_active[i]
    ]


def bench_throughput(sizes, agent_counts, steps=200) -> List[Dict]:
    rows = []
    for size in sizes:
        for na in agent_counts:
            cfg = EnvConfig(
                size=size, num_agents=na, density=0.15,
                collision_system="block_both", deadlock_tracking=True,
                on_target="nothing", seed=0,
            )
            env = drimapsim_v0(cfg)
            env.reset(seed=0)
            actions = [env.sample_actions() for _ in range(steps)]
            t0 = time.perf_counter()
            for a in actions:
                env.step(a)
            dt = time.perf_counter() - t0
            n = env.unwrapped.config.num_agents
            rows.append({
                "size": size, "agents": n, "steps": steps,
                "sec": round(dt, 4),
                "steps_per_sec": round(steps / dt, 1),
                "agent_steps_per_sec": round(n * steps / dt, 1),
            })
            print(f"  {size:3d}x{size:<3d} {n:4d} agents: "
                  f"{steps/dt:8.1f} env-steps/s  "
                  f"{n*steps/dt:10.1f} agent-steps/s")
    return rows


def measure_monitor_overhead(agent_counts, steps=200) -> List[Dict]:
    """Throughput with the deadlock monitor on vs. off (its cost)."""
    rows = []
    for na in agent_counts:
        def thr(track):
            cfg = EnvConfig(size=32, num_agents=na, density=0.15,
                            collision_system="block_both",
                            deadlock_tracking=track, on_target="nothing",
                            seed=0)
            env = drimapsim_v0(cfg)
            env.reset(seed=0)
            acts = [env.sample_actions() for _ in range(steps)]
            t0 = time.perf_counter()
            for a in acts:
                env.step(a)
            return na * steps / (time.perf_counter() - t0)
        on, off = thr(True), thr(False)
        ov = round(100 * (off - on) / off, 1) if off > 0 else 0.0
        rows.append({"agents": na, "on": round(on, 1), "off": round(off, 1),
                     "overhead_pct": ov})
        print(f"  {na:4d} agents: monitor ON {on:8.0f} | OFF {off:8.0f} "
              f"a-steps/s | overhead {ov:.0f}%")
    return rows


def check_determinism(trials=3) -> Dict:
    def rollout():
        cfg = EnvConfig(size=24, num_agents=20, density=0.15,
                        collision_system="block_both", deadlock_tracking=True,
                        on_target="nothing", seed=11)
        env = drimapsim_v0(cfg)
        env.reset(seed=11)
        traj = []
        for _ in range(60):
            env.step(env.sample_actions())
            traj.append(tuple(env.get_agents_xy()))
        return traj, env.get_deadlock_stats()
    base_t, base_s = rollout()
    ok = True
    for _ in range(trials - 1):
        t, s = rollout()
        ok = ok and (t == base_t) and (s == base_s)
    print(f"  determinism over {trials} rollouts: "
          f"{'IDENTICAL' if ok else 'DIVERGED'}")
    return {"deterministic": ok, "trials": trials}


def check_collision_free(steps=500) -> Dict:
    total_overlaps = 0
    configs = [
        dict(size=32, num_agents=64, density=0.15),
        dict(size=48, num_agents=128, density=0.1),
    ]
    for c in configs:
        cfg = EnvConfig(collision_system="block_both", deadlock_tracking=False,
                        on_target="nothing", seed=5, **c)
        env = drimapsim_v0(cfg)
        env.reset(seed=5)
        prev = _active_positions(env)
        for _ in range(steps):
            env.step(env.sample_actions())
            cur = _active_positions(env)
            # Vertex conflict: two active agents share a cell.
            if len(set(cur)) != len(cur):
                total_overlaps += 1
            prev = cur
    print(f"  collision-free at scale: {total_overlaps} overlaps over "
          f"{steps} steps x {len(configs)} configs")
    return {"overlaps": total_overlaps, "steps": steps,
            "configs": len(configs)}


def _greedy_actions(env):
    """One greedy step toward each agent's goal (Manhattan-descending)."""
    pos = env.get_agents_xy()
    tgt = env.get_targets_xy()
    acts = []
    for (r, c), (tr, tc) in zip(pos, tgt):
        if r == tr and c == tc:
            acts.append(0)
        elif abs(tr - r) >= abs(tc - c):
            acts.append(2 if tr > r else 1)
        else:
            acts.append(4 if tc > c else 3)
    return acts


def compare_wfg_vs_stagnation(steps=120) -> Dict:
    """Quantify how much the stagnation heuristic over-reports deadlocks.

    Run under two policies: random play (exploratory) and a realistic
    greedy-toward-goal policy. The point holds under both: stagnation flags
    far more timesteps as deadlocked than there are genuine wait cycles, but
    we report greedy too so the result is not an artifact of random idling.
    """
    maps = ["corridor", "bottleneck", "warehouse", "room", "maze"]
    out = {}
    for policy in ("random", "greedy"):
        stag_steps = 0   # steps where >=2 agents flagged by stagnation
        wfg_steps = 0    # steps with an active WFG-confirmed deadlock
        wfg_episodes = 0  # distinct WFG-confirmed deadlock episodes
        for mp in maps:
            for seed in [1, 2, 3]:
                cfg = EnvConfig(size=32, num_agents=30, map_type=mp,
                                density=0.2, collision_system="block_both",
                                deadlock_tracking=True, on_target="nothing",
                                seed=seed)
                env = drimapsim_v0(cfg)
                env.reset(seed=seed)
                for _ in range(steps):
                    acts = (env.sample_actions() if policy == "random"
                            else _greedy_actions(env))
                    _, _, _, _, infos = env.step(acts)
                    last = infos[-1]
                    stagged = sum(
                        1 for i in range(len(infos))
                        if infos[i].get("stagnation_count", 0) >= 5
                    )
                    if stagged >= 2:
                        stag_steps += 1
                    if last.get("active_deadlocks", 0) >= 1:
                        wfg_steps += 1
                wfg_episodes += env.get_deadlock_stats()["deadlock_count"]
        out[policy] = {"stagnation_flagged_steps": stag_steps,
                       "wfg_deadlock_steps": wfg_steps,
                       "wfg_confirmed_episodes": wfg_episodes}
        print(f"  [{policy:6s}] stagnation-flagged steps: {stag_steps:4d}  |  "
              f"WFG-deadlock steps: {wfg_steps:4d}  |  "
              f"WFG episodes: {wfg_episodes}")
    out["maps"] = maps
    out["seeds"] = [1, 2, 3]
    out["steps_per_run"] = steps
    # Back-compat top-level keys (random policy) for existing readers.
    out["stagnation_flagged_steps"] = out["random"]["stagnation_flagged_steps"]
    out["wfg_confirmed_episodes"] = out["random"]["wfg_confirmed_episodes"]
    return out


def main():
    print("DRIMAPSim environment benchmark\n" + "=" * 50)
    print("\n[1] Throughput (random play, block_both):")
    throughput = bench_throughput([16, 32, 64], [16, 64, 256])
    print("\n[2] Deadlock-monitor overhead (32x32):")
    overhead = measure_monitor_overhead([16, 64, 256])
    print("\n[3] Determinism:")
    determinism = check_determinism()
    print("\n[4] Collision-freedom at scale:")
    collisions = check_collision_free()
    print("\n[5] WFG deadlock detection vs. stagnation heuristic:")
    wfg = compare_wfg_vs_stagnation()

    out = {
        "throughput": throughput,
        "monitor_overhead": overhead,
        "determinism": determinism,
        "collision_free": collisions,
        "wfg_vs_stagnation": wfg,
    }
    os.makedirs(os.path.join(PROJECT_ROOT, "results"), exist_ok=True)
    path = os.path.join(PROJECT_ROOT, "results", "sim_benchmark.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {os.path.relpath(path, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
