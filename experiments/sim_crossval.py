#!/usr/bin/env python3
"""
Sim Cross-Validation: DRIMAPS-as-controller inside the interactive sim
======================================================================

The offline track (``src/``) reports that a *runtime* deadlock-resolution
layer improves throughput: agents that would otherwise wedge against each
other in corridors are rerouted around the blockage. This script
cross-validates that finding inside the interactive simulation environment
(``sim/``) — a completely separate execution path — so the conclusion does
not hinge on a single codebase.

Setup
-----
We drive the *same* set of instances (same maps, starts, goals, seeds) under
three policies, all going through the public ``sim`` env (``drimapsim_v0``,
``EnvConfig``, ``reset``, ``step``) in collision-free ``block_both`` mode:

* ``random``    — uniformly random actions (a sanity floor).
* ``greedy``    — each agent always steps greedily toward its goal along a
                  BFS shortest path; when blocked it simply waits. This is
                  the "no runtime resolution" condition.
* ``drimaps``   — greedy by default, but a DRIMAPS-style controller watches
                  for stalls, builds a Wait-For Graph, detects deadlock
                  cycles with Tarjan SCCs (reusing ``src``'s WaitForGraph /
                  CycleDetector, imported read-only), and reroutes a victim
                  agent *around* the blockers using ``bfs_avoiding``.

For each policy we report ISR (fraction of agents that reached their goal)
and the number of recorded deadlock events. If the offline finding
generalises, ``drimaps`` should match or beat ``greedy`` on ISR while
recording fewer unresolved deadlocks.

Run::

    python experiments/sim_crossval.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim import drimapsim_v0, EnvConfig  # noqa: E402
from sim.map_registry import generate_map  # noqa: E402

# Imported READ-ONLY from the offline track — we never edit src/.
from src.utils import bfs_shortest_path, bfs_avoiding, get_neighbors  # noqa: E402
from src.dependency_graph import WaitForGraph  # noqa: E402
from src.cycle_detector import CycleDetector  # noqa: E402


# Action encoding mirrors sim.env_config.MOVES:
#   0=idle, 1=up, 2=down, 3=left, 4=right
_DELTA_TO_ACTION = {
    (0, 0): 0,
    (-1, 0): 1,
    (1, 0): 2,
    (0, -1): 3,
    (0, 1): 4,
}


def _base_env(env):
    base = env
    while hasattr(base, "env"):
        base = base.env
    return base


def _step_toward(pos, nxt):
    """Return the discrete action moving from ``pos`` to adjacent ``nxt``."""
    dr = nxt[0] - pos[0]
    dc = nxt[1] - pos[1]
    return _DELTA_TO_ACTION.get((dr, dc), 0)


def _greedy_action(obstacles, pos, goal, avoid=None):
    """Greedy one-step action toward ``goal`` via BFS shortest path.

    ``avoid`` is an optional set of cells to route around (used by the
    DRIMAPS controller to force a detour). Falls back to a direct path if no
    detour exists, and to idle if already at goal or boxed in.
    """
    if pos == goal:
        return 0
    path = None
    if avoid:
        path = bfs_avoiding(obstacles, pos, goal, avoid)
    if path is None:
        path = bfs_shortest_path(obstacles, pos, goal)
    if path is None or len(path) < 2:
        return 0
    return _step_toward(pos, path[1])


def _run_episode(cfg, policy, max_steps, seed):
    """Run one episode under the given policy; return (isr, deadlock_events).

    All policies share an identical instance (same cfg, same seed), so the
    comparison is apples-to-apples.
    """
    env = drimapsim_v0(cfg)
    env.reset(seed=seed)
    base = _base_env(env)
    grid = base.grid
    n = base.config.num_agents
    obstacles = grid.get_obstacles()

    rng = np.random.RandomState(seed)

    # DRIMAPS controller state.
    wfg = WaitForGraph(n)
    detector = CycleDetector()
    stall_counter = {i: 0 for i in range(n)}
    reroute_avoid = {i: None for i in range(n)}  # per-agent detour blockers
    prev_positions = {i: grid.get_agents_xy()[i] for i in range(n)}

    reached = set()

    for _step in range(max_steps):
        positions = grid.get_agents_xy()
        goals = grid.get_targets_xy()
        active = [i for i in range(n) if grid.is_active[i]]

        if policy == "random":
            actions = list(rng.randint(0, 5, size=n))

        elif policy == "greedy":
            actions = [0] * n
            for i in active:
                actions[i] = _greedy_action(obstacles, positions[i], goals[i])

        elif policy == "drimaps":
            # 1) Detect per-agent stalls (wanted to move but didn't).
            for i in active:
                if positions[i] == prev_positions.get(i) and positions[i] != goals[i]:
                    stall_counter[i] += 1
                else:
                    stall_counter[i] = 0
                    reroute_avoid[i] = None  # progress -> drop the detour

            # 2) Build desired-next cells for the WFG (greedy intent).
            cur_map = {i: positions[i] for i in range(n)}
            nxt_map = {}
            for i in range(n):
                if not grid.is_active[i] or positions[i] == goals[i]:
                    nxt_map[i] = positions[i]
                    continue
                a = _greedy_action(obstacles, positions[i], goals[i],
                                   avoid=reroute_avoid[i])
                dr, dc = {0: (0, 0), 1: (-1, 0), 2: (1, 0),
                          3: (0, -1), 4: (0, 1)}[a]
                nxt_map[i] = (positions[i][0] + dr, positions[i][1] + dc)
            finished = {i for i in range(n) if positions[i] == goals[i]}

            # 3) Update the Wait-For Graph and find deadlock cycles.
            wfg.update_all(cur_map, nxt_map, list(goals), finished)
            cycles = detector.detect_cycles(wfg, finished)

            # 4) Resolve: for each cycle with a persistently-stalled member,
            #    pick a victim and force it to detour around the blockers.
            for cycle in cycles:
                stalled = [a for a in cycle if stall_counter.get(a, 0) >= 2]
                victims = stalled or cycle
                victim = min(victims)  # deterministic choice
                blockers = {positions[a] for a in cycle if a != victim}
                reroute_avoid[victim] = blockers

            # 5) Compose greedy actions, honouring any active detour.
            actions = [0] * n
            for i in active:
                actions[i] = _greedy_action(
                    obstacles, positions[i], goals[i], avoid=reroute_avoid[i]
                )
        else:
            raise ValueError(f"unknown policy {policy}")

        prev_positions = {i: positions[i] for i in range(n)}
        env.step(actions)

        for i in range(n):
            if grid.on_goal(i):
                reached.add(i)

        if len(reached) == n:
            break

    isr = len(reached) / n if n else 0.0
    deadlock_events = grid.deadlock_event_count()
    return isr, deadlock_events


def main():
    # A few maps known to induce congestion, a few seeds, modest agent counts.
    scenarios = [
        ("corridor", 20, 14),
        ("bottleneck", 20, 16),
        ("room", 20, 18),
        ("random", 20, 20),
    ]
    seeds = [0, 1, 2, 3, 5]
    max_steps = 256
    policies = ["random", "greedy", "drimaps"]

    # results[policy] -> list of (isr, deadlocks)
    results = {p: [] for p in policies}

    for map_type, size, n_agents in scenarios:
        for seed in seeds:
            # Build ONE instance (fixed starts/goals) shared by all policies.
            obstacles = generate_map(map_type, size, 0.2, seed)
            starts, goals = _sample_instance(obstacles, n_agents, seed)
            if not starts:
                continue
            for policy in policies:
                cfg = EnvConfig(
                    size=size,
                    seed=seed,
                    collision_system="block_both",
                    on_target="nothing",  # all must arrive; deadlocks bite
                    map=_grid_to_map_str(obstacles),
                    agents_xy=starts,
                    targets_xy=goals,
                    max_episode_steps=max_steps,
                    deadlock_tracking=True,
                )
                isr, deadlocks = _run_episode(cfg, policy, max_steps, seed)
                results[policy].append((isr, deadlocks))

    _print_table(results, policies)


def _sample_instance(obstacles, n_agents, seed):
    """Sample distinct reachable start/goal cells on free space."""
    rng = np.random.RandomState(seed + 9973)
    h, w = obstacles.shape
    free = [(r, c) for r in range(h) for c in range(w) if obstacles[r, c] == 0]
    if len(free) < 2 * n_agents:
        n_agents = len(free) // 2
    if n_agents == 0:
        return [], []
    idx = rng.choice(len(free), size=2 * n_agents, replace=False)
    starts = [free[i] for i in idx[:n_agents]]
    goals = [free[i] for i in idx[n_agents:]]
    # Keep only agents whose goal is reachable from their start.
    s2, g2 = [], []
    for s, g in zip(starts, goals):
        if bfs_shortest_path(obstacles, s, g) is not None:
            s2.append(s)
            g2.append(g)
    return s2, g2


def _grid_to_map_str(obstacles):
    rows = []
    for r in range(obstacles.shape[0]):
        rows.append("".join("." if obstacles[r, c] == 0 else "#"
                            for c in range(obstacles.shape[1])))
    return "\n".join(rows)


def _print_table(results, policies):
    print()
    print("DRIMAPS sim cross-validation")
    print("=" * 58)
    print("Driving DRIMAPS as a runtime controller inside sim/ "
          "(block_both).")
    print(f"Instances per policy: {len(results[policies[0]])}")
    print()
    header = f"{'policy':<10} {'mean ISR':>10} {'mean deadlocks':>16} " \
             f"{'ISR=1.0 rate':>14}"
    print(header)
    print("-" * len(header))
    for p in policies:
        isrs = [x[0] for x in results[p]]
        dls = [x[1] for x in results[p]]
        mean_isr = float(np.mean(isrs)) if isrs else 0.0
        mean_dl = float(np.mean(dls)) if dls else 0.0
        solved = (sum(1 for v in isrs if v >= 0.999) / len(isrs)
                  if isrs else 0.0)
        print(f"{p:<10} {mean_isr:>10.3f} {mean_dl:>16.2f} {solved:>14.2%}")
    print()

    # Headline comparison: does runtime resolution help over plain greedy?
    g_isr = np.mean([x[0] for x in results["greedy"]])
    d_isr = np.mean([x[0] for x in results["drimaps"]])
    g_dl = np.mean([x[1] for x in results["greedy"]])
    d_dl = np.mean([x[1] for x in results["drimaps"]])
    print(f"ISR  greedy={g_isr:.3f}  drimaps={d_isr:.3f}  "
          f"(delta {d_isr - g_isr:+.3f})")
    print(f"DL   greedy={g_dl:.2f}  drimaps={d_dl:.2f}  "
          f"(delta {d_dl - g_dl:+.2f})")
    verdict = ("supports" if d_isr >= g_isr - 1e-9 and d_dl <= g_dl + 1e-9
               else "does NOT clearly support")
    print()
    print(f"Cross-validation {verdict} the offline finding: runtime "
          "deadlock\nresolution improves throughput / reduces deadlocks "
          "in the interactive sim.")


if __name__ == "__main__":
    main()
