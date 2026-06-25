#!/usr/bin/env python3
"""
Deadlock Visualization

Visualizes deadlock scenarios on the grid map, showing agent
positions, dependency edges, and deadlock cycles.
"""

import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from src.utils import Position, Path
from src.dependency_graph import WaitForGraph


def visualize_deadlock(
    grid: np.ndarray,
    positions: dict,
    goals: list,
    wfg: WaitForGraph,
    cycle: list,
    output_path: str = "deadlock.png",
    title: str = "Deadlock Scenario",
) -> None:
    """Visualize a deadlock on the grid.

    Args:
        grid: Map grid.
        positions: Agent positions.
        goals: Agent goal positions.
        wfg: Wait-For Graph.
        cycle: Deadlock cycle agents.
        output_path: Output file path.
        title: Figure title.
    """
    if not HAS_MPL:
        print("[WARN] matplotlib not available for visualization.")
        return

    h, w = grid.shape
    fig, ax = plt.subplots(figsize=(max(8, w * 0.5), max(8, h * 0.5)))

    # Draw grid
    for r in range(h):
        for c in range(w):
            if grid[r, c] == 1:
                ax.add_patch(plt.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1,
                    facecolor="#333", edgecolor="#555"
                ))
            else:
                ax.add_patch(plt.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1,
                    facecolor="#f0f0f0", edgecolor="#ddd"
                ))

    # Draw agents
    cycle_set = set(cycle)
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(positions), 1)))

    for i, pos in positions.items():
        color = "red" if i in cycle_set else colors[i % len(colors)]
        ax.plot(pos[1], pos[0], "o", color=color, markersize=15, zorder=5)
        ax.text(pos[1], pos[0], str(i), ha="center", va="center",
                fontsize=8, fontweight="bold", color="white", zorder=6)

    # Draw goals
    for i, goal in enumerate(goals):
        if i < len(goals):
            ax.plot(goal[1], goal[0], "x", color="green",
                    markersize=12, markeredgewidth=3, zorder=4)

    # Draw WFG edges
    for i in cycle:
        for j in wfg.successors(i):
            if i in positions and j in positions:
                pi, pj = positions[i], positions[j]
                ax.annotate(
                    "", xy=(pj[1], pj[0]), xytext=(pi[1], pi[0]),
                    arrowprops=dict(
                        arrowstyle="->", color="red",
                        lw=2, connectionstyle="arc3,rad=0.2"
                    ),
                    zorder=3,
                )

    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


if __name__ == "__main__":
    # Demo
    grid = np.zeros((8, 8), dtype=int)
    grid[3, 1:7] = 1
    grid[3, 4] = 0

    wfg = WaitForGraph(3)
    wfg.add_edge(0, 1)
    wfg.add_edge(1, 2)
    wfg.add_edge(2, 0)

    positions = {0: (2, 3), 1: (2, 5), 2: (4, 4)}
    goals = [(4, 5), (4, 3), (2, 3)]

    visualize_deadlock(
        grid, positions, goals, wfg, [0, 1, 2],
        output_path="deadlock_demo.png",
        title="3-Agent Cyclic Deadlock"
    )
