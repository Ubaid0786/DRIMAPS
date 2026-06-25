#!/usr/bin/env python3
"""
Path Execution Animation

Generates frame-by-frame animation of agent path execution,
highlighting deadlocks and resolutions.
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
    import matplotlib.animation as animation
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from src.utils import Path, get_position_at_time


def animate_execution(
    grid: np.ndarray,
    paths: list,
    goals: list,
    output_path: str = "execution.gif",
    fps: int = 4,
) -> None:
    """Animate path execution.

    Args:
        grid: Map grid.
        paths: Agent paths.
        goals: Agent goals.
        output_path: Output GIF path.
        fps: Frames per second.
    """
    if not HAS_MPL:
        print("[WARN] matplotlib not available.")
        return

    h, w = grid.shape
    max_t = max(len(p) for p in paths) if paths else 1
    n = len(paths)

    fig, ax = plt.subplots(figsize=(max(6, w * 0.4), max(6, h * 0.4)))
    colors = plt.cm.tab10(np.linspace(0, 1, max(n, 1)))

    def draw_frame(t):
        ax.clear()
        # Grid
        for r in range(h):
            for c in range(w):
                color = "#333" if grid[r, c] == 1 else "#f0f0f0"
                ax.add_patch(plt.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1,
                    facecolor=color, edgecolor="#ddd"
                ))

        # Goals
        for i, goal in enumerate(goals):
            ax.plot(goal[1], goal[0], "x", color="green",
                    markersize=10, markeredgewidth=2)

        # Agents
        for i in range(n):
            pos = get_position_at_time(paths[i], t)
            at_goal = pos == goals[i]
            ax.plot(pos[1], pos[0], "o",
                    color=colors[i % len(colors)],
                    markersize=12,
                    markeredgecolor="green" if at_goal else "black",
                    markeredgewidth=2 if at_goal else 1)
            ax.text(pos[1], pos[0], str(i), ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold")

        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.set_title(f"t = {t}", fontsize=12)

    try:
        ani = animation.FuncAnimation(
            fig, draw_frame, frames=max_t,
            interval=1000 // fps, repeat=False,
        )
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        ani.save(output_path, writer="pillow", fps=fps)
        print(f"  Saved animation: {output_path}")
    except Exception as e:
        print(f"  [WARN] Animation save failed: {e}")
        # Save first and last frames as PNG instead
        draw_frame(0)
        fig.savefig(output_path.replace(".gif", "_start.png"),
                    dpi=100, bbox_inches="tight")
        draw_frame(max_t - 1)
        fig.savefig(output_path.replace(".gif", "_end.png"),
                    dpi=100, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Solve a small instance with DRIMAPS and animate the executed paths."""
    import argparse

    from src.config import DRIMAPSConfig, MapConfig
    from src.drimaps import DRIMAPS
    from experiments.generate_instances import generate_grid, generate_scenario

    parser = argparse.ArgumentParser(description="Animate DRIMAPS path execution.")
    parser.add_argument("--map-type", default="bottleneck",
                        help="Map type (random, warehouse, corridor, bottleneck, maze, room, ...).")
    parser.add_argument("--size", type=int, default=20, help="Grid side length.")
    parser.add_argument("--agents", type=int, default=12, help="Number of agents.")
    parser.add_argument("--seed", type=int, default=42, help="Instance seed.")
    parser.add_argument("--density", type=float, default=0.2, help="Obstacle density.")
    parser.add_argument("--output", default="execution.gif", help="Output GIF path.")
    parser.add_argument("--fps", type=int, default=4, help="Frames per second.")
    args = parser.parse_args()

    if not HAS_MPL:
        print("[ERROR] matplotlib is required for animation. Install it with "
              "`pip install matplotlib`.")
        return

    mc = MapConfig(args.map_type, args.map_type, args.size, args.size,
                   args.map_type, {"obstacle_density": args.density})
    grid = generate_grid(mc, args.seed)
    starts, goals = generate_scenario(grid, args.agents, args.seed)

    result = DRIMAPS(DRIMAPSConfig()).solve(grid, starts, goals)
    reached = sum(
        1 for i, p in enumerate(result.paths)
        if p and p[-1] == goals[i]
    )
    print(f"  Solved {args.map_type} ({args.size}x{args.size}, {args.agents} agents, "
          f"seed {args.seed}): {reached}/{args.agents} reached, "
          f"{result.deadlocks_detected} deadlocks detected.")
    animate_execution(grid, result.paths, goals,
                      output_path=args.output, fps=args.fps)


if __name__ == "__main__":
    main()
