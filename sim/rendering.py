"""
DRIMAPSim Rendering — ASCII, SVG, and Animation

Publication-quality rendering for MAPF environments with deadlock
overlay visualization.
"""

import numpy as np
from typing import Dict, List, Optional, Set, Tuple


# ── SVG Rendering ───────────────────────────────────────────────────

# Color palette
COLORS = [
    "#4FC3F7", "#81C784", "#FFB74D", "#E57373", "#BA68C8",
    "#4DD0E1", "#AED581", "#FFD54F", "#FF8A65", "#9575CD",
    "#26C6DA", "#66BB6A", "#FFC107", "#EF5350", "#AB47BC",
    "#00BCD4", "#43A047", "#FF9800", "#F44336", "#8E24AA",
]
OBSTACLE_COLOR = "#37474F"
FREE_COLOR = "#ECEFF1"
GRID_COLOR = "#B0BEC5"
GOAL_COLOR = "#FFD600"
DEADLOCK_COLOR = "#FF1744"


def render_svg(
    obstacles: np.ndarray,
    agents_xy: List[Tuple[int, int]],
    targets_xy: List[Tuple[int, int]],
    is_active: Dict[int, bool] = None,
    deadlock_agents: Set[int] = None,
    cell_size: int = 24,
    show_ids: bool = True,
    title: str = "",
) -> str:
    """Render the grid as an SVG string.

    Args:
        obstacles: Obstacle grid.
        agents_xy: Agent positions.
        targets_xy: Target positions.
        is_active: Agent active flags.
        deadlock_agents: Set of agents currently in deadlock.
        cell_size: Size of each cell in pixels.
        show_ids: Show agent IDs on the grid.
        title: Optional title.

    Returns:
        SVG string.
    """
    h, w = obstacles.shape
    cs = cell_size
    svg_w = w * cs + 2
    svg_h = h * cs + 2 + (30 if title else 0)
    title_offset = 30 if title else 0

    if is_active is None:
        is_active = {i: True for i in range(len(agents_xy))}
    if deadlock_agents is None:
        deadlock_agents = set()

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{svg_w}" height="{svg_h}" '
        f'viewBox="0 0 {svg_w} {svg_h}">\n',
        f'<rect width="{svg_w}" height="{svg_h}" fill="{FREE_COLOR}"/>\n',
    ]

    if title:
        parts.append(
            f'<text x="{svg_w // 2}" y="20" text-anchor="middle" '
            f'font-family="monospace" font-size="14" fill="#263238">'
            f'{title}</text>\n'
        )

    # Grid cells
    for r in range(h):
        for c in range(w):
            x = c * cs + 1
            y = r * cs + 1 + title_offset
            if obstacles[r, c] == 1:
                parts.append(
                    f'<rect x="{x}" y="{y}" width="{cs}" height="{cs}" '
                    f'fill="{OBSTACLE_COLOR}" rx="2"/>\n'
                )
            else:
                parts.append(
                    f'<rect x="{x}" y="{y}" width="{cs}" height="{cs}" '
                    f'fill="{FREE_COLOR}" stroke="{GRID_COLOR}" '
                    f'stroke-width="0.5"/>\n'
                )

    # Targets (diamonds)
    for i, (tr, tc) in enumerate(targets_xy):
        x = tc * cs + 1 + cs // 2
        y = tr * cs + 1 + title_offset + cs // 2
        s = cs // 4
        color = COLORS[i % len(COLORS)]
        parts.append(
            f'<polygon points="{x},{y - s} {x + s},{y} '
            f'{x},{y + s} {x - s},{y}" '
            f'fill="{color}" opacity="0.4" stroke="{color}" '
            f'stroke-width="1"/>\n'
        )

    # Agents (circles)
    for i, (ar, ac) in enumerate(agents_xy):
        if not is_active.get(i, True):
            continue
        x = ac * cs + 1 + cs // 2
        y = ar * cs + 1 + title_offset + cs // 2
        r = cs // 3

        color = COLORS[i % len(COLORS)]
        stroke = DEADLOCK_COLOR if i in deadlock_agents else "#263238"
        stroke_w = 3 if i in deadlock_agents else 1.5

        parts.append(
            f'<circle cx="{x}" cy="{y}" r="{r}" '
            f'fill="{color}" stroke="{stroke}" '
            f'stroke-width="{stroke_w}"/>\n'
        )

        if show_ids:
            parts.append(
                f'<text x="{x}" y="{y + 4}" text-anchor="middle" '
                f'font-family="monospace" font-size="{cs // 3}" '
                f'fill="#FFF" font-weight="bold">{i}</text>\n'
            )

    parts.append('</svg>')
    return ''.join(parts)


def save_svg(filepath: str, svg_content: str):
    """Save SVG string to file.

    Args:
        filepath: Output file path.
        svg_content: SVG content string.
    """
    with open(filepath, 'w') as f:
        f.write(svg_content)


# ── Animation (frame-based) ────────────────────────────────────────

def render_animation_frames(
    obstacles: np.ndarray,
    trajectory: List[List[Tuple[int, int]]],
    targets_xy: List[Tuple[int, int]],
    cell_size: int = 24,
) -> List[str]:
    """Render a list of SVG frames for animation.

    Args:
        obstacles: Obstacle grid.
        trajectory: List of position snapshots at each timestep.
        targets_xy: Target positions.
        cell_size: Cell size in pixels.

    Returns:
        List of SVG strings, one per frame.
    """
    frames = []
    for t, positions in enumerate(trajectory):
        svg = render_svg(
            obstacles,
            positions,
            targets_xy,
            cell_size=cell_size,
            title=f"t={t}",
        )
        frames.append(svg)
    return frames


def save_animation_html(filepath: str, frames: List[str], fps: int = 4):
    """Save animation as a self-contained HTML file with JS player.

    Args:
        filepath: Output HTML path.
        frames: List of SVG frame strings.
        fps: Frames per second.
    """
    import html as html_mod

    escaped_frames = [html_mod.escape(f) for f in frames]

    html_content = f"""<!DOCTYPE html>
<html>
<head>
<title>DRIMAPSim Animation</title>
<style>
body {{ font-family: monospace; background: #1a1a2e; color: #e0e0e0;
       display: flex; flex-direction: column; align-items: center;
       padding: 20px; }}
#controls {{ margin: 15px 0; }}
button {{ background: #16213e; color: #e0e0e0; border: 1px solid #0f3460;
         padding: 8px 16px; margin: 0 5px; cursor: pointer; border-radius: 4px; }}
button:hover {{ background: #0f3460; }}
#frame-info {{ margin: 10px 0; font-size: 14px; }}
#viewer {{ background: white; padding: 10px; border-radius: 8px; }}
</style>
</head>
<body>
<h2>DRIMAPSim Animation — {len(frames)} frames</h2>
<div id="controls">
    <button onclick="prev()">◀ Prev</button>
    <button onclick="toggle()">▶ Play</button>
    <button onclick="next()">Next ▶</button>
    <input type="range" id="slider" min="0" max="{len(frames) - 1}"
           value="0" oninput="setFrame(this.value)">
</div>
<div id="frame-info">Frame 0 / {len(frames) - 1}</div>
<div id="viewer"></div>
<script>
const frames = {frames};
let current = 0;
let playing = false;
let timer = null;

function show(i) {{
    current = Math.max(0, Math.min(i, frames.length - 1));
    const parser = new DOMParser();
    const decoded = frames[current].replace(/&lt;/g,'<').replace(/&gt;/g,'>').replace(/&amp;/g,'&').replace(/&quot;/g,'"');
    document.getElementById('viewer').innerHTML = decoded;
    document.getElementById('slider').value = current;
    document.getElementById('frame-info').textContent =
        'Frame ' + current + ' / ' + (frames.length - 1);
}}
function next() {{ show(current + 1); }}
function prev() {{ show(current - 1); }}
function setFrame(i) {{ show(parseInt(i)); }}
function toggle() {{
    playing = !playing;
    if (playing) {{
        timer = setInterval(next, {1000 // fps});
    }} else {{
        clearInterval(timer);
    }}
}}
show(0);
</script>
</body>
</html>"""

    with open(filepath, 'w') as f:
        f.write(html_content)
