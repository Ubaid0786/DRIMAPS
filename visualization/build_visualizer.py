#!/usr/bin/env python3
"""Build the self-contained HTML visualizer with embedded demo data."""

import json, os

DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(DIR, "demo_data.json")) as f:
    demo_json = json.dumps(json.load(f), separators=(',', ':'))

html = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DRIMAPS — Interactive Multi-Agent Path Finding Visualizer</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&family=JetBrains+Mono:wght@400;600&display=swap');

  *{margin:0;padding:0;box-sizing:border-box}
  body{
    font-family:'Inter',sans-serif;
    background:#0a0e1a;
    color:#e0e6f0;
    min-height:100vh;
    overflow-x:hidden;
  }

  /* Header */
  .header{
    background:linear-gradient(135deg,#0f1629 0%,#1a1f3a 50%,#0d1225 100%);
    border-bottom:1px solid rgba(99,179,237,0.15);
    padding:18px 30px;
    display:flex;align-items:center;gap:18px;
    box-shadow:0 4px 30px rgba(0,0,0,0.4);
  }
  .logo{
    width:44px;height:44px;border-radius:12px;
    background:linear-gradient(135deg,#4fc3f7,#7c4dff);
    display:flex;align-items:center;justify-content:center;
    font-size:20px;font-weight:900;color:#fff;
    box-shadow:0 0 20px rgba(79,195,247,0.3);
  }
  .header h1{font-size:22px;font-weight:700;letter-spacing:-0.5px}
  .header h1 span{
    background:linear-gradient(90deg,#4fc3f7,#b388ff);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  }
  .header .subtitle{font-size:12px;color:#8892b0;font-weight:400;margin-top:2px}

  /* Main layout */
  .main{display:flex;gap:0;height:calc(100vh - 82px)}

  /* Sidebar */
  .sidebar{
    width:300px;min-width:300px;
    background:#0f1629;
    border-right:1px solid rgba(99,179,237,0.1);
    overflow-y:auto;padding:16px;
    display:flex;flex-direction:column;gap:14px;
  }
  .sidebar h3{
    font-size:11px;text-transform:uppercase;letter-spacing:1.5px;
    color:#4fc3f7;font-weight:600;margin-bottom:4px;
  }
  .scenario-card{
    background:linear-gradient(135deg,#151b30,#1a2040);
    border:1px solid rgba(99,179,237,0.1);
    border-radius:10px;padding:12px 14px;cursor:pointer;
    transition:all 0.2s;
  }
  .scenario-card:hover{
    border-color:rgba(99,179,237,0.3);
    box-shadow:0 0 15px rgba(79,195,247,0.1);
    transform:translateY(-1px);
  }
  .scenario-card.active{
    border-color:#4fc3f7;
    box-shadow:0 0 20px rgba(79,195,247,0.2);
    background:linear-gradient(135deg,#1a2545,#1f2a55);
  }
  .scenario-card .name{font-weight:600;font-size:14px;text-transform:capitalize;margin-bottom:6px}
  .scenario-card .meta{font-size:11px;color:#8892b0;display:flex;gap:10px;flex-wrap:wrap}
  .scenario-card .meta .tag{
    background:rgba(79,195,247,0.1);color:#4fc3f7;
    padding:2px 7px;border-radius:4px;font-family:'JetBrains Mono',monospace;
  }
  .scenario-card .meta .dl{color:#ff6b6b}

  /* Stats panel */
  .stats-panel{
    background:linear-gradient(135deg,#151b30,#1a2040);
    border:1px solid rgba(99,179,237,0.1);
    border-radius:10px;padding:14px;
  }
  .stat-row{display:flex;justify-content:space-between;padding:4px 0;font-size:12px}
  .stat-row .label{color:#8892b0}
  .stat-row .value{font-family:'JetBrains Mono',monospace;font-weight:600;color:#e0e6f0}

  /* Center area */
  .center{flex:1;display:flex;flex-direction:column;overflow:hidden}

  /* Controls bar */
  .controls{
    background:linear-gradient(180deg,#111830,#0f1629);
    border-bottom:1px solid rgba(99,179,237,0.1);
    padding:10px 24px;
    display:flex;align-items:center;gap:14px;
    flex-wrap:wrap;
  }
  .btn{
    background:linear-gradient(135deg,#1a2545,#1f2a55);
    border:1px solid rgba(99,179,237,0.2);
    color:#e0e6f0;padding:7px 16px;border-radius:8px;
    cursor:pointer;font-size:13px;font-weight:500;
    transition:all 0.2s;font-family:'Inter',sans-serif;
    display:flex;align-items:center;gap:5px;
  }
  .btn:hover{border-color:#4fc3f7;box-shadow:0 0 12px rgba(79,195,247,0.15)}
  .btn.active{background:linear-gradient(135deg,#1a3a6a,#2a4a8a);border-color:#4fc3f7}
  .btn.primary{background:linear-gradient(135deg,#1565c0,#1976d2);border-color:#42a5f5}
  .btn.primary:hover{box-shadow:0 0 15px rgba(66,165,245,0.3)}

  .speed-control{display:flex;align-items:center;gap:6px;font-size:12px;color:#8892b0}
  .speed-control input[type=range]{
    width:80px;accent-color:#4fc3f7;
  }

  .step-display{
    font-family:'JetBrains Mono',monospace;font-size:13px;
    color:#4fc3f7;margin-left:auto;
    background:rgba(79,195,247,0.08);padding:5px 12px;border-radius:6px;
  }

  /* Canvas container */
  .canvas-container{
    flex:1;display:flex;align-items:center;justify-content:center;
    background:radial-gradient(ellipse at center,#0d1225 0%,#080c18 100%);
    position:relative;overflow:hidden;
  }
  canvas{
    border-radius:8px;
    box-shadow:0 0 40px rgba(0,0,0,0.5),0 0 80px rgba(79,195,247,0.05);
  }

  /* Legend */
  .legend{
    background:linear-gradient(180deg,#0f1629,#111830);
    border-top:1px solid rgba(99,179,237,0.1);
    padding:8px 24px;
    display:flex;align-items:center;gap:20px;font-size:11px;color:#8892b0;
    flex-wrap:wrap;
  }
  .legend-item{display:flex;align-items:center;gap:5px}
  .legend-dot{width:12px;height:12px;border-radius:50%}
  .legend-sq{width:12px;height:12px;border-radius:2px}

  /* Progress bar */
  .progress-wrap{flex:1;max-width:400px}
  .progress-bar{
    width:100%;height:6px;background:rgba(79,195,247,0.1);
    border-radius:3px;overflow:hidden;cursor:pointer;
  }
  .progress-fill{height:100%;background:linear-gradient(90deg,#4fc3f7,#7c4dff);border-radius:3px;transition:width 0.1s}

  /* Toast */
  .toast{
    position:fixed;bottom:20px;left:50%;transform:translateX(-50%);
    background:linear-gradient(135deg,#1a3a6a,#2a4a8a);
    border:1px solid #4fc3f7;border-radius:10px;
    padding:10px 20px;font-size:13px;
    box-shadow:0 4px 30px rgba(79,195,247,0.2);
    opacity:0;transition:opacity 0.3s;pointer-events:none;z-index:100;
  }
  .toast.show{opacity:1}
</style>
</head>
<body>

<div class="header">
  <div class="logo">D</div>
  <div>
    <h1><span>DRIMAPS</span> Interactive Visualizer</h1>
    <div class="subtitle">Detection-Guided Deadlock Escape for Reactive Multi-Agent Path Finding</div>
  </div>
</div>

<div class="main">
  <div class="sidebar">
    <h3>🗺️ Scenarios</h3>
    <div id="scenario-list"></div>

    <h3>📊 Statistics</h3>
    <div class="stats-panel" id="stats-panel">
      <div class="stat-row"><span class="label">Map Type</span><span class="value" id="st-map">—</span></div>
      <div class="stat-row"><span class="label">Grid Size</span><span class="value" id="st-size">—</span></div>
      <div class="stat-row"><span class="label">Agents</span><span class="value" id="st-agents">—</span></div>
      <div class="stat-row"><span class="label">Total Steps</span><span class="value" id="st-steps">—</span></div>
      <div class="stat-row"><span class="label">Reached Goal</span><span class="value" id="st-reached">—</span></div>
      <div class="stat-row"><span class="label">Deadlocks</span><span class="value" id="st-deadlocks">—</span></div>
      <div class="stat-row"><span class="label">Resolved</span><span class="value" id="st-resolved">—</span></div>
      <div class="stat-row"><span class="label">Agents Replanned</span><span class="value" id="st-replanned">—</span></div>
    </div>

    <h3>🎨 Display</h3>
    <div style="display:flex;flex-direction:column;gap:8px">
      <label style="font-size:12px;display:flex;align-items:center;gap:8px;cursor:pointer">
        <input type="checkbox" id="show-trails" checked> Show trails
      </label>
      <label style="font-size:12px;display:flex;align-items:center;gap:8px;cursor:pointer">
        <input type="checkbox" id="show-goals" checked> Show goals
      </label>
      <label style="font-size:12px;display:flex;align-items:center;gap:8px;cursor:pointer">
        <input type="checkbox" id="show-ids" checked> Show agent IDs
      </label>
    </div>
  </div>

  <div class="center">
    <div class="controls">
      <button class="btn" id="btn-prev" onclick="stepBack()">⏮ Prev</button>
      <button class="btn primary" id="btn-play" onclick="togglePlay()">▶ Play</button>
      <button class="btn" id="btn-next" onclick="stepForward()">Next ⏭</button>
      <button class="btn" onclick="resetSim()">⟲ Reset</button>

      <div class="progress-wrap">
        <div class="progress-bar" id="progress-bar" onclick="seekProgress(event)">
          <div class="progress-fill" id="progress-fill"></div>
        </div>
      </div>

      <div class="speed-control">
        <span>Speed</span>
        <input type="range" id="speed" min="1" max="30" value="6">
        <span id="speed-label">6 fps</span>
      </div>

      <div class="step-display" id="step-display">t = 0 / 0</div>
    </div>

    <div class="canvas-container">
      <canvas id="canvas"></canvas>
    </div>

    <div class="legend">
      <div class="legend-item"><div class="legend-dot" style="background:#4FC3F7"></div> Agent</div>
      <div class="legend-item"><div class="legend-dot" style="background:#66BB6A;opacity:0.5"></div> Goal marker</div>
      <div class="legend-item"><div class="legend-sq" style="background:#37474F"></div> Obstacle</div>
      <div class="legend-item"><div class="legend-sq" style="background:#1a2040"></div> Free cell</div>
      <div class="legend-item"><div class="legend-dot" style="background:#4CAF50;border:2px solid #fff"></div> Arrived</div>
      <div class="legend-item" style="color:#4fc3f7">Trails show recent path history</div>
    </div>
  </div>
</div>

<div class="toast" id="toast"></div>

<script>
// ─── Embedded Demo Data ──────────────────────────────────────────
const DEMOS = """ + demo_json + r""";

// ─── Agent Colors ────────────────────────────────────────────────
const AGENT_COLORS = [
  '#4FC3F7','#81C784','#FFB74D','#E57373','#BA68C8',
  '#4DD0E1','#AED581','#FFD54F','#FF8A65','#9575CD',
  '#26C6DA','#66BB6A','#FFC107','#EF5350','#AB47BC',
  '#00BCD4','#43A047','#FF9800','#F44336','#8E24AA',
  '#29B6F6','#9CCC65','#FFCA28','#EF9A9A','#CE93D8',
];

// ─── State ───────────────────────────────────────────────────────
let currentScenario = 0;
let currentStep = 0;
let playing = false;
let timer = null;
let trailLength = 8;

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// ─── Init ────────────────────────────────────────────────────────
function init() {
  buildScenarioList();
  selectScenario(0);

  document.getElementById('speed').addEventListener('input', e => {
    document.getElementById('speed-label').textContent = e.target.value + ' fps';
    if (playing) { clearInterval(timer); timer = setInterval(tick, 1000/e.target.value); }
  });
  ['show-trails','show-goals','show-ids'].forEach(id => {
    document.getElementById(id).addEventListener('change', () => draw());
  });

  window.addEventListener('keydown', e => {
    if (e.key === ' ') { e.preventDefault(); togglePlay(); }
    if (e.key === 'ArrowRight') stepForward();
    if (e.key === 'ArrowLeft') stepBack();
  });
}

function buildScenarioList() {
  const list = document.getElementById('scenario-list');
  list.innerHTML = '';
  DEMOS.forEach((d, i) => {
    const card = document.createElement('div');
    card.className = 'scenario-card' + (i === 0 ? ' active' : '');
    card.id = 'sc-' + i;
    const reached = d.reached_goal.filter(Boolean).length;
    card.innerHTML = `
      <div class="name">${d.map_type} Map</div>
      <div class="meta">
        <span class="tag">${d.size}×${d.size}</span>
        <span class="tag">${d.num_agents} agents</span>
        <span class="tag">${d.total_steps} steps</span>
        ${d.deadlocks_detected > 0 ? `<span class="tag dl">🔴 ${d.deadlocks_detected} deadlocks</span>` : '<span class="tag" style="color:#66BB6A">✓ No deadlocks</span>'}
      </div>
    `;
    card.onclick = () => selectScenario(i);
    list.appendChild(card);
  });
}

function selectScenario(idx) {
  if (playing) togglePlay();
  currentScenario = idx;
  currentStep = 0;

  document.querySelectorAll('.scenario-card').forEach((c,i) => {
    c.classList.toggle('active', i === idx);
  });

  const d = DEMOS[idx];
  const reached = d.reached_goal.filter(Boolean).length;
  document.getElementById('st-map').textContent = d.map_type;
  document.getElementById('st-size').textContent = d.size + '×' + d.size;
  document.getElementById('st-agents').textContent = d.num_agents;
  document.getElementById('st-steps').textContent = d.total_steps;
  document.getElementById('st-reached').textContent = reached + '/' + d.num_agents;
  document.getElementById('st-deadlocks').textContent = d.deadlocks_detected;
  document.getElementById('st-resolved').textContent = d.deadlocks_resolved;
  document.getElementById('st-replanned').textContent = d.agents_replanned;

  // Color reached
  const el = document.getElementById('st-reached');
  el.style.color = reached === d.num_agents ? '#66BB6A' : '#FFB74D';
  document.getElementById('st-deadlocks').style.color = d.deadlocks_detected > 0 ? '#FF6B6B' : '#66BB6A';

  resizeCanvas();
  draw();
}

function resizeCanvas() {
  const d = DEMOS[currentScenario];
  const container = document.querySelector('.canvas-container');
  const maxW = container.clientWidth - 40;
  const maxH = container.clientHeight - 40;
  const cellSize = Math.min(Math.floor(maxW / d.grid[0].length), Math.floor(maxH / d.grid.length), 40);
  canvas.width = d.grid[0].length * cellSize;
  canvas.height = d.grid.length * cellSize;
  canvas._cellSize = cellSize;
}

// ─── Drawing ─────────────────────────────────────────────────────
function draw() {
  const d = DEMOS[currentScenario];
  const cs = canvas._cellSize;
  const h = d.grid.length, w = d.grid[0].length;
  const positions = d.trajectory[currentStep];
  const showTrails = document.getElementById('show-trails').checked;
  const showGoals = document.getElementById('show-goals').checked;
  const showIds = document.getElementById('show-ids').checked;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Grid
  for (let r = 0; r < h; r++) {
    for (let c = 0; c < w; c++) {
      if (d.grid[r][c] === 1) {
        ctx.fillStyle = '#2c3345';
        ctx.fillRect(c*cs, r*cs, cs, cs);
        // subtle 3D effect
        ctx.fillStyle = '#374058';
        ctx.fillRect(c*cs, r*cs, cs, cs*0.15);
      } else {
        ctx.fillStyle = '#0e1525';
        ctx.fillRect(c*cs, r*cs, cs, cs);
        ctx.strokeStyle = 'rgba(99,179,237,0.04)';
        ctx.lineWidth = 0.5;
        ctx.strokeRect(c*cs, r*cs, cs, cs);
      }
    }
  }

  // Goal markers
  if (showGoals) {
    for (let i = 0; i < d.goals.length; i++) {
      const [gr, gc] = d.goals[i];
      const cx = gc*cs + cs/2, cy = gr*cs + cs/2;
      const s = cs*0.3;
      ctx.save();
      ctx.globalAlpha = 0.3;
      ctx.fillStyle = AGENT_COLORS[i % AGENT_COLORS.length];
      ctx.beginPath();
      ctx.moveTo(cx, cy-s); ctx.lineTo(cx+s, cy);
      ctx.lineTo(cx, cy+s); ctx.lineTo(cx-s, cy);
      ctx.closePath();
      ctx.fill();
      ctx.globalAlpha = 0.6;
      ctx.strokeStyle = AGENT_COLORS[i % AGENT_COLORS.length];
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.restore();
    }
  }

  // Trails
  if (showTrails) {
    for (let i = 0; i < d.num_agents; i++) {
      const color = AGENT_COLORS[i % AGENT_COLORS.length];
      const startT = Math.max(0, currentStep - trailLength);
      for (let t = startT; t < currentStep; t++) {
        const [pr, pc] = d.trajectory[t][i];
        const alpha = 0.08 + 0.15 * ((t - startT) / trailLength);
        ctx.fillStyle = color;
        ctx.globalAlpha = alpha;
        ctx.beginPath();
        ctx.arc(pc*cs+cs/2, pr*cs+cs/2, cs*0.15, 0, Math.PI*2);
        ctx.fill();
      }
      ctx.globalAlpha = 1;
    }
  }

  // Agents
  for (let i = 0; i < d.num_agents; i++) {
    const [ar, ac] = positions[i];
    const cx = ac*cs + cs/2, cy = ar*cs + cs/2;
    const r = cs * 0.35;
    const color = AGENT_COLORS[i % AGENT_COLORS.length];
    const atGoal = ar === d.goals[i][0] && ac === d.goals[i][1];

    // Glow
    ctx.save();
    const glow = ctx.createRadialGradient(cx, cy, 0, cx, cy, r*2);
    glow.addColorStop(0, color + '40');
    glow.addColorStop(1, 'transparent');
    ctx.fillStyle = glow;
    ctx.fillRect(cx-r*2, cy-r*2, r*4, r*4);
    ctx.restore();

    // Circle
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI*2);
    ctx.fillStyle = color;
    ctx.fill();

    // Border
    ctx.strokeStyle = atGoal ? '#4CAF50' : '#fff';
    ctx.lineWidth = atGoal ? 3 : 1.5;
    ctx.stroke();

    // Arrival checkmark
    if (atGoal) {
      ctx.save();
      ctx.fillStyle = '#4CAF50';
      ctx.beginPath();
      ctx.arc(cx+r*0.6, cy-r*0.6, r*0.35, 0, Math.PI*2);
      ctx.fill();
      ctx.fillStyle = '#fff';
      ctx.font = `bold ${Math.max(8, cs*0.18)}px 'Inter'`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('✓', cx+r*0.6, cy-r*0.6);
      ctx.restore();
    }

    // ID
    if (showIds) {
      ctx.fillStyle = '#fff';
      ctx.font = `bold ${Math.max(9, cs*0.3)}px 'JetBrains Mono'`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(i, cx, cy+1);
    }
  }

  // Update UI
  document.getElementById('step-display').textContent =
    `t = ${currentStep} / ${d.total_steps - 1}`;
  const pct = d.total_steps > 1 ? (currentStep / (d.total_steps - 1)) * 100 : 0;
  document.getElementById('progress-fill').style.width = pct + '%';
}

// ─── Playback ────────────────────────────────────────────────────
function togglePlay() {
  playing = !playing;
  const btn = document.getElementById('btn-play');
  if (playing) {
    btn.textContent = '⏸ Pause';
    btn.classList.add('active');
    const fps = parseInt(document.getElementById('speed').value);
    timer = setInterval(tick, 1000/fps);
  } else {
    btn.textContent = '▶ Play';
    btn.classList.remove('active');
    clearInterval(timer);
  }
}

function tick() {
  const d = DEMOS[currentScenario];
  if (currentStep < d.total_steps - 1) {
    currentStep++;
    draw();
  } else {
    togglePlay();
    showToast('✅ Simulation complete!');
  }
}

function stepForward() {
  const d = DEMOS[currentScenario];
  if (currentStep < d.total_steps - 1) { currentStep++; draw(); }
}

function stepBack() {
  if (currentStep > 0) { currentStep--; draw(); }
}

function resetSim() {
  if (playing) togglePlay();
  currentStep = 0;
  draw();
  showToast('⟲ Reset to t=0');
}

function seekProgress(e) {
  const bar = document.getElementById('progress-bar');
  const rect = bar.getBoundingClientRect();
  const pct = (e.clientX - rect.left) / rect.width;
  const d = DEMOS[currentScenario];
  currentStep = Math.round(pct * (d.total_steps - 1));
  currentStep = Math.max(0, Math.min(currentStep, d.total_steps - 1));
  draw();
}

function showToast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 2000);
}

// ─── Resize handling ─────────────────────────────────────────────
window.addEventListener('resize', () => { resizeCanvas(); draw(); });

// ─── Start ───────────────────────────────────────────────────────
init();
</script>
</body>
</html>"""

out_path = os.path.join(DIR, "drimaps_visualizer.html")
with open(out_path, "w") as f:
    f.write(html)
print(f"Built: {out_path}")
print(f"Size: {os.path.getsize(out_path)} bytes")
