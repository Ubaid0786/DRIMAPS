/* ═══════════════════════════════════════════════════════════════════
   DRIMAPS Research Simulator — Application Logic
   Connects the professional UI to the Flask/Python backend via
   JSON API calls.  All simulation runs on the Python side.
   ═══════════════════════════════════════════════════════════════════ */

"use strict";

// ─── State ───────────────────────────────────────────────────────
const S = {
  grid: null,           // 2D array
  starts: [],
  goals: [],
  trajectory: null,     // array of position arrays per step
  simResult: null,      // full simulation result
  step: 0,
  playing: false,
  timer: null,
  runCount: 0,
  history: [],          // all past run results
  mapType: "random",
};

// ─── Agent Colors (colorblind-friendly Tol palette + extras) ─────
const COLORS = [
  "#4477AA","#EE6677","#228833","#CCBB44","#66CCEE",
  "#AA3377","#BBBBBB","#44BB99","#EE8866","#332288",
  "#88CCEE","#CC6677","#117733","#999933","#882255",
  "#AA4499","#DDCC77","#661100","#6699CC","#44AA99",
];

// ─── View Routing ────────────────────────────────────────────────
function setView(name) {
  document.querySelectorAll(".view").forEach(v => v.classList.remove("active"));
  document.querySelectorAll(".nav-item").forEach(n => n.classList.remove("active"));
  const view = document.getElementById("view-" + name);
  if (view) view.classList.add("active");
  const nav = document.querySelector(`.nav-item[data-view="${name}"]`);
  if (nav) nav.classList.add("active");
  document.getElementById("topbar-context").textContent =
    {dashboard:"Dashboard",environment:"Environment",simulation:"Simulation",
     analytics:"Analytics",settings:"Settings"}[name] || name;
}

// ─── Init ────────────────────────────────────────────────────────
async function init() {
  // Load map types
  try {
    const types = await api("/api/maps/types");
    const sel = document.getElementById("env-map-type");
    sel.innerHTML = "";
    types.forEach(t => {
      const opt = document.createElement("option");
      opt.value = t.id; opt.textContent = t.label;
      sel.appendChild(opt);
    });
  } catch (e) { console.error("Failed to load map types:", e); }

  // Load algorithms
  try {
    const algs = await api("/api/algorithms");
    const sel = document.getElementById("sim-algorithm");
    sel.innerHTML = "";
    algs.forEach(a => {
      const opt = document.createElement("option");
      opt.value = a.id; opt.textContent = a.label;
      sel.appendChild(opt);
    });
  } catch (e) { console.error("Failed to load algorithms:", e); }

  // Load default config for settings view
  try {
    const cfg = await api("/api/config/defaults");
    buildSettingsForm(cfg);
  } catch (e) { console.error("Failed to load config:", e); }

  // Speed slider
  document.getElementById("speed-slider").addEventListener("input", e => {
    document.getElementById("speed-label").textContent = e.target.value + "fps";
    if (S.playing) { clearInterval(S.timer); S.timer = setInterval(tick, 1000 / e.target.value); }
  });

  // Display options
  ["opt-trails","opt-goals","opt-ids"].forEach(id => {
    document.getElementById(id).addEventListener("change", () => drawSim());
  });

  // Keyboard shortcuts
  window.addEventListener("keydown", e => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
    if (e.key === " ") { e.preventDefault(); togglePlay(); }
    if (e.key === "ArrowRight") simStep(1);
    if (e.key === "ArrowLeft") simStep(-1);
  });

  window.addEventListener("resize", () => {
    if (S.grid) drawEnv();
    if (S.trajectory) drawSim();
  });
}

// ─── API Helper ──────────────────────────────────────────────────
async function api(url, data) {
  const opts = data
    ? { method: "POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify(data) }
    : { method: "GET" };
  const resp = await fetch(url, opts);
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({error: resp.statusText}));
    throw new Error(err.error || resp.statusText);
  }
  return resp.json();
}

function setStatus(msg) {
  document.getElementById("sb-status").textContent = msg;
}

// ═══════════════════════════════════════════════════════════════════
//  ENVIRONMENT
// ═══════════════════════════════════════════════════════════════════

async function generateMap() {
  const mapType = document.getElementById("env-map-type").value;
  const size = parseInt(document.getElementById("env-size").value) || 32;
  const density = parseFloat(document.getElementById("env-density").value) || 0.2;
  const seed = parseInt(document.getElementById("env-seed").value) || 42;

  setStatus("Generating map...");
  try {
    const data = await api("/api/maps/generate", { map_type: mapType, size, density, seed });
    S.grid = data.grid;
    S.mapType = mapType;
    S.starts = [];
    S.goals = [];
    S.trajectory = null;
    S.simResult = null;

    // Update metrics
    document.getElementById("env-m-free").textContent = data.difficulty.free_cells;
    document.getElementById("env-m-density").textContent = data.difficulty.density;
    document.getElementById("env-m-degree").textContent = data.difficulty.avg_degree;
    document.getElementById("env-m-choke").textContent = data.difficulty.chokepoint_count;
    document.getElementById("env-m-diff").textContent = data.difficulty.difficulty_score;
    document.getElementById("env-info").textContent =
      `${mapType} ${data.height}×${data.width}  ·  ${data.difficulty.free_cells} free cells`;
    document.getElementById("sb-map").textContent =
      `${mapType} ${data.height}×${data.width}`;
    document.getElementById("env-empty").style.display = "none";

    drawEnv();
    setStatus("Map generated");
  } catch (e) {
    setStatus("Error: " + e.message);
  }
}

async function generateAgents() {
  if (!S.grid) { setStatus("Generate a map first"); return; }
  const n = parseInt(document.getElementById("env-agents").value) || 8;
  const seed = parseInt(document.getElementById("env-seed").value) || 42;

  try {
    const data = await api("/api/scenario/generate", { grid: S.grid, num_agents: n, seed });
    S.starts = data.starts;
    S.goals = data.goals;
    document.getElementById("sb-agents").textContent = data.num_agents + " agents";
    drawEnv();
    setStatus(data.num_agents + " agents placed");
  } catch (e) {
    setStatus("Error: " + e.message);
  }
}

function clearAgents() {
  S.starts = [];
  S.goals = [];
  document.getElementById("sb-agents").textContent = "0 agents";
  drawEnv();
}

async function importMovingAI(event) {
  const file = event.target.files[0];
  if (!file) return;
  const text = await file.text();
  try {
    const data = await api("/api/maps/movingai", { map_text: text });
    S.grid = data.grid;
    S.mapType = "movingai";
    S.starts = []; S.goals = [];
    document.getElementById("env-m-free").textContent = data.difficulty.free_cells;
    document.getElementById("env-m-density").textContent = data.difficulty.density;
    document.getElementById("env-m-degree").textContent = data.difficulty.avg_degree;
    document.getElementById("env-m-choke").textContent = data.difficulty.chokepoint_count;
    document.getElementById("env-m-diff").textContent = data.difficulty.difficulty_score;
    document.getElementById("env-info").textContent =
      `MovingAI ${data.height}×${data.width}  ·  ${data.difficulty.free_cells} free cells`;
    document.getElementById("env-empty").style.display = "none";
    document.getElementById("sb-map").textContent = `movingai ${data.height}×${data.width}`;
    drawEnv();
    setStatus("Imported " + file.name);
  } catch (e) {
    setStatus("Import error: " + e.message);
  }
}

function proceedToSim() {
  if (!S.grid) { setStatus("Generate a map first"); return; }
  if (!S.starts.length) { setStatus("Place agents first"); return; }
  setView("simulation");
}

// ─── Draw Environment Canvas ─────────────────────────────────────
function drawEnv() {
  if (!S.grid) return;
  const canvas = document.getElementById("env-canvas");
  const container = canvas.parentElement;
  const H = S.grid.length, W = S.grid[0].length;
  const cs = Math.min(
    Math.floor((container.clientWidth - 32) / W),
    Math.floor((container.clientHeight - 32) / H),
    40
  );
  canvas.width = W * cs;
  canvas.height = H * cs;
  const ctx = canvas.getContext("2d");

  // Grid cells
  for (let r = 0; r < H; r++) {
    for (let c = 0; c < W; c++) {
      if (S.grid[r][c] === 1) {
        ctx.fillStyle = "#495057";
        ctx.fillRect(c*cs, r*cs, cs, cs);
      } else {
        ctx.fillStyle = "#f8f9fa";
        ctx.fillRect(c*cs, r*cs, cs, cs);
        ctx.strokeStyle = "#e9ecef";
        ctx.lineWidth = 0.5;
        ctx.strokeRect(c*cs, r*cs, cs, cs);
      }
    }
  }

  // Goals (diamonds)
  for (let i = 0; i < S.goals.length; i++) {
    const [gr, gc] = S.goals[i];
    const cx = gc*cs + cs/2, cy = gr*cs + cs/2, s = cs*0.25;
    ctx.fillStyle = COLORS[i % COLORS.length];
    ctx.globalAlpha = 0.3;
    ctx.beginPath();
    ctx.moveTo(cx, cy-s); ctx.lineTo(cx+s, cy);
    ctx.lineTo(cx, cy+s); ctx.lineTo(cx-s, cy); ctx.closePath();
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.strokeStyle = COLORS[i % COLORS.length];
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  // Agents
  for (let i = 0; i < S.starts.length; i++) {
    const [ar, ac] = S.starts[i];
    const cx = ac*cs + cs/2, cy = ar*cs + cs/2, rad = cs*0.32;
    ctx.beginPath();
    ctx.arc(cx, cy, rad, 0, Math.PI*2);
    ctx.fillStyle = COLORS[i % COLORS.length];
    ctx.fill();
    ctx.strokeStyle = "#212529";
    ctx.lineWidth = 1;
    ctx.stroke();
    if (cs >= 14) {
      ctx.fillStyle = "#fff";
      ctx.font = `600 ${Math.max(8, cs*0.3)}px Inter`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(i, cx, cy + 0.5);
    }
  }
}

// ═══════════════════════════════════════════════════════════════════
//  SIMULATION
// ═══════════════════════════════════════════════════════════════════

async function runSimulation() {
  if (!S.grid) { setStatus("Generate a map first"); return; }
  if (!S.starts.length) { setStatus("Place agents first"); return; }

  const alg = document.getElementById("sim-algorithm").value;
  const btn = document.getElementById("btn-run-sim");
  btn.disabled = true;
  btn.textContent = "Running...";
  setStatus("Running " + alg + "...");
  document.getElementById("sim-empty").style.display = "none";

  try {
    const result = await api("/api/simulate", {
      grid: S.grid,
      starts: S.starts,
      goals: S.goals,
      algorithm: alg,
    });

    S.trajectory = result.trajectory;
    S.simResult = result;
    S.step = 0;
    S.runCount++;

    // Update metrics
    const m = result.metrics;
    document.getElementById("m-isr").textContent = (m.isr * 100).toFixed(1) + "%";
    document.getElementById("m-isr").className = m.isr >= 1 ? "metric-good" : m.isr > 0.5 ? "metric-warn" : "metric-bad";
    document.getElementById("m-makespan").textContent = m.makespan;
    document.getElementById("m-soc").textContent = m.sum_of_costs;
    document.getElementById("m-runtime").textContent = m.runtime_s + "s";
    document.getElementById("m-collisions").textContent = (m.vertex_conflicts + m.edge_conflicts);
    document.getElementById("m-collisions").className = m.collision_free ? "metric-good" : "metric-bad";
    document.getElementById("m-deadlocks").textContent = m.deadlocks_detected;
    document.getElementById("m-resolved").textContent = m.deadlocks_resolved;
    document.getElementById("m-replanned").textContent = m.agents_replanned;
    document.getElementById("dash-run-count").textContent = S.runCount;

    // Add to history
    S.history.push({
      algorithm: alg,
      mapType: S.mapType,
      size: S.grid.length,
      agents: S.starts.length,
      ...m,
    });
    updateResultsTable();

    drawSim();
    setStatus(`${alg}: ISR=${(m.isr*100).toFixed(0)}%, ${m.makespan} steps, ${m.runtime_s}s`);
  } catch (e) {
    setStatus("Error: " + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "▶ Run Simulation";
  }
}

function togglePlay() {
  if (!S.trajectory) return;
  S.playing = !S.playing;
  const btn = document.getElementById("btn-play");
  if (S.playing) {
    btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>';
    const fps = parseInt(document.getElementById("speed-slider").value);
    S.timer = setInterval(tick, 1000 / fps);
  } else {
    btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>';
    clearInterval(S.timer);
  }
}

function tick() {
  if (S.step < S.trajectory.length - 1) {
    S.step++;
    drawSim();
  } else {
    if (S.playing) togglePlay();
  }
}

function simStep(dir) {
  if (!S.trajectory) return;
  S.step = Math.max(0, Math.min(S.step + dir, S.trajectory.length - 1));
  drawSim();
}

function simReset() {
  S.step = 0;
  if (S.playing) togglePlay();
  drawSim();
}

function seekTimeline(e) {
  if (!S.trajectory) return;
  const bar = document.getElementById("timeline");
  const rect = bar.getBoundingClientRect();
  const pct = (e.clientX - rect.left) / rect.width;
  S.step = Math.round(pct * (S.trajectory.length - 1));
  S.step = Math.max(0, Math.min(S.step, S.trajectory.length - 1));
  drawSim();
}

// ─── Draw Simulation Canvas ──────────────────────────────────────
function drawSim() {
  if (!S.trajectory || !S.grid) return;
  const canvas = document.getElementById("sim-canvas");
  const container = canvas.parentElement;
  const H = S.grid.length, W = S.grid[0].length;
  const cs = Math.min(
    Math.floor((container.clientWidth - 32) / W),
    Math.floor((container.clientHeight - 32) / H),
    40
  );
  canvas.width = W * cs;
  canvas.height = H * cs;
  const ctx = canvas.getContext("2d");
  const positions = S.trajectory[S.step];
  const goals = S.simResult.goals;
  const showTrails = document.getElementById("opt-trails").checked;
  const showGoals = document.getElementById("opt-goals").checked;
  const showIds = document.getElementById("opt-ids").checked;

  // Grid
  for (let r = 0; r < H; r++) {
    for (let c = 0; c < W; c++) {
      ctx.fillStyle = S.grid[r][c] === 1 ? "#495057" : "#f8f9fa";
      ctx.fillRect(c*cs, r*cs, cs, cs);
      if (S.grid[r][c] === 0) {
        ctx.strokeStyle = "#e9ecef";
        ctx.lineWidth = 0.5;
        ctx.strokeRect(c*cs, r*cs, cs, cs);
      }
    }
  }

  // Goals
  if (showGoals) {
    for (let i = 0; i < goals.length; i++) {
      const [gr, gc] = goals[i];
      const cx = gc*cs + cs/2, cy = gr*cs + cs/2, s = cs*0.25;
      ctx.fillStyle = COLORS[i % COLORS.length];
      ctx.globalAlpha = 0.25;
      ctx.beginPath();
      ctx.moveTo(cx, cy-s); ctx.lineTo(cx+s, cy);
      ctx.lineTo(cx, cy+s); ctx.lineTo(cx-s, cy); ctx.closePath();
      ctx.fill();
      ctx.globalAlpha = 0.6;
      ctx.strokeStyle = COLORS[i % COLORS.length];
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
  }

  // Trails
  if (showTrails) {
    const trailLen = 8;
    for (let i = 0; i < positions.length; i++) {
      const col = COLORS[i % COLORS.length];
      for (let t = Math.max(0, S.step - trailLen); t < S.step; t++) {
        const [pr, pc] = S.trajectory[t][i];
        const alpha = 0.06 + 0.12 * ((t - Math.max(0, S.step - trailLen)) / trailLen);
        ctx.globalAlpha = alpha;
        ctx.fillStyle = col;
        ctx.beginPath();
        ctx.arc(pc*cs+cs/2, pr*cs+cs/2, cs*0.12, 0, Math.PI*2);
        ctx.fill();
      }
    }
    ctx.globalAlpha = 1;
  }

  // Agents
  for (let i = 0; i < positions.length; i++) {
    const [ar, ac] = positions[i];
    const cx = ac*cs + cs/2, cy = ar*cs + cs/2, rad = cs*0.32;
    const col = COLORS[i % COLORS.length];
    const atGoal = ar === goals[i][0] && ac === goals[i][1];

    ctx.beginPath();
    ctx.arc(cx, cy, rad, 0, Math.PI*2);
    ctx.fillStyle = col;
    ctx.fill();
    ctx.strokeStyle = atGoal ? "#2b8a3e" : "#212529";
    ctx.lineWidth = atGoal ? 2 : 1;
    ctx.stroke();

    // Arrival check
    if (atGoal) {
      ctx.fillStyle = "#2b8a3e";
      ctx.beginPath();
      ctx.arc(cx + rad*0.6, cy - rad*0.6, rad*0.32, 0, Math.PI*2);
      ctx.fill();
      ctx.fillStyle = "#fff";
      ctx.font = `bold ${Math.max(7, cs*0.15)}px Inter`;
      ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText("✓", cx + rad*0.6, cy - rad*0.6);
    }

    // ID label
    if (showIds && cs >= 14) {
      ctx.fillStyle = "#fff";
      ctx.font = `600 ${Math.max(8, cs*0.28)}px JetBrains Mono`;
      ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText(i, cx, cy + 0.5);
    }
  }

  // Update step display
  const total = S.trajectory.length - 1;
  document.getElementById("step-display").textContent = `t=${S.step} / ${total}`;
  document.getElementById("timeline-fill").style.width =
    total > 0 ? (S.step / total * 100) + "%" : "0%";
}

// ═══════════════════════════════════════════════════════════════════
//  ANALYTICS
// ═══════════════════════════════════════════════════════════════════

function updateResultsTable() {
  const tbody = document.getElementById("results-tbody");
  tbody.innerHTML = "";
  document.getElementById("analytics-empty").style.display =
    S.history.length ? "none" : "flex";

  S.history.forEach((h, i) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${i + 1}</td>
      <td>${h.algorithm}</td>
      <td>${h.mapType}</td>
      <td>${h.size}</td>
      <td>${h.agents}</td>
      <td class="${h.isr >= 1 ? 'metric-good' : 'metric-warn'}">${(h.isr*100).toFixed(1)}%</td>
      <td>${h.makespan}</td>
      <td>${h.sum_of_costs}</td>
      <td>${h.deadlocks_detected}</td>
      <td class="${h.collision_free ? 'metric-good' : 'metric-bad'}">${h.vertex_conflicts + h.edge_conflicts}</td>
      <td>${h.runtime_s}s</td>
    `;
    tbody.appendChild(tr);
  });
}

// ═══════════════════════════════════════════════════════════════════
//  SETTINGS
// ═══════════════════════════════════════════════════════════════════

function buildSettingsForm(cfg) {
  const container = document.getElementById("settings-form");
  const skip = ["seeds", "initial_solver"];
  let html = '<div class="sidebar-section">';
  for (const [key, val] of Object.entries(cfg)) {
    if (skip.includes(key)) continue;
    const type = typeof val === "boolean" ? "checkbox"
      : typeof val === "number" ? "number" : "text";
    html += `<div class="form-row">
      <label class="form-label" style="min-width:180px">${key}</label>`;
    if (type === "checkbox") {
      html += `<input type="checkbox" ${val ? "checked" : ""} disabled>`;
    } else {
      html += `<input class="form-input" type="${type}" value="${val}" disabled>`;
    }
    html += `</div>`;
  }
  html += '<p style="margin-top:12px;font-size:11px;color:var(--text-3)">Configuration is read-only in this view. Override via the API.</p></div>';
  container.innerHTML = html;
}

// ═══════════════════════════════════════════════════════════════════
//  EXPORT
// ═══════════════════════════════════════════════════════════════════

function exportJSON() {
  if (!S.simResult) { setStatus("No results to export"); return; }
  const blob = new Blob([JSON.stringify(S.simResult, null, 2)], {type: "application/json"});
  downloadBlob(blob, "drimaps_result.json");
}

function exportCSV() {
  if (!S.history.length) { setStatus("No results to export"); return; }
  const keys = Object.keys(S.history[0]);
  let csv = keys.join(",") + "\n";
  S.history.forEach(h => {
    csv += keys.map(k => h[k]).join(",") + "\n";
  });
  downloadBlob(new Blob([csv], {type: "text/csv"}), "drimaps_results.csv");
}

function exportPNG() {
  const canvas = document.getElementById("sim-canvas");
  if (!canvas.width) { setStatus("No visualization to export"); return; }
  canvas.toBlob(blob => downloadBlob(blob, "drimaps_frame.png"), "image/png");
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = filename;
  document.body.appendChild(a); a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  setStatus("Exported " + filename);
}

// ═══════════════════════════════════════════════════════════════════
//  QUICK DEMO
// ═══════════════════════════════════════════════════════════════════

async function quickRunDemo() {
  // Set environment params
  document.getElementById("env-map-type").value = "warehouse";
  document.getElementById("env-size").value = "20";
  document.getElementById("env-density").value = "0.2";
  document.getElementById("env-seed").value = "42";
  document.getElementById("env-agents").value = "12";

  setView("environment");
  await generateMap();
  await generateAgents();
  setView("simulation");
  await runSimulation();
  togglePlay();
}

// ─── Boot ────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", init);
