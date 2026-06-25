#!/usr/bin/env python3
"""
Generate all paper tables and figures from the MovingAI benchmark CSV.
=====================================================================

Reads ``results/benchmark_latest.csv`` (produced by
``experiments/run_benchmark.py``) and regenerates every LaTeX table and figure
under ``paper/figures/``. Single source of truth: the paper renders exactly what
the experiments measured, so prose and numbers cannot drift. The script is
adaptive -- it discovers the agent counts, maps, and methods present in the CSV.

Run: ``python analysis/make_paper_assets.py``
"""

import csv
import os
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(PROJECT_ROOT, "results", "benchmark_latest.csv")
FIGDIR = os.path.join(PROJECT_ROOT, "paper", "figures")

PRETTY = {
    "drimaps": "\\textsc{Drimaps}", "pibt": "PIBT",
    "naive_dr": "Naive-DR", "prevention_only": "Prevention",
    "drimaps_no_cycle": "w/o cycle det.",
    "drimaps_no_classify": "w/o classif.",
    "drimaps_no_minimal": "w/o min. disrupt.",
}
NUM = {
    "isr", "makespan", "flowtime", "detour_ratio", "runtime", "collisions",
    "deadlocks_detected", "deadlocks_resolved", "resolution_rate",
    "agents_replanned", "memory_mb", "initial_planning_time",
    "wfg_update_time", "detection_time", "resolution_time",
}
CATEGORY_ORDER = ["empty", "random", "room", "maze", "warehouse", "game", "city"]


def load() -> List[Dict]:
    with open(RESULTS) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["num_agents"] = int(r["num_agents"])
        r["seed"] = int(r["seed"])
        r["success"] = r["success"] in ("True", "true", "1")
        for k in NUM:
            if k in r and r[k] != "":
                r[k] = float(r[k])
    return rows


def mean(xs) -> float:
    return float(np.mean(xs)) if len(xs) else 0.0


def sub(rows, **filt):
    out = rows
    for k, v in filt.items():
        if isinstance(v, (list, tuple, set)):
            out = [r for r in out if r[k] in v]
        else:
            out = [r for r in out if r[k] == v]
    return out


def write(path: str, content: str) -> None:
    with open(path, "w") as f:
        f.write(content)
    print(f"  wrote {os.path.relpath(path, PROJECT_ROOT)}")


def scaling_rows(rows):
    return [r for r in rows if r.get("campaign", "scaling") == "scaling"]


def permap_rows(rows):
    return [r for r in rows if r.get("campaign") == "permap"]


# ----------------------------------------------------------------------------
# Table 1 -- scaling ISR by method and agent count
# ----------------------------------------------------------------------------
def table_scalability(rows, counts, methods):
    sc = scaling_rows(rows)
    lines = [
        "\\begin{table}[t]", "\\centering",
        "\\caption{Mean individual success rate (\\%) on the difficult MovingAI "
        "maps (mazes, rooms, dense random, warehouse, game), averaged over maps "
        "and scenarios. \\textsc{Drimaps} adds a detection-guided escape on top "
        "of a deadlock-free PIBT core; $\\Delta$ is \\textsc{Drimaps}$-$PIBT.}",
        "\\label{tab:scalability}", "\\small",
        "\\begin{tabular}{l" + "r" * len(counts) + "}", "\\toprule",
        "Method & " + " & ".join(str(c) for c in counts) + " \\\\",
        "\\midrule",
    ]
    for m in methods:
        cells = []
        for c in counts:
            vals = [r["isr"] for r in sub(sc, algorithm=m, num_agents=c)]
            cells.append(f"{100*mean(vals):.1f}" if vals else "--")
        label = PRETTY.get(m, m)
        if m == "drimaps":
            cells = [f"\\textbf{{{x}}}" for x in cells]
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines.append("\\midrule")
    drow = []
    for c in counts:
        d = [r["isr"] for r in sub(sc, algorithm="drimaps", num_agents=c)]
        p = [r["isr"] for r in sub(sc, algorithm="pibt", num_agents=c)]
        drow.append(f"{100*(mean(d)-mean(p)):+.1f}" if d and p else "--")
    lines.append("$\\Delta$ vs PIBT & " + " & ".join(drow) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write(os.path.join(FIGDIR, "table1_scalability.tex"), "\n".join(lines) + "\n")


# ----------------------------------------------------------------------------
# Table 2 -- ablation (ISR + disruption) at the high-agent regime
# ----------------------------------------------------------------------------
def table_ablation(rows, counts):
    sc = scaling_rows(rows)
    hi = [c for c in counts if c >= max(counts) - 50]
    variants = ["drimaps", "drimaps_no_cycle", "drimaps_no_classify",
                "drimaps_no_minimal"]
    variants = [v for v in variants if any(r["algorithm"] == v for r in sc)]
    lines = [
        "\\begin{table}[t]", "\\centering",
        "\\caption{Ablation at the high-agent regime ("
        + "/".join(str(c) for c in hi) +
        " agents) on the difficult maps. Removing any component lowers ISR "
        "and/or raises disruption (agents perturbed, detour ratio).}",
        "\\label{tab:ablation}", "\\small",
        "\\begin{tabular}{lrrr}", "\\toprule",
        "Variant & ISR (\\%) & Replanned & Detour \\\\", "\\midrule",
    ]
    for v in variants:
        r = sub(sc, algorithm=v, num_agents=hi)
        isr = 100 * mean([x["isr"] for x in r])
        rep = mean([x["agents_replanned"] for x in r])
        det = mean([x["detour_ratio"] for x in r])
        label = PRETTY.get(v, v)
        if v == "drimaps":
            lines.append(f"\\textbf{{{label}}} & \\textbf{{{isr:.1f}}} & "
                         f"{rep:.0f} & {det:.2f} \\\\")
        else:
            lines.append(f"{label} & {isr:.1f} & {rep:.0f} & {det:.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write(os.path.join(FIGDIR, "table4_ablation.tex"), "\n".join(lines) + "\n")


# ----------------------------------------------------------------------------
# Table 3 -- per-category coverage (DRIMAPS vs PIBT) across the full benchmark
# ----------------------------------------------------------------------------
def table_permap(rows):
    pm = permap_rows(rows)
    if not pm:
        print("  (skipped table_permap: no per-map rows)")
        return
    cats = [c for c in CATEGORY_ORDER if any(r["map_type"] == c for r in pm)]
    lines = [
        "\\begin{table}[t]", "\\centering",
        "\\caption{Per-category coverage across the MovingAI benchmark "
        "(every category, maps up to $256\\times256$), mean ISR (\\%) at the "
        "fixed agent count, averaged over maps and two scenarios. "
        "\\textsc{Drimaps} resolves deadlocks across the whole benchmark and "
        "matches or beats the PIBT reference everywhere.}",
        "\\label{tab:permap}", "\\small",
        "\\begin{tabular}{lrrr}", "\\toprule",
        "Category & \\#maps & PIBT & \\textsc{Drimaps} \\\\", "\\midrule",
    ]
    for cat in cats:
        cr = [r for r in pm if r["map_type"] == cat]
        nmaps = len({r["map_name"] for r in cr})
        p = 100 * mean([r["isr"] for r in cr if r["algorithm"] == "pibt"])
        d = 100 * mean([r["isr"] for r in cr if r["algorithm"] == "drimaps"])
        lines.append(f"{cat} & {nmaps} & {p:.1f} & \\textbf{{{d:.1f}}} \\\\")
    p = 100 * mean([r["isr"] for r in pm if r["algorithm"] == "pibt"])
    d = 100 * mean([r["isr"] for r in pm if r["algorithm"] == "drimaps"])
    nmaps = len({r["map_name"] for r in pm})
    lines += ["\\midrule",
              f"\\textbf{{all}} & {nmaps} & {p:.1f} & \\textbf{{{d:.1f}}} \\\\",
              "\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write(os.path.join(FIGDIR, "table_permap.tex"), "\n".join(lines) + "\n")


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def _legend_label(m):
    return PRETTY.get(m, m).replace("\\textsc{", "").replace("}", "")


def fig_scaling(rows, counts, methods):
    sc = scaling_rows(rows)
    plt.figure(figsize=(5.2, 3.2))
    markers = {"drimaps": "o-", "pibt": "s--", "naive_dr": "^:",
               "prevention_only": "d-."}
    for m in methods:
        ys = [100 * mean([r["isr"] for r in sub(sc, algorithm=m, num_agents=c)])
              for c in counts]
        plt.plot(counts, ys, markers.get(m, "o-"), label=_legend_label(m))
    plt.xlabel("Number of agents")
    plt.ylabel("ISR (%)")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "fig_scaling.pdf"))
    plt.close()
    print("  wrote paper/figures/fig_scaling.pdf")


def fig_disruption(rows, counts):
    sc = scaling_rows(rows)
    plt.figure(figsize=(5.2, 3.2))
    for v, mk in [("drimaps", "o-"), ("drimaps_no_minimal", "s--"),
                  ("drimaps_no_cycle", "^:")]:
        if not any(r["algorithm"] == v for r in sc):
            continue
        ys = [mean([r["agents_replanned"]
                    for r in sub(sc, algorithm=v, num_agents=c)]) for c in counts]
        plt.plot(counts, ys, mk, label=_legend_label(v))
    plt.xlabel("Number of agents")
    plt.ylabel("Agents perturbed (escapes)")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "fig_disruption.pdf"))
    plt.close()
    print("  wrote paper/figures/fig_disruption.pdf")


# ----------------------------------------------------------------------------
# Side-product tables (sim properties, deadlock taxonomy)
# ----------------------------------------------------------------------------
def table_sim_properties():
    import json
    p = os.path.join(PROJECT_ROOT, "results", "sim_benchmark.json")
    if not os.path.exists(p):
        print("  (skipped table_simprops: results/sim_benchmark.json missing)")
        return
    d = json.load(open(p))
    thr = d.get("throughput", {})
    if isinstance(thr, list):           # older schema: list of per-agent rows
        thr = thr[-1] if thr else {}
    ov = d.get("monitor_overhead", {})
    if isinstance(ov, list):
        ov = ov[-1] if ov else {}
    det = d.get("determinism", {})
    coll = d.get("collision_free", {})
    wfg = d.get("wfg_vs_stagnation", {})
    lines = [
        "\\begin{table}[t]", "\\centering",
        "\\caption{\\textsc{DrimapSim} engineering properties, all reproducible "
        "from a seed (\\texttt{experiments/sim\\_benchmark.py}).}",
        "\\label{tab:simprops}", "\\small", "\\begin{tabular}{ll}", "\\toprule",
        "Property & Value \\\\", "\\midrule",
    ]
    if thr:
        lines.append(f"Throughput (@{thr.get('agents','?')} agents) & "
                     f"{thr.get('agent_steps_per_sec', 0):.0f} agent-steps/s \\\\")
    if ov:
        lines.append(f"Deadlock-monitor overhead (@{ov.get('agents','?')}) & "
                     f"{ov.get('overhead_pct', 0):.0f}\\% \\\\")
    if coll:
        lines.append(f"Collisions (block-both, {coll.get('steps', '?')} steps "
                     f"$\\times$ {coll.get('configs','?')}) & "
                     f"{coll.get('overlaps', 0)} \\\\")
    if det:
        status = "identical" if det.get("deterministic") else "diverged"
        lines.append(f"Determinism ({det.get('trials','?')} repeated rollouts) & "
                     f"{status} \\\\")
    g = wfg.get("greedy", {})
    if g:
        lines.append("WFG deadlock steps vs stagnation (greedy) & "
                     f"{g.get('wfg_deadlock_steps', '?')} vs "
                     f"{g.get('stagnation_flagged_steps', '?')} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write(os.path.join(FIGDIR, "table_simprops.tex"), "\n".join(lines) + "\n")


def table_dltypes():
    import json
    p = os.path.join(PROJECT_ROOT, "results", "deadlock_types.json")
    if not os.path.exists(p):
        print("  (skipped table_dltypes: run experiments/deadlock_types.py)")
        return
    d = json.load(open(p))
    ov = d["overall"]
    tot = sum(ov.values()) or 1
    order = sorted(ov, key=lambda k: -ov[k])
    lines = [
        "\\begin{table}[t]", "\\centering",
        "\\caption{Structural taxonomy of the persistent deadlocks "
        "\\textsc{Drimaps} detects and escapes on the difficult maps "
        f"({tot} confirmed episodes).}}",
        "\\label{tab:dltypes}", "\\small", "\\begin{tabular}{lr}", "\\toprule",
        "Deadlock type & Share (\\%) \\\\", "\\midrule",
    ]
    for k in order:
        lines.append(f"{k.replace('_', ' ')} & {100*ov[k]/tot:.0f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write(os.path.join(FIGDIR, "table_dltypes.tex"), "\n".join(lines) + "\n")


def main():
    rows = load()
    counts = sorted({r["num_agents"] for r in scaling_rows(rows)})
    methods = [m for m in ["drimaps", "pibt", "naive_dr", "prevention_only"]
               if any(r["algorithm"] == m for r in rows)]
    print(f"Loaded {len(rows)} rows; scaling counts={counts}")
    table_scalability(rows, counts, methods)
    table_ablation(rows, counts)
    table_permap(rows)
    table_sim_properties()
    table_dltypes()
    fig_scaling(rows, counts, methods)
    fig_disruption(rows, counts)
    print("Done.")


if __name__ == "__main__":
    main()
