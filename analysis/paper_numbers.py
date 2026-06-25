#!/usr/bin/env python3
"""Print the headline numbers the paper prose cites, so the \\newcommand macros
in paper/main.tex can be synced to the benchmark CSV after a re-run.

Run: python analysis/paper_numbers.py
"""
import csv
import os
import numpy as np

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
NUM = ("isr", "agents_replanned", "detour_ratio", "deadlocks_detected",
       "deadlocks_resolved", "collisions")


def load(name):
    rows = list(csv.DictReader(open(os.path.join(R, name))))
    for r in rows:
        r["num_agents"] = int(r["num_agents"])
        for k in NUM:
            if k in r and r[k] != "":
                r[k] = float(r[k])
    return rows


def mean(xs):
    return float(np.mean(xs)) if len(xs) else 0.0


rows = load("benchmark_latest.csv")
sc = [r for r in rows if r.get("campaign", "scaling") == "scaling"]
pm = [r for r in rows if r.get("campaign") == "permap"]
counts = sorted({r["num_agents"] for r in sc})
hi = counts[-1]


def isr(rs, **f):
    out = rs
    for k, v in f.items():
        out = [r for r in out if r[k] == v]
    return 100 * mean([r["isr"] for r in out])


print(f"# rows={len(rows)} scaling={len(sc)} permap={len(pm)} counts={counts}")
print(f"\\nRuns ......... {len(rows)}")
print(f"collisions (all) ... {sum(int(r['collisions']) for r in rows)}")
print("\n--- scaling ISR by method x count ---")
for m in ["drimaps", "pibt", "naive_dr", "prevention_only"]:
    print(f"  {m:16s}", [round(isr(sc, algorithm=m, num_agents=c), 1) for c in counts])

print(f"\ndrimapsHi (@{hi}) = {isr(sc, algorithm='drimaps', num_agents=hi):.1f}")
print(f"pibtHi    (@{hi}) = {isr(sc, algorithm='pibt', num_agents=hi):.1f}")
print(f"naiveHi   (@{hi}) = {isr(sc, algorithm='naive_dr', num_agents=hi):.1f}")

# per-map coverage
print(f"\ndrimapsPermap = {isr(pm, algorithm='drimaps'):.1f}")
print(f"pibtPermap    = {isr(pm, algorithm='pibt'):.1f}")

# biggest per-condition DRIMAPS-PIBT gap on scaling (map,count,scenario)
best = (0, None)
for mn in {r["map_name"] for r in sc}:
    for c in counts:
        for scen in {r["scenario"] for r in sc}:
            d = [r["isr"] for r in sc if r["algorithm"] == "drimaps" and
                 r["map_name"] == mn and r["num_agents"] == c and r["scenario"] == scen]
            p = [r["isr"] for r in sc if r["algorithm"] == "pibt" and
                 r["map_name"] == mn and r["num_agents"] == c and r["scenario"] == scen]
            if d and p:
                gap = 100 * (mean(d) - mean(p))
                if gap > best[0]:
                    best = (gap, (mn, c, scen))
print(f"\ndeltaWarehouse (max per-condition D-P gap) = {best[0]:.0f}  at {best[1]}")

# ablation at high regime
hicnts = [c for c in counts if c >= hi - 50]
abl = [r for r in sc if r["num_agents"] in hicnts]
print(f"\n--- ablation @ {hicnts} ---")
for v in ["drimaps", "drimaps_no_cycle", "drimaps_no_classify", "drimaps_no_minimal"]:
    rs = [r for r in abl if r["algorithm"] == v]
    print(f"  {v:22s} ISR={100*mean([r['isr'] for r in rs]):.1f} "
          f"replan={mean([r['agents_replanned'] for r in rs]):.0f} "
          f"detour={mean([r['detour_ratio'] for r in rs]):.2f}")
print(f"\nablFull  = {100*mean([r['isr'] for r in abl if r['algorithm']=='drimaps']):.1f}")
print(f"ablNoMin = {100*mean([r['isr'] for r in abl if r['algorithm']=='drimaps_no_minimal']):.1f}")
