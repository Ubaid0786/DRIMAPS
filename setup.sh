#!/bin/bash
# DRIMAPS Setup Script
# One-click setup for the DRIMAPS research project.

set -e

echo "=============================================="
echo "  DRIMAPS Setup"
echo "=============================================="

# 1. Install Python dependencies
echo "[1/4] Installing Python dependencies..."
pip install -r requirements.txt 2>/dev/null || pip3 install -r requirements.txt

# 2. Create output directories
echo "[2/4] Creating output directories..."
mkdir -p results
mkdir -p paper/figures
mkdir -p tmp
mkdir -p instances

# 3. Run quick sanity test
echo "[3/4] Running sanity test..."
cd "$(dirname "$0")"
python3 -c "
import sys
sys.path.insert(0, '.')

# Test 1: DRIMAPS solver
from src.config import DRIMAPSConfig
from src.drimaps import DRIMAPS
import numpy as np

grid = np.zeros((8, 8), dtype=int)
solver = DRIMAPS(DRIMAPSConfig(timeout=5.0))
result = solver.solve(grid, [(0,0)], [(7,7)])
assert result.success, 'Sanity test 1 failed (solver)!'
print('  ✓ DRIMAPS solver works')

# Test 2: DRIMAPSim environment
from sim import drimapsim_v0, EnvConfig
env = drimapsim_v0(EnvConfig(size=8, num_agents=4, seed=42, max_episode_steps=16))
obs, info = env.reset()
obs, r, t, tr, i = env.step(env.sample_actions())
assert len(obs) == 4, 'Sanity test 2 failed (environment)!'
print('  ✓ DRIMAPSim environment works')

# Test 3: Map generators
from sim.map_registry import generate_map
for mt in ['random', 'warehouse', 'corridor', 'bottleneck', 'maze', 'room', 'open']:
    g = generate_map(mt, 16, 0.2, 42)
    assert g.shape == (16, 16), f'Map {mt} failed!'
print('  ✓ All 7 map generators work')
"

# 4. Run test suite
echo "[4/4] Running test suite..."
python3 -m pytest tests/ -q --tb=short 2>&1 | tail -3

echo ""
echo "=============================================="
echo "  Setup complete!"
echo ""
echo "  Usage:"
echo "    # Run experiments"
echo "    python experiments/run_all_experiments.py --quick"
echo ""
echo "    # Use DRIMAPSim environment"
echo "    python -c \"from sim import drimapsim_v0; env = drimapsim_v0(); env.reset()\""
echo ""
echo "    # Generate figures and tables"
echo "    python analysis/plot_results.py"
echo "    python analysis/generate_tables.py"
echo "=============================================="
