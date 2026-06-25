#!/usr/bin/env bash
#
# reproduce.sh — End-to-end reproduction for DRIMAPS
# ==================================================
#
# Reproduces the full DRIMAPS pipeline from a clean checkout:
#   1. Install Python dependencies
#   2. Run the test suite
#   3. Run the MovingAI benchmark (scaling + per-map coverage campaigns)
#   4. Tally the deadlock taxonomy and benchmark the environment
#   5. Regenerate every paper table and figure from the benchmark CSV
#
# All quantitative results in the paper trace back to
# results/benchmark_latest.csv produced by step 3. The MovingAI maps and
# scenarios are vendored under benchmarks/.
#
# Usage:
#   ./reproduce.sh
#
set -euo pipefail

# Run from the repository root regardless of where the script is invoked.
cd "$(dirname "$0")"

echo "==> [1/5] Installing dependencies (pip install -r requirements.txt)"
pip install -r requirements.txt

echo "==> [2/5] Running test suite (python -m pytest tests/ -q)"
python -m pytest tests/ -q

echo "==> [3/5] Running the MovingAI benchmark (python experiments/run_benchmark.py)"
echo "         Scaling sweep + per-map coverage over the vendored benchmark."
python experiments/run_benchmark.py

echo "==> [4/5] Deadlock taxonomy + environment benchmark"
python experiments/deadlock_types.py
python experiments/sim_benchmark.py

echo "==> [5/5] Regenerating paper assets (python analysis/make_paper_assets.py)"
python analysis/make_paper_assets.py
echo "         Headline numbers for the paper macros:"
python analysis/paper_numbers.py

echo "==> Done. See results/ and paper/figures/."
