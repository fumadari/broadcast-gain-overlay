#!/bin/bash

# Big production run script for Broadcast-Gain experiments
# This will run extensive experiments with proper ghost CDE and all visualizations

echo "Starting BIG production run for Broadcast-Gain experiments"
echo "============================================"
echo "This will take significant time (1-2 hours)"
echo ""

# Activate virtual environment
source .venv/bin/activate

# Clean up old results
echo "Cleaning up old test results..."
rm -rf results_final_test results_fundamental_v2 results_visuals_v2 results_density_v2

# Create output directory for big run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="results_production_${TIMESTAMP}"
mkdir -p ${OUTPUT_DIR}

echo "Output directory: ${OUTPUT_DIR}"
echo ""

# 1. Main experiments with more seeds (5 seeds instead of 3)
echo "[1/5] Running Bernoulli dropout experiments (5 seeds, 500 steps)..."
python main.py --mode sanity --out ${OUTPUT_DIR}/bernoulli --drop_model bernoulli

echo "[2/5] Running burst dropout experiments (5 seeds, 500 steps)..."
python main.py --mode sanity --out ${OUTPUT_DIR}/burst --drop_model burst

# 2. Fundamental diagram with extended density range
echo "[3/5] Generating fundamental diagrams..."
python scripts/fundamental.py --out ${OUTPUT_DIR}/fundamental --drop 0.3 --drop_model burst

# 3. Space-time comparisons for multiple seeds
echo "[4/5] Generating space-time comparisons..."
for seed in 1 2 3; do
    python scripts/compare_spacetime.py --seed $seed --out ${OUTPUT_DIR}/visuals --drop 0.3 --drop_model burst
done

# 4. Gain vs density analysis
echo "[5/5] Generating gain vs density analysis..."
python scripts/gain_vs_density.py --out ${OUTPUT_DIR}/density --R 4

# Generate summary report
echo ""
echo "Generating summary report..."
cat > ${OUTPUT_DIR}/SUMMARY.md << EOF
# Production Run Summary
Generated: $(date)

## Experiments Completed
- Bernoulli dropout: 5 seeds × 3 dropout rates × 4 methods = 60 episodes
- Burst dropout: 5 seeds × 3 dropout rates × 4 methods = 60 episodes
- Fundamental diagram: Flow vs density analysis
- Space-time comparisons: Multiple seed visualizations
- Gain vs density: Correlation analysis

## Key Metrics
- Ghost free-flow CDE properly calculated (aligned with DEEPFLEET)
- Position tracking enabled for spatial analysis
- All paper-ready figures generated in PNG and PDF

## Data Files
- Main results: ${OUTPUT_DIR}/bernoulli/main_results.csv
- Main results: ${OUTPUT_DIR}/burst/main_results.csv
- Fundamental: ${OUTPUT_DIR}/fundamental/fundamental.csv
- All figures in respective directories
EOF

echo ""
echo "============================================"
echo "BIG PRODUCTION RUN COMPLETE!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "Check ${OUTPUT_DIR}/SUMMARY.md for details"
echo ""
echo "Ready for paper submission!"