#!/bin/bash
# Helper script to generate plots from evaluation data
# Usage: ./plot_results.sh

cd evaluation
python3 plot_speedup.py results.csv
cd ..

echo ""
echo "Plot generated in plots/results_analysis.png"
