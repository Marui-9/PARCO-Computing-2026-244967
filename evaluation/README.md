# Evaluation and Analysis

This directory contains all benchmark results, analysis scripts, and visualizations.

## Files

### Results Data
- `results.csv` - Speedup benchmark results (execution time, speedup metrics)
- `cache_results.csv` - Cache performance metrics (L1/LLC miss rates)

### Visualizations
- `results_analysis.png` - Speedup analysis (4 plots: speedup vs threads, heatmap, efficiency, strong scaling)
- `results_speedup.png` - Additional speedup visualizations (if generated)

### Analysis Scripts
- `plot_speedup.py` - Generate speedup visualizations from results.csv

## Usage

### Generate Speedup Plots
```bash
cd evaluation
python3 plot_speedup.py results.csv
```

### Run Benchmarks (from project root)
```bash
# Speedup benchmarks
./scripts/bench_matrices.sh

# Cache performance benchmarks
./scripts/bench_cache.sh
```

## Notes
- All benchmark scripts automatically save results to this `evaluation/` directory
- Matrix files are stored in `../matrices/`
- Source code and executables are in the project root
