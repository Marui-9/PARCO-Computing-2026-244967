# PARCO-Computing-2026-244967

Parallel sparse matrix-vector multiplication benchmark using OpenMP.

## Project Structure

```
PARCO-Computing-2026-244967/
├── evaluation/              # Benchmark results and analysis
│   ├── results.csv         # Speedup benchmark data
│   ├── cache_results.csv   # Cache performance data
│   ├── results_analysis.png # Visualization plots
│   ├── plot_speedup.py     # Python analysis script
│   └── README.md           # Evaluation documentation
├── matrices/               # Sparse matrix test files (.mtx format)
│   └── *.mtx              # Renamed as: {size}_{density}.mtx
├── bench_matrices.sh       # Run speedup benchmarks
├── bench_cache.sh          # Run cache performance benchmarks
├── plot_results.sh         # Generate visualization plots
├── executable              # Compiled program
├── Makefile               # Build configuration
└── *.c, *.h               # Source files
```

## Quick Start

### Build
```bash
make
```

### Run Single Test
```bash
./executable <threads> <matrix_file>
# Example:
./executable 8 2k_0p15.mtx
```

### Run Full Benchmarks
```bash
# Speedup benchmarks (all matrices, multiple thread counts)
./bench_matrices.sh

# Cache performance analysis
./bench_cache.sh

# Generate plots from results
./plot_results.sh
```

## Matrix Naming Convention

Matrix files use descriptive names: `{size}_{density}.mtx`
- Size: Rounded dimensions in thousands (k)
- Density: Percentage with 'p' as decimal point

Examples:
- `2k_0p15.mtx` → 2k×2k matrix, 0.15% density
- `6k_3p16.mtx` → 6k×6k matrix, 3.16% density
- `5k_9p37.mtx` → 5k×6k matrix, 9.37% density

## Results

All benchmark results, visualizations, and analysis scripts are in the `evaluation/` directory.
See `evaluation/README.md` for details.
