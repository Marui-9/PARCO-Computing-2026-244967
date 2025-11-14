# PARCO-Computing-2026-244967

Parallel sparse matrix-vector multiplication benchmark using OpenMP.

## Project Structure

```
PARCO-Computing-2026-244967/
├── src/                         # C source code
│   ├── main.c                  # Main program
│   ├── test_configurations.c   # Configuration testing tool
│   ├── test_configurations_numa.c # NUMA-optimized configurations
│   ├── generator.c, m_to_csr.c # Matrix utilities
│   └── *.h                     # Header files
├── matrices/                    # Sparse matrix test files (.mtx format)
│   └── *.mtx                   # Named as: {size}_{density}.mtx
├── scripts/                     # Benchmark and analysis scripts
│   ├── bench_matrices.sh       # Run speedup benchmarks
│   ├── bench_cache.sh          # Run cache performance benchmarks
│   ├── bench_cache_valgrind.sh # Cache analysis with valgrind
│   ├── bench_configurations.sh # Compare OpenMP configs (23 variants)
│   ├── bench_configurations_numa.sh # NUMA benchmarks (24-96 threads)
│   ├── analyze_configurations.py # Analyze configuration results
│   ├── analyze_configurations_numa.py # Analyze NUMA results
│   ├── plot_configurations.py  # Generate configuration plots
│   ├── plot_speedup.py         # Generate speedup plots
│   └── plot_cache.py           # Generate cache performance plots
├── pbs_jobs/                    # PBS cluster job scripts
│   ├── run_numa_bench.pbs      # NUMA benchmark job
│   ├── run_cache_valgrind.pbs  # Cache valgrind job
│   └── *.pbs                   # Additional job files
├── results/                     # Benchmark output data (CSV/TXT)
│   ├── configurations_results.csv
│   ├── configurations_numa_results.csv
│   ├── cache_results.csv
│   └── *.txt                   # Benchmark logs
├── plots/                       # Generated figures (PNG)
│   └── *.png                   # Visualization plots
├── executable                   # Main program (compiled)
├── test_config                  # Configuration testing tool (compiled)
├── test_config_numa             # NUMA testing tool (compiled)
└── Makefile                     # Build configuration
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

### Compare OpenMP Configurations
```bash
./test_config <threads> <matrix_file> <iterations>
# Example:
./test_config 8 6k_6p3.mtx 60
```

This tests **23 different OpenMP configurations**:
- **Static scheduling**: default, chunk=8/32
- **Dynamic scheduling**: chunk=8/16/32/64
- **Guided scheduling**: chunk=8/16/32
- **Auto scheduling**: runtime-determined
- **SIMD optimizations**: Static/Dynamic/Guided + SIMD (various chunks)
- **Thread affinity**: SIMD + proc_bind(close) for cache locality

Output shows speedup, efficiency, standard deviation, and improvement percentage for each configuration.

### Run Full Benchmarks
```bash
# Speedup benchmarks (all matrices, multiple thread counts)
./scripts/bench_matrices.sh

# Cache performance analysis
./scripts/bench_cache.sh

# Configuration comparison across all matrices and thread counts
./scripts/bench_configurations.sh

# Generate plots from results
./scripts/plot_results.sh
```

### Analyze Configuration Results
```bash
# After running bench_configurations.sh
python3 scripts/analyze_configurations.py

# For NUMA benchmarks
python3 scripts/analyze_configurations_numa.py
```

This analyzes:
- SIMD impact (typically **188x improvement**)
- Thread affinity benefits (~5% gain)
- Scaling efficiency across thread counts
- Optimal configurations per matrix density/size
- Best performance: **Guided+SIMD+chunk=32** (up to **3,300x speedup**)

### Generate Plots
```bash
# Speedup analysis
python3 scripts/plot_speedup.py

# Configuration comparison
python3 scripts/plot_configurations.py

# Cache performance
python3 scripts/plot_cache.py
```

All results are saved to `results/` and plots to `plots/`.




## Matrix Naming Convention

Matrix files use descriptive names: `{size}_{density}.mtx`
- Size: Rounded dimensions in thousands (k)
- Density: Percentage with 'p' as decimal point

Examples:
- `2k_0p15.mtx` → 2k×2k matrix, 0.15% density
- `6k_3p16.mtx` → 6k×6k matrix, 3.16% density
- `5k_9p37.mtx` → 5k×6k matrix, 9.37% density

## Results

All benchmark results are in `results/` (CSV files) and visualizations in `plots/` (PNG files).

### Key Performance Findings

From comprehensive configuration benchmarking (`test_config`):

1. **SIMD Vectorization**: 188x improvement over non-SIMD (single most important optimization)
2. **Thread Affinity**: +5% improvement when using `proc_bind(close)`
3. **Sparse Matrix Advantage**: Sparser matrices achieve much better speedups
   - <0.5% density: ~1,300x average speedup
   - ≥1.0% density: ~190x average speedup
4. **Best Configuration**: Guided scheduling + SIMD + chunk=32
   - Up to **3,300x speedup** on 18 threads
   - Consistently top performer across matrix sizes
5. **Optimal Thread Scaling**:
   - Small matrices (2k): Peak at 4 threads
   - Medium/Large sparse (10-15k, <0.5% density): Scale well to 18 threads
   - Dense matrices: Limited scaling benefit

### NUMA Optimization (24-96 threads)

NUMA-aware benchmarks test 9 configurations optimized for multi-socket systems:
- Thread affinity policies: close, spread, master
- SIMD + register blocking optimizations
- Scaling from 24 to 96 threads across 4 NUMA nodes

Run with: `qsub pbs_jobs/run_numa_bench.pbs`

