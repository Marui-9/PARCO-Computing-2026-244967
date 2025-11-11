# PARCO-Computing-2026-244967

Parallel sparse matrix-vector multiplication benchmark using OpenMP.

## Project Structure

```
PARCO-Computing-2026-244967/
├── evaluation/                   # Benchmark results and analysis
│   ├── results.csv              # Speedup benchmark data
│   ├── cache_results.csv        # Cache performance data
│   ├── configurations_results.csv # OpenMP config comparison data
│   ├── results_analysis.png     # Visualization plots
│   ├── plot_speedup.py          # Python analysis script
│   └── README.md                # Evaluation documentation
├── matrices/                    # Sparse matrix test files (.mtx format)
│   └── *.mtx                   # Renamed as: {size}_{density}.mtx
├── bench_matrices.sh            # Run speedup benchmarks
├── bench_cache.sh               # Run cache performance benchmarks
├── bench_configurations.sh      # Compare OpenMP configs (23 variants)
├── plot_results.sh              # Generate visualization plots
├── analyze_configurations.py    # Analyze configuration benchmark results
├── test_config                  # OpenMP configuration comparison tool
├── executable                   # Compiled program
├── Makefile                     # Build configuration
└── *.c, *.h                    # Source files
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
./bench_matrices.sh

# Cache performance analysis
./bench_cache.sh

# Configuration comparison across all matrices and thread counts
./bench_configurations.sh

# Generate plots from results
./plot_results.sh
```

### Analyze Configuration Results
```bash
# After running bench_configurations.sh
python3 analyze_configurations.py
```

This analyzes:
- SIMD impact (typically **188x improvement**)
- Thread affinity benefits (~5% gain)
- Scaling efficiency across thread counts
- Optimal configurations per matrix density/size
- Best performance: **Guided+SIMD+chunk=32** (up to **3,300x speedup**)
### Generate Speedup Plots
```bash
cd evaluation
python3 plot_speedup.py results.csv
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

See `evaluation/README.md` for detailed analysis.
