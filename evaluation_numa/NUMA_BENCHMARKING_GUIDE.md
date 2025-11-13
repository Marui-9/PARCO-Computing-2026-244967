# NUMA Benchmarking Quick Reference

## Overview

The NUMA benchmarking system tests OpenMP configurations on multi-socket systems with high thread counts (24-96 threads). It uses **direct CSR import** to handle large sparse matrices efficiently.

## Key Files

- **Executable**: `test_config_numa`
- **Benchmark Script**: `scripts/bench_configurations_numa.sh`
- **PBS Job**: `pbs_jobs/run_numa_bench.pbs`
- **Analysis Script**: `evaluation_numa/analyze_configurations_numa.py`
- **Matrix Directory**: `matrices_large/` (falls back to `matrices/`)

## Quick Start

### 1. Setup and Compilation

```bash
# From project root directory
cd /path/to/PARCO-Computing-2026-244967

# Run setup check
bash scripts/setup_numa_benchmarks.sh

# Compile manually (if needed)
gcc -O3 -Wall -Wextra -march=native -fopenmp \
    -o test_config_numa \
    src/test_configurations_numa.c \
    src/generator.c \
    src/m_to_csr.c \
    -lm
```

### 2. Add Large Matrices

```bash
# Copy matrices to matrices_large/
cp /path/to/large_matrix.mtx matrices_large/

# Or create symlinks
ln -s /path/to/matrix/directory/*.mtx matrices_large/
```

**Recommended matrix properties**:
- Size: 10k - 500k rows/cols
- Density: 0.01% - 5%  
- Format: Matrix Market (.mtx)

### 3. Run Benchmarks

#### Local Testing (single matrix, few threads)
```bash
# Test with one matrix and limited iterations
./test_config_numa 24 citationCiteseer.mtx 5
```

#### Full Benchmark (all matrices, all thread counts)
```bash
# Run locally
bash scripts/bench_configurations_numa.sh
```

#### PBS Job Submission (recommended for cluster)
```bash
# Submit to job queue
qsub pbs_jobs/run_numa_bench.pbs

# Monitor progress
tail -f numa_bench.out

# Check queue
qstat -u $USER
```

### 4. Analyze Results

```bash
# Generate analysis and visualizations
cd evaluation_numa
python3 analyze_configurations_numa.py

# View plots
ls -lh figures_numa/*.png
```

## Configuration Testing

The benchmark tests these configurations:

### Binding Policies
- **close**: Threads placed near each other (good for single NUMA node)
- **spread**: Threads distributed across NUMA nodes (maximizes bandwidth)
- **master**: Threads kept near master thread

### Optimizations
- **Static+SIMD**: Static scheduling with SIMD vectorization
- **Dynamic+SIMD**: Dynamic load balancing
- **Guided+SIMD**: Guided scheduling
- **Register+SIMD**: Register blocking optimization
- **Affinity+SIMD**: Enhanced cache affinity

### Thread Counts
24, 28, 32, 36, 40, 44, 48, 54, 60, 66, 72, 78, 84, 90, 96

## Output Files

After benchmarking:
```
evaluation_numa/
├── configurations_numa_results.csv      # Raw benchmark data
├── configurations_numa_results.txt      # Detailed log
└── figures_numa/                        # Generated visualizations
    ├── speedup_vs_threads_binding.png
    ├── efficiency_vs_threads_binding.png
    ├── speedup_heatmap_binding_threads.png
    ├── scaling_efficiency_doubling.png
    ├── top_configs_96_threads.png
    ├── speedup_distribution_binding.png
    └── speedup_vs_threads_matrices.png
```

## Troubleshooting

### Compilation Errors
```bash
# Check GCC version (needs 4.9+)
gcc --version

# On cluster, load correct module
module load gcc91
```

### Memory Issues
```bash
# For very large matrices, monitor memory:
free -h

# If OOM, reduce thread count or use sparser matrices
./test_config_numa 24 matrix.mtx 10  # Fewer iterations
```

### No Matrices Found
```bash
# Verify directory structure
ls -lh matrices_large/*.mtx

# Check script is looking in right place
head -30 scripts/bench_configurations_numa.sh | grep MATRICES_DIR
```

### Timeout Issues
```bash
# Edit timeout in bench script (line 10)
# TIMEOUT_SECS=600  # Increase for very large matrices
nano scripts/bench_configurations_numa.sh
```

## Performance Expectations

Based on previous benchmarks:

| Matrix Density | Expected Speedup (96 threads) | Efficiency |
|----------------|------------------------------|------------|
| < 0.5%         | 1,000 - 5,000×               | 10-50%     |
| 0.5% - 1.5%    | 200 - 1,000×                 | 2-10%      |
| > 3%           | 50 - 200×                    | 0.5-2%     |

**Note**: Efficiency drops at high thread counts due to memory bandwidth saturation.

## Best Practices

1. **Start small**: Test with one matrix and 24 threads first
2. **Monitor resources**: Use `htop` or `top` to watch CPU/memory
3. **Use PBS**: For full benchmarks, submit to job queue
4. **Backup results**: Copy CSV files before re-running benchmarks
5. **Document matrices**: Keep notes on matrix sources and properties

## References

- OpenMP Binding Policies: https://www.openmp.org/spec-html/5.0/openmpsu60.html
- Matrix Market Format: https://math.nist.gov/MatrixMarket/formats.html
- NUMA Architecture: `man numactl`, `numactl --hardware`
