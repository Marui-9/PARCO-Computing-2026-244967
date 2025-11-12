# NUMA Benchmarking Setup - Summary

## Overview

Created NUMA-aware benchmarking infrastructure for testing SpMV performance on high thread counts (24-96 threads) across multiple NUMA nodes.

## What Was Created

### New Directories
- `evaluation_numa/` - NUMA benchmark results and analysis
- `matrices_large/` - Placeholder for large test matrices

### New Code Files
- `test_configurations_numa.c` - NUMA-optimized SpMV with:
  - First-touch initialization
  - proc_bind policies (close/spread/master)
  - 11 different configurations
  - Support for 24-96 threads

### New Scripts
- `bench_configurations_numa.sh` - Benchmark runner for NUMA tests
- `run_numa_bench.pbs` - PBS job submission (96 CPUs, 12 hours)
- `evaluation_numa/analyze_configurations_numa.py` - Results analysis

### Documentation
- `evaluation_numa/README.md` - Detailed NUMA benchmarking guide

## Thread Count Strategy

| Range | Threads | Increment | Purpose |
|-------|---------|-----------|---------|
| Single node | 24 | - | Baseline |
| Cross-node | 28-48 | 4 | 1-2 NUMA nodes scaling |
| Multi-node | 54-96 | 6 | 2-4 NUMA nodes scaling |

**Full list**: 24, 28, 32, 36, 40, 44, 48, 54, 60, 66, 72, 78, 84, 90, 96

## Key Differences from Single-Node Benchmarks

| Feature | Single-Node | NUMA |
|---------|-------------|------|
| Thread counts | 1-24 | 24-96 |
| Iterations | 50 | 30 |
| Timeout | 300s | 600s |
| Matrix dir | `matrices/` | `matrices_large/` |
| Binding policies | close only | close/spread/master |
| First-touch | No | Yes |
| CPUs requested | 24 | 96 |
| Memory | 10GB | 32GB |

## Configurations Tested

1. Static+SIMD+close
2. Static+SIMD+spread ⭐ (expected best for NUMA)
3. Static+SIMD+master
4. Dynamic+SIMD+spread
5. Guided+SIMD+spread
6. Static+SIMD+Register+close
7. Static+SIMD+Register+spread
8. Static+SIMD+Affinity+close
9. Static+SIMD+Affinity+spread
10. Dynamic+SIMD+spread (chunk=16)
11. Guided+SIMD+spread (chunk=16)

## Quick Start

### 1. Add Large Matrices
Place `.mtx` files in `matrices_large/` directory

### 2. Test Locally (Optional)
```bash
gcc -O3 -Wall -Wextra -fopenmp -o test_config_numa test_configurations_numa.c generator.c m_to_csr.c -lm
./test_config_numa 48 your_matrix.mtx 10
```

### 3. Submit to Cluster
```bash
qsub run_numa_bench.pbs
```

### 4. Monitor Progress
```bash
# Check job status
qstat -u $USER

# Watch output
tail -f numa_bench.out

# Check results
tail -f evaluation_numa/configurations_numa_results.txt
```

### 5. Analyze Results
```bash
python3 evaluation_numa/analyze_configurations_numa.py
```

## Expected Analysis Outputs

1. **Speedup by binding policy** - Which policy wins?
2. **Scaling efficiency** - How well does it scale from 24→96 threads?
3. **NUMA overhead** - Is there performance cost at 24 threads?
4. **Best configurations** - Recommendations for 48 and 96 threads
5. **Efficiency metrics** - Parallel efficiency percentage

## Comparison with Existing Results

Both benchmarks test **24 threads**, allowing direct comparison:

```
# Single-node at 24 threads
evaluation/configurations_results.csv (threads=24)

# NUMA-aware at 24 threads  
evaluation_numa/configurations_numa_results.csv (threads=24)
```

**Key question**: Does NUMA-aware code have overhead at single-node scale?

## Success Criteria

✅ **Good scaling**: 96-thread speedup ≈ 4x the 24-thread speedup  
✅ **High efficiency**: >50% efficiency at 96 threads  
✅ **Policy wins**: "spread" outperforms "close" at 96 threads  
✅ **No overhead**: Similar performance at 24 threads vs single-node code  

⚠️ **Concerns to watch for**:
- Efficiency drops below 30% → Memory bandwidth bottleneck
- No improvement beyond 24 threads → NUMA not helping
- "close" beats "spread" at 96 threads → Poor NUMA awareness

## Next Steps After Results

1. **If scaling is poor**:
   - Try explicit `numactl` binding
   - Implement vector replication
   - Partition CSR matrix across nodes

2. **If scaling is good**:
   - Document best configurations
   - Compare with single-node results
   - Prepare final report

3. **Further optimizations**:
   - NUMA-aware memory allocation (`numa_alloc_onnode`)
   - Hybrid MPI+OpenMP approach
   - Custom work distribution

## File Organization

```
PARCO-Computing-2026-244967/
├── matrices/                          # Small/medium matrices (existing)
├── matrices_large/                    # Large matrices (NEW - add your files)
├── evaluation/                        # Single-node results (existing)
│   ├── configurations_results.csv
│   └── analyze_configurations.py
├── evaluation_numa/                   # NUMA results (NEW)
│   ├── configurations_numa_results.csv (generated)
│   ├── configurations_numa_results.txt (generated)
│   ├── analyze_configurations_numa.py
│   └── README.md
├── test_configurations.c              # Single-node (1-24 threads)
├── test_configurations_numa.c         # NUMA-aware (24-96 threads) (NEW)
├── bench_configurations.sh            # Single-node benchmark
├── bench_configurations_numa.sh       # NUMA benchmark (NEW)
├── run_config_bench.pbs               # Single-node PBS
└── run_numa_bench.pbs                 # NUMA PBS (NEW)
```

## PBS Job Parameters

```bash
#PBS -l select=1:ncpus=96:mem=32gb
#PBS -l walltime=12:00:00
#PBS -q short_cpuQ
```

**Notes**:
- 96 CPUs = all 4 NUMA nodes (24 cores each)
- 32GB memory for large matrices
- 12 hours should be sufficient for typical benchmark suite
- Adjust if needed based on matrix count and size

---

**Status**: Setup complete, ready to run benchmarks once large matrices are added to `matrices_large/` directory.
