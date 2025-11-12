# NUMA-Aware Benchmarking

This directory contains NUMA-optimized code and benchmarks for high thread counts (24-96 threads) across multiple NUMA nodes.

## Directory Structure

```
evaluation_numa/
├── configurations_numa_results.csv    # Benchmark results (generated)
├── configurations_numa_results.txt    # Benchmark log (generated)
└── analyze_configurations_numa.py     # Analysis script
```

## Files

### Code Files (in parent directory)

- **test_configurations_numa.c** - NUMA-aware implementation with:
  - First-touch initialization for NUMA locality
  - Multiple proc_bind policies: close, spread, master
  - Optimized for 24-96 threads
  - Support for large matrices

- **bench_configurations_numa.sh** - Benchmark script:
  - Thread counts: 24, 28, 32, 36, 40, 44, 48, 54, 60, 66, 72, 78, 84, 90, 96
  - Uses `matrices_large/` directory
  - 30 iterations per configuration (reduced from 50 for large matrices)
  - 600-second timeout per run

- **run_numa_bench.pbs** - PBS job submission script:
  - Requests 96 CPUs (all 4 NUMA nodes)
  - 32GB memory
  - 12-hour walltime
  - Compiles and runs benchmarks

## Thread Count Progression

- **24 threads**: Single NUMA node baseline
- **28-48 threads** (increment 4): Cross-node scaling (1-2 NUMA nodes)
- **54-96 threads** (increment 6): Multi-node scaling (2-4 NUMA nodes)

## Configurations Tested

1. **Static+SIMD+close** - Baseline (threads stay close together)
2. **Static+SIMD+spread** - NUMA-aware spreading across nodes
3. **Static+SIMD+master** - Keep threads close to master thread
4. **Dynamic+SIMD+spread** - Dynamic load balancing with NUMA spread
5. **Guided+SIMD+spread** - Guided scheduling with NUMA awareness
6. **Static+SIMD+Register+close** - Register blocking + close binding
7. **Static+SIMD+Register+spread** - Register blocking + NUMA spread
8. **Static+SIMD+Affinity+close** - Full affinity optimization + close
9. **Static+SIMD+Affinity+spread** - Full affinity optimization + spread

## Usage

### 1. Compile NUMA-aware executable

```bash
gcc -O3 -Wall -Wextra -fopenmp -o test_config_numa src/test_configurations_numa.c src/generator.c src/m_to_csr.c -lm
```

### 2. Run single test

```bash
./test_config_numa 48 large_matrix.mtx 30
```

### 3. Run full benchmark suite locally

```bash
chmod +x scripts/bench_configurations_numa.sh
./scripts/bench_configurations_numa.sh
```

### 4. Submit to cluster

```bash
qsub pbs_jobs/run_numa_bench.pbs
```

### 5. Analyze results

```bash
python3 evaluation_numa/analyze_configurations_numa.py
```

## Expected Results

The analysis will show:

- **Scaling efficiency** from 24 to 96 threads
- **Binding policy impact** (close vs spread vs master)
- **NUMA overhead** (if any)
- **Best configurations** for high thread counts
- **Efficiency degradation** patterns

## Key Metrics to Watch

1. **Speedup at 48 threads** - Should be ~2x speedup from 24 threads (2 NUMA nodes)
2. **Speedup at 96 threads** - Should be ~4x speedup from 24 threads (4 NUMA nodes)
3. **Efficiency at 96 threads** - Should stay above 50% if NUMA optimizations work
4. **spread vs close at 96 threads** - "spread" should win if NUMA locality matters

## Comparison with Single-Node Results

To compare with single-node results (from `evaluation/`):

1. Both benchmark suites test 24 threads
2. Compare 24-thread performance:
   - If NUMA code is slower at 24 threads, there's overhead
   - If NUMA code is similar at 24 threads, optimizations are zero-cost
3. Check if NUMA optimizations help beyond 24 threads

## System Requirements

- 96 CPUs across 4 NUMA nodes (24 cores each)
- At least 32GB memory for large matrices
- GCC 9.1.0 or later with OpenMP support
- Optional: `numactl` for explicit NUMA control

## Troubleshooting

**Problem**: Low efficiency at high thread counts  
**Solution**: Try different binding policies, check for memory bandwidth bottleneck

**Problem**: No scaling beyond 24 threads  
**Solution**: Check NUMA topology, verify threads are actually spreading across nodes

**Problem**: Timeout on large matrices  
**Solution**: Increase TIMEOUT_SECS in bench_configurations_numa.sh

**Problem**: Out of memory  
**Solution**: Reduce matrix size or request more memory in PBS script
