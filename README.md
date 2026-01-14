# PARCO-Computing-2026-244967 
# Deliverable 2

Distributed sparse matrix-vector multiplication (SpMV) benchmark using MPI + OpenMP hybrid parallelization.

This project implements a distributed-memory parallel SpMV with:
1. **MPI Inter-node Communication**: Row-wise matrix distribution across MPI ranks
2. **OpenMP Intra-node Parallelism**: Multithreaded computation within each rank
3. **Communication Optimization**: Multiple communication strategies (blocking, non-blocking, pipelined)
4. **Load Balancing**: Row-based and NNZ-based distribution strategies

## Project Structure

```
deliv_2/
├── src/                         # C source code
│   ├── main.c                  # Main MPI+OpenMP SpMV program
│   ├── test_configurations_mpi.c # MPI communication mode testing
│   ├── test_pipelined_mpi.c    # Pipelined communication experiments
│   ├── test_weak_scaling.c     # Weak scaling benchmark
│   ├── test_load_balance_sweep.c # Load balancing strategy comparison
│   ├── load_balance.c/h        # Load balancing implementations
│   ├── generator.c/h           # Matrix generation utilities
│   ├── m_to_csr.c/h            # Matrix Market to CSR conversion
│   ├── multiply.c/h            # SpMV kernel implementations
│   └── mtrvec.c                # Matrix-vector multiplication utilities
├── matrices/                    # Sparse matrices (.mtx) - download required
│   └── *.mtx                   # Large benchmark matrices (916k-4.8M rows)
├── scripts/                     # Benchmark and analysis scripts
│   ├── configurations.sh       # MPI communication mode sweep
│   ├── pipelined_sweep.sh      # Pipelined communication benchmark
│   ├── bench_weak_scaling.sh   # Weak scaling benchmark
│   ├── load_balance_sweep.sh   # Load balancing strategy sweep
│   ├── download_matrices_mpi.sh # Download benchmark matrices
│   └── python_scripts/         # Analysis and plotting scripts
│       ├── analyze_configurations.py   # Communication mode analysis
│       ├── analyze_pipelined.py        # Pipelined mode analysis
│       ├── analyze_weak_scaling.py     # Weak scaling analysis
│       ├── analyze_strong_scaling.py   # Strong scaling analysis
│       ├── analyze_load_balance.py     # Load balancing analysis
│       ├── plot_configurations.py      # Generate configuration plots
│       ├── plot_mpi_speedup.py         # MPI speedup plots
│       └── plot_mpi_efficiency.py      # MPI efficiency plots
├── pbs_jobs/                    # PBS cluster job scripts
│   ├── configurations.pbs      # Communication mode benchmark job
│   ├── pipelined_sweep.pbs     # Pipelined benchmark job
│   ├── weak_scaling.pbs        # Weak scaling benchmark job
│   └── load_balance_sweep.pbs  # Load balance benchmark job
├── results/                     # Benchmark output data
│   ├── configurations_results.csv  # Communication mode results
│   ├── pipelined_results.csv       # Pipelined mode results
│   ├── weak_scaling_results.csv    # Weak scaling results
│   └── load_balance_results.csv    # Load balancing results
├── plots/                       # Generated figures (PNG)
│   ├── efficiency_vs_procs.png
│   ├── speedup_vs_procs.png
│   ├── strong_scaling.png
│   ├── config_comparison_bars.png
│   └── configurations_combined.png
├── Makefile                     # Build configuration (MPI + OpenMP)
└── README.md                    # This file (project documentation)
```

## Quick Start

### Prerequisites
- **MPI**: MPICH 3.2.1 or compatible MPI implementation
- **Compiler**: GCC ≥ 9.1 with OpenMP support
- **System**: Linux cluster with multi-node capability
- **Memory**: ≥ 16GB RAM per node recommended
- **Python** (optional): For analysis scripts (matplotlib, pandas, numpy)

### Build All Executables

**Using Makefile:**
```bash
cd deliv_2
make all          # Build main executable (mtrvec)
make test_mpi     # Build MPI configuration tester
make test_pipelined  # Build pipelined benchmark
make test_weak    # Build weak scaling benchmark
```

This compiles the following executables:
- `mtrvec` - Main MPI+OpenMP SpMV program
- `test_config_mpi` - Communication mode benchmark
- `test_pipelined_mpi` - Pipelined communication experiments
- `test_weak_scaling` - Weak scaling benchmark

**Manual Compilation:**

```bash
# Set MPI compiler path (adjust for your system)
export MPICC=/apps/mpich-3.2.1--gcc-9.1.0/bin/mpicc

# Compile main executable
$MPICC -O3 -Wall -Wextra -march=native -fopenmp -DMPI_ENABLED -o mtrvec \
    src/main.c src/generator.c src/m_to_csr.c -lm

# Compile MPI configuration tester
$MPICC -O3 -Wall -fopenmp -o test_config_mpi \
    src/test_configurations_mpi.c src/generator.c src/m_to_csr.c -lm

# Compile pipelined benchmark
$MPICC -O3 -Wall -fopenmp -o test_pipelined_mpi \
    src/test_pipelined_mpi.c src/generator.c src/m_to_csr.c -lm

# Compile weak scaling benchmark
$MPICC -O3 -Wall -fopenmp -o test_weak_scaling \
    src/test_weak_scaling.c src/generator.c src/m_to_csr.c -lm
```

**Notes:**
- `-O3`: Aggressive optimization
- `-march=native`: Optimize for current CPU
- `-fopenmp`: Enable OpenMP support
- `-DMPI_ENABLED`: Enable MPI code paths
- `-lm`: Link math library

### Download Benchmark Matrices

```bash
# Download matrices for MPI benchmarks (916k - 4.8M rows)
./scripts/download_matrices_mpi.sh
```

### Clean Build

```bash
make clean
make all test_mpi test_pipelined test_weak
```

---

## Running Executables

### 1. Basic MPI+OpenMP SpMV Execution

Run distributed sparse matrix-vector multiplication:

```bash
mpirun -np <num_ranks> ./mtrvec <threads_per_rank> <matrix_file>
```

**Examples:**
```bash
# Run with 4 MPI ranks, 4 threads each on a medium matrix
mpirun -np 4 ./mtrvec 4 matrices/916k_0p0006.mtx

# Run with 8 MPI ranks, 4 threads each
mpirun -np 8 ./mtrvec 4 matrices/1508k_0p0012.mtx

# Run with 2 MPI ranks (minimum) for testing
mpirun -np 2 ./mtrvec 4 matrices/916k_0p0006.mtx
```

**Output:**
- Matrix dimensions, density, and NNZ count
- MPI rank and thread configuration
- Execution time (milliseconds)
- Speedup and efficiency metrics

---

### 2. MPI Communication Mode Testing

Compare different MPI communication strategies:

```bash
mpirun -np <num_ranks> ./src/test_config_mpi <matrix_file> [iterations]
```

**Examples:**
```bash
# Test with 4 MPI ranks, 12 iterations
mpirun -np 4 ./src/test_config_mpi matrices/916k_0p0006.mtx 12

# Test with 8 MPI ranks
mpirun -np 8 ./src/test_config_mpi matrices/1508k_0p0012.mtx 12

# Test with maximum ranks
mpirun -np 128 ./src/test_config_mpi matrices/1438k_0p0016.mtx 12
```

**Communication modes tested:**
1. **MPI_Bcast+Gatherv** - Standard blocking collective operations
2. **Ibcast+Igatherv** - Non-blocking collectives with computation overlap

**Output format:**
```
Mode                  Time(ms)  Comm(ms)  Compute(ms)  Speedup  Efficiency
MPI_Bcast+Gatherv     12.34     10.5      1.84         2.5×     62.5%
Ibcast+Overlap        11.89     9.8       2.09         2.6×     65.0%
```

---

### 3. Pipelined Communication Testing

Test pipelined communication-computation overlap:

```bash
mpirun -np <num_ranks> ./src/test_pipelined_mpi <matrix_file> [iterations]
```

**Examples:**
```bash
# Test pipelined mode with 4 ranks
mpirun -np 4 ./src/test_pipelined_mpi matrices/916k_0p0006.mtx 12

# Test with 32 ranks
mpirun -np 32 ./src/test_pipelined_mpi matrices/1508k_0p0012.mtx 12
```

**Communication strategies compared:**
1. **Standard blocking** - MPI_Bcast + MPI_Gatherv
2. **Non-blocking** - Ibcast + Igatherv
3. **Async collectives** - Ibcast + Igatherv with explicit overlap
4. **Pipelined chunked** - Chunked broadcast with immediate computation

---

### 4. Weak Scaling Benchmark

Test weak scaling (constant work per process):

```bash
mpirun -np <num_ranks> ./src/test_weak_scaling <rows_per_proc> <nnz_per_proc> [iterations]
```

**Examples:**
```bash
# Default: 200k rows/proc, 40M NNZ/proc
mpirun -np 4 ./src/test_weak_scaling 200000 40000000 10

# Larger problem: 500k rows/proc
mpirun -np 8 ./src/test_weak_scaling 500000 100000000 10
```

**Weak scaling strategy:**
- Each process handles `rows_per_proc` rows
- Matrix size = rows_per_proc × num_procs (grows with P)
- NNZ per process = constant (same work per process)
- Ideal weak scaling: constant execution time as P increases

---

## Running Benchmark Scripts

### Communication Mode Sweep

```bash
./scripts/configurations.sh [matrix_file] [iterations]
```

**What it does:**
- Tests 2 communication modes (Bcast+Gatherv, Ibcast+Overlap)
- Sweeps process counts: 2, 4, 8, 16, 32, 64, 96, 128
- Uses 4 threads per MPI rank
- 12 iterations per configuration
- Output: `results/configurations_results.csv`

**Use when:** Comparing communication strategies at different scales

---

### Pipelined Communication Sweep

```bash
./scripts/pipelined_sweep.sh
```

**What it does:**
- Tests 4 pipelined communication strategies
- Sweeps process counts: 2, 4, 8, 16, 32, 64, 96, 128
- Tests multiple matrices (916k - 4.8M rows)
- Output: `results/pipelined_results.csv`

**Use when:** Evaluating pipelined communication-computation overlap

---

### Weak Scaling Benchmark

```bash
./scripts/bench_weak_scaling.sh [rows_per_proc] [nnz_per_proc] [iterations]
```

**What it does:**
- Tests weak scaling (constant work per process)
- Default: 200k rows/proc, 40M NNZ/proc
- Sweeps process counts: 2, 4, 8, 16, 32, 64, 128
- Output: `results/weak_scaling_results.csv`

**Use when:** Measuring scalability with growing problem size

---

### Load Balancing Strategy Sweep

```bash
./scripts/load_balance_sweep.sh
```

**What it does:**
- Tests 2 load balancing strategies: ROW-BASED, NNZ-BASED
- Sweeps process counts: 2, 4, 8, 16, 32, 64, 96, 128
- Tests all matrices in `matrices/` directory
- Output: `results/load_balance_results.csv`

**Use when:** Comparing row-based vs NNZ-based distribution

---

### Cluster Execution (PBS)

```bash
# Submit communication mode benchmark
qsub pbs_jobs/configurations.pbs

# Submit pipelined benchmark
qsub pbs_jobs/pipelined_sweep.pbs

# Submit weak scaling benchmark
qsub pbs_jobs/weak_scaling.pbs

# Submit load balance benchmark
qsub pbs_jobs/load_balance_sweep.pbs

# Check job status
qstat

# View output
cat configurations.out
cat configurations.err
```

---

## Analysis and Visualization

### Analyze Communication Mode Results

```bash
python3 scripts/python_scripts/analyze_configurations.py
```

**Analyzes:**
- Communication mode performance comparison
- Speedup and efficiency across process counts
- Communication vs computation time breakdown
- Best communication strategy per scale

### Analyze Pipelined Results

```bash
python3 scripts/python_scripts/analyze_pipelined.py
```

**Analyzes:**
- Pipelined communication overhead reduction
- Speedup gains from computation-communication overlap
- Optimal chunk sizes for pipelining
- Comparison with baseline strategies

### Analyze Weak Scaling Results

```bash
python3 scripts/python_scripts/analyze_weak_scaling.py
```

**Analyzes:**
- Weak scaling efficiency (ideal = 100%)
- Communication overhead growth with scale
- Scaling breakdown points
- Comparison across communication modes

### Analyze Strong Scaling Results

```bash
python3 scripts/python_scripts/analyze_strong_scaling.py
```

**Analyzes:**
- Strong scaling speedup (ideal = P×)
- Parallel efficiency at different scales
- Amdahl's law fit
- Performance saturation points

### Analyze Load Balancing Results

```bash
python3 scripts/python_scripts/analyze_load_balance.py
```

**Analyzes:**
- ROW-BASED vs NNZ-BASED performance
- Load imbalance impact on efficiency
- Optimal strategy per matrix structure

### Generate Visualizations

```bash
# Configuration comparison plots
python3 scripts/python_scripts/plot_configurations.py

# MPI speedup plots
python3 scripts/python_scripts/plot_mpi_speedup.py

# MPI efficiency plots
python3 scripts/python_scripts/plot_mpi_efficiency.py
```

**Output:** PNG files in `plots/` directory

---

## Understanding Results Files

### configurations_results.csv

Communication mode benchmark results containing:
- Matrix name, dimensions, density, NNZ count
- Process count (2-128)
- Communication mode (MPI_Bcast+Gatherv, Ibcast+Overlap)
- Total execution time (milliseconds)
- Communication time and compute time breakdown
- Speedup and parallel efficiency

### pipelined_results.csv

Pipelined communication results containing:
- Matrix name and properties
- Process count
- Communication strategy (Standard, Non-blocking, Async, Pipelined)
- Execution time and overhead metrics
- Speedup relative to blocking baseline

### weak_scaling_results.csv

Weak scaling benchmark results containing:
- Process count
- Rows per process, NNZ per process
- Total matrix size
- Execution time (should be constant for ideal scaling)
- Weak scaling efficiency

### load_balance_results.csv

Load balancing comparison results containing:
- Matrix name and properties
- Process count
- Load balancing strategy (ROW-BASED, NNZ-BASED)
- Execution time
- Load imbalance metrics

---

## Matrix Files

### Naming Convention

Matrix files follow the pattern: `{size}_{density}.mtx`

- **Size**: Rounded dimensions in thousands (k) or millions (M)
- **Density**: Percentage with 'p' as decimal point

**Examples:**
- `916k_0p0006.mtx` → 916,000×916,000 matrix, 0.0006% density
- `1508k_0p0012.mtx` → 1,508,000×1,508,000 matrix, 0.0012% density
- `2097k_0p0001.mtx` → 2,097,000×2,097,000 matrix, 0.0001% density

### Benchmark Matrices (Download Required)

Use `./scripts/download_matrices_mpi.sh` to download matrices.

| File | Dimensions | Density | NNZ (approx) |
|------|------------|---------|--------------|
| 916k_0p0006.mtx | 916k × 916k | 0.0006% | ~5M |
| 1438k_0p0016.mtx | 1,438k × 1,438k | 0.0016% | ~33M |
| 1508k_0p0012.mtx | 1,508k × 1,508k | 0.0012% | ~27M |
| 1565k_0p0024.mtx | 1,565k × 1,565k | 0.0024% | ~59M |
| 2097k_0p0001.mtx | 2,097k × 2,097k | 0.0001% | ~4.4M |

**Note:** Large matrices (>1GB) are not included in the repository. Download scripts fetch from SuiteSparse Matrix Collection.

### Matrix Format

All matrices use **Matrix Market coordinate format (.mtx)**:
```
%%MatrixMarket matrix coordinate real general
% comments
<rows> <cols> <nnz>
<row> <col> <value>
<row> <col> <value>
...
```

Converted to **CSR (Compressed Sparse Row)** format at runtime for efficient SpMV.

---

## Key Performance Results

### Communication Mode Analysis

From benchmark measurements across 2-128 MPI processes:

**Communication Strategies:**
- **MPI_Bcast+Gatherv**: Best overall performance at all scales
- **Ibcast+Overlap**: Marginal improvement at 4-64 processes

**Scaling Observations:**
- Parallel efficiency drops from ~50% at 2 processes to <1% at 128 processes
- Communication overhead dominates: 93%+ of total time at 2 processes, 99%+ at 128 processes
- Memory-bound workload limits scalability beyond 8-16 processes

### Pipelined Communication Results

**Pipelined vs Blocking:**
- Average 8.5% overhead reduction in pipelined mode
- 5-7% speedup gains in 4-64 process range
- Limited benefit at extreme scales (2 or 128 processes)

### Weak Scaling Analysis

**Weak Scaling Efficiency:**
- ~22% average efficiency across all scales
- Efficiency collapse beyond 2-4 processes
- Communication overhead grows faster than computation

### Load Balancing Comparison

**ROW-BASED vs NNZ-BASED:**
- ROW-BASED marginally better (1.02× average speedup)
- Both strategies achieve ~15% efficiency
- Matrix structure has larger impact than distribution strategy

---

## Reproducibility

All experimental artifacts are version-controlled for full reproducibility.

**Repository:** https://github.com/Marui-9/PARCO-Computing-2026-244967

**Hardware Requirements:**
- Linux cluster with MPI support
- Multi-node capability (2-128 MPI ranks)
- ≥ 16GB RAM per node
- High-speed interconnect recommended (InfiniBand, etc.)

**Software Requirements:**
- MPICH 3.2.1 or compatible MPI implementation
- GCC ≥ 9.1 with OpenMP support
- Python 3 (optional): matplotlib, pandas, numpy for analysis

**Complete Workflow:**
```bash
# 1. Clone repository
git clone https://github.com/Marui-9/PARCO-Computing-2026-244967.git
cd PARCO-Computing-2026-244967/deliv_2

# 2. Download benchmark matrices
./scripts/download_matrices_mpi.sh

# 3. Build executables
make all test_mpi test_pipelined test_weak

# 4. Quick test (2 MPI ranks)
mpirun -np 2 ./mtrvec 4 matrices/916k_0p0006.mtx

# 5. Run communication mode benchmark
./scripts/configurations.sh

# 6. Run pipelined benchmark
./scripts/pipelined_sweep.sh

# 7. Run weak scaling benchmark
./scripts/bench_weak_scaling.sh

# 8. Analyze results
python3 scripts/python_scripts/analyze_configurations.py
python3 scripts/python_scripts/analyze_weak_scaling.py

# 9. Generate visualizations
python3 scripts/python_scripts/plot_configurations.py

# 10. Cluster submission (PBS)
qsub pbs_jobs/configurations.pbs
qsub pbs_jobs/weak_scaling.pbs
```

**Expected Results:**
- `results/configurations_results.csv`: Communication mode benchmark data
- `results/pipelined_results.csv`: Pipelined communication data
- `results/weak_scaling_results.csv`: Weak scaling benchmark data
- `plots/`: Efficiency and speedup visualizations

---