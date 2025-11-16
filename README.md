# PARCO-Computing-2026-244967

Parallel sparse matrix-vector multiplication (SpMV) benchmark using OpenMP with SIMD optimizations.

This project implements a two-phase optimization approach:
1. **Phase 1**: Single-node optimization (1-24 threads) - **COMPLETED**
2. **Phase 2**: NUMA-aware multi-node optimization (24-96 threads) - **IN PROGRESS**

## Project Structure

```
PARCO-Computing-2026-244967/
├── src/                         # C source code
│   ├── main.c                  # Main SpMV program
│   ├── test_configurations.c   # Single-node config testing (Phase 1)
│   ├── test_configurations_numa.c # NUMA-aware config testing (Phase 2)
│   ├── mtrvec.c                # Matrix-vector multiplication utilities
│   ├── generator.c             # Matrix generation utilities
│   ├── m_to_csr.c              # Matrix Market to CSR conversion
│   ├── multiply.c              # SpMV kernel implementations
│   ├── generator.h, multiply.h, m_to_csr.h  # Header files
│   └── *.o                     # Compiled object files
├── matrices/                    # Small/medium sparse matrices (.mtx)
│   └── *.mtx                   # 9 matrices: 2k-15k, 0.23%-9.37% density
├── scripts/                     # Benchmark and analysis scripts
│   ├── bench_matrices.sh       # Speedup benchmarks (Phase 1)
│   ├── bench_configurations.sh # Compare 30 OpenMP configs (Phase 1)
│   ├── bench_configurations_numa.sh # NUMA benchmarks (Phase 2)
│   ├── bench_cache.sh          # Cache performance benchmarks
│   ├── bench_cache_valgrind.sh # Cache analysis with valgrind
│   ├── analyze_configurations.py # Analyze Phase 1 results
│   ├── analyze_configurations_numa.py # Analyze Phase 2 results
│   ├── analyze_strong_scaling.py # Strong scaling analysis
│   ├── plot_speedup.py         # Generate speedup plots
│   ├── plot_configurations.py  # Configuration comparison plots
│   ├── plot_cache.py           # Cache performance plots
│   ├── plot_results.sh         # Batch plot generation
│   ├── setup_numa_benchmarks.sh # NUMA benchmark setup
│   └── test_bench_small.sh     # Quick benchmark test
├── pbs_jobs/                    # PBS cluster job scripts
│   ├── run_numa_bench.pbs      # NUMA benchmark job (Phase 2)
│   ├── run_config_bench.pbs    # Configuration benchmark job
│   ├── run_benchmarks.pbs      # General benchmark job
│   ├── run_cache_valgrind.pbs  # Cache valgrind job
│   ├── test_single_matrix.pbs  # Single matrix test job
│   ├── array.pbs               # PBS array job script
│   ├── *.out, *.err            # Job output/error logs
│   └── *.pbs                   # Additional job scripts
├── results/                     # Benchmark output data
│   ├── configurations_results.csv # Phase 1: 2,700 measurements
│   ├── configurations_results.txt # Phase 1: Detailed log
│   ├── configurations_numa_results.csv # Phase 2: NUMA results
│   ├── configurations_numa_results.txt # Phase 2: Detailed log
│   ├── matrices_results.csv    # Speedup benchmark results
│   └── cache_valgrind_results.csv # Cache analysis results
├── plots/                       # Generated figures (PNG)
│   ├── strong_scaling/         # Strong scaling analysis plots
│   ├── speedup_vs_threads.png  # Thread scaling visualization
│   ├── speedup_vs_density.png  # Density impact analysis
│   ├── speedup_heatmap.png     # Configuration heatmap
│   ├── parallel_efficiency.png # Efficiency curves
│   ├── top_configurations.png  # Best config comparison
│   └── *.png                   # Additional visualizations
├── executable                   # Main SpMV program (compiled)
├── test_config                  # Phase 1 testing tool (compiled)
├── test_config_numa             # Phase 2 testing tool (compiled)
├── executable_cache_valgrind    # Valgrind-instrumented executable
├── mtrvec, rowmajor, tocsr      # Utility executables
├── *.o                          # Compiled object files
├── Makefile                     # Build configuration
├── README.md                    # This file (project documentation)
├── TEST_CONFIG_README.md        # Configuration testing guide
├── NUMA_SETUP.md                # NUMA optimization guide
├── PERFORMANCE_NOTES.md         # Performance analysis notes
├── numa_optimization_guide.md   # Additional NUMA documentation
├── .gitignore                   # Git ignore rules
└── *.err, *.out                 # Build/run logs
```

**Note:** `matrices_large/` directory is expected on the cluster for Phase 2 NUMA benchmarks but not included in the repository due to file size.


## Quick Start

### Prerequisites
- **Compiler**: GCC ≥ 9.1 with OpenMP 4.5+ support
- **System**: x86-64 CPU with AVX2 (AVX-512 optional)
- **Memory**: ≥ 8GB RAM recommended
- **Python** (optional): For analysis scripts (matplotlib, pandas, numpy)

### Build All Executables

```bash
make
```

This compiles three executables:
- `executable` - Main SpMV program for single matrix runs
- `test_config` - Phase 1: Single-node configuration testing (1-24 threads)
- `test_config_numa` - Phase 2: NUMA-aware configuration testing (24-96 threads)

### Clean Build

```bash
make clean
make
```

---

## Running Executables

### 1. Basic SpMV Execution

Run sparse matrix-vector multiplication with specified thread count:

```bash
./executable <threads> <matrix_file>
```

**Examples:**
```bash
# Run with 8 threads on a small matrix
./executable 8 matrices/2k_0p23.mtx

# Run with 24 threads on a larger matrix
./executable 24 matrices/15k_0p41.mtx

# Serial execution (1 thread)
./executable 1 matrices/6k_6p28.mtx
```

**Output:**
- Matrix dimensions and density
- Execution time (milliseconds)
- Speedup vs. serial baseline

---

### 2. Phase 1: Single-Node Configuration Testing

Compare 30 OpenMP configurations on a single NUMA node (1-24 threads):

```bash
./test_config <threads> <matrix_file> <iterations>
```

**Examples:**
```bash
# Quick test: 8 threads, 10 iterations
./test_config 8 matrices/6k_6p28.mtx 10

# Full statistical test: 24 threads, 30 iterations (recommended)
./test_config 24 matrices/10k_1p5.mtx 30

# Low thread count comparison
./test_config 4 matrices/2k_0p23.mtx 30
```

**What it tests (30 configurations):**
- **Static scheduling**: default, chunk=8/16/32/64
- **Dynamic scheduling**: chunk=8/16/32/64
- **Guided scheduling**: chunk=8/16/32/64
- **Auto scheduling**: chunk=8/16/32/64
- **SIMD variants**: Static/Dynamic/Guided + SIMD (chunk 8/16/32/64)
- **Thread affinity**: SIMD + `proc_bind(close)` for cache locality
- **Cache alignment**: Combined optimizations

**Output format:**
```
Configuration Name    Time(ms)  Speedup  Efficiency  Improvement
Static+SIMD+c32       1.234     925.0x   96.8%       +12.5%
Guided+SIMD+c32       1.156     1047.2x  98.2%       +0.0% (baseline)
...
```

---

### 3. Phase 2: NUMA-Aware Configuration Testing

Test NUMA-optimized configurations across multiple sockets (24-96 threads):

```bash
./test_config_numa <threads> <matrix_file> [iterations]
```

**Examples:**
```bash
# Test on single socket (24 threads)
./test_config_numa 24 matrices_large/citationCiteseer.mtx 10

# Test on two sockets (48 threads)
./test_config_numa 48 matrices_large/F1.mtx 10

# Test on all four sockets (96 threads)
./test_config_numa 96 matrices_large/cnr-2000.mtx 10

# Default iterations (10 if omitted)
./test_config_numa 72 matrices_large/RM07R.mtx
```

**What it tests (7 NUMA configurations):**
1. **Static+SIMD+chunk=32+close** - Single-socket optimized
2. **Static+SIMD+chunk=32+spread** - Multi-socket thread distribution
3. **Static+SIMD+chunk=32+master** - Master thread binding
4. **Guided+SIMD+chunk=32+close** - Guided with single-socket binding
5. **Guided+SIMD+chunk=32+spread** - Guided with multi-socket distribution
6. **Dynamic+SIMD+chunk=32+spread** - Dynamic load balancing
7. **Static+SIMD+Reg+close** - With register blocking hints

**Output format:**
Similar to Phase 1, but optimized for NUMA effects and cross-socket communication.

**Note:** Phase 2 results are currently being collected. Use `matrices_large/` directory for testing.

---

## Running Benchmark Scripts

### Phase 1: Full Single-Node Benchmarks

#### Configuration Comparison (Comprehensive)

```bash
./scripts/bench_configurations.sh
```

**What it does:**
- Tests all 30 configurations
- Across all 9 matrices in `matrices/`
- At 10 thread counts: 1, 2, 4, 8, 12, 16, 18, 20, 22, 24
- **Total: 2,700 measurements**
- Runtime: ~2-3 hours
- Output: `results/configurations_results.csv` and `results/configurations_results.txt`

**Use when:** You want comprehensive optimization analysis

#### Speedup-Focused Benchmarks

```bash
./scripts/bench_matrices.sh
```

**What it does:**
- Tests best configurations for speedup analysis
- Multiple thread counts per matrix
- Faster than full configuration comparison
- Output: `results/` directory

**Use when:** You want quick speedup curves

---

### Phase 2: NUMA-Aware Benchmarks

#### Local Execution

```bash
./scripts/bench_configurations_numa.sh
```

**What it does:**
- Tests 7 NUMA-aware configurations
- Across 5 large matrices in `matrices_large/`
- At 4 thread counts: 24, 48, 72, 96
- **Total: 140 measurements** (7 configs × 5 matrices × 4 thread counts)
- Runtime: ~3-6 hours (15 min timeout per test)
- Output: `results/configurations_numa_results.csv` and `.txt`

**Requirements:**
- Multi-socket NUMA system
- Matrices in `matrices_large/` directory
- `test_config_numa` executable compiled

#### Cluster Execution (PBS)

```bash
# Submit to cluster queue
qsub pbs_jobs/run_numa_bench.pbs

# Check job status
qstat

# View output
cat numa_bench.out
cat numa_bench.err
```

**PBS script details:**
- Queue: `short_cpuQ`
- Walltime: 6 hours
- Resources: 4 nodes × 24 cores = 96 cores
- Auto-compiles `test_config_numa` with GCC 9.1.0
- Runs `bench_configurations_numa.sh`
- Results in `results/configurations_numa_results.csv`

---

## Analysis and Visualization

### Analyze Phase 1 Results

```bash
# Comprehensive configuration analysis
python3 scripts/analyze_configurations.py

# Strong scaling analysis with detailed plots
python3 scripts/analyze_strong_scaling.py
```

**Generates:**
- SIMD impact analysis (typically 188× improvement)
- Thread affinity benefits (~5% gain)
- Optimal configurations per matrix type
- Scaling efficiency across thread counts
- Best performance: **925× average speedup**, **3,300× peak**
- Plots saved to `plots/strong_scaling/`

### Analyze Phase 2 Results (When Available)

```bash
python3 scripts/analyze_configurations_numa.py
```

**Analyzes:**
- NUMA node scaling efficiency (24→48→72→96 threads)
- Cross-socket communication overhead
- Memory affinity impact
- Optimal binding policies (close vs. spread)

### Generate Visualizations

```bash
# Speedup curves
python3 scripts/plot_speedup.py

# Configuration comparison heatmaps
python3 scripts/plot_configurations.py

# Cache performance analysis
python3 scripts/plot_cache.py
```

**Output:** PNG files in `plots/` directory

---

## Understanding Results Files

### configurations_results.csv (Phase 1)

2,700 rows containing:
- Matrix name, dimensions, density, NNZ count
- Thread count (1-24)
- Configuration name and binding policy
- Execution time (milliseconds)
- Speedup vs. serial baseline
- Parallel efficiency percentage
- Standard deviation
- Improvement vs. best configuration

### configurations_numa_results.csv (Phase 2)

140 rows (when complete) containing:
- Matrix name, dimensions, density, NNZ count  
- Thread count (24, 48, 72, 96)
- NUMA configuration and binding policy
- Execution time (milliseconds)
- Speedup and efficiency metrics
- Cross-socket scaling characteristics

### Benchmark Logs (.txt files)

Human-readable detailed output:
- Timestamp and system information
- Per-matrix, per-thread-count results
- Configuration rankings
- Debug output for failed tests

---## Matrix Files

### Naming Convention

Matrix files follow the pattern: `{size}_{density}.mtx`

- **Size**: Rounded dimensions in thousands (k)
- **Density**: Percentage with 'p' as decimal point

**Examples:**
- `2k_0p23.mtx` → 2,000×2,000 matrix, 0.23% density
- `6k_6p28.mtx` → 6,000×6,000 matrix, 6.28% density
- `15k_0p41.mtx` → 15,449×15,449 matrix, 0.41% density

### Available Matrices

**Phase 1 (matrices/):** 9 matrices
| File | Dimensions | Density | NNZ | Category |
|------|------------|---------|-----|----------|
| 2k_0p23.mtx | 1,624 × 1,624 | 0.23% | 6,080 | Ultra-sparse |
| 5k_0p23.mtx | 4,690 × 4,690 | 0.23% | 20,316 | Ultra-sparse |
| 6k_6p28.mtx | 6,156 × 6,156 | 6.28% | 2,380,515 | Dense |
| 10k_0p31.mtx | 9,800 × 9,800 | 0.31% | 52,329 | Sparse |
| 10k_1p5.mtx | 10,000 × 10,000 | 1.50% | 684,692 | Moderately sparse |
| 10k_3p17.mtx | 10,638 × 10,638 | 3.17% | 1,096,948 | Moderately dense |
| 15k_0p41.mtx | 15,449 × 15,449 | 0.41% | 973,227 | Sparse |
| 5k_9p37.mtx | 5,005 × 5,005 | 9.37% | 2,462,970 | Very dense |
| 2k_0p73.mtx | 1,952 × 1,952 | 0.73% | 3,837 | Sparse |

**Phase 2 (matrices_large/):** 5 large matrices for NUMA testing
- `citationCiteseer.mtx`
- `cnr-2000.mtx`
- `F1.mtx`
- `n4c6-b9.mtx`
- `RM07R.mtx`

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

---## Key Performance Results

### Phase 1: Single-Node Optimization (COMPLETED)

From 2,700 benchmark measurements across 30 configurations:

**Overall Performance:**
- **Average speedup**: 925× across all matrices and configurations
- **Peak speedup**: 3,300× on ultra-sparse matrices at 18 threads
- **Best configuration**: Static+SIMD+chunk=32+Alignment+Affinity

**Optimization Impact Hierarchy:**
1. **SIMD Vectorization**: 188× improvement (dominant factor)
   - Transforms 1.02× (non-SIMD) to 73× (SIMD) at single thread
   - Non-negotiable for performance
2. **Cache Alignment**: +28% average (+35% on ultra-sparse)
3. **Static Scheduling**: +16% over dynamic/guided (925× vs 805×)
4. **Thread Affinity**: +5% via `proc_bind(close)`

**Scaling by Matrix Density:**
- **Ultra-sparse** (<0.5% density): 1,647× average speedup
  - 85%+ parallel efficiency through 8 threads
  - 50-60% efficiency at 18 threads
- **Moderately sparse** (0.5-1.5%): Scale well to 12 threads (75%+ efficiency)
- **Dense** (>3%): 190× average speedup
  - Peak at 4-8 threads due to memory bandwidth saturation

**Thread Scaling:**
- Near-linear scaling to 8-12 threads
- Plateau at 16-24 threads (single socket limit)
- Memory bandwidth becomes bottleneck beyond 18 threads

### Phase 2: NUMA Multi-Node Optimization (IN PROGRESS)

Testing NUMA-aware configurations for 24-96 threads across 4 sockets.

**Expected gains:**
- Cross-socket scaling efficiency
- Memory locality optimization
- Thread-to-socket binding strategies

**Status:** Benchmarks currently being collected with `bench_configurations_numa.sh`

---

## Reproducibility

All experimental artifacts are version-controlled for full reproducibility.

**Repository:** https://github.com/Marui-9/PARCO-Computing-2026-244967

**Hardware Requirements:**
- x86-64 CPU with AVX2 support (AVX-512 optional)
- ≥ 8GB RAM
- Linux environment
- Multi-socket NUMA system recommended for Phase 2

**Software Requirements:**
- GCC ≥ 9.1 with OpenMP 4.5+
- Python 3 (optional): matplotlib, pandas, numpy for analysis

**Complete Workflow:**
```bash
# 1. Clone repository
git clone https://github.com/Marui-9/PARCO-Computing-2026-244967.git
cd PARCO-Computing-2026-244967

# 2. Build executables
make

# 3. Quick test
./executable 8 matrices/10k_1p5.mtx

# 4. Configuration comparison (single matrix)
./test_config 24 matrices/6k_6p28.mtx 30

# 5. Full Phase 1 benchmarks (~2-3 hours)
./scripts/bench_configurations.sh

# 6. Analyze results
python3 scripts/analyze_configurations.py
python3 scripts/analyze_strong_scaling.py

# 7. Generate visualizations
python3 scripts/plot_speedup.py

# 8. NUMA benchmarks (cluster with 96 cores)
qsub pbs_jobs/run_numa_bench.pbs
```

**Expected Results:**
- `results/configurations_results.csv`: 2,700 measurements
- `plots/strong_scaling/`: Speedup curves, efficiency analysis
- Performance matching published results (925× average speedup)

---

## Troubleshooting

### Compilation Issues

**Error: OpenMP not supported**
```bash
# Check GCC version
gcc --version  # Should be ≥ 9.1

# Install newer GCC if needed
# On Ubuntu/Debian:
sudo apt-get install gcc-9 g++-9
```

**Error: Undefined reference to `posix_memalign`**
```bash
# Ensure -lm flag is included (already in Makefile)
make clean && make
```

### Runtime Issues

**Error: Matrix file not found**
```bash
# Check file exists
ls -lh matrices/*.mtx

# Use relative path from project root
./executable 8 matrices/2k_0p23.mtx  # Correct
./executable 8 2k_0p23.mtx           # Wrong (unless in matrices/)
```

**Error: Segmentation fault**
```bash
# Likely insufficient memory for large matrices
# Check available memory
free -h

# Try smaller matrix or fewer threads
./executable 4 matrices/2k_0p23.mtx
```

**NUMA benchmarks produce no output**
```bash
# Check matrices_large/ directory exists and has files
ls -lh matrices_large/

# Verify test_config_numa is compiled
ls -lh test_config_numa

# Check error log for details
cat results/configurations_numa_results.txt

# Try manual test
./test_config_numa 24 matrices_large/citationCiteseer.mtx 1
```

### Analysis Script Issues

**Error: Module not found (matplotlib/pandas/numpy)**
```bash
# Install Python dependencies
pip3 install matplotlib pandas numpy

# Or using conda
conda install matplotlib pandas numpy
```

**Error: No data to plot**
```bash
# Ensure benchmark CSV files exist
ls -lh results/*.csv

# Run benchmarks first
./scripts/bench_configurations.sh
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{marchesin2026parco,
  author = {Marchesin, Jacopo},
  title = {Parallel Sparse Matrix-Vector Multiplication Using OpenMP with SIMD Optimizations},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Marui-9/PARCO-Computing-2026-244967}},
  note = {Course: Introduction to Parallel Computing, University of Trento}
}
```

---

## License

This project is developed for educational purposes as part of the Introduction to Parallel Computing course (2025-2026) at the University of Trento.

## Author

**Jacopo Marchesin** (ID: 244967)  
Email: jacopo.marchesin@studenti.unitn.it  
Course: Introduction to Parallel Computing (2025-2026)