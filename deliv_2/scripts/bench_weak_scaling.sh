#!/bin/bash
#
# Weak Scaling Benchmark Script
# 
# Usage: ./bench_weak_scaling.sh [rows_per_proc] [density_pct] [iterations]
# Example: ./bench_weak_scaling.sh 25000 0.01 10
#
# Weak Scaling: matrix size scales with process count, work per process constant
#

set -e

# Default parameters
# IMPORTANT: 200k rows/proc provides sufficient compute/comm ratio
# Previous 25k was too small - communication dominated (96%+ overhead)
ROWS_PER_PROC=${1:-200000}
DENSITY_PCT=${2:-0.05}
ITERATIONS=10
THREADS_PER_RANK=4

# Process counts to test (limited to 2-32 for valid weak scaling)
# Beyond 32 procs, communication overhead becomes prohibitive
PROCESS_COUNTS=(2 4 8 16 32)

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$PROJECT_DIR/results"
SRC_DIR="$PROJECT_DIR/src"

# Create results directory
mkdir -p "$RESULTS_DIR"

# Remove old results
rm -f "$RESULTS_DIR/weak_scaling_results.csv"

# Set OpenMP environment
export OMP_NUM_THREADS=$THREADS_PER_RANK
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_STACKSIZE=64M

echo "=============================================="
echo "  Weak Scaling Benchmark"
echo "=============================================="
echo "Date: $(date)"
echo "Rows per process: $ROWS_PER_PROC"
echo "Target density: $DENSITY_PCT%"
echo "Iterations: $ITERATIONS"
echo "Threads per rank: $THREADS_PER_RANK"
echo "Process counts: ${PROCESS_COUNTS[*]}"
echo ""

# Compile if needed
if [ ! -f "$SRC_DIR/test_weak_scaling" ] || [ "$SRC_DIR/test_weak_scaling.c" -nt "$SRC_DIR/test_weak_scaling" ]; then
    echo "Compiling test_weak_scaling..."
    cd "$SRC_DIR"
    mpicc -std=c99 -g -Wall -O3 -fopenmp -o test_weak_scaling \
        test_weak_scaling.c generator.c m_to_csr.c -lm
    cd "$PROJECT_DIR"
    echo "Compilation successful."
    echo ""
fi

# Run benchmarks
for NP in "${PROCESS_COUNTS[@]}"; do
    TOTAL_ROWS=$((NP * ROWS_PER_PROC))
    
    echo "----------------------------------------------"
    echo "Process Count: $NP"
    echo "Matrix Size: $TOTAL_ROWS × $TOTAL_ROWS"
    echo "----------------------------------------------"
    
    mpirun -np $NP \
        --bind-to none \
        "$SRC_DIR/test_weak_scaling" $ROWS_PER_PROC $DENSITY_PCT $ITERATIONS
    
    echo ""
done

echo "=============================================="
echo "  Weak Scaling Benchmark Complete"
echo "=============================================="
echo "Finished: $(date)"
echo "Results: $RESULTS_DIR/weak_scaling_results.csv"
echo ""
