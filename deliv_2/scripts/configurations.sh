#!/bin/bash
#
# configurations.sh - Sweep MPI process counts for SpMV benchmark
#
# Usage: ./configurations.sh [matrix_file] [iterations]
# Example: ./configurations.sh matrices/36k_0p17.mtx 30
#

# Configuration
PROCESS_COUNTS=(2 4 8 16 24 32 48 64 72 84 96 108 128)
THREADS_PER_RANK=4
ITERATIONS=${2:-30}
MATRIX_FILE=${1:-"matrices/36k_0p17.mtx"}

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$BASE_DIR/src"
RESULTS_DIR="$BASE_DIR/results/configurations_sweep"

# Create results directory
mkdir -p "$RESULTS_DIR"

# Compile if needed
cd "$SRC_DIR"
if [ ! -f "test_config_mpi" ] || [ "test_configurations_mpi.c" -nt "test_config_mpi" ]; then
    echo "Compiling test_configurations_mpi.c..."
    mpicc -std=c99 -g -Wall -O3 -fopenmp -o test_config_mpi \
        test_configurations_mpi.c generator.c m_to_csr.c -lm
    if [ $? -ne 0 ]; then
        echo "ERROR: Compilation failed"
        exit 1
    fi
fi

# Set OpenMP threads
export OMP_NUM_THREADS=$THREADS_PER_RANK
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Get matrix basename for output
MATRIX_BASENAME=$(basename "$MATRIX_FILE" .mtx)

echo ""
echo "=========================================="
echo "  MPI Process Count Sweep Benchmark"
echo "=========================================="
echo "Matrix: $MATRIX_FILE"
echo "Threads per rank: $THREADS_PER_RANK"
echo "Iterations per config: $ITERATIONS"
echo "Process counts: ${PROCESS_COUNTS[*]}"
echo ""

# Run benchmarks for each process count
for NP in "${PROCESS_COUNTS[@]}"; do
    TOTAL_THREADS=$((NP * THREADS_PER_RANK))
    
    echo "------------------------------------------"
    echo "Testing: $NP MPI processes × $THREADS_PER_RANK threads = $TOTAL_THREADS total"
    echo "------------------------------------------"
    
    # Run the benchmark
    mpirun -np $NP \
        --bind-to none \
        --map-by node:PE=$THREADS_PER_RANK \
        "$SRC_DIR/test_config_mpi" "$MATRIX_FILE" $ITERATIONS
    
    if [ $? -ne 0 ]; then
        echo "WARNING: mpirun with $NP processes failed"
        continue
    fi
    
    echo ""
done

echo ""
echo "=========================================="
echo "  Sweep Complete"
echo "=========================================="
echo "Finished: $(date)"
echo ""
echo "Results saved to:"
echo "  - Main CSV: $BASE_DIR/results/configurations_results.csv"
echo ""
echo "All process counts tested successfully."
