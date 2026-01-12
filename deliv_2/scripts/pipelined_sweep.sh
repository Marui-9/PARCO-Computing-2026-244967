#!/bin/bash
#
# Script: pipelined_sweep.sh
# Purpose: Run pipelined MPI benchmarks for process counts 2-128
# Usage: bash scripts/pipelined_sweep.sh
#

set -e  # Exit on error
set -u  # Exit on undefined variable

# Configuration
PROCESS_COUNTS=(2 4 8 16 24 32 48 64 72 84 96 108 128)
THREADS_PER_RANK=4
ITERATIONS=30

# Matrices to test (all matrices in deliv_2/matrices/)
MATRICES=(
    "matrices/2k_0p22.mtx"
    "matrices/2k_0p52.mtx"
    "matrices/5k_9p37.mtx"
    "matrices/6k_6p3.mtx"
    "matrices/9k_0p77.mtx"
    "matrices/10k_1p5.mtx"
    "matrices/11k_0p35.mtx"
    "matrices/11k_0p38.mtx"
    "matrices/15k_0p41.mtx"
    "matrices/20k_0p38.mtx"
    "matrices/25k_0p03.mtx"
    "matrices/30k_0p05.mtx"
    "matrices/40k_0p02.mtx"
    "matrices/50k_0p008.mtx"
    "matrices/60k_0p005.mtx"
)

# Paths
EXE="./src/test_pipelined_mpi"
RESULTS_DIR="results"

# Create results directory
mkdir -p "$RESULTS_DIR"

# Set OpenMP environment
export OMP_NUM_THREADS=$THREADS_PER_RANK
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "=============================================="
echo "  Pipelined MPI Process Count Sweep"
echo "=============================================="
echo "Started: $(date)"
echo ""
echo "Configuration:"
echo "  Threads per MPI rank: $THREADS_PER_RANK"
echo "  Iterations per test: $ITERATIONS"
echo "  Process counts: ${PROCESS_COUNTS[*]}"
echo "  Matrices: ${#MATRICES[@]} total"
echo ""

# Check if executable exists
if [ ! -f "$EXE" ]; then
    echo "Executable not found. Compiling..."
    cd src
    mpicc -g -Wall -O3 -fopenmp -o test_pipelined_mpi \
        test_pipelined_mpi.c generator.c m_to_csr.c -lm
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Compilation failed"
        exit 1
    fi
    echo "Compilation successful."
    cd ..
fi

# Run benchmarks
for MATRIX in "${MATRICES[@]}"; do
    MATRIX_BASENAME=$(basename "$MATRIX" .mtx)
    
    # Check if matrix exists
    if [ ! -f "$MATRIX" ]; then
        echo "WARNING: Matrix $MATRIX not found, skipping..."
        continue
    fi
    
    echo "=============================================="
    echo "  Matrix: $MATRIX"
    echo "=============================================="
    
    for NP in "${PROCESS_COUNTS[@]}"; do
        TOTAL_THREADS=$((NP * THREADS_PER_RANK))
        
        echo ""
        echo "----------------------------------------------"
        echo "Testing: $NP MPI ranks × $THREADS_PER_RANK threads = $TOTAL_THREADS total"
        echo "----------------------------------------------"
        
        # Run the benchmark
        mpirun -np $NP "$EXE" "$MATRIX" $ITERATIONS
        
        RUN_STATUS=$?
        
        if [ $RUN_STATUS -eq 0 ]; then
            echo "✓ Completed successfully"
        else
            echo "✗ FAILED (exit code: $RUN_STATUS)"
        fi
        
        echo ""
    done
done

echo ""
echo "=============================================="
echo "  Benchmark Complete"
echo "=============================================="
echo "Finished: $(date)"
echo ""
echo "Results saved to: ${RESULTS_DIR}/pipelined_results.csv"
echo ""
