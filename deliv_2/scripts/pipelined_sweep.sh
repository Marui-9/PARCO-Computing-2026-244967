#!/bin/bash
#
# Script: pipelined_sweep.sh
# Purpose: Run pipelined MPI benchmarks for process counts 2-128
# Usage: bash scripts/pipelined_sweep.sh
#

set -e  # Exit on error
set -u  # Exit on undefined variable

# Configuration
PROCESS_COUNTS=(2 4 8 16 32 64 96 128)
THREADS_PER_RANK=4
ITERATIONS=12

# Matrices to test (all matrices in deliv_2/matrices/)
MATRICES=(
    "1438k_0p0016.mtx" 
     "1565k_0p0024.mtx"  
     "41292k_0p0001.mtx"  
     "916k_0p0006.mtx"
     "1508k_0p0012.mtx"  
     "2097k_0p0001.mtx"  
     "4848k_0p0003.mtx"
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
