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

# Matrices to test
MATRICES=(
    "matrices/36k_0p17.mtx"
    "matrices/98k_0p52.mtx"
    "matrices/262k_0p0011.mtx"
    "matrices/265k_0p0006.mtx"
    "matrices/326k_0p0030.mtx"
    "matrices/524k_0p0006.mtx"
    "matrices/916k_0p0006.mtx"
    "matrices/1438k_0p0016.mtx"
    "matrices/1508k_0p0012.mtx"
    "matrices/1585k_0p0002.mtx"
)

# Paths
EXE="./src/test_pipelined_mpi"
RESULTS_DIR="results"
SUMMARY_FILE="${RESULTS_DIR}/pipelined_job_summary.csv"

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

# Initialize summary file
echo "num_procs,threads_per_proc,total_threads,matrix,status,runtime_sec" > "$SUMMARY_FILE"

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
        
        # Output file for this configuration
        OUTPUT_FILE="${RESULTS_DIR}/${MATRIX_BASENAME}_${NP}procs_pipelined.txt"
        
        # Record start time
        START_TIME=$(date +%s)
        
        # Run the benchmark
        mpirun -np $NP "$EXE" "$MATRIX" $ITERATIONS 2>&1 | tee "$OUTPUT_FILE"
        
        RUN_STATUS=$?
        END_TIME=$(date +%s)
        RUNTIME=$((END_TIME - START_TIME))
        
        if [ $RUN_STATUS -eq 0 ]; then
            echo "$NP,$THREADS_PER_RANK,$TOTAL_THREADS,$MATRIX_BASENAME,success,$RUNTIME" >> "$SUMMARY_FILE"
            echo "✓ Completed in ${RUNTIME}s"
        else
            echo "$NP,$THREADS_PER_RANK,$TOTAL_THREADS,$MATRIX_BASENAME,failed,$RUNTIME" >> "$SUMMARY_FILE"
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
echo "Results saved to:"
echo "  - Main CSV: ${RESULTS_DIR}/pipelined_results.csv"
echo "  - Job summary: $SUMMARY_FILE"
echo ""
