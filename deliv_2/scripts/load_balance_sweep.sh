#!/bin/bash
#
# Load balance strategy sweep benchmark (local execution)
# Tests: ROW-BASED, NNZ-BASED, HYBRID-0.5, HYBRID-0.7
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "================================================================================"
echo "  LOAD BALANCE STRATEGY SWEEP BENCHMARK (Local Execution)"
echo "================================================================================"
echo ""
echo "This script benchmarks load balancing strategies by:"
echo "  1. Testing 4 strategies: ROW-BASED, NNZ-BASED, HYBRID (α=0.5), HYBRID (α=0.7)"
echo "  2. Sweeping MPI process counts: 2, 4, 8, 16, 24, 32, 48, 64, 72, 84, 96, 108, 128"
echo "  3. Testing all matrices in deliv_2/matrices/"
echo "  4. Using 4 threads per MPI rank (optimal from configurations)"
echo ""
echo "================================================================================"
echo ""

# Compile the benchmark
echo "Compiling load balance sweep benchmark..."
mpicc -std=c99 -g -Wall -O3 -fopenmp -o test_lb_sweep \
    src/test_load_balance_sweep.c \
    src/load_balance.c \
    src/generator.c \
    src/m_to_csr.c \
    -lm

if [ $? -ne 0 ]; then
    echo "ERROR: Compilation failed"
    exit 1
fi

echo "✓ Compilation successful"
echo ""

# Create results directory
mkdir -p results
rm -f results/load_balance_results.csv

# Configuration
PROCESS_COUNTS=(2 4 8 16 24 32 48 64 72 84 96 108 128)
THREADS_PER_RANK=4
ITERATIONS=30
TIMEOUT=3600  # 1 hour per test

# Set OpenMP environment
export OMP_NUM_THREADS=$THREADS_PER_RANK
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_STACKSIZE=64M

# Find all matrices
MATRICES=($(find matrices -name "*.mtx" -type f | sort))

if [ ${#MATRICES[@]} -eq 0 ]; then
    echo "ERROR: No matrix files found in matrices/"
    exit 1
fi

echo "Test Configuration:"
echo "  Matrices found: ${#MATRICES[@]}"
for m in "${MATRICES[@]}"; do
    echo "    - $m"
done
echo ""
echo "  Process counts: ${PROCESS_COUNTS[*]}"
echo "  Threads/rank: $THREADS_PER_RANK"
echo "  Iterations: $ITERATIONS"
echo "  Timeout: ${TIMEOUT}s per test"
echo "  Strategies: ROW-BASED, NNZ-BASED, HYBRID-0.5, HYBRID-0.7"
echo ""

# Logging
LOG_FILE="results/load_balance_sweep.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Starting sweep at $(date)"
echo ""

# Counters
TOTAL_TESTS=0
SUCCESSFUL_TESTS=0
FAILED_TESTS=0
TIMEOUT_TESTS=0

# Run benchmarks
for MATRIX in "${MATRICES[@]}"; do
    MATRIX_BASENAME=$(basename "$MATRIX" .mtx)
    
    echo "=============================================="
    echo "  Matrix: $MATRIX"
    echo "=============================================="
    
    for NP in "${PROCESS_COUNTS[@]}"; do
        TOTAL_TESTS=$((TOTAL_TESTS + 1))
        TOTAL_THREADS=$((NP * THREADS_PER_RANK))
        
        echo ""
        echo "----------------------------------------------"
        echo "Test $TOTAL_TESTS: $NP ranks × $THREADS_PER_RANK threads = $TOTAL_THREADS total"
        echo "----------------------------------------------"
        
        # Create temp file for output
        TEMP_OUT=$(mktemp)
        
        # Run with timeout
        timeout $TIMEOUT mpirun -np $NP \
            --bind-to none \
            ./test_lb_sweep "$MATRIX" $ITERATIONS > "$TEMP_OUT" 2>&1
        
        EXIT_CODE=$?
        
        if [ $EXIT_CODE -eq 0 ]; then
            SUCCESSFUL_TESTS=$((SUCCESSFUL_TESTS + 1))
            echo "✓ Completed successfully"
            cat "$TEMP_OUT"
        elif [ $EXIT_CODE -eq 124 ]; then
            TIMEOUT_TESTS=$((TIMEOUT_TESTS + 1))
            echo "✗ TIMEOUT after ${TIMEOUT}s"
            echo "Partial output:"
            tail -20 "$TEMP_OUT"
        else
            FAILED_TESTS=$((FAILED_TESTS + 1))
            echo "✗ FAILED with exit code $EXIT_CODE"
            echo "Error output:"
            tail -20 "$TEMP_OUT"
        fi
        
        rm -f "$TEMP_OUT"
        echo ""
    done
done

echo ""
echo "================================================================================"
echo "  SWEEP COMPLETE"
echo "================================================================================"
echo "Finished: $(date)"
echo ""
echo "Summary:"
echo "  Total tests:      $TOTAL_TESTS"
echo "  Successful:       $SUCCESSFUL_TESTS"
echo "  Failed:           $FAILED_TESTS"
echo "  Timeouts:         $TIMEOUT_TESTS"
echo ""
echo "Results saved to:"
echo "  CSV:  results/load_balance_results.csv"
echo "  Log:  results/load_balance_sweep.log"
echo ""

if [ -f results/load_balance_results.csv ]; then
    RESULT_LINES=$(wc -l < results/load_balance_results.csv)
    echo "Result lines in CSV: $RESULT_LINES (including header)"
    echo ""
    
    # Quick analysis
    echo "Quick Analysis:"
    echo "  Unique matrices tested:"
    tail -n +2 results/load_balance_results.csv | cut -d',' -f1 | sort -u | wc -l
    echo "  Unique process counts:"
    tail -n +2 results/load_balance_results.csv | cut -d',' -f2 | sort -u | wc -l
    echo ""
    
    echo "To analyze results:"
    echo "  - Best speedup: sort -t',' -k10 -rn results/load_balance_results.csv | head -10"
    echo "  - NNZ-BASED only: grep NNZ-BASED results/load_balance_results.csv"
    echo "  - Specific matrix: grep '36k_0p17' results/load_balance_results.csv"
    echo ""
fi
