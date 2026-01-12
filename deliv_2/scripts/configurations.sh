#!/bin/bash
#
# configurations.sh - Sweep MPI process counts for SpMV benchmark
#
# Usage: ./configurations.sh [matrix_file] [iterations]
# Example: ./configurations.sh matrices/36k_0p17.mtx 30
#
set -u

EXE="./test_config_mpi"
OUT_CSV="results/configurations_results.csv"
OUT_LOG="results/configurations.log"
TIMEOUT_SECS=3600      # 60 minutes per matrix combo

# Configuration
PROCESS_COUNTS=(2 4 8 16 24 32 48 64 72 84 96 108 128)
THREADS_PER_RANK=4
ITERATIONS=12
MATRIX_FILE=${1:-"matrices/36k_0p17.mtx"}

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$BASE_DIR/src"

# Navigate to base directory
cd "$BASE_DIR" || exit 1

# Create results directory
mkdir -p results

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

cd "$BASE_DIR"

# Verify executable exists
if [ ! -f "$EXE" ]; then
    echo "ERROR: executable not found at $(pwd)/$EXE" >&2
    exit 2
fi
if [ ! -x "$EXE" ]; then
    echo "ERROR: '$EXE' exists but is not executable." >&2
    exit 3
fi

# Verify matrix exists
if [ ! -f "$MATRIX_FILE" ]; then
    echo "ERROR: Matrix not found: $MATRIX_FILE" >&2
    exit 2
fi

# Set OpenMP threads
export OMP_NUM_THREADS=$THREADS_PER_RANK
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Get matrix basename for output
MATRIX_BASENAME=$(basename "$MATRIX_FILE" .mtx)

# Initialize log file
{
    echo "========================================================================"
    echo "MPI CONFIGURATION SWEEP BENCHMARK LOG"
    echo "========================================================================"
    echo "Started at: $(date)"
    echo "Executable: $EXE"
    echo "Timeout per run: ${TIMEOUT_SECS}s"
    echo "Iterations per mode: $ITERATIONS"
    echo "Process counts: ${PROCESS_COUNTS[*]}"
    echo "Matrix: $MATRIX_FILE"
    echo "Threads per rank: $THREADS_PER_RANK"
    echo "========================================================================"
    echo ""
} > "$OUT_LOG"

echo ""
echo "=========================================="
echo "  MPI Process Count Sweep Benchmark"
echo "=========================================="
echo "Matrix: $MATRIX_FILE"
echo "Threads per rank: $THREADS_PER_RANK"
echo "Iterations per config: $ITERATIONS"
echo "Process counts: ${PROCESS_COUNTS[*]}"
echo "Output CSV: $OUT_CSV"
echo "Output log: $OUT_LOG"
echo ""

total_runs=0
failed_runs=0

# Run benchmarks for each process count
for NP in "${PROCESS_COUNTS[@]}"; do
    TOTAL_THREADS=$((NP * THREADS_PER_RANK))
    
    echo "------------------------------------------"
    echo "Testing: $NP MPI processes × $THREADS_PER_RANK threads = $TOTAL_THREADS total"
    echo "------------------------------------------"
    echo "=== Testing: $NP processes ==" >> "$OUT_LOG"
    
    # Create temp files for output
    tmpout=$(mktemp)
    tmperr=$(mktemp)
    trap "rm -f $tmpout $tmperr" EXIT
    
    echo "  Running test... (timeout in ${TIMEOUT_SECS}s)"
    echo "  Start time: $(date)" >> "$OUT_LOG"
    
    # Run MPI test with timeout
    if [ -n "${PBS_NODEFILE:-}" ] && [ -f "${PBS_NODEFILE:-}" ]; then
        timeout "$TIMEOUT_SECS" mpirun -np "$NP" -hostfile "$PBS_NODEFILE" \
            --bind-to none --map-by node:PE=$THREADS_PER_RANK \
            "$EXE" "$MATRIX_FILE" "$ITERATIONS" > "$tmpout" 2> "$tmperr"
        exitcode=$?
    else
        timeout "$TIMEOUT_SECS" mpirun -np "$NP" \
            --bind-to none --map-by node:PE=$THREADS_PER_RANK \
            "$EXE" "$MATRIX_FILE" "$ITERATIONS" > "$tmpout" 2> "$tmperr"
        exitcode=$?
    fi
    
    echo "  Exit code: $exitcode" >> "$OUT_LOG"
    echo "  End time: $(date)" >> "$OUT_LOG"
    
    # Check for timeout or failure
    if [ "$exitcode" -eq 124 ]; then
        echo "  ERROR: Test TIMEOUT (exceeded ${TIMEOUT_SECS}s)"
        echo "  ERROR: Test TIMEOUT" >> "$OUT_LOG"
        failed_runs=$((failed_runs + 1))
    elif [ "$exitcode" -ne 0 ]; then
        echo "  ERROR: Test failed with exit code $exitcode"
        echo "  ERROR: Test failed with exit code $exitcode" >> "$OUT_LOG"
        failed_runs=$((failed_runs + 1))
    else
        echo "  SUCCESS: Test completed"
        echo "  SUCCESS: Test completed" >> "$OUT_LOG"
    fi
    
    # Log stderr if present
    if [ -s "$tmperr" ]; then
        echo "  Stderr output:" >> "$OUT_LOG"
        head -20 "$tmperr" | sed 's/^/    /' >> "$OUT_LOG"
    fi
    
    total_runs=$((total_runs + 1))
    
    # Cleanup
    rm -f "$tmpout" "$tmperr"
    
    echo ""
done

# Summary
echo ""
echo "========================================================================"
echo "BENCHMARK SUMMARY"
echo "========================================================================"
echo "Total runs: $total_runs"
echo "Successful: $((total_runs - failed_runs))"
echo "Failed: $failed_runs"
echo "Completed at: $(date)"
echo ""
echo "Results saved to: $OUT_CSV"

{
    echo ""
    echo "========================================================================"
    echo "BENCHMARK SUMMARY"
    echo "========================================================================"
    echo "Total runs: $total_runs"
    echo "Successful: $((total_runs - failed_runs))"
    echo "Failed: $failed_runs"
    echo "Completed at: $(date)"
} >> "$OUT_LOG"

# Exit with error if any tests failed
if [ "$failed_runs" -gt 0 ]; then
    echo "WARNING: $failed_runs test(s) failed. Check log: $OUT_LOG"
    exit 1
fi

exit 0
