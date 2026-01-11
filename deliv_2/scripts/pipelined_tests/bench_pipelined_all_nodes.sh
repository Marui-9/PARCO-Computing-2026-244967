#!/bin/bash
# bench_pipelined_all_nodes.sh
# Benchmark pipelined MPI communication mode across 2-8 nodes
# Tests single executable with increasing process counts
# Usage: ./bench_pipelined_all_nodes.sh
# This script is designed to run within a PBS job that allocates 8 nodes
set -u

EXE="./test_pipelined_mpi"
OUT_BASE="results/pipelined_results/test_pipelined_mpi_results"
OUT_LOG="results/pipelined_all_nodes.log"
TIMEOUT_SECS=3600      # 60 minutes per matrix combo
ITERATIONS=30          # Iterations per configuration for statistics

# Node counts to test - will run 2 through 8
NODE_COUNTS="2 3 4 5 6 7 8"

# Test matrices - same as baseline tests
MATRICES_DIR="matrices"
TEST_MATRICES=(
    "36k_0p17.mtx"           # 36K rows
    "98k_0p52.mtx"           # 98K rows
    "262k_0p0011.mtx"        # 262K rows, very sparse
    "265k_0p0006.mtx"        # 265K rows, sparse
    "326k_0p0030.mtx"        # 326K rows, sparse
    "524k_0p0006.mtx"        # 524K rows, sparse
    "916k_0p0006.mtx"        # 916K rows, very sparse
    "1438k_0p0016.mtx"       # 1438K rows, ultra sparse
    "1508k_0p0012.mtx"       # 1508K rows, ultra sparse
    "1585k_0p0002.mtx"       # 1585K rows, ultra sparse
)

if [ ! -d "$MATRICES_DIR" ]; then
  echo "ERROR: matrices/ directory not found at $(pwd)/$MATRICES_DIR" >&2
  exit 1
fi

# Check that test matrices exist
echo "Verifying test matrices exist..."
for mtx in "${TEST_MATRICES[@]}"; do
    if [ ! -f "$MATRICES_DIR/$mtx" ]; then
        echo "ERROR: Matrix not found: $MATRICES_DIR/$mtx" >&2
        exit 2
    fi
done
echo "All ${#TEST_MATRICES[@]} test matrices verified."

# Verify executable exists
if [ ! -f "$EXE" ]; then
  echo "ERROR: executable not found at $(pwd)/$EXE" >&2
  echo "Compile with: mpicc -O3 -fopenmp -o test_pipelined_mpi \\" >&2
  echo "  src/test_pipelined_mpi.c src/generator.c src/m_to_csr.c -lm" >&2
  exit 2
fi
if [ ! -x "$EXE" ]; then
  echo "ERROR: '$EXE' exists but is not executable. Try 'chmod +x $EXE'." >&2
  exit 3
fi

# Create output directories
mkdir -p results/pipelined_results

# Initialize log file
{
  echo "========================================================================"
  echo "PIPELINED MPI COMMUNICATION MODE BENCHMARK LOG (2-8 NODES)"
  echo "========================================================================"
  echo "Started at: $(date)"
  echo "Executable: $EXE"
  echo "Timeout per run: ${TIMEOUT_SECS}s"
  echo "Iterations per mode: $ITERATIONS"
  echo "Node counts: $NODE_COUNTS"
  echo "Test matrices: ${#TEST_MATRICES[@]}"
  echo "  - ${TEST_MATRICES[*]}"
  echo "========================================================================"
  echo ""
} > "$OUT_LOG"

trap 'echo "Benchmark interrupted at $(date)" >> "$OUT_LOG"' INT TERM

echo "Starting Pipelined MPI Benchmark (2-8 NODES)"
echo "Executable: $EXE"
echo "Timeout per run: ${TIMEOUT_SECS}s"
echo "Iterations per mode: $ITERATIONS"
echo "Node counts: $NODE_COUNTS"
echo "Test matrices: ${TEST_MATRICES[@]}"
echo "Output log: $OUT_LOG"
echo ""

total_runs=0
failed_runs=0
timeout_runs=0

# Test mpirun connectivity before benchmarking
echo ""
echo "=== Testing MPI connectivity ==="
echo "=== Testing MPI connectivity ===" >> "$OUT_LOG"
if [ -n "$PBS_NODEFILE" ] && [ -f "$PBS_NODEFILE" ]; then
    echo "PBS_NODEFILE: $PBS_NODEFILE"
    echo "Allocated nodes:"
    cat "$PBS_NODEFILE" | sort -u
    echo "PBS_NODEFILE: $PBS_NODEFILE" >> "$OUT_LOG"
    echo "Allocated nodes:" >> "$OUT_LOG"
    cat "$PBS_NODEFILE" | sort -u >> "$OUT_LOG"
fi

# Test with 2 processes to verify MPI is working
echo "Testing MPI with 2 processes..."
timeout 30 mpirun -np 2 /bin/hostname >/dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ MPI connectivity test passed"
    echo "✓ MPI connectivity test passed" >> "$OUT_LOG"
else
    echo "✗ MPI connectivity test failed (but continuing anyway)"
    echo "✗ MPI connectivity test failed" >> "$OUT_LOG"
fi
echo ""

# Main benchmark loop over node counts
for num_nodes in $NODE_COUNTS; do
    echo "========================================================================"
    echo "Testing with $num_nodes nodes (processes)"
    echo "========================================================================"
    echo "========================================================================"
    echo "Testing with $num_nodes nodes (processes)" | tee -a "$OUT_LOG"
    echo "========================================================================"
    echo ""
    
    # Test each matrix
    for matrix in "${TEST_MATRICES[@]}"; do
        matrix_name=$(basename "$matrix" .mtx)
        
        echo "  Matrix: $matrix ($matrix_name)"
        echo "  Matrix: $matrix ($matrix_name)" >> "$OUT_LOG"
        
        cmd="timeout ${TIMEOUT_SECS} mpirun -np ${num_nodes} $EXE $matrix $ITERATIONS"
        
        {
            echo ""
            echo "Running: $cmd"
            echo "Started at: $(date)"
        } >> "$OUT_LOG"
        
        # Run benchmark
        if eval "$cmd" >> "$OUT_LOG" 2>&1; then
            echo "    ✓ Success"
            ((total_runs++))
        else
            exit_code=$?
            if [ $exit_code -eq 124 ]; then
                echo "    ✗ TIMEOUT after ${TIMEOUT_SECS}s"
                echo "    TIMEOUT after ${TIMEOUT_SECS}s" >> "$OUT_LOG"
                ((timeout_runs++))
            else
                echo "    ✗ FAILED (exit code: $exit_code)"
                echo "    FAILED (exit code: $exit_code)" >> "$OUT_LOG"
                ((failed_runs++))
            fi
        fi
        
        {
            echo "Completed at: $(date)"
            echo ""
        } >> "$OUT_LOG"
    done
    
    echo ""
done

# Summary
echo "========================================================================"
echo "BENCHMARK COMPLETED"
echo "========================================================================"
echo "Total runs: $total_runs"
echo "Failed runs: $failed_runs"
echo "Timeout runs: $timeout_runs"
echo ""
echo "Results stored in: results/pipelined_results/"
echo ""
echo "To view results:"
echo "  cat results/pipelined_results/test_pipelined_mpi_results_*.csv"
echo ""
echo "========================================================================"

{
    echo "========================================================================"
    echo "BENCHMARK COMPLETED at $(date)"
    echo "========================================================================"
    echo "Total runs: $total_runs"
    echo "Failed runs: $failed_runs"
    echo "Timeout runs: $timeout_runs"
    echo ""
} >> "$OUT_LOG"

exit 0
