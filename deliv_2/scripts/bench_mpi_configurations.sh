#!/bin/bash
# bench_mpi_configurations.sh
# Benchmark MPI communication modes across nodes (2, 3, 4)
# Tests various inter-node communication strategies
# Usage: ./bench_mpi_configurations.sh
set -eu

EXE="./test_config_mpi"
OUT_BASE="results/test_config_mpi_results"
OUT_LOG="results/bench_mpi_configurations.log"
TIMEOUT_SECS=1800      # 30 minutes per node count + matrix combo
ITERATIONS=10          # 10 iterations per configuration

# Node counts to test
NODE_COUNTS="2 3 4"

# Test matrices - comment/uncomment as needed
# Small matrices: quick validation
MATRICES_DIR="matrices"
TEST_MATRICES=(
    # === Small matrices (for quick testing) ===
    "2k_0p22.mtx"      # 2K rows
    "2k_0p52.mtx"      # 2K rows
    "5k_9p37.mtx"      # 5K rows
    "6k_6p3.mtx"       # 6K rows
    "9k_0p77.mtx"      # 9K rows
    "10k_1p5.mtx"      # 10K rows
    "11k_0p35.mtx"     # 11K rows, sparse - quick baseline
    "11k_0p38.mtx"     # 11K rows
    "15k_0p41.mtx"     # 15K rows
    
    # === Medium matrices (moderate workload) ===
    "20k_0p38.mtx"     # 20K rows - medium workload
    "25k_0p03.mtx"     # 25K rows, very sparse
    "30k_0p05.mtx"     # 30K rows, very sparse
    "40k_0p02.mtx"     # 40K rows, very sparse
    
    # === Large matrices (heavy workload) ===
    "50k_0p008.mtx"    # 50K rows, very sparse - large but light computation
    "60k_0p005.mtx"    # 60K rows, very sparse
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
  echo "Compile with: mpicc -O3 -fopenmp -o test_config_mpi \\" >&2
  echo "  src/test_configurations_mpi.c src/generator.c src/m_to_csr.c -lm" >&2
  exit 2
fi
if [ ! -x "$EXE" ]; then
  echo "ERROR: '$EXE' exists but is not executable. Try 'chmod +x $EXE'." >&2
  exit 3
fi

# Create output directory
mkdir -p results

# Initialize log file
{
  echo "========================================================================"
  echo "MPI COMMUNICATION MODES BENCHMARK LOG"
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

echo "Starting MPI Communication Configurations Benchmark"
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

# Loop over node counts
for num_nodes in $NODE_COUNTS; do
    OUT_CSV="${OUT_BASE}_${num_nodes}nodes.csv"
    
    echo ""
    echo "========================================================================"
    echo "Testing with $num_nodes MPI ranks (nodes)"
    echo "========================================================================"
    echo "=== Testing with $num_nodes MPI ranks ===" >> "$OUT_LOG"
    echo "Output CSV: $OUT_CSV" | tee -a "$OUT_LOG"
    echo ""
    
    # Create or clear CSV for this node count
    if [ ! -f "$OUT_CSV" ]; then
        echo "Creating new CSV: $OUT_CSV"
        {
            echo "num_nodes,matrix,rows,cols,nnz,density_pct,config_name,"
            echo "avg_time_ms,std_dev_ms,min_time_ms,max_time_ms,"
            echo "comm_time_ms,compute_time_ms,speedup,efficiency_pct,iterations,notes"
        } | paste -sd',' - > "$OUT_CSV"
    fi
    
    # Loop over test matrices
    for mtx in "${TEST_MATRICES[@]}"; do
        mtx_path="$MATRICES_DIR/$mtx"
        
        echo "=== Matrix: $mtx (nodes=$num_nodes) ==="
        echo "=== Matrix: $mtx (nodes=$num_nodes) ===" >> "$OUT_LOG"
        echo "  File: $mtx_path"
        echo "  Iterations: $ITERATIONS"
        
        if [ ! -f "$mtx_path" ]; then
            echo "  ERROR: Matrix file not found: $mtx_path"
            echo "  ERROR: Matrix file not found: $mtx_path" >> "$OUT_LOG"
            continue
        fi
        
        # Create temp files for output
        tmpout=$(mktemp)
        tmperr=$(mktemp)
        trap "rm -f $tmpout $tmperr" EXIT
        
        echo "  Running test... (timeout in ${TIMEOUT_SECS}s)"
        echo "  Start time: $(date)" >> "$OUT_LOG"
        
        # Run MPI test with timeout
        # Note: Pass matrix path relative to current directory
        # The C program will open it as-is or search in matrices/
        timeout "$TIMEOUT_SECS" mpirun -np "$num_nodes" "$EXE" "$mtx_path" "$ITERATIONS" \
            > "$tmpout" 2> "$tmperr"
        exitcode=$?
        
        echo "  End time: $(date)" >> "$OUT_LOG"
        echo "  Exit code: $exitcode" >> "$OUT_LOG"
        
        # Check for timeout
        if [ "$exitcode" -eq 124 ]; then
            echo "  ERROR: Test TIMEOUT (exceeded ${TIMEOUT_SECS}s)"
            echo "  ERROR: Test TIMEOUT" >> "$OUT_LOG"
            timeout_runs=$((timeout_runs + 1))
            failed_runs=$((failed_runs + 1))
        elif [ "$exitcode" -ne 0 ]; then
            echo "  ERROR: Test failed with exit code $exitcode"
            echo "  ERROR: Test failed with exit code $exitcode" >> "$OUT_LOG"
            failed_runs=$((failed_runs + 1))
            
            # Log stderr
            if [ -s "$tmperr" ]; then
                echo "  stderr output:" >> "$OUT_LOG"
                head -20 "$tmperr" | sed 's/^/    /' >> "$OUT_LOG"
            fi
        else
            echo "  SUCCESS: Test completed"
            echo "  SUCCESS: Test completed" >> "$OUT_LOG"
            
            # Log output and CSV was written by C program
            if [ -s "$tmpout" ]; then
                echo "  Test output:" >> "$OUT_LOG"
                head -30 "$tmpout" | sed 's/^/    /' >> "$OUT_LOG"
            fi
        fi
        
        # Log any errors
        if [ -s "$tmperr" ]; then
            echo "  stderr size: $(wc -c < $tmperr) bytes" >> "$OUT_LOG"
            head -10 "$tmperr" >> "$OUT_LOG" 2>/dev/null || true
        fi
        
        total_runs=$((total_runs + 1))
        
        # Sync filesystem
        sync
        sleep 1
        
        # Cleanup
        rm -f "$tmpout" "$tmperr"
    done
done

# Summary
echo ""
echo "========================================================================"
echo "BENCHMARK SUMMARY"
echo "========================================================================"
echo "Total runs: $total_runs"
echo "Successful: $((total_runs - failed_runs))"
echo "Failed: $failed_runs"
echo "Timeouts: $timeout_runs"
echo "Timestamp: $(date)"
echo ""
echo "Output CSVs:"
for num_nodes in $NODE_COUNTS; do
    OUT_CSV="${OUT_BASE}_${num_nodes}nodes.csv"
    if [ -f "$OUT_CSV" ]; then
        lines=$(wc -l < "$OUT_CSV")
        echo "  - $OUT_CSV ($lines lines)"
    fi
done

{
    echo ""
    echo "========================================================================"
    echo "BENCHMARK SUMMARY"
    echo "========================================================================"
    echo "Total runs: $total_runs"
    echo "Successful: $((total_runs - failed_runs))"
    echo "Failed: $failed_runs"
    echo "Timeouts: $timeout_runs"
    echo "Completed at: $(date)"
} >> "$OUT_LOG"

# Exit with error if any tests failed
if [ "$failed_runs" -gt 0 ]; then
    echo "WARNING: $failed_runs test(s) failed. Check log: $OUT_LOG"
    exit 1
fi

exit 0
