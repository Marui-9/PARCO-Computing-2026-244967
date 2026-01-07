#!/bin/bash
# bench_mpi_6nodes.sh
# Benchmark MPI communication modes across 6 nodes
# Tests various inter-node communication strategies
# Usage: ./bench_mpi_6nodes.sh
set -u

EXE="./test_config_mpi"
OUT_BASE="results/test_config_mpi_results"
OUT_LOG="results/bench_mpi_6nodes.log"
TIMEOUT_SECS=3600      # 60 minutes per matrix combo
ITERATIONS=10          # 10 iterations per configuration

# Node counts to test - FIXED TO 6 NODES
NODE_COUNTS="6"

# Test matrices - comment/uncomment as needed
MATRICES_DIR="matrices"
TEST_MATRICES=(
    "36k_0p17.mtx"           # 36K rows
    "98k_0p52.mtx"           # 98K rows
    "262k_0p0011.mtx"        # 262K rows, very sparse
    "326k_0p0030.mtx"        # 326K rows, very sparse
    "916k_0p0006.mtx"        # 916K rows, very sparse
    "1135k_0p0002.mtx"       # 1135K rows, ultra sparse
    "1508k_0p0012.mtx"       # 1508K rows, ultra sparse
    "1585k_0p0002.mtx"       # 1585K rows, ultra sparse
    "3998k_0p0002.mtx"       # 3998K rows, ultra sparse
    "4194k_0p0001.mtx"       # 4194K rows, ultra sparse
    "5155k_0p0004.mtx"       # 5155K rows, ultra sparse
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
  echo "MPI COMMUNICATION MODES BENCHMARK LOG (6 NODES)"
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

echo "Starting MPI Communication Configurations Benchmark (6 NODES)"
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
    echo "PBS_NODEFILE: $PBS_NODEFILE" | tee -a "$OUT_LOG"
    echo "Node file contents:" | tee -a "$OUT_LOG"
    cat "$PBS_NODEFILE" | tee -a "$OUT_LOG"
    
    # Test mpirun with a simple hostname command
    echo "Testing mpirun with hostname..." | tee -a "$OUT_LOG"
    timeout 30 mpirun -np 6 -hostfile "$PBS_NODEFILE" hostname >> "$OUT_LOG" 2>&1 || {
        echo "WARNING: mpirun hostname test failed with exit code $?" | tee -a "$OUT_LOG"
    }
else
    echo "WARNING: PBS_NODEFILE not set, mpirun may not work correctly" | tee -a "$OUT_LOG"
fi
echo ""

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
        # Use hostfile if PBS_NODEFILE exists, otherwise just specify -np
        if [ -n "$PBS_NODEFILE" ] && [ -f "$PBS_NODEFILE" ]; then
            echo "  Using PBS_NODEFILE: $PBS_NODEFILE" >> "$OUT_LOG"
            echo "  Running: timeout $TIMEOUT_SECS mpirun -np $num_nodes -hostfile $PBS_NODEFILE $EXE $mtx_path $ITERATIONS" >> "$OUT_LOG"
            timeout "$TIMEOUT_SECS" mpirun -np "$num_nodes" -hostfile "$PBS_NODEFILE" "$EXE" "$mtx_path" "$ITERATIONS" \
                > "$tmpout" 2> "$tmperr"
            exitcode=$?
        else
            echo "  No PBS_NODEFILE, using -np only" >> "$OUT_LOG"
            echo "  Running: timeout $TIMEOUT_SECS mpirun -np $num_nodes $EXE $mtx_path $ITERATIONS" >> "$OUT_LOG"
            timeout "$TIMEOUT_SECS" mpirun -np "$num_nodes" "$EXE" "$mtx_path" "$ITERATIONS" \
                > "$tmpout" 2> "$tmperr"
            exitcode=$?
        fi
        
        echo "  Exit code: $exitcode" >> "$OUT_LOG"
        echo "  End time: $(date)" >> "$OUT_LOG"
        
        # Log all output for debugging
        if [ -s "$tmpout" ]; then
            echo "  Stdout output (first 50 lines):" >> "$OUT_LOG"
            head -50 "$tmpout" | sed 's/^/    /' >> "$OUT_LOG"
        fi
        
        # Log stderr
        if [ -s "$tmperr" ]; then
            echo "  Stderr output (first 50 lines):" >> "$OUT_LOG"
            head -50 "$tmperr" | sed 's/^/    /' >> "$OUT_LOG"
        fi
        
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
