#!/bin/bash
# bench_pipelined_1nodes.sh
# Benchmark pipelined MPI communication mode across 1 node
# Tests pipelined chunked x distribution
# Usage: ./bench_pipelined_1nodes.sh
set -u

EXE="./test_pipelined_mpi"
OUT_BASE="results/pipelined_results/test_pipelined_mpi_results"
OUT_LOG="results/bench_pipelined_1nodes.log"
TIMEOUT_SECS=3600
ITERATIONS=30

NODE_COUNTS="1"

MATRICES_DIR="matrices"
TEST_MATRICES=(
    "36k_0p17.mtx"
    "98k_0p52.mtx"
    "262k_0p0011.mtx"
    "265k_0p0006.mtx"
    "326k_0p0030.mtx"
    "524k_0p0006.mtx"
    "916k_0p0006.mtx"
    "1438k_0p0016.mtx"
    "1508k_0p0012.mtx"
    "1585k_0p0002.mtx"
)

if [ ! -d "$MATRICES_DIR" ]; then
  echo "ERROR: matrices/ directory not found at $(pwd)/$MATRICES_DIR" >&2
  exit 1
fi

echo "Verifying test matrices exist..."
for mtx in "${TEST_MATRICES[@]}"; do
    if [ ! -f "$MATRICES_DIR/$mtx" ]; then
        echo "ERROR: Matrix not found: $MATRICES_DIR/$mtx" >&2
        exit 2
    fi
done
echo "All ${#TEST_MATRICES[@]} test matrices verified."

if [ ! -f "$EXE" ]; then
  echo "ERROR: executable not found at $(pwd)/$EXE" >&2
  exit 2
fi
if [ ! -x "$EXE" ]; then
  echo "ERROR: '$EXE' exists but is not executable." >&2
  exit 3
fi

mkdir -p results/pipelined_results

{
  echo "========================================================================"
  echo "PIPELINED MPI COMMUNICATION MODE BENCHMARK LOG (1 NODE)"
  echo "========================================================================"
  echo "Started at: $(date)"
  echo "Executable: $EXE"
  echo "Timeout per run: ${TIMEOUT_SECS}s"
  echo "Iterations per mode: $ITERATIONS"
  echo "Node counts: $NODE_COUNTS"
  echo "Test matrices: ${#TEST_MATRICES[@]}"
  echo "========================================================================"
  echo ""
} > "$OUT_LOG"

trap 'echo "Benchmark interrupted at $(date)" >> "$OUT_LOG"' INT TERM

echo "Starting Pipelined MPI Benchmark (1 NODE)"
echo "Executable: $EXE"
echo "Timeout: ${TIMEOUT_SECS}s per run"
echo "Iterations: $ITERATIONS"
echo "Output log: $OUT_LOG"
echo ""

total_runs=0
failed_runs=0
timeout_runs=0

for num_nodes in $NODE_COUNTS; do
    OUT_CSV="${OUT_BASE}_${num_nodes}nodes.csv"
    
    echo "========================================================================"
    echo "Testing with $num_nodes MPI rank"
    echo "========================================================================"
    
    if [ ! -f "$OUT_CSV" ]; then
        echo "num_nodes,matrix,rows,cols,nnz,density_pct,config_name,avg_time_ms,std_dev_ms,min_time_ms,max_time_ms,comm_time_ms,compute_time_ms,speedup,efficiency_pct,iterations,notes" > "$OUT_CSV"
    fi
    
    for mtx in "${TEST_MATRICES[@]}"; do
        mtx_path="$MATRICES_DIR/$mtx"
        
        echo "  Matrix: $mtx"
        echo "  Matrix: $mtx" >> "$OUT_LOG"
        
        if [ ! -f "$mtx_path" ]; then
            echo "    ERROR: Matrix file not found"
            continue
        fi
        
        cmd="timeout ${TIMEOUT_SECS} mpirun -np ${num_nodes} $EXE $mtx_path $ITERATIONS"
        
        {
            echo ""
            echo "Running: $cmd"
            echo "Started at: $(date)"
        } >> "$OUT_LOG"
        
        if eval "$cmd" >> "$OUT_LOG" 2>&1; then
            echo "    ✓ Success"
            ((total_runs++))
        else
            exit_code=$?
            if [ $exit_code -eq 124 ]; then
                echo "    ✗ TIMEOUT"
                ((timeout_runs++))
            else
                echo "    ✗ FAILED (exit code: $exit_code)"
                ((failed_runs++))
            fi
        fi
        
        echo "Completed at: $(date)" >> "$OUT_LOG"
    done
done

echo ""
echo "========================================================================"
echo "BENCHMARK COMPLETED"
echo "Total runs: $total_runs | Failed: $failed_runs | Timeout: $timeout_runs"
echo "Results: results/pipelined_results/"
echo "========================================================================"

{
    echo "========================================================================"
    echo "COMPLETED at $(date)"
    echo "Total: $total_runs | Failed: $failed_runs | Timeout: $timeout_runs"
} >> "$OUT_LOG"

exit 0
