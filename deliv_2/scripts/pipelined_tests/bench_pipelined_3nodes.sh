#!/bin/bash
# bench_pipelined_3nodes.sh
set -u
EXE="./test_pipelined_mpi"
OUT_BASE="results/pipelined_results/test_pipelined_mpi_results"
OUT_LOG="results/bench_pipelined_3nodes.log"
TIMEOUT_SECS=3600
ITERATIONS=30
NODE_COUNTS="3"
MATRICES_DIR="matrices"
TEST_MATRICES=("36k_0p17.mtx" "98k_0p52.mtx" "262k_0p0011.mtx" "265k_0p0006.mtx" "326k_0p0030.mtx" "524k_0p0006.mtx" "916k_0p0006.mtx" "1438k_0p0016.mtx" "1508k_0p0012.mtx" "1585k_0p0002.mtx")
[ ! -d "$MATRICES_DIR" ] && { echo "ERROR: matrices/ not found" >&2; exit 1; }
for mtx in "${TEST_MATRICES[@]}"; do [ ! -f "$MATRICES_DIR/$mtx" ] && { echo "ERROR: $mtx not found" >&2; exit 2; }; done
[ ! -f "$EXE" ] && { echo "ERROR: $EXE not found" >&2; exit 2; }
[ ! -x "$EXE" ] && { echo "ERROR: $EXE not executable" >&2; exit 3; }
mkdir -p results/pipelined_results
{ echo "========================================================================"; echo "PIPELINED MPI BENCHMARK LOG (3 NODES)"; echo "========================================================================"; echo "Started: $(date)"; echo "========================================================================"; echo ""; } > "$OUT_LOG"
trap 'echo "Interrupted at $(date)" >> "$OUT_LOG"' INT TERM
echo "Starting Pipelined MPI Benchmark (3 NODES)"
total_runs=0; failed_runs=0; timeout_runs=0
for num_nodes in $NODE_COUNTS; do
    OUT_CSV="${OUT_BASE}_${num_nodes}nodes.csv"
    echo "Testing with $num_nodes nodes"
    [ ! -f "$OUT_CSV" ] && echo "num_nodes,matrix,rows,cols,nnz,density_pct,config_name,avg_time_ms,std_dev_ms,min_time_ms,max_time_ms,comm_time_ms,compute_time_ms,speedup,efficiency_pct,iterations,notes" > "$OUT_CSV"
    for mtx in "${TEST_MATRICES[@]}"; do
        mtx_path="$MATRICES_DIR/$mtx"
        echo "  Matrix: $mtx"
        [ ! -f "$mtx_path" ] && continue
        cmd="timeout ${TIMEOUT_SECS} mpirun -np ${num_nodes} $EXE $mtx_path $ITERATIONS"
        { echo ""; echo "Running: $cmd"; echo "Started: $(date)"; } >> "$OUT_LOG"
        if eval "$cmd" >> "$OUT_LOG" 2>&1; then
            echo "    ✓ Success"; ((total_runs++))
        else
            exit_code=$?
            if [ $exit_code -eq 124 ]; then echo "    ✗ TIMEOUT"; ((timeout_runs++)); else echo "    ✗ FAILED"; ((failed_runs++)); fi
        fi
        echo "Completed: $(date)" >> "$OUT_LOG"
    done
done
echo ""; echo "COMPLETED: $total_runs runs | Failed: $failed_runs | Timeout: $timeout_runs"
{ echo "COMPLETED at $(date)"; echo "Total: $total_runs | Failed: $failed_runs | Timeout: $timeout_runs"; } >> "$OUT_LOG"
exit 0
