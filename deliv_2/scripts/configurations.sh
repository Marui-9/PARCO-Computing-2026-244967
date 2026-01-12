#!/bin/bash
#
# configurations.sh - Sweep MPI process counts for SpMV benchmark
#
# Usage: ./configurations.sh [matrix_file] [iterations]
# Example: ./configurations.sh matrices/36k_0p17.mtx 30
#

# Configuration
PROCESS_COUNTS=(2 4 8 16 24 32 48 64 72 84 96 108 128)
THREADS_PER_RANK=4
ITERATIONS=${2:-30}
MATRIX_FILE=${1:-"matrices/36k_0p17.mtx"}

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$BASE_DIR/src"
RESULTS_DIR="$BASE_DIR/results/configurations_sweep"

# Create results directory
mkdir -p "$RESULTS_DIR"

# Compile if needed
cd "$SRC_DIR"
if [ ! -f "test_config_mpi" ] || [ "test_configurations_mpi.c" -nt "test_config_mpi" ]; then
    echo "Compiling test_configurations_mpi.c..."
    mpicc -g -Wall -O3 -fopenmp -o test_config_mpi \
        test_configurations_mpi.c generator.c m_to_csr.c -lm
    if [ $? -ne 0 ]; then
        echo "ERROR: Compilation failed"
        exit 1
    fi
fi

# Set OpenMP threads
export OMP_NUM_THREADS=$THREADS_PER_RANK
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Summary CSV file
SUMMARY_FILE="$RESULTS_DIR/process_sweep_summary.csv"
echo "num_procs,threads_per_proc,total_threads,matrix,config_name,avg_time_ms,std_dev_ms,comm_time_ms,compute_time_ms,speedup,efficiency_pct" > "$SUMMARY_FILE"

# Get matrix basename for output
MATRIX_BASENAME=$(basename "$MATRIX_FILE" .mtx)

echo ""
echo "=========================================="
echo "  MPI Process Count Sweep Benchmark"
echo "=========================================="
echo "Matrix: $MATRIX_FILE"
echo "Threads per rank: $THREADS_PER_RANK"
echo "Iterations per config: $ITERATIONS"
echo "Process counts: ${PROCESS_COUNTS[*]}"
echo ""

# Run benchmarks for each process count
for NP in "${PROCESS_COUNTS[@]}"; do
    TOTAL_THREADS=$((NP * THREADS_PER_RANK))
    
    echo "------------------------------------------"
    echo "Testing: $NP MPI processes × $THREADS_PER_RANK threads = $TOTAL_THREADS total"
    echo "------------------------------------------"
    
    # Create output file for this configuration
    OUTPUT_FILE="$RESULTS_DIR/results_${NP}procs_${MATRIX_BASENAME}.txt"
    
    # Run the benchmark
    mpirun -np $NP \
        --bind-to none \
        --map-by node:PE=$THREADS_PER_RANK \
        "$SRC_DIR/test_config_mpi" "$MATRIX_FILE" $ITERATIONS 2>&1 | tee "$OUTPUT_FILE"
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "WARNING: mpirun with $NP processes failed"
        continue
    fi
    
    echo ""
done

echo ""
echo "=========================================="
echo "  Sweep Complete"
echo "=========================================="
echo "Results saved to: $RESULTS_DIR"
echo ""

# Consolidate all CSV results
echo "Consolidating CSV results..."
CONSOLIDATED_CSV="$RESULTS_DIR/all_configurations_results.csv"
FIRST=1
for CSV_FILE in "$RESULTS_DIR"/../test_results_Xnodes/test_config_mpi_results_*nodes.csv; do
    if [ -f "$CSV_FILE" ]; then
        if [ $FIRST -eq 1 ]; then
            cat "$CSV_FILE" > "$CONSOLIDATED_CSV"
            FIRST=0
        else
            tail -n +2 "$CSV_FILE" >> "$CONSOLIDATED_CSV"
        fi
    fi
done

if [ -f "$CONSOLIDATED_CSV" ]; then
    echo "Consolidated results: $CONSOLIDATED_CSV"
fi

echo "Done."
