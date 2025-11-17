#!/bin/bash
# Script: bench_memory_optimizations.sh
# Purpose: Benchmark memory-level optimizations (Phase 3)
# Tests register blocking and software prefetching on representative matrices

set -e

EXECUTABLE="./test_memory_opt"
RESULTS_DIR="results"
OUTPUT_CSV="${RESULTS_DIR}/memory_optimizations_results.csv"
OUTPUT_TXT="${RESULTS_DIR}/memory_optimizations_results.txt"

# Representative matrices: focus on larger ones where memory bottleneck dominates
MATRICES=(
    "10k_1p5.mtx"
    "20k_0p38.mtx"
    "40k_0p02.mtx"
    "60k_0p005.mtx"
)

# Thread counts to test (focus on higher thread counts where bandwidth is saturated)
THREAD_COUNTS=(1 8 16 24)

ITERATIONS=30

echo "=== Phase 3: Memory-Level Optimization Benchmarking ===" | tee "$OUTPUT_TXT"
echo "Testing register blocking and software prefetching" | tee -a "$OUTPUT_TXT"
echo "Date: $(date)" | tee -a "$OUTPUT_TXT"
echo "Matrices: ${MATRICES[*]}" | tee -a "$OUTPUT_TXT"
echo "Thread counts: ${THREAD_COUNTS[*]}" | tee -a "$OUTPUT_TXT"
echo "Iterations: $ITERATIONS" | tee -a "$OUTPUT_TXT"
echo "" | tee -a "$OUTPUT_TXT"

# Check executable exists
if [ ! -f "$EXECUTABLE" ]; then
    echo "Error: $EXECUTABLE not found. Run 'make test_memory' first." >&2
    exit 1
fi

# Create results directory
mkdir -p "$RESULTS_DIR"

# CSV header
echo "Matrix,Threads,Config,Optimization,Time_ms,Speedup_vs_Serial,Speedup_vs_Phase1,Improvement_Percent" > "$OUTPUT_CSV"

# Run benchmarks
for matrix in "${MATRICES[@]}"; do
    matrix_file="matrices/$matrix"
    
    if [ ! -f "$matrix_file" ]; then
        echo "Warning: $matrix_file not found, skipping..." | tee -a "$OUTPUT_TXT"
        continue
    fi
    
    echo "============================================" | tee -a "$OUTPUT_TXT"
    echo "Matrix: $matrix" | tee -a "$OUTPUT_TXT"
    echo "============================================" | tee -a "$OUTPUT_TXT"
    
    for threads in "${THREAD_COUNTS[@]}"; do
        echo "" | tee -a "$OUTPUT_TXT"
        echo "--- Testing with $threads threads ---" | tee -a "$OUTPUT_TXT"
        
        # Run test and capture output
        output=$("$EXECUTABLE" "$threads" "$matrix" "$ITERATIONS" 2>&1 | tee -a "$OUTPUT_TXT")
        
        # Parse results from table output
        # Extract timing data (this is a simplified parser - adjust based on actual output format)
        while IFS= read -r line; do
            # Parse lines like: "║ RegBlock-4 + Baseline           ║   1.2345 ║  925.1x  ║  +12.34% ║ +12.34%║"
            if [[ $line =~ ║[[:space:]]*([^║]+)[[:space:]]*║[[:space:]]*([0-9.]+)[[:space:]]*║[[:space:]]*([0-9.]+)x[[:space:]]*║[[:space:]]*([+-]?[0-9.]+)%[[:space:]]*║[[:space:]]*([+-]?[0-9.]+)%[[:space:]]*║ ]]; then
                config="${BASH_REMATCH[1]}"
                time_ms="${BASH_REMATCH[2]}"
                speedup_serial="${BASH_REMATCH[3]}"
                speedup_phase1="${BASH_REMATCH[4]}"
                improvement="${BASH_REMATCH[5]}"
                
                # Clean up config name
                config=$(echo "$config" | xargs)
                
                # Determine optimization type
                if [[ $config == *"Baseline"* ]]; then
                    opt_type="Baseline"
                elif [[ $config == *"RegBlock"* ]] && [[ $config == *"Prefetch"* ]]; then
                    opt_type="Combined"
                elif [[ $config == *"RegBlock"* ]]; then
                    opt_type="Register_Blocking"
                elif [[ $config == *"Prefetch"* ]]; then
                    opt_type="Software_Prefetch"
                else
                    opt_type="Other"
                fi
                
                echo "$matrix,$threads,$config,$opt_type,$time_ms,$speedup_serial,$speedup_phase1,$improvement" >> "$OUTPUT_CSV"
            fi
        done <<< "$output"
        
        echo "" | tee -a "$OUTPUT_TXT"
    done
    
    echo "" | tee -a "$OUTPUT_TXT"
done

echo "============================================" | tee -a "$OUTPUT_TXT"
echo "Benchmarking complete!" | tee -a "$OUTPUT_TXT"
echo "Results saved to:" | tee -a "$OUTPUT_TXT"
echo "  - CSV: $OUTPUT_CSV" | tee -a "$OUTPUT_TXT"
echo "  - TXT: $OUTPUT_TXT" | tee -a "$OUTPUT_TXT"
echo "============================================" | tee -a "$OUTPUT_TXT"
