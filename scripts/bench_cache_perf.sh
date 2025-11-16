#!/bin/bash

# Benchmark script for cache performance analysis using perf stat
# Output: results/cache_perf_results.csv with cache performance metrics
# NOTE: Works with perf_event_paranoid=2 (user-space events only)

OUTPUT_CSV="results/cache_perf_results.csv"
EXECUTABLE="${EXECUTABLE:-./executable}"  # Use environment variable if set, otherwise default
TIMEOUT_SECONDS=60  # perf is much faster than valgrind

# Thread counts to test
THREAD_COUNTS=(1 4 8 12 16 24)

# Check if executable exists
if [ ! -f "$EXECUTABLE" ]; then
    echo "Error: $EXECUTABLE not found. Please compile first with 'make'."
    exit 1
fi

# Check if perf is available
if ! command -v perf &> /dev/null; then
    echo "Error: perf command not found. Please install perf."
    exit 1
fi

# Create results directory
mkdir -p results

# Create CSV header
echo "matrix,rows,cols,density_pct,threads,cycles,instructions,l1d_loads,l1d_misses,l1d_miss_rate,llc_loads,llc_misses,llc_miss_rate,branches,branch_misses,branch_miss_rate,time_ms,exit_code,notes" > "$OUTPUT_CSV"

echo "=== Cache Performance Benchmarking with perf stat ==="
echo "Output file: $OUTPUT_CSV"
echo "Note: Using user-space performance counters (perf_event_paranoid=2)"
echo ""

# Find all .mtx files in matrices/ directory, excluding those with two-digit numbers before 'k'
# This excludes patterns like 10k_, 11k_, 15k_, etc. to reduce runtime
MATRIX_FILES=$(find matrices/ -name "*.mtx" -type f | grep -v '/[0-9][0-9]k_' | sort)

if [ -z "$MATRIX_FILES" ]; then
    echo "Error: No .mtx files found in matrices/ directory (after filtering)"
    exit 1
fi

echo "Testing matrices (excluded two-digit prefixes like 10k_, 11k_, 15k_):"
echo "$MATRIX_FILES" | wc -l | xargs echo "  Total matrices:"
echo ""

TOTAL_COUNT=0
for mtx in $MATRIX_FILES; do
    for threads in "${THREAD_COUNTS[@]}"; do
        ((TOTAL_COUNT++))
    done
done

CURRENT=0

# Loop through each matrix file
for mtx in $MATRIX_FILES; do
    # Extract just the filename (without path)
    mtx_basename=$(basename "$mtx")
    
    echo "Matrix: $mtx_basename"
    
    # Loop through different thread counts
    for threads in "${THREAD_COUNTS[@]}"; do
        ((CURRENT++))
        echo -n "  [$CURRENT/$TOTAL_COUNT] Testing with $threads threads... "
        
        # Create temporary files for perf and program output
        PERF_OUTPUT=$(mktemp)
        PROG_OUTPUT=$(mktemp)
        PROG_STDERR=$(mktemp)
        
        # Run with perf stat
        # -e: specify events to measure
        # L1-dcache-loads, L1-dcache-load-misses: L1 data cache
        # LLC-loads, LLC-load-misses: Last level cache
        # branches, branch-misses: Branch prediction
        # Using timeout to prevent hanging
        timeout $TIMEOUT_SECONDS perf stat -e cycles,instructions,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses,branches,branch-misses \
            "$EXECUTABLE" "$threads" "$mtx" > "$PROG_OUTPUT" 2>"$PERF_OUTPUT"
        
        EXIT_CODE=$?
        
        if [ $EXIT_CODE -eq 124 ]; then
            # Timeout occurred
            echo "TIMEOUT"
            echo "\"$mtx_basename\",NA,NA,NA,$threads,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,124,\"timeout after ${TIMEOUT_SECONDS}s\"" >> "$OUTPUT_CSV"
            rm -f "$PERF_OUTPUT" "$PROG_OUTPUT" "$PROG_STDERR"
            continue
        elif [ $EXIT_CODE -ne 0 ]; then
            echo "ERROR (exit code: $EXIT_CODE)"
            # Show error details for debugging
            if [ -s "$PERF_OUTPUT" ]; then
                echo "    PERF OUTPUT:"
                head -10 "$PERF_OUTPUT" | sed 's/^/      /'
            fi
            if [ -s "$PROG_OUTPUT" ]; then
                echo "    STDOUT:"
                head -10 "$PROG_OUTPUT" | sed 's/^/      /'
            fi
            echo "\"$mtx_basename\",NA,NA,NA,$threads,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,$EXIT_CODE,\"execution error\"" >> "$OUTPUT_CSV"
            rm -f "$PERF_OUTPUT" "$PROG_OUTPUT" "$PROG_STDERR"
            continue
        fi
        
        # Parse perf stat output
        # Format examples:
        #     1,234,567      cycles
        #       123,456      L1-dcache-loads
        #    <not supported> some-event
        
        CYCLES=$(grep -E '^\s*[0-9,]+\s+cycles' "$PERF_OUTPUT" | awk '{gsub(/,/,""); print $1}')
        INSTRUCTIONS=$(grep -E '^\s*[0-9,]+\s+instructions' "$PERF_OUTPUT" | awk '{gsub(/,/,""); print $1}')
        
        L1D_LOADS=$(grep -E '^\s*[0-9,]+\s+L1-dcache-loads' "$PERF_OUTPUT" | awk '{gsub(/,/,""); print $1}')
        L1D_MISSES=$(grep -E '^\s*[0-9,]+\s+L1-dcache-load-misses' "$PERF_OUTPUT" | awk '{gsub(/,/,""); print $1}')
        
        LLC_LOADS=$(grep -E '^\s*[0-9,]+\s+LLC-loads' "$PERF_OUTPUT" | awk '{gsub(/,/,""); print $1}')
        LLC_MISSES=$(grep -E '^\s*[0-9,]+\s+LLC-load-misses' "$PERF_OUTPUT" | awk '{gsub(/,/,""); print $1}')
        
        BRANCHES=$(grep -E '^\s*[0-9,]+\s+branches' "$PERF_OUTPUT" | awk '{gsub(/,/,""); print $1}')
        BRANCH_MISSES=$(grep -E '^\s*[0-9,]+\s+branch-misses' "$PERF_OUTPUT" | awk '{gsub(/,/,""); print $1}')
        
        # Extract execution time from perf output (e.g., "0.123456789 seconds time elapsed")
        TIME_SEC=$(grep -E 'seconds time elapsed' "$PERF_OUTPUT" | awk '{print $1}')
        
        # Calculate miss rates if we have the data
        if [[ -n "$L1D_LOADS" && -n "$L1D_MISSES" && "$L1D_LOADS" != "0" ]]; then
            L1D_MISS_RATE=$(echo "scale=4; ($L1D_MISSES / $L1D_LOADS) * 100" | bc)
        else
            L1D_MISS_RATE="NA"
        fi
        
        if [[ -n "$LLC_LOADS" && -n "$LLC_MISSES" && "$LLC_LOADS" != "0" ]]; then
            LLC_MISS_RATE=$(echo "scale=4; ($LLC_MISSES / $LLC_LOADS) * 100" | bc)
        else
            LLC_MISS_RATE="NA"
        fi
        
        if [[ -n "$BRANCHES" && -n "$BRANCH_MISSES" && "$BRANCHES" != "0" ]]; then
            BRANCH_MISS_RATE=$(echo "scale=4; ($BRANCH_MISSES / $BRANCHES) * 100" | bc)
        else
            BRANCH_MISS_RATE="NA"
        fi
        
        # Convert time to milliseconds
        if [[ -n "$TIME_SEC" ]]; then
            TIME_MS=$(echo "scale=2; $TIME_SEC * 1000" | bc)
        else
            TIME_MS="NA"
        fi
        
        # Extract matrix info from program output
        ROWS=$(grep -E "Matrix dimensions:" "$PROG_OUTPUT" | awk '{print $3}' | tr -d ',')
        COLS=$(grep -E "Matrix dimensions:" "$PROG_OUTPUT" | awk '{print $5}')
        DENSITY=$(grep -E "non-zero entries" "$PROG_OUTPUT" | grep -oE "[0-9]+\.[0-9]+%" | head -1)
        
        # Set defaults for missing values
        CYCLES=${CYCLES:-NA}
        INSTRUCTIONS=${INSTRUCTIONS:-NA}
        L1D_LOADS=${L1D_LOADS:-NA}
        L1D_MISSES=${L1D_MISSES:-NA}
        LLC_LOADS=${LLC_LOADS:-NA}
        LLC_MISSES=${LLC_MISSES:-NA}
        BRANCHES=${BRANCHES:-NA}
        BRANCH_MISSES=${BRANCH_MISSES:-NA}
        ROWS=${ROWS:-NA}
        COLS=${COLS:-NA}
        DENSITY=${DENSITY:-NA}
        
        # Write to CSV
        echo "\"$mtx_basename\",$ROWS,$COLS,\"$DENSITY\",$threads,$CYCLES,$INSTRUCTIONS,$L1D_LOADS,$L1D_MISSES,$L1D_MISS_RATE,$LLC_LOADS,$LLC_MISSES,$LLC_MISS_RATE,$BRANCHES,$BRANCH_MISSES,$BRANCH_MISS_RATE,$TIME_MS,$EXIT_CODE,\"success\"" >> "$OUTPUT_CSV"
        
        echo "OK (L1D miss: ${L1D_MISS_RATE}%, LLC miss: ${LLC_MISS_RATE}%, time: ${TIME_MS}ms)"
        
        rm -f "$PERF_OUTPUT" "$PROG_OUTPUT" "$PROG_STDERR"
    done
    echo ""
done

echo "=== Benchmarking Complete ==="
echo "Results saved to: $OUTPUT_CSV"
echo ""
echo "Summary:"
wc -l < "$OUTPUT_CSV" | awk '{print $1-1 " test runs completed"}'
echo ""
echo "Note: perf captures all threads' cache events, providing accurate"
echo "multi-threaded cache performance data."
