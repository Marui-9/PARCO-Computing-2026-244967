#!/bin/bash

# Benchmark script for cache performance analysis using perf stat
# Output: evaluation/cache_results.csv with cache performance metrics

OUTPUT_CSV="evaluation/cache_results.csv"
EXECUTABLE="./executable"
TIMEOUT_SECONDS=120

# Thread counts to test
THREAD_COUNTS=(1 2 4 8 12 16 18 20 22 24)

# Check if executable exists
if [ ! -f "$EXECUTABLE" ]; then
    echo "Error: $EXECUTABLE not found. Please compile first with 'make'."
    exit 1
fi

# Check if perf is available
if ! command -v perf &> /dev/null; then
    echo "Error: perf command not found. Please install linux-tools or perf."
    exit 1
fi

# Create CSV header
echo "matrix,rows,cols,density_pct,threads,l1_dcache_loads,l1_dcache_misses,l1_miss_rate,llc_loads,llc_misses,llc_miss_rate,cache_refs,cache_misses,cache_miss_rate,time_ms,exit_code,notes" > "$OUTPUT_CSV"

echo "=== Cache Performance Benchmarking ==="
echo "Output file: $OUTPUT_CSV"
echo ""

# Find all .mtx files in matrices/ directory
MATRIX_FILES=$(find matrices/ -name "*.mtx" -type f | sort)

if [ -z "$MATRIX_FILES" ]; then
    echo "Error: No .mtx files found in matrices/ directory"
    exit 1
fi

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
        
        # Create temporary file for perf output
        PERF_OUTPUT=$(mktemp)
        
        # Run with perf stat and timeout
        timeout $TIMEOUT_SECONDS perf stat -e L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses,cache-misses,cache-references \
            "$EXECUTABLE" "$threads" "$mtx_basename" 2>"$PERF_OUTPUT" >/dev/null
        
        EXIT_CODE=$?
        
        if [ $EXIT_CODE -eq 124 ]; then
            # Timeout occurred
            echo "TIMEOUT"
            echo "\"$mtx_basename\",NA,NA,NA,$threads,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,124,\"timeout after ${TIMEOUT_SECONDS}s\"" >> "$OUTPUT_CSV"
            rm -f "$PERF_OUTPUT"
            continue
        elif [ $EXIT_CODE -ne 0 ]; then
            echo "ERROR (exit code: $EXIT_CODE)"
            echo "\"$mtx_basename\",NA,NA,NA,$threads,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,$EXIT_CODE,\"execution error\"" >> "$OUTPUT_CSV"
            rm -f "$PERF_OUTPUT"
            continue
        fi
        
        # Parse perf output
        L1_LOADS=$(grep -E "L1-dcache-loads" "$PERF_OUTPUT" | awk '{gsub(/,/,""); print $1}')
        L1_MISSES=$(grep -E "L1-dcache-load-misses" "$PERF_OUTPUT" | awk '{gsub(/,/,""); print $1}')
        LLC_LOADS=$(grep -E "LLC-loads" "$PERF_OUTPUT" | awk '{gsub(/,/,""); print $1}')
        LLC_MISSES=$(grep -E "LLC-load-misses" "$PERF_OUTPUT" | awk '{gsub(/,/,""); print $1}')
        CACHE_REFS=$(grep -E "cache-references" "$PERF_OUTPUT" | awk '{gsub(/,/,""); print $1}')
        CACHE_MISSES=$(grep -E "cache-misses" "$PERF_OUTPUT" | awk '{gsub(/,/,""); print $1}')
        TIME_SEC=$(grep -E "seconds time elapsed" "$PERF_OUTPUT" | awk '{print $1}')
        
        # Extract matrix info from program output (need to capture stdout)
        PROG_OUTPUT=$(mktemp)
        timeout $TIMEOUT_SECONDS "$EXECUTABLE" "$threads" "$mtx_basename" > "$PROG_OUTPUT" 2>/dev/null
        
        ROWS=$(grep -E "Matrix dimensions:" "$PROG_OUTPUT" | awk '{print $3}' | tr -d ',')
        COLS=$(grep -E "Matrix dimensions:" "$PROG_OUTPUT" | awk '{print $5}')
        DENSITY=$(grep -E "Conversion completed" "$PROG_OUTPUT" | grep -oE "[0-9]+\.[0-9]+%" | head -1)
        
        rm -f "$PROG_OUTPUT"
        
        # Calculate miss rates
        if [ -n "$L1_LOADS" ] && [ -n "$L1_MISSES" ] && [ "$L1_LOADS" != "0" ]; then
            L1_MISS_RATE=$(awk "BEGIN {printf \"%.4f\", ($L1_MISSES / $L1_LOADS) * 100}")
        else
            L1_MISS_RATE="NA"
        fi
        
        if [ -n "$LLC_LOADS" ] && [ -n "$LLC_MISSES" ] && [ "$LLC_LOADS" != "0" ]; then
            LLC_MISS_RATE=$(awk "BEGIN {printf \"%.4f\", ($LLC_MISSES / $LLC_LOADS) * 100}")
        else
            LLC_MISS_RATE="NA"
        fi
        
        if [ -n "$CACHE_REFS" ] && [ -n "$CACHE_MISSES" ] && [ "$CACHE_REFS" != "0" ]; then
            CACHE_MISS_RATE=$(awk "BEGIN {printf \"%.4f\", ($CACHE_MISSES / $CACHE_REFS) * 100}")
        else
            CACHE_MISS_RATE="NA"
        fi
        
        # Convert time to milliseconds
        if [ -n "$TIME_SEC" ]; then
            TIME_MS=$(awk "BEGIN {printf \"%.2f\", $TIME_SEC * 1000}")
        else
            TIME_MS="NA"
        fi
        
        # Set defaults for missing values
        L1_LOADS=${L1_LOADS:-NA}
        L1_MISSES=${L1_MISSES:-NA}
        LLC_LOADS=${LLC_LOADS:-NA}
        LLC_MISSES=${LLC_MISSES:-NA}
        CACHE_REFS=${CACHE_REFS:-NA}
        CACHE_MISSES=${CACHE_MISSES:-NA}
        ROWS=${ROWS:-NA}
        COLS=${COLS:-NA}
        DENSITY=${DENSITY:-NA}
        
        # Write to CSV
        echo "\"$mtx_basename\",$ROWS,$COLS,\"$DENSITY\",$threads,$L1_LOADS,$L1_MISSES,$L1_MISS_RATE,$LLC_LOADS,$LLC_MISSES,$LLC_MISS_RATE,$CACHE_REFS,$CACHE_MISSES,$CACHE_MISS_RATE,$TIME_MS,$EXIT_CODE,\"success\"" >> "$OUTPUT_CSV"
        
        echo "OK (Time: ${TIME_MS}ms, L1 miss: ${L1_MISS_RATE}%, LLC miss: ${LLC_MISS_RATE}%)"
        
        rm -f "$PERF_OUTPUT"
    done
    echo ""
done

echo "=== Benchmarking Complete ==="
echo "Results saved to: $OUTPUT_CSV"
echo ""
echo "Summary:"
wc -l < "$OUTPUT_CSV" | awk '{print $1-1 " test runs completed"}'
