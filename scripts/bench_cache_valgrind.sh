#!/bin/bash

# Benchmark script for cache performance analysis using valgrind cachegrind
# Output: results/cache_valgrind_results.csv with cache performance metrics
# NOTE: Valgrind may not support all AVX-512 instructions on all systems

OUTPUT_CSV="results/cache_valgrind_results.csv"
EXECUTABLE="${EXECUTABLE:-./executable}"  # Use environment variable if set, otherwise default
TIMEOUT_SECONDS=300  # Valgrind is slower than perf

# Thread counts to test (valgrind can be slow, so fewer thread counts)
THREAD_COUNTS=(1 4 8 12 16 24)

# Check if executable exists
if [ ! -f "$EXECUTABLE" ]; then
    echo "Error: $EXECUTABLE not found. Please compile first with 'make'."
    exit 1
fi

# Check if valgrind is available
if ! command -v valgrind &> /dev/null; then
    echo "Error: valgrind command not found. Please install valgrind."
    echo "On Fedora/RHEL: sudo dnf install valgrind"
    echo "On Ubuntu/Debian: sudo apt install valgrind"
    exit 1
fi

# Create output directory
mkdir -p evaluation

# Create CSV header
echo "matrix,rows,cols,density_pct,threads,i_refs,i1_misses,i1_miss_rate,d_refs,d1_misses,d1_miss_rate,d1_writes,d1_write_misses,ll_refs,ll_misses,ll_miss_rate,branches,branch_misses,branch_miss_rate,exit_code,notes" > "$OUTPUT_CSV"

echo "=== Cache Performance Benchmarking with Valgrind Cachegrind ==="
echo "Output file: $OUTPUT_CSV"
echo "WARNING: Valgrind adds significant overhead. This will take longer than perf."
echo ""

# Find all .mtx files in matrices/ directory, excluding those with two-digit numbers before 'k'
# This excludes patterns like 10k_, 11k_, 15k_, etc. to reduce runtime
MATRIX_FILES=$(find matrices/ -name "*.mtx" -type f | grep -v '/[0-9][0-9]k_' | sort)

if [ -z "$MATRIX_FILES" ]; then
    echo "Error: No .mtx files found in matrices/ directory (after filtering)"
    exit 1
fi

echo "Excluded matrices with two-digit prefixes (e.g., 10k_, 11k_, 15k_) to fit within walltime"

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
        
        # Create temporary files for cachegrind output
        CACHEGRIND_OUT=$(mktemp -u --suffix=.cachegrind)
        PROG_OUTPUT=$(mktemp)
        PROG_STDERR=$(mktemp)
        
        # Run with valgrind cachegrind and timeout
        # Note: --tool=cachegrind profiles cache behavior
        timeout $TIMEOUT_SECONDS valgrind --tool=cachegrind \
            --cachegrind-out-file="$CACHEGRIND_OUT" \
            --cache-sim=yes \
            --branch-sim=yes \
            "$EXECUTABLE" "$threads" "$mtx_basename" > "$PROG_OUTPUT" 2>"$PROG_STDERR"
        
        EXIT_CODE=$?
        
        if [ $EXIT_CODE -eq 124 ]; then
            # Timeout occurred
            echo "TIMEOUT"
            echo "\"$mtx_basename\",NA,NA,NA,$threads,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,124,\"timeout after ${TIMEOUT_SECONDS}s\"" >> "$OUTPUT_CSV"
            rm -f "$CACHEGRIND_OUT" "$PROG_OUTPUT" "$PROG_STDERR"
            continue
        elif [ $EXIT_CODE -ne 0 ]; then
            echo "ERROR (exit code: $EXIT_CODE)"
            # Show error details for debugging
            if [ -s "$PROG_STDERR" ]; then
                echo "    STDERR:"
                head -10 "$PROG_STDERR" | sed 's/^/      /'
            fi
            if [ -s "$PROG_OUTPUT" ]; then
                echo "    STDOUT:"
                head -10 "$PROG_OUTPUT" | sed 's/^/      /'
            fi
            echo "\"$mtx_basename\",NA,NA,NA,$threads,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,$EXIT_CODE,\"execution error\"" >> "$OUTPUT_CSV"
            rm -f "$CACHEGRIND_OUT" "$PROG_OUTPUT" "$PROG_STDERR"
            continue
        fi
        
        # Check if cachegrind output was generated
        if [ ! -f "$CACHEGRIND_OUT" ]; then
            echo "ERROR (no cachegrind output)"
            echo "\"$mtx_basename\",NA,NA,NA,$threads,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,1,\"no cachegrind output\"" >> "$OUTPUT_CSV"
            rm -f "$PROG_OUTPUT" "$PROG_STDERR"
            continue
        fi
        
        # Parse cachegrind output using cg_annotate
        CG_SUMMARY=$(mktemp)
        cg_annotate "$CACHEGRIND_OUT" > "$CG_SUMMARY" 2>/dev/null
        
        # Extract cache statistics from cachegrind summary
        # The summary section looks like:
        # I   refs:      123,456,789
        # I1  misses:        234,567
        # LLi misses:         12,345
        # I1  miss rate:       0.19%
        # LLi miss rate:       0.01%
        
        # D   refs:      456,789,012
        # D1  misses:      1,234,567
        # LLd misses:         23,456
        # D1  miss rate:       0.27%
        # LLd miss rate:       0.01%
        
        # LL refs:         1,469,134
        # LL misses:          35,801
        # LL miss rate:        2.4%
        
        # Branches:       89,012,345
        # Mispredicts:       123,456
        # Mispred rate:        0.1%
        
        I_REFS=$(grep -E "^I\s+refs:" "$CG_SUMMARY" | awk '{gsub(/,/,""); print $3}')
        I1_MISSES=$(grep -E "^I1\s+misses:" "$CG_SUMMARY" | awk '{gsub(/,/,""); print $3}')
        I1_MISS_RATE=$(grep -E "^I1\s+miss rate:" "$CG_SUMMARY" | awk '{gsub(/%/,""); print $4}')
        
        D_REFS=$(grep -E "^D\s+refs:" "$CG_SUMMARY" | awk '{gsub(/,/,""); print $3}')
        D1_MISSES=$(grep -E "^D1\s+misses:" "$CG_SUMMARY" | awk '{gsub(/,/,""); print $3}')
        D1_MISS_RATE=$(grep -E "^D1\s+miss rate:" "$CG_SUMMARY" | awk '{gsub(/%/,""); print $4}')
        
        # D writes are part of D refs - estimate from D1 write misses if available
        D1_WRITES=$(grep -E "^D1\s+wr misses:" "$CG_SUMMARY" 2>/dev/null | awk '{gsub(/,/,""); print $4}')
        D1_WRITE_MISSES=$(grep -E "^D1\s+wr misses:" "$CG_SUMMARY" 2>/dev/null | awk '{gsub(/,/,""); print $4}')
        
        LL_REFS=$(grep -E "^LL\s+refs:" "$CG_SUMMARY" | awk '{gsub(/,/,""); print $3}')
        LL_MISSES=$(grep -E "^LL\s+misses:" "$CG_SUMMARY" | awk '{gsub(/,/,""); print $3}')
        LL_MISS_RATE=$(grep -E "^LL\s+miss rate:" "$CG_SUMMARY" | awk '{gsub(/%/,""); print $4}')
        
        BRANCHES=$(grep -E "^Branches:" "$CG_SUMMARY" | awk '{gsub(/,/,""); print $2}')
        BRANCH_MISSES=$(grep -E "^Mispredicts:" "$CG_SUMMARY" | awk '{gsub(/,/,""); print $2}')
        BRANCH_MISS_RATE=$(grep -E "^Mispred rate:" "$CG_SUMMARY" | awk '{gsub(/%/,""); print $3}')
        
        # Extract matrix info from program output
        ROWS=$(grep -E "Matrix dimensions:" "$PROG_OUTPUT" | awk '{print $3}' | tr -d ',')
        COLS=$(grep -E "Matrix dimensions:" "$PROG_OUTPUT" | awk '{print $5}')
        DENSITY=$(grep -E "non-zero entries" "$PROG_OUTPUT" | grep -oE "[0-9]+\.[0-9]+%" | head -1)
        
        # Set defaults for missing values
        I_REFS=${I_REFS:-NA}
        I1_MISSES=${I1_MISSES:-NA}
        I1_MISS_RATE=${I1_MISS_RATE:-NA}
        D_REFS=${D_REFS:-NA}
        D1_MISSES=${D1_MISSES:-NA}
        D1_MISS_RATE=${D1_MISS_RATE:-NA}
        D1_WRITES=${D1_WRITES:-NA}
        D1_WRITE_MISSES=${D1_WRITE_MISSES:-NA}
        LL_REFS=${LL_REFS:-NA}
        LL_MISSES=${LL_MISSES:-NA}
        LL_MISS_RATE=${LL_MISS_RATE:-NA}
        BRANCHES=${BRANCHES:-NA}
        BRANCH_MISSES=${BRANCH_MISSES:-NA}
        BRANCH_MISS_RATE=${BRANCH_MISS_RATE:-NA}
        ROWS=${ROWS:-NA}
        COLS=${COLS:-NA}
        DENSITY=${DENSITY:-NA}
        
        # Write to CSV
        echo "\"$mtx_basename\",$ROWS,$COLS,\"$DENSITY\",$threads,$I_REFS,$I1_MISSES,$I1_MISS_RATE,$D_REFS,$D1_MISSES,$D1_MISS_RATE,$D1_WRITES,$D1_WRITE_MISSES,$LL_REFS,$LL_MISSES,$LL_MISS_RATE,$BRANCHES,$BRANCH_MISSES,$BRANCH_MISS_RATE,$EXIT_CODE,\"success\"" >> "$OUTPUT_CSV"
        
        echo "OK (D1 miss: ${D1_MISS_RATE}%, LL miss: ${LL_MISS_RATE}%, Branch miss: ${BRANCH_MISS_RATE}%)"
        
        rm -f "$CACHEGRIND_OUT" "$CG_SUMMARY" "$PROG_OUTPUT" "$PROG_STDERR"
    done
    echo ""
done

echo "=== Benchmarking Complete ==="
echo "Results saved to: $OUTPUT_CSV"
echo ""
echo "Summary:"
wc -l < "$OUTPUT_CSV" | awk '{print $1-1 " test runs completed"}'
echo ""
echo "Note: Valgrind provides more detailed cache simulation than perf,"
echo "including instruction cache, data cache, and last-level cache statistics."
