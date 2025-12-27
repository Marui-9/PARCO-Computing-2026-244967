#!/bin/bash

# Benchmark script for memory-leak and error summary using Valgrind memcheck
# Output: results/cache_valgrind_results.csv with memcheck leak-summary metrics

OUTPUT_CSV="results/cache_valgrind_results.csv"
EXECUTABLE="${EXECUTABLE:-./executable}"  # Use environment variable if set, otherwise default
TIMEOUT_SECONDS=300  # Valgrind memcheck is slow

# Thread counts to test
THREAD_COUNTS=(1 4 8 12 16 24)

# Check if executable exists
if [ ! -f "$EXECUTABLE" ]; then
    echo "Error: $EXECUTABLE not found. Please compile first with 'make'."
    exit 1
fi

# Check if valgrind is available
if ! command -v valgrind &> /dev/null; then
    echo "Error: valgrind command not found. Please install valgrind."
    exit 1
fi

# Create output directory
mkdir -p evaluation results

# Create CSV header (memcheck leak summary fields)
echo "matrix,rows,cols,density_pct,threads,definitely_lost_bytes,definitely_lost_blocks,indirectly_lost_bytes,indirectly_lost_blocks,possibly_lost_bytes,possibly_lost_blocks,still_reachable_bytes,still_reachable_blocks,suppressed_bytes,suppressed_blocks,error_summary_count,exit_code,notes" > "$OUTPUT_CSV"

echo "=== Valgrind memcheck Benchmarking ==="
echo "Output file: $OUTPUT_CSV"
echo "WARNING: Valgrind memcheck adds significant overhead. This will take longer than perf."
echo ""

# Find .mtx files (exclude very large prefixes to fit within walltime)
MATRIX_FILES=$(find matrices/ -name "*.mtx" -type f | grep -v '/[0-9][0-9]k_' | sort)

if [ -z "$MATRIX_FILES" ]; then
    echo "Error: No .mtx files found in matrices/ directory (after filtering)"
    exit 1
fi

TOTAL_COUNT=0
for mtx in $MATRIX_FILES; do
    for threads in "${THREAD_COUNTS[@]}"; do
        ((TOTAL_COUNT++))
    done
done

CURRENT=0

for mtx in $MATRIX_FILES; do
    mtx_basename=$(basename "$mtx")
    echo "Matrix: $mtx_basename"
    for threads in "${THREAD_COUNTS[@]}"; do
        ((CURRENT++))
        echo -n "  [$CURRENT/$TOTAL_COUNT] Testing with $threads threads... "

        PROG_OUTPUT=$(mktemp)
        PROG_STDERR=$(mktemp)

        # Run valgrind memcheck
        timeout $TIMEOUT_SECONDS valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all \
            "$EXECUTABLE" "$threads" "$mtx" > "$PROG_OUTPUT" 2>"$PROG_STDERR"

        EXIT_CODE=$?

        if [ $EXIT_CODE -eq 124 ]; then
            echo "TIMEOUT"
            echo "\"$mtx_basename\",NA,NA,NA,$threads,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,124,\"timeout after ${TIMEOUT_SECONDS}s\"" >> "$OUTPUT_CSV"
            rm -f "$PROG_OUTPUT" "$PROG_STDERR"
            continue
        elif [ $EXIT_CODE -ne 0 ]; then
            echo "ERROR (exit code: $EXIT_CODE)"
            if [ -s "$PROG_STDERR" ]; then
                echo "    STDERR:"
                head -10 "$PROG_STDERR" | sed 's/^/      /'
            fi
            if [ -s "$PROG_OUTPUT" ]; then
                echo "    STDOUT:"
                head -10 "$PROG_OUTPUT" | sed 's/^/      /'
            fi
            echo "\"$mtx_basename\",NA,NA,NA,$threads,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,$EXIT_CODE,\"execution error\"" >> "$OUTPUT_CSV"
            rm -f "$PROG_OUTPUT" "$PROG_STDERR"
            continue
        fi

        # Parse memcheck leak summary
        DEF_LOST_BYTES=$(grep -E "^==[0-9]+==\s+definitely lost:" "$PROG_STDERR" | awk '{gsub(/,/,"",$0); print $4}')
        DEF_LOST_BLOCKS=$(grep -E "^==[0-9]+==\s+definitely lost:" "$PROG_STDERR" | awk '{gsub(/,/,"",$0); print $6}')
        IND_LOST_BYTES=$(grep -E "^==[0-9]+==\s+indirectly lost:" "$PROG_STDERR" | awk '{gsub(/,/,"",$0); print $4}')
        IND_LOST_BLOCKS=$(grep -E "^==[0-9]+==\s+indirectly lost:" "$PROG_STDERR" | awk '{gsub(/,/,"",$0); print $6}')
        POSS_LOST_BYTES=$(grep -E "^==[0-9]+==\s+possibly lost:" "$PROG_STDERR" | awk '{gsub(/,/,"",$0); print $4}')
        POSS_LOST_BLOCKS=$(grep -E "^==[0-9]+==\s+possibly lost:" "$PROG_STDERR" | awk '{gsub(/,/,"",$0); print $6}')
        STILL_REACH_BYTES=$(grep -E "^==[0-9]+==\s+still reachable:" "$PROG_STDERR" | awk '{gsub(/,/,"",$0); print $4}')
        STILL_REACH_BLOCKS=$(grep -E "^==[0-9]+==\s+still reachable:" "$PROG_STDERR" | awk '{gsub(/,/,"",$0); print $6}')
        SUPP_BYTES=$(grep -E "^==[0-9]+==\s+suppressed:" "$PROG_STDERR" | awk '{gsub(/,/,"",$0); print $4}')
        SUPP_BLOCKS=$(grep -E "^==[0-9]+==\s+suppressed:" "$PROG_STDERR" | awk '{gsub(/,/,"",$0); print $6}')
        ERROR_SUMMARY=$(grep -E "ERROR SUMMARY:" "$PROG_STDERR" | awk '{print $4}')

        # Extract matrix info from program output
        ROWS=$(grep -E "Matrix dimensions:" "$PROG_OUTPUT" | awk '{print $3}' | tr -d ',')
        COLS=$(grep -E "Matrix dimensions:" "$PROG_OUTPUT" | awk '{print $5}')
        DENSITY=$(grep -E "non-zero entries" "$PROG_OUTPUT" | grep -oE "[0-9]+\.[0-9]+%" | head -1)

        DEF_LOST_BYTES=${DEF_LOST_BYTES:-NA}
        DEF_LOST_BLOCKS=${DEF_LOST_BLOCKS:-NA}
        IND_LOST_BYTES=${IND_LOST_BYTES:-NA}
        IND_LOST_BLOCKS=${IND_LOST_BLOCKS:-NA}
        POSS_LOST_BYTES=${POSS_LOST_BYTES:-NA}
        POSS_LOST_BLOCKS=${POSS_LOST_BLOCKS:-NA}
        STILL_REACH_BYTES=${STILL_REACH_BYTES:-NA}
        STILL_REACH_BLOCKS=${STILL_REACH_BLOCKS:-NA}
        SUPP_BYTES=${SUPP_BYTES:-NA}
        SUPP_BLOCKS=${SUPP_BLOCKS:-NA}
        ERROR_SUMMARY=${ERROR_SUMMARY:-NA}
        ROWS=${ROWS:-NA}
        COLS=${COLS:-NA}
        DENSITY=${DENSITY:-NA}

        # Write to CSV
        echo "\"$mtx_basename\",$ROWS,$COLS,"$DENSITY",$threads,$DEF_LOST_BYTES,$DEF_LOST_BLOCKS,$IND_LOST_BYTES,$IND_LOST_BLOCKS,$POSS_LOST_BYTES,$POSS_LOST_BLOCKS,$STILL_REACH_BYTES,$STILL_REACH_BLOCKS,$SUPP_BYTES,$SUPP_BLOCKS,$ERROR_SUMMARY,$EXIT_CODE,\"success\"" >> "$OUTPUT_CSV"

        echo "OK (definitely_lost=${DEF_LOST_BYTES} bytes, possibly_lost=${POSS_LOST_BYTES} bytes, errors=${ERROR_SUMMARY})"

        rm -f "$PROG_OUTPUT" "$PROG_STDERR"
    done
    echo ""
done

echo "=== Benchmarking Complete ==="
echo "Results saved to: $OUTPUT_CSV"
echo ""
echo "Summary:"
wc -l < "$OUTPUT_CSV" | awk '{print $1-1 " test runs completed"}'
echo ""
echo "Note: Valgrind memcheck reports leak summaries and error counts."
