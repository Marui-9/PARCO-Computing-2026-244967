#!/bin/bash
# Test bench_configurations.sh with just one small matrix
set -euo pipefail

# Temporarily modify to use only one small matrix
EXE="./test_config"
OUT_CSV="evaluation/test_config_small.csv"
TIMEOUT_SECS=60
ITERATIONS=20
THREAD_COUNTS="4 8"
mtx_basename="2k_0p22.mtx"

printf '%s\n' "matrix,rows,cols,density_pct,nnz,threads,configuration,schedule,chunk_size,time_ms,speedup,efficiency_pct,std_dev_ms,improvement_pct,exit_code,notes" > "$OUT_CSV"

for threads in $THREAD_COUNTS; do
  echo "Testing $mtx_basename with $threads threads..."
  TMP_OUTPUT=$(mktemp)
  
  timeout $TIMEOUT_SECS "$EXE" "$threads" "$mtx_basename" "$ITERATIONS" > "$TMP_OUTPUT" 2>&1
  EXIT_CODE=$?
  
  if [ $EXIT_CODE -ne 0 ]; then
    echo "Failed with exit code: $EXIT_CODE"
    cat "$TMP_OUTPUT"
    rm -f "$TMP_OUTPUT"
    continue
  fi
  
  # Parse metadata
  ROWS=$(grep -oP 'Matrix: \K[0-9]+(?= ×)' "$TMP_OUTPUT" || echo "NA")
  COLS=$(grep -oP '× \K[0-9]+(?=,)' "$TMP_OUTPUT" || echo "NA")
  DENSITY=$(grep -oP '\K[0-9]+\.[0-9]+(?=% non-zero)' "$TMP_OUTPUT" || echo "NA")
  NNZ=$(grep -oP 'non-zero, \K[0-9]+(?= NNZ)' "$TMP_OUTPUT" || echo "NA")
  
  echo "  Found: ${ROWS}x${COLS}, ${DENSITY}%, ${NNZ} NNZ"
  
  # Parse results
  CONFIG_COUNT=0
  while IFS= read -r line; do
    CONFIG=$(echo "$line" | cut -c1-32 | sed 's/[[:space:]]*$//')
    REMAINING=$(echo "$line" | cut -c33-)
    TIME_MS=$(echo "$REMAINING" | awk '{print $1}')
    SPEEDUP=$(echo "$REMAINING" | awk '{print $2}' | tr -d 'x')
    EFFICIENCY=$(echo "$REMAINING" | awk '{print $3}' | tr -d '%')
    STDDEV=$(echo "$REMAINING" | awk '{print $4}' | tr -d 'ms')
    IMPROVEMENT=$(echo "$REMAINING" | awk '{print $5}' | tr -d '%')
    
    echo "\"$mtx_basename\",$ROWS,$COLS,$DENSITY,$NNZ,$threads,\"$CONFIG\",\"NA\",\"NA\",$TIME_MS,$SPEEDUP,$EFFICIENCY,$STDDEV,$IMPROVEMENT,0,\"success\"" >> "$OUT_CSV"
    ((CONFIG_COUNT++))
  done < <(grep -E "^[A-Za-z].*[0-9]+\.[0-9]+.*[0-9]+\.[0-9]+x" "$TMP_OUTPUT" || true)
  
  echo "  Extracted $CONFIG_COUNT configurations"
  rm -f "$TMP_OUTPUT"
done

echo "Results written to: $OUT_CSV"
wc -l "$OUT_CSV"
