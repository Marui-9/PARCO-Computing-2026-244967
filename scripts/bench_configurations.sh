#!/usr/bin/env bash
# bench_configurations.sh
# Benchmark all configurations from test_config across all matrices and thread counts
# Usage: ./bench_configurations.sh
set -euo pipefail

EXE="./test_config"
OUT_CSV="results/configurations_results.csv"
OUT_LOG="results/configurations_results.txt"
TIMEOUT_SECS=900   # kill any run that exceeds this (15 minutes per matrix/thread combo)
ITERATIONS=30      # number of iterations per configuration

# Thread counts to test
THREAD_COUNTS="1 2 4 8 12 16 18 20 22 24"

# Automatically find all .mtx files in matrices/ directory
MATRICES_DIR="matrices"
if [ ! -d "$MATRICES_DIR" ]; then
  echo "ERROR: matrices directory '$MATRICES_DIR' not found." >&2
  exit 2
fi

# Build array of all .mtx files
MATRICES=()
while IFS= read -r -d '' mtx; do
  MATRICES+=("$mtx")
done < <(find "$MATRICES_DIR" -name "*.mtx" -type f -print0 | sort -z)

if [ ${#MATRICES[@]} -eq 0 ]; then
  echo "ERROR: No .mtx files found in '$MATRICES_DIR'." >&2
  exit 3
fi

echo "Found ${#MATRICES[@]} matrix files in '$MATRICES_DIR'"

# Quick checks
if [ ! -f "$EXE" ]; then
  echo "ERROR: executable not found at '$EXE'. Adjust EXE variable." >&2
  exit 2
fi
if [ ! -x "$EXE" ]; then
  echo "ERROR: '$EXE' exists but is not executable. Try 'chmod +x $EXE'." >&2
  exit 3
fi

# Create output directory if needed
mkdir -p "$(dirname "$OUT_CSV")"

# CSV header
printf '%s\n' "matrix,rows,cols,density_pct,nnz,threads,configuration,time_ms,speedup,efficiency_pct,std_dev_ms,improvement_pct,exit_code,notes" > "$OUT_CSV"

# Initialize log file
{
  echo "========================================================================"
  echo "CONFIGURATION BENCHMARK LOG"
  echo "========================================================================"
  echo "Started at: $(date)"
  echo "Executable: $EXE"
  echo "Timeout per run: ${TIMEOUT_SECS}s"
  echo "Iterations per config: $ITERATIONS"
  echo "Thread counts: $THREAD_COUNTS"
  echo "Matrices found: ${#MATRICES[@]}"
  echo "========================================================================"
  echo ""
} > "$OUT_LOG"

cleanup() { :; }
trap cleanup EXIT

echo "Starting configuration benchmark. EXE=$EXE. Timeout per run=${TIMEOUT_SECS}s"
echo "Iterations per config: $ITERATIONS"
echo "Thread counts: $THREAD_COUNTS"
echo ""

for mtx in "${MATRICES[@]}"; do
  echo "=== Matrix: $mtx ==="
  echo "=== Processing Matrix: $mtx ===" >> "$OUT_LOG"
  if [ ! -f "$mtx" ]; then
    echo "  SKIP: file not found: $mtx"
    echo "  SKIP: file not found: $mtx" >> "$OUT_LOG"
    printf '%s\n' "\"$mtx\",NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,FILE_NOT_FOUND,\"file not found\"" >> "$OUT_CSV"
    continue
  fi

  # Extract just the filename
  mtx_basename=$(basename "$mtx")

  for threads in $THREAD_COUNTS; do
    echo "  threads=$threads"
    tmpout=$(mktemp)
    notes=""

    # Run test_config with timeout
    timeout "$TIMEOUT_SECS" "$EXE" "$threads" "$mtx_basename" "$ITERATIONS" > "$tmpout" 2>&1 || true
    exitcode=$?

    # timeout returns 124 when timed out
    if [ "$exitcode" -eq 124 ]; then
      notes="TIMED_OUT"
      echo "    Run timed out after ${TIMEOUT_SECS}s"
      printf '%s\n' "\"$mtx_basename\",NA,NA,NA,NA,$threads,NA,NA,NA,NA,NA,NA,$exitcode,\"$notes\"" >> "$OUT_CSV"
      rm -f "$tmpout"
      continue
    fi

    # check for output presence
    if [ ! -s "$tmpout" ]; then
      notes="NO_OUTPUT"
      echo "    WARNING: no output captured from run."
      printf '%s\n' "\"$mtx_basename\",NA,NA,NA,NA,$threads,NA,NA,NA,NA,NA,NA,$exitcode,\"$notes\"" >> "$OUT_CSV"
      rm -f "$tmpout"
      continue
    fi

    # diagnostics if nonzero exit
    if [ "$exitcode" -ne 0 ]; then
      echo "    Run exited with code $exitcode. Last 20 lines of output:"
      tail -n 20 "$tmpout" | sed 's/^/      /'
      notes="EXIT_${exitcode}"
      # Still try to parse what we can
    fi

    # Show output size for debugging
    if [ -f "$tmpout" ]; then
      output_size=$(wc -l < "$tmpout")
      echo "    Output: $output_size lines, exit code: $exitcode"
      if [ "$output_size" -lt 5 ]; then
        echo "    Output content:"
        cat "$tmpout" | sed 's/^/      /'
      fi
    fi

    # Parse matrix metadata
    ROWS=$(grep -oP 'Matrix: \K[0-9]+(?= ×)' "$tmpout" 2>/dev/null || echo "NA")
    COLS=$(grep -oP '× \K[0-9]+(?=,)' "$tmpout" 2>/dev/null || echo "NA")
    DENSITY=$(grep -oP '\K[0-9]+\.[0-9]+(?=% non-zero)' "$tmpout" 2>/dev/null || echo "NA")
    NNZ=$(grep -oP 'non-zero, \K[0-9]+(?= NNZ)' "$tmpout" 2>/dev/null || echo "NA")

    if [ "$ROWS" = "NA" ]; then
      echo "    WARNING: Could not parse matrix metadata"
      notes="${notes}${notes:+; }PARSE_FAILED"
    fi

    # Parse each configuration result line
    # Format: "ConfigName    time    speedup    efficiency    stddev    improvement"
    config_count=0
    while IFS= read -r line; do
      # Use awk to extract numeric values (simpler approach)
      # Pattern: config_name followed by 5 numeric fields
      TIME_MS=$(echo "$line" | awk '{for(i=1;i<=NF;i++) if($i~/^[0-9]+\.[0-9]+$/) {print $i; exit}}')
      SPEEDUP=$(echo "$line" | awk '{for(i=1;i<=NF;i++) if($i~/^[0-9]+\.[0-9]+x$/) {print $i; exit}}' | tr -d 'x')
      EFFICIENCY=$(echo "$line" | awk '{for(i=1;i<=NF;i++) if($i~/^[0-9]+\.[0-9]+%$/) {print $i; exit}}' | tr -d '%')
      STDDEV=$(echo "$line" | awk '{for(i=1;i<=NF;i++) if($i~/^[0-9]+\.[0-9]+ms$/) {print $i; exit}}' | tr -d 'ms')
      IMPROVEMENT=$(echo "$line" | awk '{n=0; for(i=1;i<=NF;i++) if($i~/^[0-9]+\.[0-9]+%$/) {n++; if(n==2) {print $i; exit}}}' | tr -d '%')
      
      # Extract configuration name (everything before first numeric field)
      CONFIG=$(echo "$line" | awk '{
        for(i=1; i<=NF; i++) {
          if($i ~ /^[0-9]/) {
            for(j=1; j<i; j++) printf "%s ", $j
            exit
          }
        }
      }' | sed 's/[[:space:]]*$//')
      
      # Write to CSV
      printf '%s\n' "\"$mtx_basename\",$ROWS,$COLS,$DENSITY,$NNZ,$threads,\"$CONFIG\",$TIME_MS,$SPEEDUP,$EFFICIENCY,$STDDEV,$IMPROVEMENT,$exitcode,\"$notes\"" >> "$OUT_CSV"
      config_count=$((config_count + 1))
    done < <(grep -E "^[A-Za-z].*[0-9]+\.[0-9]+.*[0-9]+\.[0-9]+x" "$tmpout" 2>/dev/null || true)

    if [ "$config_count" -eq 0 ]; then
      echo "    WARNING: No configuration results parsed"
      echo "    WARNING: No configuration results parsed for $mtx_basename, threads=$threads" >> "$OUT_LOG"
      notes="${notes}${notes:+; }NO_CONFIGS"
      printf '%s\n' "\"$mtx_basename\",$ROWS,$COLS,$DENSITY,$NNZ,$threads,\"NONE\",NA,NA,NA,NA,NA,$exitcode,\"$notes\"" >> "$OUT_CSV"
    else
      echo "    Parsed $config_count configurations"
      echo "    $mtx_basename (threads=$threads): Parsed $config_count configurations" >> "$OUT_LOG"
    fi

    rm -f "$tmpout"
  done
  echo ""
done

echo "=== Benchmark Complete ==="
echo "Results written to: $OUT_CSV"
echo "Log written to: $OUT_LOG"
wc -l "$OUT_CSV"

# Append completion info to log
{
  echo ""
  echo "========================================================================"
  echo "BENCHMARK COMPLETED"
  echo "========================================================================"
  echo "Completed at: $(date)"
  echo "CSV results: $OUT_CSV"
  echo "Total data rows: $(( $(wc -l < "$OUT_CSV") - 1 ))"
  echo "========================================================================"
} >> "$OUT_LOG"
