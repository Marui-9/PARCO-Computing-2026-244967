#!/usr/bin/env bash
# bench_fixed_matrices_debug.sh
# Safer, verbose runner for fixed matrix list.
# Usage: ./bench_fixed_matrices_debug.sh
set -euo pipefail

EXE="./executable"
OUT_CSV="results/matrices_results.csv"
TIMEOUT_SECS=120   # kill any run that exceeds this

# Thread configuration - customize as needed
# Option 1: Sequential (1 to 16)
# THREAD_COUNTS=$(seq 1 16)

# Option 2: Powers of 2 (more common for benchmarking)
THREAD_COUNTS="2 4 8 12 16 18 20 22 24"

# Option 3: Specific values
# THREAD_COUNTS="1 4 8 12 16 20 24"

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

# quick checks
if [ ! -f "$EXE" ]; then
  echo "ERROR: executable not found at '$EXE'. Adjust EXE variable." >&2
  exit 2
fi
if [ ! -x "$EXE" ]; then
  echo "ERROR: '$EXE' exists but is not executable. Try 'chmod +x $EXE'." >&2
  exit 3
fi

# header
printf '%s\n' "matrix,rows,cols,density_pct,threads,speedup_x,serial_ms,parallel_ms,exit_code,notes" > "$OUT_CSV"

cleanup() { :; }
trap cleanup EXIT

parse_metadata() {
  local tmp="$1"
  local line
  line=$(grep -m1 -E "Matrix dimensions" "$tmp" || true)
  if [ -n "$line" ]; then
    rows=$(echo "$line" | sed -E 's/.*: *([0-9]+)[[:space:]]*rows.*/\1/')
    cols=$(echo "$line" | sed -E 's/.*rows[^0-9]*([0-9]+)[[:space:]]*cols.*/\1/')
    dens=$(echo "$line" | sed -E 's/.* ([0-9]+(\.[0-9]+)?)% .*non-*/\1/')
  else
    rows=NA; cols=NA; dens=NA
  fi
}

parse_results() {
  local tmp="$1"
  speedup=$(grep -m1 "Average speedup" "$tmp" | sed -E 's/.*: *([0-9]+(\.[0-9]+)?)x.*/\1/' || echo NA)
  serial=$(grep -m1 "Average serial execution time" "$tmp" | sed -E 's/.*: *([0-9]+(\.[0-9]+)?)\s*milliseconds.*/\1/' || echo NA)
  parallel=$(grep -m1 "Average parallel execution time" "$tmp" | sed -E 's/.*: *([0-9]+(\.[0-9]+)?)\s*milliseconds.*/\1/' || echo NA)
}

echo "Starting benchmark. EXE=$EXE. Timeout per run=${TIMEOUT_SECS}s"
echo "Thread counts: $THREAD_COUNTS"

for mtx in "${MATRICES[@]}"; do
  echo "=== Matrix: $mtx ==="
  if [ ! -f "$mtx" ]; then
    echo "  SKIP: file not found: $mtx"
    printf '%s\n' "\"$mtx\",NA,NA,NA,NA,NA,NA,NA,FILE_NOT_FOUND" >> "$OUT_CSV"
    continue
  fi

  # Extract just the filename (main.c adds "matrices/" prefix)
  mtx_basename=$(basename "$mtx")

  for threads in $THREAD_COUNTS; do
    echo "  threads=$threads"
    tmpout=$(mktemp)
    notes=""

    # Run with timeout. adjust command if your program expects OMP_NUM_THREADS instead of arg.
    # If your program uses OMP_NUM_THREADS, uncomment the export line and remove threads arg.
    # export OMP_NUM_THREADS="$threads"
    # timeout "$TIMEOUT_SECS" "$EXE" "$mtx" > "$tmpout" 2>&1 || true

    timeout "$TIMEOUT_SECS" "$EXE" "$threads" "$mtx_basename" > "$tmpout" 2>&1 || true
    exitcode=$?

    # timeout returns 124 when timed out
    if [ "$exitcode" -eq 124 ]; then
      notes="TIMED_OUT"
      echo "    Run timed out after ${TIMEOUT_SECS}s"
    fi

    # check for output presence
    if [ ! -s "$tmpout" ]; then
      notes="${notes}${notes:+; }NO_OUTPUT"
      echo "    WARNING: no output captured from run. Check whether program expects stdin or different args."
      # add an empty or dummy line in CSV with note
      printf '%s\n' "\"$mtx\",NA,NA,NA,$threads,NA,NA,NA,$exitcode,\"$notes\"" >> "$OUT_CSV"
      rm -f "$tmpout"
      continue
    fi

    # quick diagnostics print last 20 lines if nonzero exit
    if [ "$exitcode" -ne 0 ] && [ "$exitcode" -ne 124 ]; then
      echo "    Run exited with code $exitcode. Last 20 lines of output:"
      tail -n 20 "$tmpout" | sed 's/^/      /'
      notes="${notes}${notes:+; }EXIT_${exitcode}"
    fi

    # parse metadata & results
    parse_metadata "$tmpout"
    parse_results "$tmpout"

    # if parsing failed, dump a short snippet for debugging
    if [ "$rows" = "NA" ] || [ "$speedup" = "NA" ]; then
      echo "    Parsing incomplete. Showing first 40 and last 20 lines of output:"
      echo "----- head -----"
      head -n 40 "$tmpout" | sed 's/^/      /'
      echo "----- tail -----"
      tail -n 20 "$tmpout" | sed 's/^/      /'
      notes="${notes}${notes:+; }PARSE_INCOMPLETE"
    fi

    # write CSV line
    safe_mtx=$(printf '%s' "$mtx" | sed 's/"/""/g')
    printf '%s\n' "\"$safe_mtx\",$rows,$cols,$dens,$threads,$speedup,$serial,$parallel,$exitcode,\"$notes\"" >> "$OUT_CSV"

    rm -f "$tmpout"
  done
done

echo "Done. Results: $OUT_CSV"
