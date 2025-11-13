#!/usr/bin/env bash
# bench_configurations_numa.sh
# Benchmark NUMA-aware configurations across large matrices and high thread counts (24-96)
# Usage: ./bench_configurations_numa.sh
set -euo pipefail

EXE="./test_config_numa"
OUT_CSV="evaluation_numa/configurations_numa_results.csv"
OUT_LOG="evaluation_numa/configurations_numa_results.txt"
TIMEOUT_SECS=600   # 10 minutes per matrix/thread combo (large matrices take longer)
ITERATIONS=30      # Reduced iterations for large matrices

# Thread counts: 24-48 (increment 4), 48-96 (increment 6)
THREAD_COUNTS="24 28 32 36 40 44 48 54 60 66 72 78 84 90 96"

# Use matrices_large/ for NUMA benchmarks (direct CSR import, no dense allocation)
MATRICES_DIR="matrices_large"

# Fall back to matrices/ if matrices_large/ doesn't exist
if [ ! -d "$MATRICES_DIR" ]; then
  echo "INFO: matrices_large/ not found, falling back to matrices/" >&2
  MATRICES_DIR="matrices"
fi

if [ ! -d "$MATRICES_DIR" ]; then
  echo "ERROR: Neither matrices_large/ nor matrices/ directory found." >&2
  exit 1
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
  echo "ERROR: executable not found at '$EXE'. Compile with:" >&2
  echo "  gcc -O3 -Wall -Wextra -march=native -fopenmp -o test_config_numa src/test_configurations_numa.c src/generator.c src/m_to_csr.c -lm" >&2
  exit 2
fi
if [ ! -x "$EXE" ]; then
  echo "ERROR: '$EXE' exists but is not executable. Try 'chmod +x $EXE'." >&2
  exit 3
fi

# Create output directory if needed
mkdir -p "$(dirname "$OUT_CSV")"

# CSV header
printf '%s\n' "matrix,rows,cols,density_pct,nnz,threads,configuration,bind_policy,time_ms,speedup,efficiency_pct,std_dev_ms,improvement_pct,exit_code,notes" > "$OUT_CSV"

# Initialize log file
{
  echo "========================================================================"
  echo "NUMA-AWARE CONFIGURATION BENCHMARK LOG"
  echo "========================================================================"
  echo "Started at: $(date)"
  echo "Executable: $EXE"
  echo "Timeout per run: ${TIMEOUT_SECS}s"
  echo "Iterations per config: $ITERATIONS"
  echo "Thread counts: $THREAD_COUNTS"
  echo "Matrices found: ${#MATRICES[@]}"
  echo "Matrix directory: $MATRICES_DIR"
  echo "NUMA nodes: 4 (assumed 24 cores each)"
  echo "========================================================================"
  echo ""
} > "$OUT_LOG"

cleanup() { :; }
trap cleanup EXIT

echo "Starting NUMA-aware configuration benchmark. EXE=$EXE. Timeout per run=${TIMEOUT_SECS}s"
echo "Iterations per config: $ITERATIONS"
echo "Thread counts: $THREAD_COUNTS"
echo "Output: $OUT_CSV"
echo ""

for mtx in "${MATRICES[@]}"; do
  echo "=== Matrix: $mtx ==="
  echo "=== Processing Matrix: $mtx ===" >> "$OUT_LOG"
  if [ ! -f "$mtx" ]; then
    echo "  SKIP: file not found: $mtx"
    echo "  SKIP: file not found: $mtx" >> "$OUT_LOG"
    printf '%s\n' "\"$mtx\",NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,FILE_NOT_FOUND,\"file not found\"" >> "$OUT_CSV"
    continue
  fi

  # Extract just the filename
  mtx_basename=$(basename "$mtx")

  for threads in $THREAD_COUNTS; do
    echo "  threads=$threads"
    tmpout=$(mktemp)
    notes=""

    # Run test_config_numa with timeout
    timeout "$TIMEOUT_SECS" "$EXE" "$threads" "$mtx_basename" "$ITERATIONS" > "$tmpout" 2>&1 || true
    exitcode=$?

    # timeout returns 124 when timed out
    if [ "$exitcode" -eq 124 ]; then
      echo "    TIMEOUT after ${TIMEOUT_SECS}s"
      echo "    TIMEOUT: $mtx_basename, threads=$threads" >> "$OUT_LOG"
      notes="TIMEOUT"
      printf '%s\n' "\"$mtx_basename\",NA,NA,NA,NA,$threads,\"TIMEOUT\",NA,NA,NA,NA,NA,NA,$exitcode,\"$notes\"" >> "$OUT_CSV"
      rm -f "$tmpout"
      continue
    fi

    # Extract matrix info from output
    # Format: "Matrix loaded: 10000 × 10000, 150000 NNZ (1.5000% density)"
    ROWS=$(grep "Matrix loaded:" "$tmpout" | awk '{print $3}' | head -1)
    COLS=$(grep "Matrix loaded:" "$tmpout" | awk '{print $5}' | tr -d ',' | head -1)
    NNZ=$(grep "Matrix loaded:" "$tmpout" | awk '{print $6}' | head -1)
    DENSITY=$(grep "Matrix loaded:" "$tmpout" | awk -F'[()]' '{print $2}' | awk '{print $1}' | tr -d '%' | head -1)

    if [ -z "$ROWS" ]; then ROWS="NA"; fi
    if [ -z "$COLS" ]; then COLS="NA"; fi
    if [ -z "$DENSITY" ]; then DENSITY="NA"; fi
    if [ -z "$NNZ" ]; then NNZ="NA"; fi

    # Parse configuration results
    # Expected format: "ConfigName   Time   Speedup   Efficiency  Improvement"
    config_count=0
    while IFS= read -r line; do
      # Extract fields from the output line
      TIME_MS=$(echo "$line" | awk '{for(i=1;i<=NF;i++) if($i~/^[0-9]+\.[0-9]+$/) {print $i; exit}}')
      SPEEDUP=$(echo "$line" | awk '{for(i=1;i<=NF;i++) if($i~/^[0-9]+\.[0-9]+x$/) {print $i; exit}}' | tr -d 'x')
      EFFICIENCY=$(echo "$line" | awk '{for(i=1;i<=NF;i++) if($i~/^[0-9]+\.[0-9]+%$/) {print $i; exit}}' | tr -d '%')
      STDDEV="0.0"  # Not extracted from simple output, could be added
      IMPROVEMENT=$(echo "$line" | awk '{n=0; for(i=1;i<=NF;i++) if($i~/^[+-]?[0-9]+\.[0-9]+%$/) {n++; if(n==2) {print $i; exit}}}' | tr -d '%')
      
      # Extract configuration name and binding policy
      CONFIG=$(echo "$line" | awk '{
        for(i=1; i<=NF; i++) {
          if($i ~ /^[0-9]/) {
            for(j=1; j<i; j++) printf "%s ", $j
            exit
          }
        }
      }' | sed 's/[[:space:]]*$//')
      
      # Extract binding policy from config name
      BIND_POLICY="unknown"
      if [[ "$CONFIG" == *"+close"* ]]; then
        BIND_POLICY="close"
      elif [[ "$CONFIG" == *"+spread"* ]]; then
        BIND_POLICY="spread"
      elif [[ "$CONFIG" == *"+master"* ]]; then
        BIND_POLICY="master"
      fi
      
      # Write to CSV
      printf '%s\n' "\"$mtx_basename\",$ROWS,$COLS,$DENSITY,$NNZ,$threads,\"$CONFIG\",\"$BIND_POLICY\",$TIME_MS,$SPEEDUP,$EFFICIENCY,$STDDEV,$IMPROVEMENT,$exitcode,\"$notes\"" >> "$OUT_CSV"
      config_count=$((config_count + 1))
    done < <(grep -E "^[A-Za-z].*[0-9]+\.[0-9]+.*[0-9]+\.[0-9]+x" "$tmpout" 2>/dev/null || true)

    if [ "$config_count" -eq 0 ]; then
      echo "    WARNING: No configuration results parsed"
      echo "    WARNING: No configuration results parsed for $mtx_basename, threads=$threads" >> "$OUT_LOG"
      notes="${notes}${notes:+; }NO_CONFIGS"
      printf '%s\n' "\"$mtx_basename\",$ROWS,$COLS,$DENSITY,$NNZ,$threads,\"NONE\",NA,NA,NA,NA,NA,NA,$exitcode,\"$notes\"" >> "$OUT_CSV"
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
