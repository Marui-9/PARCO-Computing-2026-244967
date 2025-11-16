#!/bin/bash
# bench_configurations_numa.sh
# Benchmark NUMA-aware configurations across large matrices and high thread counts (24-96)
# Usage: ./bench_configurations_numa.sh
set -eu

EXE="./test_config_numa"
OUT_CSV="results/configurations_numa_results.csv"
OUT_LOG="results/configurations_numa_results.txt"
TIMEOUT_SECS=900    # 15 minutes per matrix/thread combo
ITERATIONS=10       # 10 iterations for better statistical accuracy

# Thread counts: reduced to 4 key points for 6-hour walltime
THREAD_COUNTS="24 48 72 96"

# Use matrices/ for NUMA benchmarks (direct CSR import, no dense allocation)
MATRICES_DIR="matrices"

if [ ! -d "$MATRICES_DIR" ]; then
  echo "ERROR: matrices/ directory not found." >&2
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
echo "Log: $OUT_LOG"
echo ""

# Flush output
sync

for mtx in "${MATRICES[@]}"; do
  echo "=== Matrix: $mtx ==="
  echo "=== Processing Matrix: $mtx ===" >> "$OUT_LOG"
  
  # Flush log after each matrix header
  sync
  if [ ! -f "$mtx" ]; then
    echo "  SKIP: file not found: $mtx"
    echo "  SKIP: file not found: $mtx" >> "$OUT_LOG"
    printf '%s\n' "\"$mtx\",NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,FILE_NOT_FOUND,\"file not found\"" >> "$OUT_CSV"
    continue
  fi

  # Extract just the filename for display/CSV
  mtx_basename=$(basename "$mtx")
  
  # For test_config_numa, pass just the basename since it adds matrices/ prefix
  # But verify the file exists first
  if [ ! -f "$MATRICES_DIR/$mtx_basename" ] && [ ! -f "matrices/$mtx_basename" ]; then
    echo "  ERROR: Matrix file not accessible: $mtx" | tee -a "$OUT_LOG"
    printf '%s\n' "\"$mtx_basename\",NA,NA,NA,NA,$threads,\"FILE_NOT_FOUND\",NA,NA,NA,NA,NA,NA,1,\"matrix file not accessible\"" >> "$OUT_CSV"
    continue
  fi

  for threads in $THREAD_COUNTS; do
    echo "  threads=$threads"
    echo "  Processing: $mtx_basename, threads=$threads" >> "$OUT_LOG"
    tmpout=$(mktemp)
    tmperr=$(mktemp)
    notes=""
    
    # Verify executable exists and is executable
    if [ ! -x "$EXE" ]; then
      echo "    ERROR: Executable not found or not executable: $EXE" | tee -a "$OUT_LOG"
      printf '%s\n' "\"$mtx_basename\",NA,NA,NA,NA,$threads,\"EXE_NOT_FOUND\",NA,NA,NA,NA,NA,NA,127,\"executable not found\"" >> "$OUT_CSV"
      rm -f "$tmpout" "$tmperr"
      continue
    fi
    
    # Verify matrix file exists (C code will look for matrices/<basename>)
    expected_matrix_path="matrices/$mtx_basename"
    if [ ! -f "$expected_matrix_path" ]; then
      echo "    ERROR: Matrix file not found: $expected_matrix_path" | tee -a "$OUT_LOG"
      printf '%s\n' "\"$mtx_basename\",NA,NA,NA,NA,$threads,\"MATRIX_NOT_FOUND\",NA,NA,NA,NA,NA,NA,2,\"matrix file not found at $expected_matrix_path\"" >> "$OUT_CSV"
      rm -f "$tmpout" "$tmperr"
      continue
    fi
    
    echo "    Running: $EXE $threads $mtx_basename $ITERATIONS" >> "$OUT_LOG"
    echo "    tmpout: $tmpout" >> "$OUT_LOG"
    echo "    tmperr: $tmperr" >> "$OUT_LOG"
    echo "    Start time: $(date)" >> "$OUT_LOG"
    
    # Flush log before potentially long-running command
    sync

    # Test if executable can be invoked at all (quick sanity check)
    echo "    Testing if executable responds..." >> "$OUT_LOG"
    sync
    if ! "$EXE" 2>&1 | head -1 >> "$OUT_LOG"; then
      echo "    Warning: Executable test invocation failed" >> "$OUT_LOG"
    fi
    sync

    # Run test_config_numa with timeout
    # Note: test_config_numa expects basename and adds matrices/ prefix internally
    echo "    Executing timeout command..." >> "$OUT_LOG"
    echo "    Command: timeout $TIMEOUT_SECS $EXE $threads $mtx_basename $ITERATIONS" >> "$OUT_LOG"
    sync
    
    # Add a shorter timeout initially to detect hangs faster
    timeout "$TIMEOUT_SECS" "$EXE" "$threads" "$mtx_basename" "$ITERATIONS" > "$tmpout" 2> "$tmperr" &
    timeout_pid=$!
    echo "    Process started with PID: $timeout_pid" >> "$OUT_LOG"
    sync
    
    # Wait for it to complete
    wait $timeout_pid
    exitcode=$?
    
    echo "    Command completed" >> "$OUT_LOG"
    echo "    Exit code: $exitcode" >> "$OUT_LOG"
    echo "    End time: $(date)" >> "$OUT_LOG"
    
    # Check what's in stderr immediately
    if [ -f "$tmperr" ] && [ -s "$tmperr" ]; then
      echo "    stderr content:" >> "$OUT_LOG"
      head -20 "$tmperr" >> "$OUT_LOG"
    else
      echo "    stderr is empty" >> "$OUT_LOG"
    fi
    sync
    
    # Force sync to ensure files are written
    sync
    sleep 0.5
    
    echo "    Exit code: $exitcode" >> "$OUT_LOG"
    
    # Use stat or ls to check file sizes more reliably
    if [ -f "$tmpout" ]; then
      tmpout_size=$(stat -f%z "$tmpout" 2>/dev/null || stat -c%s "$tmpout" 2>/dev/null || wc -c < "$tmpout" 2>/dev/null || echo "unknown")
      echo "    tmpout size: $tmpout_size bytes" >> "$OUT_LOG"
    else
      echo "    tmpout: FILE NOT FOUND" >> "$OUT_LOG"
    fi
    
    if [ -f "$tmperr" ]; then
      tmperr_size=$(stat -f%z "$tmperr" 2>/dev/null || stat -c%s "$tmperr" 2>/dev/null || wc -c < "$tmperr" 2>/dev/null || echo "unknown")
      echo "    tmperr size: $tmperr_size bytes" >> "$OUT_LOG"
    else
      echo "    tmperr: FILE NOT FOUND" >> "$OUT_LOG"
    fi

    # timeout returns 124 when timed out
    if [ "$exitcode" -eq 124 ]; then
      echo "    TIMEOUT after ${TIMEOUT_SECS}s"
      echo "    TIMEOUT: $mtx_basename, threads=$threads" >> "$OUT_LOG"
      notes="TIMEOUT"
      printf '%s\n' "\"$mtx_basename\",NA,NA,NA,NA,$threads,\"TIMEOUT\",NA,NA,NA,NA,NA,NA,$exitcode,\"$notes\"" >> "$OUT_CSV"
      rm -f "$tmpout" "$tmperr"
      continue
    fi

    # Check for errors
    if [ "$exitcode" -ne 0 ]; then
      echo "    ERROR: Exit code $exitcode" | tee -a "$OUT_LOG"
      echo "    STDERR:" | tee -a "$OUT_LOG"
      cat "$tmperr" | tee -a "$OUT_LOG"
      echo "    STDOUT sample:" | tee -a "$OUT_LOG"
      head -20 "$tmpout" | tee -a "$OUT_LOG"
      notes="EXIT_${exitcode}"
      printf '%s\n' "\"$mtx_basename\",NA,NA,NA,NA,$threads,\"ERROR_EXIT_$exitcode\",NA,NA,NA,NA,NA,NA,$exitcode,\"$notes\"" >> "$OUT_CSV"
      rm -f "$tmpout" "$tmperr"
      continue
    fi

    # check for output presence
    if [ ! -s "$tmpout" ]; then
      notes="${notes}${notes:+;}NO_OUTPUT"
      echo "    WARNING: no stdout output captured from run." | tee -a "$OUT_LOG"
      if [ -s "$tmperr" ]; then
        echo "    STDERR was:" | tee -a "$OUT_LOG"
        cat "$tmperr" | tee -a "$OUT_LOG"
      fi
      printf '%s\n' "\"$mtx_basename\",NA,NA,NA,NA,$threads,\"NO_OUTPUT\",NA,NA,NA,NA,NA,NA,$exitcode,\"$notes\"" >> "$OUT_CSV"
      rm -f "$tmpout" "$tmperr"
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
    # Expected format: "ConfigName                             5.00       2.16x      9.00%     +0.00%"
    # The table appears after "=== Configuration Comparison" line
    config_count=0
    
    # Find the start of the results table and process lines after it
    in_table=0
    while IFS= read -r line; do
      # Detect when we enter the results table
      if [[ "$line" == *"Configuration Comparison"* ]]; then
        in_table=1
        continue
      fi
      
      # Skip header and separator lines
      if [[ "$line" == *"Configuration"*"Time"* ]] || [[ "$line" == *"---"* ]] || [[ "$line" == "" ]]; then
        continue
      fi
      
      # Only process lines if we're in the table section
      if [ $in_table -eq 0 ]; then
        continue
      fi
      
      # Stop if we hit the "Best configuration" line
      if [[ "$line" == *"Best configuration"* ]]; then
        break
      fi
      
      # Parse the line - configuration name should start at beginning, followed by numbers
      # Use awk to parse: first field is config name, rest are numeric values
      CONFIG=$(echo "$line" | awk '{print $1}')
      TIME_MS=$(echo "$line" | awk '{print $2}')
      SPEEDUP=$(echo "$line" | awk '{print $3}' | tr -d 'x')
      EFFICIENCY=$(echo "$line" | awk '{print $4}' | tr -d '%')
      IMPROVEMENT=$(echo "$line" | awk '{print $5}' | tr -d '%+')
      STDDEV="0.0"
      
      # Validate we got actual data (config name and numeric time)
      if [[ -z "$CONFIG" ]] || [[ ! "$TIME_MS" =~ ^[0-9]+\.?[0-9]*$ ]]; then
        continue
      fi
      
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
      echo "      Parsed config $config_count: $CONFIG (time=$TIME_MS ms)" >> "$OUT_LOG"
    done < "$tmpout"

    if [ "$config_count" -eq 0 ]; then
      echo "    WARNING: No configuration results parsed"
      echo "    WARNING: No configuration results parsed for $mtx_basename, threads=$threads" >> "$OUT_LOG"
      echo "    DEBUG: Exit code = $exitcode" >> "$OUT_LOG"
      echo "    DEBUG: First 50 lines of stdout:" >> "$OUT_LOG"
      head -50 "$tmpout" >> "$OUT_LOG" 2>&1 || true
      echo "    DEBUG: Last 50 lines of stdout:" >> "$OUT_LOG"
      tail -50 "$tmpout" >> "$OUT_LOG" 2>&1 || true
      if [ -s "$tmperr" ]; then
        echo "    DEBUG: STDERR content:" >> "$OUT_LOG"
        cat "$tmperr" >> "$OUT_LOG"
      else
        echo "    DEBUG: STDERR was empty" >> "$OUT_LOG"
      fi
      notes="${notes}${notes:+; }NO_CONFIGS"
      printf '%s\n' "\"$mtx_basename\",$ROWS,$COLS,$DENSITY,$NNZ,$threads,\"NONE\",NA,NA,NA,NA,NA,NA,$exitcode,\"$notes\"" >> "$OUT_CSV"
    else
      echo "    Parsed $config_count configurations"
      echo "    $mtx_basename (threads=$threads): Parsed $config_count configurations" >> "$OUT_LOG"
    fi

    rm -f "$tmpout" "$tmperr"
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
