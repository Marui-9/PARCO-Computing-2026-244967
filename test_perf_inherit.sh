#!/bin/bash
# Test script to verify perf --inherit captures OpenMP thread events

echo "Testing perf with --inherit flag for OpenMP threads"
echo "===================================================="
echo ""

MATRIX="matrices/2k_0p52.mtx"

if [ ! -f "$MATRIX" ]; then
    echo "ERROR: Matrix file not found: $MATRIX"
    echo "Please run from the project root directory"
    exit 1
fi

echo "Test 1: Without --inherit (will show 0 for threaded work)"
echo "-----------------------------------------------------------"
perf stat -e L1-dcache-loads,L1-dcache-load-misses,branch-loads,branch-load-misses \
    ./executable 8 "$MATRIX" 2>&1 | tail -20

echo ""
echo ""
echo "Test 2: With --inherit (should show actual values)"
echo "---------------------------------------------------"
perf stat --inherit -e L1-dcache-loads,L1-dcache-load-misses,branch-loads,branch-load-misses \
    ./executable 8 "$MATRIX" 2>&1 | tail -20

echo ""
echo "===================================================="
echo "If Test 2 shows non-zero values, the --inherit flag is working!"
