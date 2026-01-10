#!/bin/bash
#
# Load balancing validation benchmark
# Tests row-based vs NNZ-based distribution on irregular matrices
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "================================================================================"
echo "  LOAD BALANCING VALIDATION BENCHMARK"
echo "================================================================================"
echo ""
echo "This script validates load balancing improvements by:"
echo "  1. Testing the most irregular matrix (1585k_0p0002.mtx)"
echo "  2. Testing at problematic node counts (6-7 nodes)"
echo "  3. Comparing row-based vs NNZ-based distribution"
echo ""
echo "================================================================================"
echo ""

# Compile the validation test
echo "Compiling validation test..."
mpicc -g -Wall -O3 -fopenmp -o test_lb_validation \
    src/test_load_balance_validation.c \
    src/load_balance.c \
    src/generator.c \
    src/m_to_csr.c \
    -lm

if [ $? -ne 0 ]; then
    echo "ERROR: Compilation failed"
    exit 1
fi

echo "✓ Compilation successful"
echo ""

# Create results directory
mkdir -p results/load_balancing

# Test matrix (most irregular from analysis)
MATRIX="matrices/1585k_0p0002.mtx"

if [ ! -f "$MATRIX" ]; then
    echo "ERROR: Matrix file not found: $MATRIX"
    echo "Available matrices:"
    ls -1 matrices/*.mtx 2>/dev/null || echo "  No matrices found in matrices/"
    exit 1
fi

echo "Test Configuration:"
echo "  Matrix:       $MATRIX"
echo "  Iterations:   20 per test"
echo "  Threads/rank: 48"
echo ""

# Test 1: 6 nodes (high communication overhead region)
echo "================================================================================"
echo "TEST 1: 6 NODES (288 total cores)"
echo "================================================================================"
echo ""

OUTPUT_FILE="results/load_balancing/validation_6nodes.txt"

mpirun -np 6 ./test_lb_validation "$MATRIX" 20 2>&1 | tee "$OUTPUT_FILE"

echo ""
echo "Results saved to: $OUTPUT_FILE"
echo ""

# Test 2: 7 nodes (worst efficiency in current results)
echo ""
echo "================================================================================"
echo "TEST 2: 7 NODES (336 total cores)"
echo "================================================================================"
echo ""

OUTPUT_FILE="results/load_balancing/validation_7nodes.txt"

mpirun -np 7 ./test_lb_validation "$MATRIX" 20 2>&1 | tee "$OUTPUT_FILE"

echo ""
echo "Results saved to: $OUTPUT_FILE"
echo ""

# Summary
echo ""
echo "================================================================================"
echo "  VALIDATION COMPLETE"
echo "================================================================================"
echo ""
echo "Results saved in: results/load_balancing/"
echo ""
echo "Next Steps:"
echo "  1. Review validation results above"
echo "  2. If improvement > 10%, consider full implementation"
echo "  3. If improvement < 10%, current row-based is sufficient"
echo ""
echo "To analyze results:"
echo "  cat results/load_balancing/validation_6nodes.txt | grep 'CONCLUSION' -A 10"
echo "  cat results/load_balancing/validation_7nodes.txt | grep 'CONCLUSION' -A 10"
echo ""
