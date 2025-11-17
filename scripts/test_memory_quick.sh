#!/bin/bash
# Quick test of Phase 3 memory optimizations

echo "=== Testing Phase 3: Memory Optimizations ==="
echo ""

echo "Step 1: Compiling test_memory_opt..."
make test_memory

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

echo "✓ Compilation successful"
echo ""

echo "Step 2: Quick test run (10k matrix, 8 threads, 10 iterations)..."
./test_memory_opt 8 matrices/10k_1p5.mtx 10

if [ $? -ne 0 ]; then
    echo "Test run failed!"
    exit 1
fi

echo ""
echo "✓ Phase 3 test complete!"
echo ""
echo "To run full benchmarks:"
echo "  chmod +x scripts/bench_memory_optimizations.sh"
echo "  ./scripts/bench_memory_optimizations.sh"
