#!/bin/bash
# setup_numa_benchmarks.sh
# Quick setup script for NUMA benchmarking
# Usage: bash setup_numa_benchmarks.sh

echo "=========================================="
echo "NUMA Benchmark Setup"
echo "=========================================="
echo ""

# Make scripts executable
echo "Making scripts executable..."
chmod +x bench_configurations_numa.sh
chmod +x evaluation_numa/analyze_configurations_numa.py

echo "✓ Scripts are executable"
echo ""

# Check for compiler
echo "Checking for GCC..."
if command -v gcc &> /dev/null; then
    gcc --version | head -1
    echo "✓ GCC found"
else
    echo "✗ GCC not found - load module on cluster with: module load gcc91"
fi
echo ""

# Check for required source files
echo "Checking source files..."
required_files=("src/test_configurations_numa.c" "src/generator.c" "src/m_to_csr.c" "src/generator.h" "src/m_to_csr.h")
all_found=true

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "✗ $file - MISSING"
        all_found=false
    fi
done
echo ""

# Check for matrices
echo "Checking matrix directories..."
if [ -d "matrices" ]; then
    count=$(find matrices -name "*.mtx" -type f 2>/dev/null | wc -l)
    echo "✓ matrices/ exists ($count .mtx files found)"
    if [ "$count" -eq 0 ]; then
        echo "  ⚠️  No .mtx files found - add matrices to matrices/"
    fi
else
    echo "✗ matrices/ - MISSING"
    all_found=false
fi

if [ -d "evaluation_numa" ]; then
    echo "✓ evaluation_numa/ exists"
else
    echo "✗ evaluation_numa/ - MISSING"
    all_found=false
fi
echo ""

# Try to compile (if gcc available)
if command -v gcc &> /dev/null && [ "$all_found" = true ]; then
    echo "Attempting test compilation..."
    if gcc -O3 -Wall -Wextra -fopenmp -c src/test_configurations_numa.c -o test_config_numa_test.o 2>/dev/null; then
        echo "✓ test_configurations_numa.c compiles successfully"
        rm -f test_config_numa_test.o
    else
        echo "✗ Compilation failed - check code for errors"
    fi
    echo ""
fi

# Summary
echo "=========================================="
echo "Setup Summary"
echo "=========================================="
if [ "$all_found" = true ]; then
    echo "✓ All required files present"
    echo ""
    echo "Next steps:"
    echo "1. Add .mtx files to matrices/ directory (if needed)"
    echo "2. Submit job: qsub pbs_jobs/run_numa_bench.pbs"
    echo "3. Monitor: tail -f numa_bench.out"
    echo "4. Analyze: python3 evaluation_numa/analyze_configurations_numa.py"
else
    echo "✗ Some files are missing - check errors above"
fi
echo "=========================================="
