# Test Configurations Tool

A standalone program for comparing different OpenMP scheduling strategies for sparse matrix-vector multiplication.

## Purpose

Test and compare various OpenMP configurations without affecting your production benchmarking workflow. Helps identify the optimal scheduling strategy and chunk size for your specific matrices.

## Usage

```bash
# Compile
make test

# Run with default iterations (50)
./test_config <threads> <matrix_file>

# Run with custom iterations
./test_config <threads> <matrix_file> <iterations>

# Examples
./test_config 8 2k_0p22.mtx         # Quick test with 50 iterations
./test_config 8 2k_0p22.mtx 30      # Faster test with 30 iterations
./test_config 16 6k_3p16.mtx 100    # More thorough test
```

## What It Tests

The program automatically tests these configurations:
- **Static scheduling**: Default, chunk=8, chunk=32
- **Dynamic scheduling**: chunk=8, 16, 32, 64
- **Guided scheduling**: chunk=8, 16, 32
- **Auto scheduling**: Runtime-determined

## Output

Produces a comparison table showing:
- **Time(ms)**: Average execution time
- **Speedup**: Compared to serial baseline
- **Efficiency**: Speedup / max_threads (%)
- **StdDev**: Standard deviation of measurements
- **Improvement**: Percentage gain over first config (static default)

The best configuration is marked with `← BEST`.

## Workflow

### Development Phase (Testing Optimizations)
```bash
make test
./test_config 8 6k_3p16.mtx 50      # Test on representative matrix
# Identify best configuration from output
# Apply to main.c
```

### Production Phase (Full Benchmarks)
```bash
make executable
./scripts/bench_matrices.sh                 # Run full benchmark suite
```

## Tips

- **Quick testing**: Use 30-50 iterations to get fast results
- **Final validation**: Use 100+ iterations for statistical confidence
- **Test multiple matrices**: Results may vary by matrix size/density
- **Look for patterns**: Best config should be consistent across similar matrices

## Difference from main.c

| Feature | test_config | main.c (executable) |
|---------|-------------|---------------------|
| Purpose | Compare configurations | Production benchmarking |
| Output | Comparison table | CSV-ready format |
| Iterations | 30-100 (faster) | 166 (thorough) |
| Used by scripts | No | Yes (bench_matrices.sh) |
| Configurations | 11 variants tested | 1 (whatever is coded) |

## Example Output

```
=== Performance Comparison Results ===

Configuration                 Time(ms)    Speedup   Effic(%)     StdDev    Improv(%)
----------------------------------------------------------------------------------------------------
Static (default)                0.526      5.32x      22.2%     0.006ms        0.00% (baseline)
Dynamic, chunk=8                0.438      6.39x      26.6%     0.008ms       16.76% ← BEST
Dynamic, chunk=16               0.441      6.34x      26.4%     0.005ms       16.05%
Guided, chunk=8                 0.474      5.90x      24.6%     0.031ms        9.78%
...

Notes:
  - Best config: Dynamic, chunk=8 (6.39x speedup, 16.76% improvement)
```

## Cleanup

```bash
make clean      # Removes both test_config and executable
```
