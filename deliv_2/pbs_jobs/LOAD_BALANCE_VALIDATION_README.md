# Load Balancing Validation PBS Jobs

This directory contains PBS job scripts for validating load balancing improvements on the cluster.

## Available Scripts

### Individual Node Tests

1. **`lb_validation_6nodes.pbs`** - Tests 6 nodes (288 cores)
   - Resources: 6 nodes × 48 cores × 32GB RAM
   - Runtime: 2 hours
   - Tests the most irregular matrix (1585k_0p0002.mtx)

2. **`lb_validation_7nodes.pbs`** - Tests 7 nodes (336 cores)
   - Resources: 7 nodes × 48 cores × 32GB RAM
   - Runtime: 2 hours
   - Tests at the worst-performing node count from current analysis

### Combined Test

3. **`lb_validation_both.pbs`** - Tests both 6 and 7 nodes in one job
   - Resources: 7 nodes × 48 cores × 32GB RAM (runs 6-node test first)
   - Runtime: 4 hours
   - Most efficient option - runs both tests sequentially

## Submission Instructions

### On the Cluster

```bash
# Navigate to project directory
cd ~/PARCO-Computing-2026-244967/deliv_2

# Submit combined test (recommended)
qsub pbs_jobs/lb_validation_both.pbs

# OR submit individual tests
qsub pbs_jobs/lb_validation_6nodes.pbs
qsub pbs_jobs/lb_validation_7nodes.pbs

# Check job status
qstat -u $USER

# Monitor output (once job starts)
tail -f lb_validation_both.out
# or
tail -f lb_validation_6nodes.out
```

## What Gets Tested

Each validation test:

1. **Compiles** the load balancing validation program with all dependencies
2. **Runs two benchmarks** on the same matrix:
   - **ROW-BASED**: Current static row partitioning (baseline)
   - **NNZ-BASED**: Work-balanced distribution by NNZ count
3. **Measures**:
   - Total execution time
   - Compute time breakdown
   - Communication time breakdown
   - Load imbalance factor across ranks
4. **Reports**:
   - Performance improvement (%)
   - Speedup factor
   - Recommendation on whether to implement

## Expected Results

Based on analysis predictions:

| Nodes | Current Performance | Predicted Improvement |
|-------|---------------------|----------------------|
| 6     | 9.87× speedup (165% eff) | +10-15% improvement |
| 7     | 8.72× speedup (125% eff) | +15-20% improvement |

**Decision Criteria:**
- **> 15% improvement**: Implement NNZ-based for production
- **10-15% improvement**: Consider for irregular matrices
- **< 10% improvement**: Current row-based is sufficient

## Output Files

Results are saved to:
```
results/load_balancing/
├── validation_6nodes_cluster.txt    # Detailed 6-node results
└── validation_7nodes_cluster.txt    # Detailed 7-node results
```

Also captured in PBS output files:
```
lb_validation_both.out              # Combined stdout
lb_validation_both.err              # Combined stderr
lb_validation_6nodes.out            # Individual stdout
lb_validation_7nodes.out            # Individual stdout
```

## Quick Results Check

After job completes:

```bash
# View summary from combined test
grep -A 15 "VALIDATION TESTS SUMMARY" lb_validation_both.out

# View specific conclusions
grep -A 5 "CONCLUSION" results/load_balancing/validation_6nodes_cluster.txt
grep -A 5 "CONCLUSION" results/load_balancing/validation_7nodes_cluster.txt

# Check speedup improvements
grep "Speedup Factor" results/load_balancing/*.txt
```

## Troubleshooting

If compilation fails:
- Check that all source files exist (load_balance.c, load_balance.h)
- Verify gcc91 and mpich modules are loaded
- Check PBS error file for compiler messages

If runtime fails:
- Verify matrix file exists: `ls matrices/1585k_0p0002.mtx`
- Check that OMP_NUM_THREADS is set to 48
- Review PBS stderr file for MPI errors

## Next Steps After Validation

1. **If improvement > 15%**:
   - Integrate NNZ-based distribution into production code
   - Run full benchmarks with all matrices
   - Document performance gains in report

2. **If improvement 10-15%**:
   - Document as "optional optimization"
   - Implement for irregular matrices only
   - Keep framework for future use

3. **If improvement < 10%**:
   - Document as "explored optimization"
   - Note predicted vs actual improvement
   - Keep framework for reference
   - Focus on current excellent results (14× speedup)
