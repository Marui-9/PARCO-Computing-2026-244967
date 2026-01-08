#!/usr/bin/env python3
"""
Analyze MPI scaling with single-node (1 MPI rank) as baseline.

Combines results from:
- mpi_matrices_results.csv (1-node baseline: OMP only performance)
- test_config_mpi_results_*nodes.csv (multi-node MPI results)

Calculates speedup and efficiency relative to single-node performance.
"""

import pandas as pd
import numpy as np
import glob
import os

results_dir = "results"
csv_results_subdir = "test_results_Xnodes"  # Subdirectory for test_config_mpi CSVs

# Load baseline (1-node) results
baseline_file = os.path.join(results_dir, "mpi_matrices_results.csv")
print(f"Loading baseline from: {baseline_file}")
df_baseline = pd.read_csv(baseline_file)

# Extract single-node times (OMP time is single-threaded baseline)
# Use mpi_time_ms as 1-node baseline (single MPI rank, 48 threads via OpenMP)
baseline_data = {}
for _, row in df_baseline.iterrows():
    matrix = row['matrix']
    # Normalize matrix path
    if not matrix.startswith('matrices/'):
        matrix = f"matrices/{matrix}"
    baseline_data[matrix] = row['mpi_time_ms']

print(f"\nLoaded {len(baseline_data)} matrix baselines")
print(f"Sample: {list(baseline_data.items())[:3]}")

# Load multi-node results
node_counts = [2, 3, 4, 5, 6, 7, 8]
all_data = []

for nodes in node_counts:
    csv_file = os.path.join(results_dir, csv_results_subdir, f"test_config_mpi_results_{nodes}nodes.csv")
    if os.path.exists(csv_file):
        print(f"\nLoading {nodes}-node results from: {csv_file}")
        df = pd.read_csv(csv_file)
        print(f"  Loaded {len(df)} rows")
        
        for _, row in df.iterrows():
            matrix = row['matrix']
            avg_time = row['avg_time_ms']
            
            # Look up baseline
            if matrix in baseline_data:
                baseline_time = baseline_data[matrix]
                
                # Skip unmeasurable cases
                if baseline_time < 0.0001 or avg_time < 0.0001 or avg_time == 0:
                    continue
                
                speedup = baseline_time / avg_time
                efficiency = (speedup / nodes) * 100
                efficiency_capped = min(efficiency, 100.0)
                
                all_data.append({
                    'num_nodes': nodes,
                    'matrix': matrix,
                    'config': row['config_name'],
                    'baseline_1node_ms': baseline_time,
                    'avg_time_ms': avg_time,
                    'speedup': speedup,
                    'efficiency': efficiency_capped
                })
            else:
                print(f"  WARNING: Matrix {matrix} not found in baseline!")
    else:
        print(f"  File not found: {csv_file}")

results_df = pd.DataFrame(all_data)

print("\n" + "=" * 90)
print("SCALING ANALYSIS: 1-NODE BASELINE")
print("=" * 90)

print(f"\nTotal measurable test cases: {len(results_df)}")

# Summary statistics
print("\n=== SPEEDUP SUMMARY (1-NODE BASELINE) ===\n")

speedup_stats = results_df.groupby('config').agg({
    'speedup': ['mean', 'std', 'min', 'max', 'count'],
    'efficiency': ['mean', 'std']
}).round(3)

speedup_stats.columns = ['Avg_Speedup', 'Std_Speedup', 'Min_Speedup', 'Max_Speedup',
                         'N_Tests', 'Avg_Efficiency', 'Std_Efficiency']
speedup_stats = speedup_stats.sort_values('Avg_Speedup', ascending=False)

print(speedup_stats.to_string())

print("\n\n=== RANKING BY SPEEDUP ===\n")
for i, (config, row) in enumerate(speedup_stats.iterrows(), 1):
    print(f"{i}. {config:20s}  Speedup: {row['Avg_Speedup']:6.2f}x (±{row['Std_Speedup']:6.2f})  "
          f"Efficiency: {row['Avg_Efficiency']:5.1f}%  Tests: {int(row['N_Tests'])}")

print("\n\n=== EFFICIENCY BY NODE COUNT (1-NODE BASELINE) ===\n")

node_eff = results_df.groupby(['num_nodes', 'config'])['efficiency'].mean().unstack()
print(node_eff.round(1).to_string())

# Per-configuration scaling curves
print("\n\n=== SCALING CURVES (Efficiency % vs Node Count) ===\n")

for config in sorted(results_df['config'].unique()):
    config_data = results_df[results_df['config'] == config].groupby('num_nodes')['efficiency'].mean()
    values = " → ".join([f"{eff:5.1f}%" for eff in config_data.values])
    print(f"{config:20s}: {values}")

# Export combined results
output_file = os.path.join(results_dir, "mpi_scaling_1node_baseline.csv")
results_df.to_csv(output_file, index=False)
print(f"\n✅ Results saved to: {output_file}")

print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)
print(f"Baseline: 1 MPI rank (OpenMP only)")
print(f"Test nodes: {node_counts}")
print(f"Configurations: {results_df['config'].nunique()}")
print(f"Total matrices: {results_df['matrix'].nunique()}")
print(f"Total measurements: {len(results_df)}")
print("=" * 90)

