#!/usr/bin/env python3
"""
Clean analysis script that filters anomalous data points
"""
import pandas as pd
import numpy as np
import sys

# Load all data
nodes = [1, 2, 3, 4, 5, 6, 7, 8]
dfs = []
for n in nodes:
    df = pd.read_csv(f'results/test_results_Xnodes/test_config_mpi_results_{n}nodes.csv')
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)

# Filter valid matrices (nnz > 0)
df_valid = df_all[df_all['nnz'] > 0].copy()

# === ANOMALY DETECTION ===
# Remove rows where execution time is suspiciously low
# Physical constraint: Even with perfect scaling, time cannot be < baseline/nodes
df_1node = df_valid[df_valid['num_nodes'] == 1]

# Calculate minimum expected time for each matrix and config
def filter_anomalies(df, df_baseline):
    """Remove data points with physically impossible timing"""
    filtered_rows = []
    
    for _, row in df.iterrows():
        n = row['num_nodes']
        config = row['config_name']
        matrix = row['matrix']
        time = row['avg_time_ms']
        
        # Get baseline time for this specific matrix+config
        baseline_row = df_baseline[
            (df_baseline['config_name'] == config) & 
            (df_baseline['matrix'] == matrix)
        ]
        
        if len(baseline_row) > 0:
            baseline_time = baseline_row['avg_time_ms'].values[0]
            
            # Physical limit: Cannot be faster than perfect linear scaling
            # Add 20% margin for measurement noise
            min_expected_time = (baseline_time / n) * 0.8
            
            # Also filter if time is too fast relative to baseline (< 10% of 1-node)
            if n > 1 and time < min_expected_time and time < baseline_time * 0.1:
                # Skip this anomalous data point
                continue
        
        filtered_rows.append(row)
    
    return pd.DataFrame(filtered_rows)

# Apply filtering
print("=" * 120)
print("DATA CLEANING AND ANOMALY FILTERING")
print("=" * 120)
print(f"\nOriginal valid data points: {len(df_valid)}")

df_cleaned = filter_anomalies(df_valid, df_1node)

print(f"After anomaly filtering:    {len(df_cleaned)}")
print(f"Removed anomalies:          {len(df_valid) - len(df_cleaned)}")

# Show what was removed
df_removed = df_valid[~df_valid.index.isin(df_cleaned.index)]
if len(df_removed) > 0:
    print("\n\nREMOVED ANOMALOUS DATA POINTS:")
    print("-" * 120)
    print(f"{'Nodes':<7} {'Config':<22} {'Matrix':<25} {'Time (ms)':<12} {'Expected >':<15} {'Issue'}")
    print("-" * 120)
    
    for _, row in df_removed.iterrows():
        n = row['num_nodes']
        config = row['config_name']
        matrix = row['matrix']
        time = row['avg_time_ms']
        
        baseline_row = df_1node[
            (df_1node['config_name'] == config) & 
            (df_1node['matrix'] == matrix)
        ]
        
        if len(baseline_row) > 0:
            baseline_time = baseline_row['avg_time_ms'].values[0]
            min_expected = (baseline_time / n) * 0.8
            issue = f"Too fast ({time/baseline_time*100:.1f}% of baseline)"
            print(f"{n:<7} {config:<22} {matrix:<25} {time:>8.2f}    {min_expected:>8.2f}      {issue}")

# Save cleaned dataset
df_cleaned.to_csv('results/cleaned_mpi_results.csv', index=False)
print(f"\n\nCleaned dataset saved to: results/cleaned_mpi_results.csv")

# Re-run analysis on cleaned data
configs = ['MPI_Bcast+Gatherv', 'Ibcast/Igatherv', 'Async_Collectives']

print("\n\n" + "=" * 120)
print("CLEANED PERFORMANCE ANALYSIS")
print("=" * 120)

# Get cleaned 1-node baseline
df_1node_clean = df_cleaned[df_cleaned['num_nodes'] == 1]

print("\n1. SPEEDUP ANALYSIS (Cleaned Data)")
print("-" * 120)
print(f"{'Nodes':<8} {'Cores':<8} {'Bcast+Gatherv':<18} {'Ibcast/Igatherv':<18} {'Async_Collectives':<18}")
print("-" * 120)

for n in nodes:
    df_n = df_cleaned[df_cleaned['num_nodes'] == n]
    cores = n * 48
    
    speedups = []
    for config in configs:
        df_config = df_n[df_n['config_name'] == config]
        if len(df_config) > 0:
            avg_time = df_config['avg_time_ms'].mean()
            baseline_config = df_1node_clean[df_1node_clean['config_name'] == config]['avg_time_ms'].mean()
            if baseline_config > 0:
                speedup = baseline_config / avg_time
                speedups.append(speedup)
            else:
                speedups.append(0)
        else:
            speedups.append(0)
    
    print(f"{n:<8} {cores:<8} {speedups[0]:6.2f}×           {speedups[1]:6.2f}×           {speedups[2]:6.2f}×")

print("\n\n2. PARALLEL EFFICIENCY (Cleaned Data)")
print("-" * 120)
print(f"{'Nodes':<8} {'Cores':<8} {'Bcast+Gatherv':<18} {'Ibcast/Igatherv':<18} {'Async_Collectives':<18}")
print("-" * 120)

for n in nodes:
    df_n = df_cleaned[df_cleaned['num_nodes'] == n]
    cores = n * 48
    
    efficiencies = []
    for config in configs:
        df_config = df_n[df_n['config_name'] == config]
        if len(df_config) > 0:
            avg_time = df_config['avg_time_ms'].mean()
            baseline_config = df_1node_clean[df_1node_clean['config_name'] == config]['avg_time_ms'].mean()
            if baseline_config > 0:
                speedup = baseline_config / avg_time
                efficiency = 100 * speedup / n
                efficiencies.append(efficiency)
            else:
                efficiencies.append(0)
        else:
            efficiencies.append(0)
    
    print(f"{n:<8} {cores:<8} {efficiencies[0]:6.2f}%          {efficiencies[1]:6.2f}%          {efficiencies[2]:6.2f}%")

print("\n\n3. KEY OBSERVATIONS")
print("-" * 120)
print("- Anomalous data points filtered based on physical constraints")
print("- Removed data where execution time was < 10% of baseline (impossible speedups)")
print("- Cleaned dataset provides more realistic scaling analysis")
print("\n\nNOTE: Re-running benchmarks is recommended to replace filtered data points.")
