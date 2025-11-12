#!/usr/bin/env python3
"""
Analyze NUMA-aware configurations_numa_results.csv to examine:
- Scaling behavior from 24-96 threads
- NUMA binding policy impact (close vs spread vs master)
- Cross-NUMA node performance
- Efficiency degradation at high thread counts
"""

import pandas as pd
import numpy as np
import os

# Determine the correct path to the CSV file
if os.path.exists('configurations_numa_results.csv'):
    csv_path = 'configurations_numa_results.csv'
elif os.path.exists('evaluation_numa/configurations_numa_results.csv'):
    csv_path = 'evaluation_numa/configurations_numa_results.csv'
else:
    print("Error: configurations_numa_results.csv not found!")
    print("Run this script from either the project root or evaluation_numa/ directory")
    exit(1)

# Read the CSV file
df = pd.read_csv(csv_path)

# Clean up configuration names
df['configuration'] = df['configuration'].str.strip()

# Remove rows with missing configuration values
df = df.dropna(subset=['configuration'])

# Create matrix size metric
df['matrix_size'] = df['rows'] * df['cols']
df['matrix_size_M'] = df['matrix_size'] / 1e6

# Categorize by binding policy
def categorize_binding(row):
    if pd.isna(row['bind_policy']):
        return 'Unknown'
    return row['bind_policy']

df['bind_category'] = df.apply(categorize_binding, axis=1)

# Categorize by optimization type
def categorize_optimization(config):
    if not isinstance(config, str):
        return 'Unknown'
    if 'Register' in config:
        return 'Register'
    elif 'Affinity' in config:
        return 'Affinity'
    elif 'SIMD' in config:
        return 'SIMD'
    else:
        return 'Basic'

df['opt_category'] = df['configuration'].apply(categorize_optimization)

print("="*80)
print("NUMA-AWARE CONFIGURATION BENCHMARK ANALYSIS")
print("="*80)
print(f"\nTotal rows: {len(df)}")
print(f"Matrices tested: {df['matrix'].nunique()}")
print(f"Thread counts: {sorted(df['threads'].unique())}")
print(f"Configurations tested: {df['configuration'].nunique()}")
print(f"Binding policies: {sorted(df['bind_category'].unique())}")

# Matrix information
print("\n" + "="*80)
print("MATRIX CHARACTERISTICS")
print("="*80)
matrix_info = df.groupby('matrix').agg({
    'rows': 'first',
    'cols': 'first',
    'density_pct': 'first',
    'nnz': 'first',
    'matrix_size_M': 'first'
}).sort_values('matrix_size_M')
print(matrix_info.to_string())

# Binding policy impact
print("\n" + "="*80)
print("SPEEDUP BY BINDING POLICY")
print("="*80)
binding_stats = df.groupby('bind_category')['speedup'].agg([
    'count', 'mean', 'std', 'min', 'max'
]).sort_values('mean', ascending=False)
print(binding_stats.to_string())

# Speedup vs threads by binding policy
print("\n" + "="*80)
print("SPEEDUP BY THREADS AND BINDING POLICY")
print("="*80)
thread_bind_speedup = df.groupby(['threads', 'bind_category'])['speedup'].mean().unstack()
print(thread_bind_speedup.to_string())

# Efficiency vs threads by binding policy
print("\n" + "="*80)
print("EFFICIENCY BY THREADS AND BINDING POLICY")
print("="*80)
thread_bind_efficiency = df.groupby(['threads', 'bind_category'])['efficiency_pct'].mean().unstack()
print(thread_bind_efficiency.to_string())

# Scaling analysis: Compare performance at key thread counts
print("\n" + "="*80)
print("NUMA NODE SCALING ANALYSIS")
print("="*80)
print("\nAverage speedup at key thread counts:")
key_threads = [24, 48, 72, 96]
for threads in key_threads:
    if threads in df['threads'].values:
        avg_speedup = df[df['threads'] == threads]['speedup'].mean()
        avg_efficiency = df[df['threads'] == threads]['efficiency_pct'].mean()
        print(f"  {threads:3d} threads: {avg_speedup:7.2f}x speedup, {avg_efficiency:5.2f}% efficiency")

# Scaling efficiency (speedup ratio when doubling threads)
print("\n" + "="*80)
print("SCALING EFFICIENCY (speedup ratio when doubling threads)")
print("="*80)
thread_pairs = [(24, 48), (48, 96)]
for t1, t2 in thread_pairs:
    if t1 in df['threads'].values and t2 in df['threads'].values:
        # Group by matrix and configuration, compare thread counts
        scaling_data = []
        for matrix in df['matrix'].unique():
            for config in df['configuration'].unique():
                data_t1 = df[(df['threads'] == t1) & (df['matrix'] == matrix) & (df['configuration'] == config)]
                data_t2 = df[(df['threads'] == t2) & (df['matrix'] == matrix) & (df['configuration'] == config)]
                
                if not data_t1.empty and not data_t2.empty:
                    speedup_t1 = data_t1['speedup'].values[0]
                    speedup_t2 = data_t2['speedup'].values[0]
                    if speedup_t1 > 0:
                        ratio = speedup_t2 / speedup_t1
                        scaling_data.append(ratio)
        
        if scaling_data:
            print(f"\n{t1} → {t2} threads (ideal: {t2/t1:.1f}x):")
            print(f"  Mean scaling: {np.mean(scaling_data):.2f}x")
            print(f"  Min:  {np.min(scaling_data):.2f}x")
            print(f"  Max:  {np.max(scaling_data):.2f}x")
            print(f"  Std:  {np.std(scaling_data):.2f}")

# Best configurations at high thread counts
print("\n" + "="*80)
print("TOP 5 CONFIGURATIONS AT HIGH THREAD COUNTS")
print("="*80)
high_thread_counts = [48, 72, 96]
for threads in high_thread_counts:
    if threads in df['threads'].values:
        thread_data = df[df['threads'] == threads]
        top5 = thread_data.nlargest(5, 'speedup')[['configuration', 'bind_policy', 'speedup', 'efficiency_pct']]
        print(f"\n{threads} threads:")
        print(top5.to_string(index=False))

# Binding policy comparison at 96 threads
if 96 in df['threads'].values:
    print("\n" + "="*80)
    print("BINDING POLICY COMPARISON AT 96 THREADS")
    print("="*80)
    data_96 = df[df['threads'] == 96]
    bind_comparison = data_96.groupby('bind_category').agg({
        'speedup': ['mean', 'std', 'max'],
        'efficiency_pct': ['mean', 'std']
    })
    print(bind_comparison.to_string())

# NUMA overhead analysis (compare 24 threads across datasets)
print("\n" + "="*80)
print("NUMA OPTIMIZATION EFFECTIVENESS")
print("="*80)
if 24 in df['threads'].values:
    data_24 = df[df['threads'] == 24]
    
    print("\nAt 24 threads (single NUMA node):")
    bind_24 = data_24.groupby('bind_category')['speedup'].mean().sort_values(ascending=False)
    print(bind_24.to_string())
    
    print("\nComparison: Best at 24 threads vs best at 96 threads")
    best_24 = data_24['speedup'].max()
    best_24_config = data_24.loc[data_24['speedup'].idxmax(), 'configuration']
    
    if 96 in df['threads'].values:
        data_96 = df[df['threads'] == 96]
        best_96 = data_96['speedup'].max()
        best_96_config = data_96.loc[data_96['speedup'].idxmax(), 'configuration']
        
        print(f"  Best @24: {best_24:.2f}x ({best_24_config})")
        print(f"  Best @96: {best_96:.2f}x ({best_96_config})")
        print(f"  Scaling: {best_96/best_24:.2f}x (ideal: 4.0x for 4 NUMA nodes)")

# Key findings
print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)

# 1. Best binding policy
print("\n1. Best Binding Policy:")
best_bind = df.groupby('bind_category')['speedup'].mean().sort_values(ascending=False)
print(f"   Overall best: {best_bind.index[0]} ({best_bind.values[0]:.2f}x avg speedup)")

# 2. Efficiency at high thread counts
if 96 in df['threads'].values:
    efficiency_96 = df[df['threads'] == 96]['efficiency_pct'].mean()
    print(f"\n2. Efficiency at 96 threads:")
    print(f"   Average: {efficiency_96:.2f}%")
    if efficiency_96 < 50:
        print("   WARNING: Low efficiency suggests NUMA bottleneck or poor scaling")

# 3. Configuration recommendation
print("\n3. Recommended Configurations:")
for threads in [48, 96]:
    if threads in df['threads'].values:
        best_config = df[df['threads'] == threads].nlargest(1, 'speedup')
        if not best_config.empty:
            config_name = best_config['configuration'].values[0]
            bind_policy = best_config['bind_policy'].values[0]
            speedup = best_config['speedup'].values[0]
            efficiency = best_config['efficiency_pct'].values[0]
            print(f"   {threads} threads: {config_name} ({bind_policy}) - {speedup:.2f}x ({efficiency:.1f}%)")

print("\n" + "="*80)
print("END OF ANALYSIS")
print("="*80)
