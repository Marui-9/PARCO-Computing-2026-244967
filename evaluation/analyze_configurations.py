#!/usr/bin/env python3
"""
Analyze configurations_results.csv to examine speedup behavior across:
- Matrix size (rows × cols)
- Density (percentage of nonzero entries)
- Number of threads
- Configuration types
"""

import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('evaluation/configurations_results.csv')

# Clean up configuration names (remove trailing commas)
df['configuration'] = df['configuration'].str.strip()

# Create matrix size metric (total elements)
df['matrix_size'] = df['rows'] * df['cols']
df['matrix_size_M'] = df['matrix_size'] / 1e6  # in millions

# Create configuration categories
def categorize_config(config):
    if 'SIMD+Affinity' in config:
        return 'SIMD+Affinity'
    elif 'SIMD' in config:
        return 'SIMD'
    elif 'default' in config:
        return 'Default'
    else:
        return 'Scheduling'

df['config_category'] = df['configuration'].apply(categorize_config)

print("="*80)
print("CONFIGURATION BENCHMARK ANALYSIS")
print("="*80)
print(f"\nTotal rows: {len(df)}")
print(f"Matrices tested: {df['matrix'].nunique()}")
print(f"Thread counts: {sorted(df['threads'].unique())}")
print(f"Configurations per matrix/thread: {df['configuration'].nunique()}")

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

# Overall speedup statistics by configuration category
print("\n" + "="*80)
print("SPEEDUP BY CONFIGURATION CATEGORY (all matrices, all threads)")
print("="*80)
category_stats = df.groupby('config_category')['speedup'].agg([
    'count', 'mean', 'std', 'min', 'max'
]).sort_values('mean', ascending=False)
print(category_stats.to_string())

# Speedup vs threads for each configuration category
print("\n" + "="*80)
print("SPEEDUP BY THREADS AND CONFIGURATION CATEGORY")
print("="*80)
thread_speedup = df.groupby(['threads', 'config_category'])['speedup'].mean().unstack()
print(thread_speedup.to_string())

# Best configurations per thread count
print("\n" + "="*80)
print("TOP 5 CONFIGURATIONS BY THREAD COUNT")
print("="*80)
for threads in sorted(df['threads'].unique()):
    thread_data = df[df['threads'] == threads]
    top5 = thread_data.nlargest(5, 'speedup')[['configuration', 'speedup', 'efficiency_pct']]
    print(f"\n{threads} threads:")
    print(top5.to_string(index=False))

# Speedup vs matrix size
print("\n" + "="*80)
print("SPEEDUP VS MATRIX SIZE (by category, at 12 threads)")
print("="*80)
size_speedup = df[df['threads'] == 12].groupby(['matrix', 'config_category'])['speedup'].mean().unstack()
size_speedup = size_speedup.join(matrix_info[['matrix_size_M', 'density_pct']])
size_speedup = size_speedup.sort_values('matrix_size_M')
print(size_speedup.to_string())

# Speedup vs density
print("\n" + "="*80)
print("SPEEDUP VS DENSITY (SIMD configs only, at 12 threads)")
print("="*80)
density_data = df[(df['threads'] == 12) & (df['config_category'].isin(['SIMD', 'SIMD+Affinity']))]
density_speedup = density_data.groupby('matrix').agg({
    'density_pct': 'first',
    'nnz': 'first',
    'speedup': 'mean'
}).sort_values('density_pct')
print(density_speedup.to_string())

# Efficiency analysis
print("\n" + "="*80)
print("PARALLEL EFFICIENCY BY THREAD COUNT (SIMD configs)")
print("="*80)
simd_data = df[df['config_category'].isin(['SIMD', 'SIMD+Affinity'])]
efficiency = simd_data.groupby('threads')['efficiency_pct'].agg(['mean', 'std', 'min', 'max'])
print(efficiency.to_string())

# Scaling analysis: speedup improvement with threads
print("\n" + "="*80)
print("SCALING ANALYSIS: Speedup gain per thread doubling (SIMD configs)")
print("="*80)
simd_scaling = simd_data.groupby(['matrix', 'threads'])['speedup'].mean().unstack()
print("\nSpeedup by matrix and thread count:")
print(simd_scaling.to_string())

# Calculate scaling efficiency (speedup ratio when doubling threads)
print("\nScaling efficiency (speedup ratio when doubling threads):")
for col_pair in [(1, 2), (2, 4), (4, 8), (8, 16)]:
    if col_pair[0] in simd_scaling.columns and col_pair[1] in simd_scaling.columns:
        ratio = simd_scaling[col_pair[1]] / simd_scaling[col_pair[0]]
        print(f"\n{col_pair[0]} → {col_pair[1]} threads (ideal: 2.0x):")
        print(f"  Mean: {ratio.mean():.2f}x")
        print(f"  Min:  {ratio.min():.2f}x")
        print(f"  Max:  {ratio.max():.2f}x")

# Key findings
print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)

# 1. SIMD impact
no_simd = df[df['config_category'] == 'Default']['speedup'].mean()
with_simd = df[df['config_category'].isin(['SIMD', 'SIMD+Affinity'])]['speedup'].mean()
print(f"\n1. SIMD Impact:")
print(f"   Average speedup without SIMD: {no_simd:.2f}x")
print(f"   Average speedup with SIMD:    {with_simd:.2f}x")
print(f"   SIMD improvement:              {with_simd/no_simd:.2f}x")

# 2. Thread affinity impact
simd_only = df[df['config_category'] == 'SIMD']['speedup'].mean()
simd_affinity = df[df['config_category'] == 'SIMD+Affinity']['speedup'].mean()
print(f"\n2. Thread Affinity Impact (on SIMD configs):")
print(f"   SIMD without affinity: {simd_only:.2f}x")
print(f"   SIMD with affinity:    {simd_affinity:.2f}x")
print(f"   Affinity improvement:  {((simd_affinity/simd_only - 1) * 100):.2f}%")

# 3. Matrix size impact
small_matrices = df[df['matrix_size_M'] < 5]['speedup'].mean()
large_matrices = df[df['matrix_size_M'] >= 10]['speedup'].mean()
print(f"\n3. Matrix Size Impact:")
print(f"   Small matrices (<5M elements): {small_matrices:.2f}x avg speedup")
print(f"   Large matrices (≥10M elements): {large_matrices:.2f}x avg speedup")

# 4. Density impact (on SIMD configs)
simd_df = df[df['config_category'].isin(['SIMD', 'SIMD+Affinity'])]
sparse_matrices = simd_df[simd_df['density_pct'] < 0.5]['speedup'].mean()
dense_matrices = simd_df[simd_df['density_pct'] >= 1.0]['speedup'].mean()
print(f"\n4. Density Impact (SIMD configs only):")
print(f"   Sparse matrices (<0.5% density): {sparse_matrices:.2f}x avg speedup")
print(f"   Dense matrices (≥1.0% density):  {dense_matrices:.2f}x avg speedup")

# 5. Optimal thread count
print(f"\n5. Optimal Thread Count (SIMD configs):")
for matrix in df['matrix'].unique():
    matrix_data = df[(df['matrix'] == matrix) & (df['config_category'].isin(['SIMD', 'SIMD+Affinity']))]
    best_threads = matrix_data.groupby('threads')['speedup'].mean().idxmax()
    best_speedup = matrix_data.groupby('threads')['speedup'].mean().max()
    print(f"   {matrix:20s}: {best_threads:2d} threads ({best_speedup:.1f}x speedup)")

print("\n" + "="*80)
print("END OF ANALYSIS")
print("="*80)
