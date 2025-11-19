#!/usr/bin/env python3
"""
Plot cache performance metrics from perf benchmark results.
Creates visualizations of L1 cache behavior, IPC, branch prediction, and execution time.
Usage: python3 plot_cache.py [cache_perf_results.csv]
"""

import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import re
import numpy as np


# Read CSV file
csv_file = sys.argv[1] if len(sys.argv) > 1 else 'results/cache_valgrind_results.csv'

if not os.path.exists(csv_file):
    print(f"Error: {csv_file} not found")
    sys.exit(1)

# Read the data
df = pd.read_csv(csv_file)

# Clean column names (strip quotes and whitespace)
df.columns = df.columns.str.strip().str.replace('"', '')
df['matrix'] = df['matrix'].str.strip().str.replace('"', '')

# Fix columns: cols is not numeric, so set to 0 or skip size calc
df['rows'] = pd.to_numeric(df['rows'], errors='coerce')
df['threads'] = pd.to_numeric(df['threads'], errors='coerce')

# Use d_refs for L1 data cache loads, d1_misses for misses, d1_miss_rate for miss rate
# Use branches, branch_misses, branch_miss_rate for branch metrics
# Extract density percentage from density_pct column (e.g., "0.23%" -> 0.23)
if 'density_pct' in df.columns:
    df['density_clean'] = df['density_pct'].apply(
        lambda x: float(re.findall(r'\d+\.\d+', str(x))[0]) if pd.notna(x) and re.findall(r'\d+\.\d+', str(x)) else 0.0
    )
else:
    df['density_clean'] = 0.0

# Remove rows with NA values in critical fields
df = df.dropna(subset=['d1_miss_rate', 'branch_miss_rate', 'threads'])

if len(df) == 0:
    print("Error: No valid data found in CSV file after cleaning.")
    print("The cache_valgrind_results.csv may be malformed or empty.")
    sys.exit(1)

# Extract matrix basename for cleaner labels
df['matrix_name'] = df['matrix'].apply(lambda x: os.path.basename(x).replace('.mtx', ''))

# Matrix size: only use rows (since cols is not numeric)
df['matrix_size'] = df['rows']



# Get unique matrices and threads
matrices = sorted(df['matrix_name'].unique())
thread_counts = sorted(df['threads'].unique())

# Create figure with 3 subplots (1x3) since no time_ms
fig = plt.figure(figsize=(18, 6))
gs = fig.add_gridspec(1, 3, hspace=0.35, wspace=0.3)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])

# ==================== PLOT 1: L1 Cache Miss Rate vs Threads ====================
for matrix in matrices:
    matrix_data = df[df['matrix_name'] == matrix].sort_values('threads')
    ax1.plot(matrix_data['threads'], matrix_data['d1_miss_rate'], 
             marker='o', linewidth=2, markersize=6, label=matrix, alpha=0.8)

ax1.set_xlabel('Number of Threads', fontsize=12, fontweight='bold')
ax1.set_ylabel('L1 Data Cache Miss Rate (%)', fontsize=12, fontweight='bold')
ax1.set_title('L1 Cache Miss Rate vs Number of Threads', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(loc='best', fontsize=8, ncol=2)
ax1.set_xticks(thread_counts)

# ==================== PLOT 2: Branch Miss Rate vs Threads ====================
for matrix in matrices:
    matrix_data = df[df['matrix_name'] == matrix].sort_values('threads')
    ax2.plot(matrix_data['threads'], matrix_data['branch_miss_rate'], 
             marker='s', linewidth=2, markersize=6, label=matrix, alpha=0.8)

ax2.set_xlabel('Number of Threads', fontsize=12, fontweight='bold')
ax2.set_ylabel('Branch Miss Rate (%)', fontsize=12, fontweight='bold')
ax2.set_title('Branch Misprediction Rate vs Number of Threads', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(loc='best', fontsize=8, ncol=2)
ax2.set_xticks(thread_counts)

# ==================== PLOT 3: L1 Cache Miss Rate vs Matrix Size ====================
# Get average performance for each matrix across all thread counts
avg_perf = df.groupby('matrix_name').agg({
    'd1_miss_rate': 'mean',
    'branch_miss_rate': 'mean',
    'matrix_size': 'first',
    'density_clean': 'first'
}).reset_index().sort_values('matrix_size')


# Scatter plot
scatter1 = ax3.scatter(avg_perf['matrix_size'], avg_perf['d1_miss_rate'],
           s=200, alpha=0.7, c=avg_perf['density_clean'], 
           cmap='viridis', edgecolors='black', linewidth=1.5, zorder=3)

# Add trend line
if len(avg_perf) > 2:
    valid_data = avg_perf[avg_perf['matrix_size'] > 0]
    if len(valid_data) > 2:
        z = np.polyfit(np.log10(valid_data['matrix_size']), 
                       valid_data['d1_miss_rate'], 1)
        p = np.poly1d(z)
        x_line = np.logspace(np.log10(valid_data['matrix_size'].min()),
                             np.log10(valid_data['matrix_size'].max()), 100)
        ax3.plot(x_line, p(np.log10(x_line)), 'r--', linewidth=2.5, 
                 alpha=0.7, label='Trend', zorder=2)

# Add labels for each point
for idx, row in avg_perf.iterrows():
    ax3.annotate(row['matrix_name'], 
                (row['matrix_size'], row['d1_miss_rate']),
                fontsize=7, alpha=0.8, 
                xytext=(5, 5), textcoords='offset points')

ax3.set_xlabel('Matrix Size (rows)', fontsize=12, fontweight='bold')
ax3.set_ylabel('L1 Cache Miss Rate (%, average)', fontsize=12, fontweight='bold')
ax3.set_title('L1 Cache Miss Rate vs Matrix Size', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3, linestyle='--', zorder=1)
ax3.set_xscale('log')
if len(avg_perf) > 2:
    ax3.legend(loc='best', fontsize=9)
cbar4 = plt.colorbar(scatter1, ax=ax3)
cbar4.set_label('Density (%)', fontsize=10)

plt.tight_layout()

# Save the plot to plots directory
os.makedirs('plots', exist_ok=True)
base_name = os.path.basename(csv_file).replace('.csv', '_analysis.png')
output_file = os.path.join('plots', base_name)
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Plot saved to: {output_file}")

# Print summary statistics

print("\n=== Cache Performance Summary (Average across all threads) ===")
print(f"{'Matrix':<25s} {'L1 Miss %':>10s} {'Branch Miss %':>14s} {'Size':>12s} {'Density':>10s}")
print("=" * 75)
for idx, row in avg_perf.sort_values('d1_miss_rate').iterrows():
    print(f"{row['matrix_name']:<25s} {row['d1_miss_rate']:>9.2f}% {row['branch_miss_rate']:>13.2f}% "
          f"{row['matrix_size']:>12,} {row['density_clean']:>9.2f}%")


print("\n=== Performance by Thread Count ===")
# Group by threads and show average metrics
thread_summary = df.groupby('threads').agg({
    'd1_miss_rate': 'mean',
    'branch_miss_rate': 'mean'
}).reset_index().sort_values('threads')

print(f"{'Threads':<10s} {'L1 Miss %':>10s} {'Branch Miss %':>14s}")
print("-" * 40)
for _, row in thread_summary.iterrows():
    print(f"{int(row['threads']):<10d} {row['d1_miss_rate']:>9.2f}% {row['branch_miss_rate']:>13.2f}%")


print("\n=== Sample Matrix Detail (Thread Scaling) ===")
# Show how performance changes with threading for first matrix
sample_matrix = matrices[0]
sample_data = df[df['matrix_name'] == sample_matrix].sort_values('threads')
print(f"\nMatrix: {sample_matrix}")
print(f"{'Threads':<10s} {'L1 Miss %':>10s} {'Branch Miss %':>14s}")
print("-" * 40)
for _, row in sample_data.iterrows():
    print(f"{int(row['threads']):<10d} {row['d1_miss_rate']:>9.2f}% {row['branch_miss_rate']:>13.2f}%")


print("\n=== Key Insights ===")
# Calculate and show key insights
best_cache = avg_perf.loc[avg_perf['d1_miss_rate'].idxmin()]
worst_cache = avg_perf.loc[avg_perf['d1_miss_rate'].idxmax()]

print(f"Best L1 cache:  {best_cache['matrix_name']} ({best_cache['d1_miss_rate']:.2f}% miss rate)")
print(f"Worst L1 cache: {worst_cache['matrix_name']} ({worst_cache['d1_miss_rate']:.2f}% miss rate)")

# Thread scaling insights
if len(thread_summary) > 1:
    thread_increase = thread_summary.iloc[-1]['threads'] / thread_summary.iloc[0]['threads']
    cache_change = thread_summary.iloc[-1]['d1_miss_rate'] - thread_summary.iloc[0]['d1_miss_rate']
    print(f"\nThread scaling ({int(thread_summary.iloc[0]['threads'])}→{int(thread_summary.iloc[-1]['threads'])} threads, {thread_increase:.1f}× increase):")
    print(f"  L1 cache miss rate change: {cache_change:+.2f}%")

print("\nMetrics collected: L1 data cache loads/misses, branch-loads/misses (Valgrind, all OpenMP threads)")
