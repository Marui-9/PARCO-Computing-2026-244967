#!/usr/bin/env python3
"""
Plot cache performance metrics from perf benchmark results.
Creates comprehensive visualizations of L1 D-cache and LLC performance.
Usage: python3 plot_cache.py [cache_perf_results.csv]
"""

import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import re
import numpy as np

# Read CSV file
csv_file = sys.argv[1] if len(sys.argv) > 1 else 'results/cache_perf_results.csv'

if not os.path.exists(csv_file):
    print(f"Error: {csv_file} not found")
    sys.exit(1)

# Read the data
df = pd.read_csv(csv_file)

# Clean column names (strip quotes and whitespace)
df.columns = df.columns.str.strip().str.replace('"', '')
df['matrix'] = df['matrix'].str.strip().str.replace('"', '')

# Convert numeric columns for perf data
numeric_cols = ['rows', 'cols', 'threads', 'cycles', 'instructions',
                'l1d_loads', 'l1d_misses', 'l1d_miss_rate',
                'llc_loads', 'llc_misses', 'llc_miss_rate',
                'branches', 'branch_misses', 'branch_miss_rate', 'time_ms']

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Remove rows with NA values in critical fields
df = df.dropna(subset=['l1d_miss_rate', 'llc_miss_rate', 'threads', 'time_ms'])

if len(df) == 0:
    print("Error: No valid data found in CSV file after cleaning.")
    print("The cache_perf_results.csv may be malformed or empty.")
    print("\nPlease re-run the benchmark with: bash scripts/bench_cache_perf.sh")
    sys.exit(1)

# Extract matrix basename for cleaner labels
df['matrix_name'] = df['matrix'].apply(lambda x: os.path.basename(x).replace('.mtx', ''))

# Calculate matrix size and clean density
df['matrix_size'] = df['rows'] * df['cols']

# Extract density percentage from density_pct column (e.g., "0.23%" -> 0.23)
if 'density_pct' in df.columns:
    df['density_clean'] = df['density_pct'].apply(
        lambda x: float(re.findall(r'\d+\.\d+', str(x))[0]) if pd.notna(x) and re.findall(r'\d+\.\d+', str(x)) else 0.0
    )
else:
    df['density_clean'] = 0.0

# Get unique matrices and threads
matrices = sorted(df['matrix_name'].unique())
thread_counts = sorted(df['threads'].unique())

# Create figure with 6 subplots (3x2)
fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])
ax5 = fig.add_subplot(gs[2, 0])
ax6 = fig.add_subplot(gs[2, 1])

# ==================== PLOT 1: D1 Cache Miss Rate vs Threads ====================
for matrix in matrices:
    matrix_data = df[df['matrix_name'] == matrix].sort_values('threads')
    ax1.plot(matrix_data['threads'], matrix_data['l1d_miss_rate'], 
             marker='o', linewidth=2, markersize=6, label=matrix, alpha=0.8)

ax1.set_xlabel('Number of Threads', fontsize=12, fontweight='bold')
ax1.set_ylabel('L1 D-Cache Miss Rate (%)', fontsize=12, fontweight='bold')
ax1.set_title('L1 D-Cache Miss Rate vs Number of Threads', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(loc='best', fontsize=8, ncol=2)
ax1.set_xticks(thread_counts)

# ==================== PLOT 2: LL Cache Miss Rate vs Threads ====================
for matrix in matrices:
    matrix_data = df[df['matrix_name'] == matrix].sort_values('threads')
    ax2.plot(matrix_data['threads'], matrix_data['llc_miss_rate'], 
             marker='s', linewidth=2, markersize=6, label=matrix, alpha=0.8)

ax2.set_xlabel('Number of Threads', fontsize=12, fontweight='bold')
ax2.set_ylabel('Last Level Cache Miss Rate (%)', fontsize=12, fontweight='bold')
ax2.set_title('LLC Miss Rate vs Number of Threads', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(loc='best', fontsize=8, ncol=2)
ax2.set_xticks(thread_counts)

# ==================== PLOT 3: L1D Misses vs Threads (Absolute Count) ====================
for matrix in matrices:
    matrix_data = df[df['matrix_name'] == matrix].sort_values('threads')
    ax3.plot(matrix_data['threads'], matrix_data['l1d_misses'], 
             marker='o', linewidth=2, markersize=6, label=matrix, alpha=0.8)

ax3.set_xlabel('Number of Threads', fontsize=12, fontweight='bold')
ax3.set_ylabel('L1 D-Cache Misses (count)', fontsize=12, fontweight='bold')
ax3.set_title('L1 D-Cache Misses vs Number of Threads', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.legend(loc='best', fontsize=8, ncol=2)
ax3.set_xticks(thread_counts)
ax3.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

# ==================== PLOT 4: LLC Misses vs Threads (Absolute Count) ====================
for matrix in matrices:
    matrix_data = df[df['matrix_name'] == matrix].sort_values('threads')
    ax4.plot(matrix_data['threads'], matrix_data['llc_misses'], 
             marker='s', linewidth=2, markersize=6, label=matrix, alpha=0.8)

ax4.set_xlabel('Number of Threads', fontsize=12, fontweight='bold')
ax4.set_ylabel('Last Level Cache Misses (count)', fontsize=12, fontweight='bold')
ax4.set_title('LLC Misses vs Number of Threads', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3, linestyle='--')
ax4.legend(loc='best', fontsize=8, ncol=2)
ax4.set_xticks(thread_counts)
ax4.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

# ==================== PLOT 5: L1D Miss Rate vs Matrix Size ====================
# Get average cache performance for each matrix across all thread counts
avg_cache_perf = df.groupby('matrix_name').agg({
    'l1d_misses': 'mean',
    'l1d_miss_rate': 'mean',
    'llc_misses': 'mean',
    'llc_miss_rate': 'mean',
    'branch_misses': 'mean',
    'branch_miss_rate': 'mean',
    'time_ms': 'mean',
    'matrix_size': 'first',
    'density_clean': 'first'
}).reset_index().sort_values('matrix_size')

# Scatter plot
scatter1 = ax5.scatter(avg_cache_perf['matrix_size'], avg_cache_perf['l1d_miss_rate'],
           s=200, alpha=0.7, c=avg_cache_perf['density_clean'], 
           cmap='viridis', edgecolors='black', linewidth=1.5, zorder=3)

# Add trend line
if len(avg_cache_perf) > 2:
    valid_data = avg_cache_perf[avg_cache_perf['matrix_size'] > 0]
    if len(valid_data) > 2:
        z = np.polyfit(np.log10(valid_data['matrix_size']), 
                       valid_data['l1d_miss_rate'], 1)
        p = np.poly1d(z)
        x_line = np.logspace(np.log10(valid_data['matrix_size'].min()),
                             np.log10(valid_data['matrix_size'].max()), 100)
        ax5.plot(x_line, p(np.log10(x_line)), 'r--', linewidth=2.5, 
                 alpha=0.7, label='Trend', zorder=2)

# Add labels for each point
for idx, row in avg_cache_perf.iterrows():
    ax5.annotate(row['matrix_name'], 
                (row['matrix_size'], row['l1d_miss_rate']),
                fontsize=7, alpha=0.8, 
                xytext=(5, 5), textcoords='offset points')

ax5.set_xlabel('Matrix Size (rows × cols)', fontsize=12, fontweight='bold')
ax5.set_ylabel('L1 D-Cache Miss Rate (% average)', fontsize=12, fontweight='bold')
ax5.set_title('L1 D-Cache Miss Rate vs Matrix Size', fontsize=14, fontweight='bold')
ax5.grid(True, alpha=0.3, linestyle='--', zorder=1)
ax5.set_xscale('log')
if len(avg_cache_perf) > 2:
    ax5.legend(loc='best', fontsize=9)
cbar5 = plt.colorbar(scatter1, ax=ax5)
cbar5.set_label('Density (%)', fontsize=10)

# ==================== PLOT 6: LLC Miss Rate vs Matrix Size ====================
# Scatter plot
scatter2 = ax6.scatter(avg_cache_perf['matrix_size'], avg_cache_perf['llc_miss_rate'],
           s=200, alpha=0.7, c=avg_cache_perf['density_clean'], 
           cmap='plasma', edgecolors='black', linewidth=1.5, zorder=3)

# Add trend line
if len(avg_cache_perf) > 2:
    valid_data = avg_cache_perf[avg_cache_perf['matrix_size'] > 0]
    if len(valid_data) > 2:
        z = np.polyfit(np.log10(valid_data['matrix_size']), 
                       valid_data['llc_miss_rate'], 1)
        p = np.poly1d(z)
        x_line = np.logspace(np.log10(valid_data['matrix_size'].min()),
                             np.log10(valid_data['matrix_size'].max()), 100)
        ax6.plot(x_line, p(np.log10(x_line)), 'r--', linewidth=2.5, 
                 alpha=0.7, label='Trend', zorder=2)

# Add labels
for idx, row in avg_cache_perf.iterrows():
    ax6.annotate(row['matrix_name'], 
                (row['matrix_size'], row['llc_miss_rate']),
                fontsize=7, alpha=0.8,
                xytext=(5, 5), textcoords='offset points')

ax6.set_xlabel('Matrix Size (rows × cols)', fontsize=12, fontweight='bold')
ax6.set_ylabel('Last Level Cache Miss Rate (% average)', fontsize=12, fontweight='bold')
ax6.set_title('LLC Miss Rate vs Matrix Size', fontsize=14, fontweight='bold')
ax6.grid(True, alpha=0.3, linestyle='--', zorder=1)
ax6.set_xscale('log')
if len(avg_cache_perf) > 2:
    ax6.legend(loc='best', fontsize=9)
cbar6 = plt.colorbar(scatter2, ax=ax6)
cbar6.set_label('Density (%)', fontsize=10)

plt.tight_layout()

# Save the plot to plots directory
os.makedirs('plots', exist_ok=True)
base_name = os.path.basename(csv_file).replace('.csv', '_analysis.png')
output_file = os.path.join('plots', base_name)
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Plot saved to: {output_file}")

# Print summary statistics
print("\n=== Cache Performance Summary (Average across all threads) ===")
print(f"{'Matrix':<25s} {'L1D Miss %':>12s} {'LLC Miss %':>12s} {'Branch Miss %':>14s} {'Time (ms)':>12s} {'Size':>12s} {'Density':>10s}")
print("=" * 110)
for idx, row in avg_cache_perf.sort_values('l1d_miss_rate', ascending=False).iterrows():
    print(f"{row['matrix_name']:<25s} {row['l1d_miss_rate']:>11.2f}% {row['llc_miss_rate']:>11.2f}% "
          f"{row['branch_miss_rate']:>13.2f}% {row['time_ms']:>11.2f} {row['matrix_size']:>12,} {row['density_clean']:>9.2f}%")

print("\n=== Cache Misses by Thread Count ===")
# Group by threads and show average miss rates
thread_summary = df.groupby('threads').agg({
    'l1d_miss_rate': 'mean',
    'llc_miss_rate': 'mean',
    'branch_miss_rate': 'mean',
    'l1d_misses': 'mean',
    'llc_misses': 'mean',
    'time_ms': 'mean'
}).reset_index().sort_values('threads')

print(f"{'Threads':<10s} {'L1D Miss %':>12s} {'LLC Miss %':>12s} {'Branch Miss %':>14s} {'Avg L1D Misses':>16s} {'Avg LLC Misses':>16s} {'Time (ms)':>12s}")
print("-" * 110)
for _, row in thread_summary.iterrows():
    print(f"{int(row['threads']):<10d} {row['l1d_miss_rate']:>11.2f}% {row['llc_miss_rate']:>11.2f}% "
          f"{row['branch_miss_rate']:>13.2f}% {row['l1d_misses']:>16,.0f} {row['llc_misses']:>16,.0f} {row['time_ms']:>11.2f}")

print("\n=== Sample Matrix Detail (Thread Scaling) ===")
# Show how cache misses change with threading for first matrix
sample_matrix = matrices[0]
sample_data = df[df['matrix_name'] == sample_matrix].sort_values('threads')
print(f"\nMatrix: {sample_matrix}")
print(f"{'Threads':<10s} {'L1D Misses':>16s} {'LLC Misses':>16s} {'L1D Miss %':>12s} {'LLC Miss %':>12s} {'Time (ms)':>12s}")
print("-" * 90)
for _, row in sample_data.iterrows():
    print(f"{int(row['threads']):<10d} {row['l1d_misses']:>16,.0f} {row['llc_misses']:>16,.0f} "
          f"{row['l1d_miss_rate']:>11.2f}% {row['llc_miss_rate']:>11.2f}% {row['time_ms']:>11.2f}")

print("\n=== Key Insights ===")
# Calculate and show key insights
best_l1d = avg_cache_perf.loc[avg_cache_perf['l1d_miss_rate'].idxmin()]
worst_l1d = avg_cache_perf.loc[avg_cache_perf['l1d_miss_rate'].idxmax()]
best_llc = avg_cache_perf.loc[avg_cache_perf['llc_miss_rate'].idxmin()]
worst_llc = avg_cache_perf.loc[avg_cache_perf['llc_miss_rate'].idxmax()]

print(f"Best L1D cache performance:  {best_l1d['matrix_name']} ({best_l1d['l1d_miss_rate']:.2f}% miss rate)")
print(f"Worst L1D cache performance: {worst_l1d['matrix_name']} ({worst_l1d['l1d_miss_rate']:.2f}% miss rate)")
print(f"Best LLC performance:        {best_llc['matrix_name']} ({best_llc['llc_miss_rate']:.2f}% miss rate)")
print(f"Worst LLC performance:       {worst_llc['matrix_name']} ({worst_llc['llc_miss_rate']:.2f}% miss rate)")

# Thread scaling insights
if len(thread_summary) > 1:
    thread_increase = thread_summary.iloc[-1]['threads'] / thread_summary.iloc[0]['threads']
    l1d_change = thread_summary.iloc[-1]['l1d_miss_rate'] - thread_summary.iloc[0]['l1d_miss_rate']
    llc_change = thread_summary.iloc[-1]['llc_miss_rate'] - thread_summary.iloc[0]['llc_miss_rate']
    time_speedup = thread_summary.iloc[0]['time_ms'] / thread_summary.iloc[-1]['time_ms']
    
    print(f"\nThread scaling ({int(thread_summary.iloc[0]['threads'])}→{int(thread_summary.iloc[-1]['threads'])} threads, {thread_increase:.1f}× increase):")
    print(f"  L1D miss rate change: {l1d_change:+.2f} percentage points")
    print(f"  LLC miss rate change: {llc_change:+.2f} percentage points")
    print(f"  Speedup: {time_speedup:.2f}×")
