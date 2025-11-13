#!/usr/bin/env python3
"""
Plot cache performance metrics from benchmark results.
Creates comprehensive visualizations of L1 D-cache and LLC performance.
Usage: python3 plot_cache.py [cache_results.csv]
"""

import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import re
import numpy as np

# Read CSV file
csv_file = sys.argv[1] if len(sys.argv) > 1 else 'cache_results.csv'

if not os.path.exists(csv_file):
    print(f"Error: {csv_file} not found")
    sys.exit(1)

# Read the data with special handling for encoding issues
df = pd.read_csv(csv_file, encoding='utf-8', encoding_errors='ignore')

# Clean up the data - convert to numeric, coercing errors to NaN
df['l1_miss_rate'] = pd.to_numeric(df['l1_miss_rate'], errors='coerce')
df['llc_miss_rate'] = pd.to_numeric(df['llc_miss_rate'], errors='coerce')
df['cache_miss_rate'] = pd.to_numeric(df['cache_miss_rate'], errors='coerce')
df['l1_dcache_loads'] = pd.to_numeric(df['l1_dcache_loads'], errors='coerce')
df['l1_dcache_misses'] = pd.to_numeric(df['l1_dcache_misses'], errors='coerce')
df['llc_loads'] = pd.to_numeric(df['llc_loads'], errors='coerce')
df['llc_misses'] = pd.to_numeric(df['llc_misses'], errors='coerce')
df['time_ms'] = pd.to_numeric(df['time_ms'], errors='coerce')
df['threads'] = pd.to_numeric(df['threads'], errors='coerce')

# Try to parse rows - handle the special character
df['rows'] = pd.to_numeric(df['rows'], errors='coerce')

# Remove rows with NA values in critical fields
df = df.dropna(subset=['l1_miss_rate', 'cache_miss_rate', 'time_ms', 'threads'])

if len(df) == 0:
    print("Error: No valid data found in CSV file after cleaning.")
    print("The cache_results.csv may be malformed.")
    print("\nPlease re-run the benchmark with: ./bench_cache.sh")
    sys.exit(1)

# Extract matrix basename for cleaner labels
df['matrix_name'] = df['matrix'].apply(lambda x: os.path.basename(x.strip('"')).replace('.mtx', ''))

# Try to get matrix info from the results.csv if available
results_csv = 'results.csv'
if os.path.exists(results_csv):
    results_df = pd.read_csv(results_csv)
    results_df['matrix_name'] = results_df['matrix'].apply(lambda x: os.path.basename(x.strip('"')).replace('.mtx', ''))
    
    # Merge to get proper matrix dimensions and density
    matrix_info = results_df[['matrix_name', 'rows', 'cols', 'density_pct']].drop_duplicates()
    df = df.drop(columns=['rows', 'cols', 'density_pct'], errors='ignore')
    df = df.merge(matrix_info, on='matrix_name', how='left')

# Calculate matrix size
if 'rows' in df.columns and 'cols' in df.columns:
    df['rows'] = pd.to_numeric(df['rows'], errors='coerce')
    df['cols'] = pd.to_numeric(df['cols'], errors='coerce')
    df['matrix_size'] = df['rows'] * df['cols']
    # Extract density percentage
    df['density_clean'] = df['density_pct'].apply(lambda x: float(re.findall(r'\d+\.\d+', str(x))[0]) if re.findall(r'\d+\.\d+', str(x)) else 0.0)
else:
    # If still no rows/cols, use placeholder
    df['matrix_size'] = 1
    df['density_clean'] = 0.0
    print("Warning: Could not determine matrix dimensions. Some plots may be limited.")

# Get unique matrices and threads
matrices = sorted(df['matrix_name'].unique())
thread_counts = sorted(df['threads'].unique())

# Create figure with 4 subplots (2x2)
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

# ==================== PLOT 1: L1 Cache Misses vs Threads ====================
for matrix in matrices:
    matrix_data = df[df['matrix_name'] == matrix].sort_values('threads')
    ax1.plot(matrix_data['threads'], matrix_data['l1_dcache_misses'], 
             marker='o', linewidth=2, markersize=6, label=matrix, alpha=0.8)

ax1.set_xlabel('Number of Threads', fontsize=12, fontweight='bold')
ax1.set_ylabel('L1 D-Cache Misses (count)', fontsize=12, fontweight='bold')
ax1.set_title('L1 D-Cache Misses vs Number of Threads', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(loc='best', fontsize=8, ncol=2)
ax1.set_xticks(thread_counts)
ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

# ==================== PLOT 2: Cache Misses vs Threads ====================
for matrix in matrices:
    matrix_data = df[df['matrix_name'] == matrix].sort_values('threads')
    ax2.plot(matrix_data['threads'], matrix_data['cache_misses'], 
             marker='s', linewidth=2, markersize=6, label=matrix, alpha=0.8)

ax2.set_xlabel('Number of Threads', fontsize=12, fontweight='bold')
ax2.set_ylabel('Overall Cache Misses (count)', fontsize=12, fontweight='bold')
ax2.set_title('Overall Cache Misses vs Number of Threads', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(loc='best', fontsize=8, ncol=2)
ax2.set_xticks(thread_counts)
ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

# ==================== PLOT 3: L1 Cache Misses vs Matrix Size ====================
if 'matrix_size' in df.columns and df['matrix_size'].max() > 1:
    # Get average cache misses for each matrix across all thread counts
    avg_cache_perf = df.groupby('matrix_name').agg({
        'l1_dcache_misses': 'mean',
        'cache_misses': 'mean',
        'l1_miss_rate': 'mean',
        'cache_miss_rate': 'mean',
        'matrix_size': 'first',
        'density_clean': 'first'
    }).reset_index().sort_values('matrix_size')

    # Scatter plot
    scatter1 = ax3.scatter(avg_cache_perf['matrix_size'], avg_cache_perf['l1_dcache_misses'],
               s=200, alpha=0.7, c=avg_cache_perf['density_clean'], 
               cmap='viridis', edgecolors='black', linewidth=1.5, zorder=3)

    # Add trend line
    if len(avg_cache_perf) > 2:
        z = np.polyfit(np.log10(avg_cache_perf['matrix_size']), 
                       avg_cache_perf['l1_dcache_misses'], 1)
        p = np.poly1d(z)
        x_line = np.logspace(np.log10(avg_cache_perf['matrix_size'].min()),
                             np.log10(avg_cache_perf['matrix_size'].max()), 100)
        ax3.plot(x_line, p(np.log10(x_line)), 'r--', linewidth=2.5, 
                 alpha=0.7, label='Trend', zorder=2)

    # Add labels for each point
    for idx, row in avg_cache_perf.iterrows():
        ax3.annotate(row['matrix_name'], 
                    (row['matrix_size'], row['l1_dcache_misses']),
                    fontsize=7, alpha=0.8, 
                    xytext=(5, 5), textcoords='offset points')

    ax3.set_xlabel('Matrix Size (rows × cols)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('L1 D-Cache Misses (average)', fontsize=12, fontweight='bold')
    ax3.set_title('L1 D-Cache Misses vs Matrix Size', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--', zorder=1)
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.ticklabel_format(style='plain', axis='y')
    if len(avg_cache_perf) > 2:
        ax3.legend(loc='best', fontsize=9)
    cbar3 = plt.colorbar(scatter1, ax=ax3)
    cbar3.set_label('Density (%)', fontsize=10)
else:
    ax3.text(0.5, 0.5, 'Matrix size data not available', 
             ha='center', va='center', transform=ax3.transAxes, fontsize=14)

# ==================== PLOT 4: Overall Cache Misses vs Matrix Size ====================
if 'matrix_size' in df.columns and df['matrix_size'].max() > 1:
    # Scatter plot
    scatter2 = ax4.scatter(avg_cache_perf['matrix_size'], avg_cache_perf['cache_misses'],
               s=200, alpha=0.7, c=avg_cache_perf['density_clean'], 
               cmap='plasma', edgecolors='black', linewidth=1.5, zorder=3)

    # Add trend line
    if len(avg_cache_perf) > 2:
        z = np.polyfit(np.log10(avg_cache_perf['matrix_size']), 
                       avg_cache_perf['cache_misses'], 1)
        p = np.poly1d(z)
        x_line = np.logspace(np.log10(avg_cache_perf['matrix_size'].min()),
                             np.log10(avg_cache_perf['matrix_size'].max()), 100)
        ax4.plot(x_line, p(np.log10(x_line)), 'r--', linewidth=2.5, 
                 alpha=0.7, label='Trend', zorder=2)

    # Add labels
    for idx, row in avg_cache_perf.iterrows():
        ax4.annotate(row['matrix_name'], 
                    (row['matrix_size'], row['cache_misses']),
                    fontsize=7, alpha=0.8,
                    xytext=(5, 5), textcoords='offset points')

    ax4.set_xlabel('Matrix Size (rows × cols)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Overall Cache Misses (average)', fontsize=12, fontweight='bold')
    ax4.set_title('Overall Cache Misses vs Matrix Size', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle='--', zorder=1)
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.ticklabel_format(style='plain', axis='y')
    if len(avg_cache_perf) > 2:
        ax4.legend(loc='best', fontsize=9)
    cbar4 = plt.colorbar(scatter2, ax=ax4)
    cbar4.set_label('Density (%)', fontsize=10)
else:
    ax4.text(0.5, 0.5, 'Matrix size data not available', 
             ha='center', va='center', transform=ax4.transAxes, fontsize=14)

plt.tight_layout()

# Save the plot to figures directory
os.makedirs('figures', exist_ok=True)
base_name = os.path.basename(csv_file).replace('.csv', '_analysis.png')
output_file = os.path.join('figures', base_name)
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Plot saved to: {output_file}")

# Print summary statistics
print("\n=== Cache Misses Summary (Average across all threads) ===")
if 'matrix_size' in df.columns and df['matrix_size'].max() > 1:
    print(f"{'Matrix':<25s} {'L1 Misses':>15s} {'Cache Misses':>15s} {'L1 Miss %':>10s} {'Size':>12s}")
    print("=" * 80)
    for idx, row in avg_cache_perf.sort_values('l1_dcache_misses', ascending=False).iterrows():
        print(f"{row['matrix_name']:<25s} {row['l1_dcache_misses']:>15,.0f} {row['cache_misses']:>15,.0f} "
              f"{row['l1_miss_rate']:>9.2f}% {row['matrix_size']:>12,}")
else:
    # Simple summary without matrix info
    summary = df.groupby('matrix_name').agg({
        'l1_dcache_misses': 'mean',
        'cache_misses': 'mean',
        'l1_miss_rate': 'mean',
        'cache_miss_rate': 'mean'
    }).sort_values('l1_dcache_misses', ascending=False)
    
    print(f"{'Matrix':<25s} {'L1 Misses':>15s} {'Cache Misses':>15s} {'L1 Miss %':>10s}")
    print("=" * 70)
    for matrix_name, row in summary.iterrows():
        print(f"{matrix_name:<25s} {row['l1_dcache_misses']:>15,.0f} {row['cache_misses']:>15,.0f} "
              f"{row['l1_miss_rate']:>9.2f}%")

print("\n=== Cache Misses vs Threading ===")
# Show how cache misses change with threading for a sample matrix
sample_matrix = matrices[0]
sample_data = df[df['matrix_name'] == sample_matrix].sort_values('threads')
print(f"\nMatrix: {sample_matrix}")
print(f"{'Threads':<10s} {'L1 Misses':>15s} {'Cache Misses':>15s} {'Time (ms)':>12s}")
print("-" * 55)
for _, row in sample_data.iterrows():
    print(f"{row['threads']:<10d} {row['l1_dcache_misses']:>15,.0f} {row['cache_misses']:>15,.0f} {row['time_ms']:>11.2f}")
