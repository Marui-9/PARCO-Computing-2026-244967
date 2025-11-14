#!/usr/bin/env python3
"""
Plot speedup vs threads for different matrices from benchmark results.
Creates 4 different visualizations focused on performance analysis.
Usage: python3 plot_speedup.py [results.csv]
"""

import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import re
import numpy as np

# Read CSV file
csv_file = sys.argv[1] if len(sys.argv) > 1 else 'matrices_results.csv'

if not os.path.exists(csv_file):
    print(f"Error: {csv_file} not found")
    sys.exit(1)

# Read the data
df = pd.read_csv(csv_file)

# Clean up the data - remove rows with NA speedup
df = df[df['speedup_x'] != 'NA']
df['speedup_x'] = pd.to_numeric(df['speedup_x'])
df['serial_ms'] = pd.to_numeric(df['serial_ms'])
df['parallel_ms'] = pd.to_numeric(df['parallel_ms'])

# Extract matrix basename for cleaner labels
df['matrix_name'] = df['matrix'].apply(lambda x: os.path.basename(x.strip('"')).replace('.mtx', ''))

# Calculate efficiency (speedup / threads)
df['efficiency'] = df['speedup_x'] / df['threads']

# Calculate matrix size (total elements)
df['matrix_size'] = df['rows'] * df['cols']

# Extract density percentage (clean up the format)
df['density_clean'] = df['density_pct'].apply(lambda x: float(re.findall(r'\d+\.\d+', str(x))[0]) if re.findall(r'\d+\.\d+', str(x)) else 0.0)

# Get unique matrices and threads
matrices = sorted(df['matrix_name'].unique())
thread_counts = sorted(df['threads'].unique())

# Create figure with 4 subplots (2x2)
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

# ==================== PLOT 1: Speedup vs Threads (Classic) ====================
for matrix in matrices:
    matrix_data = df[df['matrix_name'] == matrix].sort_values('threads')
    ax1.plot(matrix_data['threads'], matrix_data['speedup_x'], 
             marker='o', linewidth=2, markersize=6, label=matrix, alpha=0.8)

# Add ideal speedup line
max_threads = df['threads'].max()
ax1.plot([1, max_threads], [1, max_threads], 'k--', linewidth=2, 
         alpha=0.5, label='Ideal (linear)')

ax1.set_xlabel('Number of Threads', fontsize=12, fontweight='bold')
ax1.set_ylabel('Speedup', fontsize=12, fontweight='bold')
ax1.set_title('Speedup vs Number of Threads', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(loc='best', fontsize=8, ncol=2)
ax1.set_xticks(thread_counts)
ax1.set_ylim(bottom=0)

# ==================== PLOT 2: Heatmap - Speedup by Matrix × Threads ====================
# Create pivot table for heatmap
heatmap_data = df.pivot(index='matrix_name', columns='threads', values='speedup_x')
heatmap_data = heatmap_data.reindex(matrices)  # Sort by matrix name

# Create heatmap
im = ax2.imshow(heatmap_data.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=max_threads)

# Set ticks and labels
ax2.set_xticks(np.arange(len(thread_counts)))
ax2.set_yticks(np.arange(len(matrices)))
ax2.set_xticklabels(thread_counts)
ax2.set_yticklabels(matrices, fontsize=9)

# Add colorbar
cbar = plt.colorbar(im, ax=ax2)
cbar.set_label('Speedup', fontsize=11, fontweight='bold')

# Annotate each cell with speedup value
for i in range(len(matrices)):
    for j in range(len(thread_counts)):
        value = heatmap_data.iloc[i, j]
        if not pd.isna(value):
            text_color = 'white' if value > max_threads/2 else 'black'
            ax2.text(j, i, f'{value:.1f}', ha='center', va='center',
                    color=text_color, fontsize=8, fontweight='bold')

ax2.set_xlabel('Number of Threads', fontsize=12, fontweight='bold')
ax2.set_ylabel('Matrix', fontsize=12, fontweight='bold')
ax2.set_title('Speedup Heatmap: Matrix × Threads', fontsize=14, fontweight='bold')

# ==================== PLOT 3: Efficiency (Speedup/Threads) ====================
for matrix in matrices:
    matrix_data = df[df['matrix_name'] == matrix].sort_values('threads')
    ax3.plot(matrix_data['threads'], matrix_data['efficiency'] * 100, 
             marker='s', linewidth=2, markersize=6, label=matrix, alpha=0.8)

# Add perfect efficiency line (100%)
ax3.axhline(y=100, color='k', linestyle='--', linewidth=2, alpha=0.5, label='Perfect (100%)')

ax3.set_xlabel('Number of Threads', fontsize=12, fontweight='bold')
ax3.set_ylabel('Parallel Efficiency (%)', fontsize=12, fontweight='bold')
ax3.set_title('Parallel Efficiency vs Number of Threads', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.legend(loc='best', fontsize=8, ncol=2)
ax3.set_xticks(thread_counts)
ax3.set_ylim(bottom=0, top=110)

# ==================== PLOT 4: Speedup vs Matrix Density ====================
# Calculate average speedup for each matrix across all thread counts
matrix_avg_speedup = df.groupby('matrix_name').agg({
    'speedup_x': 'mean',
    'density_clean': 'first',
    'matrix_size': 'first'
}).sort_values('density_clean')

# Create scatter plot
scatter = ax4.scatter(matrix_avg_speedup['density_clean'], 
                      matrix_avg_speedup['speedup_x'],
                      s=200, alpha=0.7, c=np.log10(matrix_avg_speedup['matrix_size']),
                      cmap='viridis', edgecolors='black', linewidth=1.5)

# Add matrix labels
for idx, row in matrix_avg_speedup.iterrows():
    ax4.annotate(idx, (row['density_clean'], row['speedup_x']),
                fontsize=8, ha='left', va='bottom',
                xytext=(3, 3), textcoords='offset points')

ax4.set_xlabel('Matrix Density (%)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Average Speedup (all threads)', fontsize=12, fontweight='bold')
ax4.set_title('Speedup vs Matrix Density', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3, linestyle='--')
ax4.set_xscale('log')

# Add colorbar for matrix size
cbar = plt.colorbar(scatter, ax=ax4)
cbar.set_label('log₁₀(Matrix Size)', fontsize=10, fontweight='bold')

# Create figures directory and save
os.makedirs('plots', exist_ok=True)
output_file = 'plots/speedup_analysis.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✓ Plot saved to: {output_file}")

# Print summary statistics
print("\n=== Speedup Summary ===")
for matrix in matrices:
    matrix_data = df[df['matrix_name'] == matrix]
    max_speedup = matrix_data['speedup_x'].max()
    max_threads_for_matrix = matrix_data.loc[matrix_data['speedup_x'].idxmax(), 'threads']
    efficiency_at_max = (max_speedup / max_threads_for_matrix) * 100
    size = matrix_data['matrix_size'].iloc[0]
    density = matrix_data['density_clean'].iloc[0]
    print(f"{matrix:25s}: Max speedup = {max_speedup:5.2f}x @ {max_threads_for_matrix:2.0f} threads "
          f"({efficiency_at_max:4.1f}% eff) | Size: {size:>10,} | Density: {density:5.2f}%")
