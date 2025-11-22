#!/usr/bin/env python3
"""
Plot speedup vs threads for different matrices from benchmark results.
Creates 4 different visualizations focused on performance analysis.
Usage: python3 plot_speedup.py [results.csv]
"""

import pandas as pd
import matplotlib
# Use non-interactive backend for headless environments (prevents Qt/QSocketNotifier errors)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
import re
import numpy as np

# Read CSV file
csv_file = sys.argv[1] if len(sys.argv) > 1 else 'results/matrices_results.csv'

if not os.path.exists(csv_file):
    print(f"Error: {csv_file} not found")
    sys.exit(1)

# Read the data
df = pd.read_csv(csv_file)


# Clean up the data - remove rows with NA or missing speedup_x
df = df[df['speedup_x'] != 'NA']
df['speedup_x'] = pd.to_numeric(df['speedup_x'], errors='coerce')
df['serial_ms'] = pd.to_numeric(df['serial_ms'], errors='coerce')
df['parallel_ms'] = pd.to_numeric(df['parallel_ms'], errors='coerce')
df = df.dropna(subset=['speedup_x', 'threads', 'matrix'])

# Extract matrix basename for cleaner labels
df['matrix_name'] = df['matrix'].apply(lambda x: os.path.basename(str(x).strip('"')).replace('.mtx', ''))

# Calculate efficiency (speedup / threads)
df['efficiency'] = df['speedup_x'] / df['threads']

# Calculate matrix size (total elements)
df['matrix_size'] = df['rows'] * df['cols']

# Extract density percentage (clean up the format)
df['density_clean'] = df['density_pct'].apply(lambda x: float(re.findall(r'\d+\.\d+', str(x))[0]) if re.findall(r'\d+\.\d+', str(x)) else 0.0)

# Get unique matrices and threads
matrices = sorted(df['matrix_name'].unique())
thread_counts = sorted(df['threads'].unique())

# === PLOT: Speedup of Best Configuration vs Serial vs Number of Threads ===
# For each matrix, find the row with the maximum speedup (best config) for each thread count
filtered = df.dropna(subset=['speedup_x'])
best_by_thread = filtered.loc[filtered.groupby(['matrix_name', 'threads'])['speedup_x'].idxmax()]



# Plot only the average speedup vs threads (single line), no y-axis break, keep shorter height
plt.figure(figsize=(10, 5))
avg_speedup_by_thread = best_by_thread.groupby('threads')['speedup_x'].mean().sort_index()
plt.plot(avg_speedup_by_thread.index, avg_speedup_by_thread.values, marker='o', linewidth=2.5, markersize=8, color='tab:blue', label='Average Speedup')

plt.xlabel('Number of Threads', fontsize=12, fontweight='bold')
plt.ylabel('Average Speedup (Best Config)', fontsize=12, fontweight='bold')
plt.title('Average Best Configuration Speedup vs Serial by Number of Threads', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(loc='best', fontsize=11)
plt.xticks(thread_counts)
plt.ylim(bottom=0)
os.makedirs('plots', exist_ok=True)
plt.savefig('plots/speedup_vs_threads_best.png', dpi=300, bbox_inches='tight')
print("\u2713 Plot saved to: plots/speedup_vs_threads_best.png")

