#!/usr/bin/env python3
"""
Plot benchmark results from configurations_results.csv
Generates multiple visualizations for performance analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
colors = plt.cm.Set2(np.linspace(0, 1, 8))

# Determine the correct path to the CSV file
if os.path.exists('configurations_results.csv'):
    csv_path = 'configurations_results.csv'
elif os.path.exists('results/configurations_results.csv'):
    csv_path = 'results/configurations_results.csv'
else:
    print("Error: configurations_results.csv not found!")
    exit(1)

# Read data
df = pd.read_csv(csv_path)
df['configuration'] = df['configuration'].str.strip()

# Create configuration categories
def categorize_config(config):
    if 'Align+Affin' in config:
        return 'SIMD+Align+Affin'
    elif 'SIMD+Affinity' in config:
        return 'SIMD+Affinity'
    elif 'SIMD+Register' in config:
        return 'SIMD+Register'
    elif 'SIMD' in config:
        return 'SIMD'
    elif 'Task-based' in config:
        return 'Task-based'
    elif 'default' in config:
        return 'Default'
    else:
        return 'Scheduling'

df['config_category'] = df['configuration'].apply(categorize_config)
df['matrix_size_M'] = (df['rows'] * df['cols']) / 1e6

print(f"Generating plots from {csv_path}...")
print(f"Data points: {len(df)}")

# Create figure directory
os.makedirs('plots', exist_ok=True)

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Density (%)', fontsize=12, fontweight='bold')

# Only plot: Speedup vs Thread Count by Configuration Category
fig, ax = plt.subplots(figsize=(12, 7))
categories = ['Default', 'Scheduling', 'SIMD', 'SIMD+Affinity', 'SIMD+Register', 'SIMD+Align+Affin']
for i, cat in enumerate(categories):
    cat_data = df[df['config_category'] == cat]
    if len(cat_data) > 0:
        speedup_by_threads = cat_data.groupby('threads')['speedup'].mean()
        ax.plot(speedup_by_threads.index, speedup_by_threads.values, 
                marker='o', linewidth=2.5, markersize=8, label=cat, color=colors[i])

ax.set_xlabel('Number of Threads', fontsize=13, fontweight='bold')
ax.set_ylabel('Average Speedup', fontsize=13, fontweight='bold')
ax.set_title('Speedup vs Thread Count by Configuration Category', fontsize=15, fontweight='bold')
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xticks(sorted(df['threads'].unique()))
plt.tight_layout()
plt.savefig('plots/speedup_vs_threads.png', dpi=300, bbox_inches='tight')
print("✓ Generated: plots/speedup_vs_threads.png")
plt.close()
