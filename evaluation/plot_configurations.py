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
elif os.path.exists('evaluation/configurations_results.csv'):
    csv_path = 'evaluation/configurations_results.csv'
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
os.makedirs('figures', exist_ok=True)

# ============================================================================
# 1. Speedup vs Thread Count (by category)
# ============================================================================
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
plt.savefig('figures/speedup_vs_threads.png', dpi=300, bbox_inches='tight')
print("✓ Generated: figures/speedup_vs_threads.png")
plt.close()

# ============================================================================
# 2. Speedup Heatmap: Configuration × Matrix
# ============================================================================
# Select top configurations
top_configs = df.groupby('configuration')['speedup'].mean().nlargest(15).index
heatmap_data = df[df['configuration'].isin(top_configs) & (df['threads'] == 12)]
pivot_table = heatmap_data.pivot_table(values='speedup', 
                                        index='configuration', 
                                        columns='matrix', 
                                        aggfunc='mean')

fig, ax = plt.subplots(figsize=(14, 8))
im = ax.imshow(pivot_table.values, aspect='auto', cmap='YlOrRd', interpolation='nearest')

# Set ticks and labels
ax.set_xticks(np.arange(len(pivot_table.columns)))
ax.set_yticks(np.arange(len(pivot_table.index)))
ax.set_xticklabels(pivot_table.columns, rotation=45, ha='right', fontsize=10)
ax.set_yticklabels(pivot_table.index, fontsize=10)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Speedup', fontsize=12, fontweight='bold')

# Add text annotations
for i in range(len(pivot_table.index)):
    for j in range(len(pivot_table.columns)):
        value = pivot_table.values[i, j]
        if not np.isnan(value):
            text = ax.text(j, i, f'{value:.0f}', ha='center', va='center', 
                          color='black' if value < pivot_table.values.max()/2 else 'white',
                          fontsize=8, fontweight='bold')

ax.set_title('Speedup Heatmap: Top 15 Configurations vs Matrices (12 threads)', 
             fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('figures/speedup_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Generated: figures/speedup_heatmap.png")
plt.close()

# ============================================================================
# 3. Speedup vs Matrix Density (SIMD variants only)
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 7))

simd_data = df[(df['config_category'].isin(['SIMD', 'SIMD+Affinity', 'SIMD+Align+Affin'])) & 
               (df['threads'] == 12)]

for cat in ['SIMD', 'SIMD+Affinity', 'SIMD+Align+Affin']:
    cat_data = simd_data[simd_data['config_category'] == cat]
    if len(cat_data) > 0:
        # Group by density and calculate mean
        density_speedup = cat_data.groupby('density_pct')['speedup'].mean().sort_index()
        ax.scatter(density_speedup.index, density_speedup.values, s=150, alpha=0.7, label=cat)
        ax.plot(density_speedup.index, density_speedup.values, alpha=0.5, linewidth=2)

ax.set_xlabel('Matrix Density (%)', fontsize=13, fontweight='bold')
ax.set_ylabel('Average Speedup', fontsize=13, fontweight='bold')
ax.set_title('Speedup vs Matrix Density (SIMD variants, 12 threads)', fontsize=15, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
ax.set_yscale('log')
plt.tight_layout()
plt.savefig('figures/speedup_vs_density.png', dpi=300, bbox_inches='tight')
print("✓ Generated: figures/speedup_vs_density.png")
plt.close()

# ============================================================================
# 4. Parallel Efficiency vs Thread Count
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 7))

simd_cats = ['SIMD', 'SIMD+Affinity', 'SIMD+Register', 'SIMD+Align+Affin']
for cat in simd_cats:
    cat_data = df[df['config_category'] == cat]
    if len(cat_data) > 0:
        efficiency = cat_data.groupby('threads')['efficiency_pct'].mean()
        ax.plot(efficiency.index, efficiency.values, 
                marker='s', linewidth=2.5, markersize=8, label=cat)

# Add ideal efficiency line
threads = sorted(df['threads'].unique())
ax.plot(threads, [100] * len(threads), 'k--', linewidth=2, alpha=0.5, label='Ideal (100%)')

ax.set_xlabel('Number of Threads', fontsize=13, fontweight='bold')
ax.set_ylabel('Parallel Efficiency (%)', fontsize=13, fontweight='bold')
ax.set_title('Parallel Efficiency vs Thread Count', fontsize=15, fontweight='bold')
ax.legend(fontsize=11, loc='lower left')
ax.grid(True, alpha=0.3)
ax.set_xticks(threads)
plt.tight_layout()
plt.savefig('figures/parallel_efficiency.png', dpi=300, bbox_inches='tight')
print("✓ Generated: figures/parallel_efficiency.png")
plt.close()

# ============================================================================
# 5. Box Plot: Speedup Distribution by Category
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 7))

categories_for_box = ['Default', 'Scheduling', 'SIMD', 'SIMD+Affinity', 'SIMD+Align+Affin']
box_data = [df[df['config_category'] == cat]['speedup'].values for cat in categories_for_box]

bp = ax.boxplot(box_data, labels=categories_for_box, patch_artist=True, 
                widths=0.6, showmeans=True, meanline=True)

# Color the boxes
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_xlabel('Configuration Category', fontsize=13, fontweight='bold')
ax.set_ylabel('Speedup', fontsize=13, fontweight='bold')
ax.set_title('Speedup Distribution by Configuration Category', fontsize=15, fontweight='bold')
ax.set_yscale('log')
ax.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig('figures/speedup_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Generated: figures/speedup_distribution.png")
plt.close()

# ============================================================================
# 6. Scaling Efficiency: Speedup gain when doubling threads
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 7))

simd_data = df[df['config_category'].isin(['SIMD', 'SIMD+Affinity', 'SIMD+Align+Affin'])]
matrices = simd_data['matrix'].unique()

thread_pairs = [(1, 2), (2, 4), (4, 8), (8, 16)]
pair_labels = ['1→2', '2→4', '4→8', '8→16']

scaling_results = []
for pair in thread_pairs:
    t1, t2 = pair
    ratios = []
    for matrix in matrices:
        speedup_t1 = simd_data[(simd_data['matrix'] == matrix) & (simd_data['threads'] == t1)]['speedup'].mean()
        speedup_t2 = simd_data[(simd_data['matrix'] == matrix) & (simd_data['threads'] == t2)]['speedup'].mean()
        if not np.isnan(speedup_t1) and not np.isnan(speedup_t2) and speedup_t1 > 0:
            ratios.append(speedup_t2 / speedup_t1)
    scaling_results.append(ratios)

bp = ax.boxplot(scaling_results, labels=pair_labels, patch_artist=True, widths=0.5)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.axhline(y=2.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Ideal (2.0x)')
ax.set_xlabel('Thread Count Doubling', fontsize=13, fontweight='bold')
ax.set_ylabel('Speedup Ratio', fontsize=13, fontweight='bold')
ax.set_title('Scaling Efficiency: Speedup Gain When Doubling Threads\n(SIMD configurations)', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('figures/scaling_efficiency.png', dpi=300, bbox_inches='tight')
print("✓ Generated: figures/scaling_efficiency.png")
plt.close()

# ============================================================================
# 7. Top Configurations Comparison (12 threads)
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 8))

top_10_configs = df[df['threads'] == 12].groupby('configuration')['speedup'].mean().nlargest(10)

bars = ax.barh(range(len(top_10_configs)), top_10_configs.values, color=colors[3], alpha=0.8)
ax.set_yticks(range(len(top_10_configs)))
ax.set_yticklabels(top_10_configs.index, fontsize=11)
ax.set_xlabel('Average Speedup', fontsize=13, fontweight='bold')
ax.set_title('Top 10 Configurations (12 threads, averaged across all matrices)', 
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

# Add value labels on bars
for i, (bar, value) in enumerate(zip(bars, top_10_configs.values)):
    ax.text(value + value*0.02, bar.get_y() + bar.get_height()/2, 
            f'{value:.1f}x', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('figures/top_configurations.png', dpi=300, bbox_inches='tight')
print("✓ Generated: figures/top_configurations.png")
plt.close()

# ============================================================================
# 8. Matrix Size Impact on Speedup
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 7))

simd_data = df[(df['config_category'].isin(['SIMD', 'SIMD+Affinity', 'SIMD+Align+Affin'])) & 
               (df['threads'] == 12)]

# Group by matrix and calculate average speedup and size
matrix_stats = simd_data.groupby('matrix').agg({
    'speedup': 'mean',
    'matrix_size_M': 'first',
    'density_pct': 'first'
}).sort_values('matrix_size_M')

# Color by density
scatter = ax.scatter(matrix_stats['matrix_size_M'], matrix_stats['speedup'], 
                     c=matrix_stats['density_pct'], s=300, alpha=0.7, 
                     cmap='viridis', edgecolors='black', linewidth=1.5)

# Add labels
for idx, row in matrix_stats.iterrows():
    ax.annotate(idx, (row['matrix_size_M'], row['speedup']), 
                fontsize=9, ha='left', va='bottom', 
                xytext=(5, 5), textcoords='offset points')

ax.set_xlabel('Matrix Size (Million elements)', fontsize=13, fontweight='bold')
ax.set_ylabel('Average Speedup (SIMD variants)', fontsize=13, fontweight='bold')
ax.set_title('Matrix Size Impact on Speedup (12 threads, SIMD configs)', 
             fontsize=14, fontweight='bold')
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Density (%)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('figures/matrix_size_impact.png', dpi=300, bbox_inches='tight')
print("✓ Generated: figures/matrix_size_impact.png")
plt.close()

# ============================================================================
# Summary statistics
# ============================================================================
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

print(f"\nBest overall configuration:")
best_config = df.loc[df['speedup'].idxmax()]
print(f"  {best_config['configuration']}")
print(f"  Matrix: {best_config['matrix']}")
print(f"  Threads: {best_config['threads']}")
print(f"  Speedup: {best_config['speedup']:.1f}x")

print(f"\nAverage speedup by category (all data):")
for cat in ['Default', 'Scheduling', 'SIMD', 'SIMD+Affinity', 'SIMD+Register', 'SIMD+Align+Affin']:
    avg_speedup = df[df['config_category'] == cat]['speedup'].mean()
    if not np.isnan(avg_speedup):
        print(f"  {cat:25s}: {avg_speedup:8.1f}x")

print("\n" + "="*60)
print("All plots saved to: figures/")
print("="*60)
