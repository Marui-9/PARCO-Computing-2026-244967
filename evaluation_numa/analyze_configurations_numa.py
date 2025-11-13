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
import matplotlib.pyplot as plt
import os

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
colors = plt.cm.Set2(np.linspace(0, 1, 8))

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

# ============================================================================
# GENERATE VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# Create figures directory
os.makedirs('figures_numa', exist_ok=True)

# ============================================================================
# 1. Speedup vs Thread Count by Binding Policy
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 7))

for i, bind_policy in enumerate(sorted(df['bind_category'].unique())):
    bind_data = df[df['bind_category'] == bind_policy]
    speedup_by_threads = bind_data.groupby('threads')['speedup'].mean()
    if len(speedup_by_threads) > 0:
        ax.plot(speedup_by_threads.index, speedup_by_threads.values, 
                marker='o', linewidth=2.5, markersize=8, label=bind_policy, color=colors[i])

ax.set_xlabel('Number of Threads', fontsize=13, fontweight='bold')
ax.set_ylabel('Average Speedup', fontsize=13, fontweight='bold')
ax.set_title('NUMA: Speedup vs Thread Count by Binding Policy', fontsize=15, fontweight='bold')
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xticks(sorted(df['threads'].unique()))
plt.tight_layout()
plt.savefig('figures_numa/speedup_vs_threads_binding.png', dpi=300, bbox_inches='tight')
print("✓ Generated: figures_numa/speedup_vs_threads_binding.png")
plt.close()

# ============================================================================
# 2. Efficiency vs Thread Count by Binding Policy
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 7))

for i, bind_policy in enumerate(sorted(df['bind_category'].unique())):
    bind_data = df[df['bind_category'] == bind_policy]
    efficiency_by_threads = bind_data.groupby('threads')['efficiency_pct'].mean()
    if len(efficiency_by_threads) > 0:
        ax.plot(efficiency_by_threads.index, efficiency_by_threads.values, 
                marker='s', linewidth=2.5, markersize=8, label=bind_policy, color=colors[i])

# Add ideal efficiency line
threads = sorted(df['threads'].unique())
ax.plot(threads, [100] * len(threads), 'k--', linewidth=2, alpha=0.5, label='Ideal (100%)')

ax.set_xlabel('Number of Threads', fontsize=13, fontweight='bold')
ax.set_ylabel('Parallel Efficiency (%)', fontsize=13, fontweight='bold')
ax.set_title('NUMA: Parallel Efficiency vs Thread Count by Binding Policy', fontsize=15, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xticks(threads)
plt.tight_layout()
plt.savefig('figures_numa/efficiency_vs_threads_binding.png', dpi=300, bbox_inches='tight')
print("✓ Generated: figures_numa/efficiency_vs_threads_binding.png")
plt.close()

# ============================================================================
# 3. Speedup Heatmap: Binding Policy × Thread Count
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 8))

pivot_speedup = df.pivot_table(values='speedup', 
                                index='bind_category', 
                                columns='threads', 
                                aggfunc='mean')

if len(pivot_speedup) > 0:
    im = ax.imshow(pivot_speedup.values, aspect='auto', cmap='YlOrRd', interpolation='nearest')

    # Set ticks and labels
    ax.set_xticks(np.arange(len(pivot_speedup.columns)))
    ax.set_yticks(np.arange(len(pivot_speedup.index)))
    ax.set_xticklabels(pivot_speedup.columns, fontsize=11)
    ax.set_yticklabels(pivot_speedup.index, fontsize=11)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Average Speedup', fontsize=12, fontweight='bold')

    # Add text annotations
    for i in range(len(pivot_speedup.index)):
        for j in range(len(pivot_speedup.columns)):
            value = pivot_speedup.values[i, j]
            if not np.isnan(value):
                text = ax.text(j, i, f'{value:.0f}', ha='center', va='center', 
                              color='black' if value < pivot_speedup.values.max()/2 else 'white',
                              fontsize=10, fontweight='bold')

    ax.set_xlabel('Number of Threads', fontsize=13, fontweight='bold')
    ax.set_ylabel('Binding Policy', fontsize=13, fontweight='bold')
    ax.set_title('NUMA: Speedup Heatmap by Binding Policy and Thread Count', 
                 fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('figures_numa/speedup_heatmap_binding_threads.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: figures_numa/speedup_heatmap_binding_threads.png")
plt.close()

# ============================================================================
# 4. Scaling Efficiency (speedup ratio when doubling threads)
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 7))

thread_pairs = [(24, 48), (48, 96)]
pair_labels = ['24→48', '48→96']
ideal_ratios = [2.0, 2.0]

scaling_results = {}
for bind_policy in df['bind_category'].unique():
    bind_data = df[df['bind_category'] == bind_policy]
    policy_results = []
    
    for t1, t2 in thread_pairs:
        ratios = []
        for matrix in bind_data['matrix'].unique():
            for config in bind_data['configuration'].unique():
                data_t1 = bind_data[(bind_data['threads'] == t1) & 
                                   (bind_data['matrix'] == matrix) & 
                                   (bind_data['configuration'] == config)]
                data_t2 = bind_data[(bind_data['threads'] == t2) & 
                                   (bind_data['matrix'] == matrix) & 
                                   (bind_data['configuration'] == config)]
                
                if not data_t1.empty and not data_t2.empty:
                    speedup_t1 = data_t1['speedup'].values[0]
                    speedup_t2 = data_t2['speedup'].values[0]
                    if speedup_t1 > 0:
                        ratios.append(speedup_t2 / speedup_t1)
        
        if ratios:
            policy_results.append(np.mean(ratios))
        else:
            policy_results.append(np.nan)
    
    scaling_results[bind_policy] = policy_results

# Plot grouped bars
x = np.arange(len(pair_labels))
width = 0.2
multiplier = 0

for i, (bind_policy, values) in enumerate(scaling_results.items()):
    offset = width * multiplier
    ax.bar(x + offset, values, width, label=bind_policy, color=colors[i], alpha=0.8)
    multiplier += 1

# Add ideal line
ax.plot(x, ideal_ratios, 'k--', linewidth=2, marker='*', markersize=12, 
        label='Ideal', alpha=0.7)

ax.set_xlabel('Thread Count Doubling', fontsize=13, fontweight='bold')
ax.set_ylabel('Speedup Ratio', fontsize=13, fontweight='bold')
ax.set_title('NUMA: Scaling Efficiency When Doubling Threads', fontsize=15, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(pair_labels)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('figures_numa/scaling_efficiency_doubling.png', dpi=300, bbox_inches='tight')
print("✓ Generated: figures_numa/scaling_efficiency_doubling.png")
plt.close()

# ============================================================================
# 5. Configuration Performance Comparison at High Thread Counts
# ============================================================================
high_thread = df['threads'].max()
if high_thread >= 48:
    fig, ax = plt.subplots(figsize=(14, 8))

    high_thread_data = df[df['threads'] == high_thread]
    top_configs = high_thread_data.groupby('configuration')['speedup'].mean().nlargest(10)

    bars = ax.barh(range(len(top_configs)), top_configs.values, color=colors[3], alpha=0.8)
    ax.set_yticks(range(len(top_configs)))
    ax.set_yticklabels(top_configs.index, fontsize=11)
    ax.set_xlabel('Average Speedup', fontsize=13, fontweight='bold')
    ax.set_title(f'NUMA: Top 10 Configurations at {high_thread} Threads', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, top_configs.values)):
        ax.text(value + value*0.02, bar.get_y() + bar.get_height()/2, 
                f'{value:.1f}x', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'figures_numa/top_configs_{high_thread}_threads.png', dpi=300, bbox_inches='tight')
    print(f"✓ Generated: figures_numa/top_configs_{high_thread}_threads.png")
    plt.close()

# ============================================================================
# 6. Box Plot: Speedup Distribution by Binding Policy
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 7))

binding_policies = sorted(df['bind_category'].unique())
box_data = [df[df['bind_category'] == policy]['speedup'].values for policy in binding_policies]

bp = ax.boxplot(box_data, tick_labels=binding_policies, patch_artist=True, 
                widths=0.6, showmeans=True, meanline=True)

# Color the boxes
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_xlabel('Binding Policy', fontsize=13, fontweight='bold')
ax.set_ylabel('Speedup', fontsize=13, fontweight='bold')
ax.set_title('NUMA: Speedup Distribution by Binding Policy', fontsize=15, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig('figures_numa/speedup_distribution_binding.png', dpi=300, bbox_inches='tight')
print("✓ Generated: figures_numa/speedup_distribution_binding.png")
plt.close()

# ============================================================================
# 7. Speedup vs Thread Count for Different Matrices
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 7))

matrices = df['matrix'].unique()[:5]  # Plot up to 5 matrices
for i, matrix in enumerate(matrices):
    matrix_data = df[df['matrix'] == matrix]
    speedup_by_threads = matrix_data.groupby('threads')['speedup'].mean()
    if len(speedup_by_threads) > 0:
        ax.plot(speedup_by_threads.index, speedup_by_threads.values, 
                marker='o', linewidth=2.5, markersize=8, 
                label=f'{matrix} ({matrix_data["density_pct"].iloc[0]:.2f}%)', 
                color=colors[i])

ax.set_xlabel('Number of Threads', fontsize=13, fontweight='bold')
ax.set_ylabel('Average Speedup', fontsize=13, fontweight='bold')
ax.set_title('NUMA: Speedup vs Thread Count by Matrix', fontsize=15, fontweight='bold')
ax.legend(fontsize=10, loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xticks(sorted(df['threads'].unique()))
plt.tight_layout()
plt.savefig('figures_numa/speedup_vs_threads_matrices.png', dpi=300, bbox_inches='tight')
print("✓ Generated: figures_numa/speedup_vs_threads_matrices.png")
plt.close()

print("\n" + "="*80)
print("All NUMA visualizations saved to: figures_numa/")
print("="*80)
