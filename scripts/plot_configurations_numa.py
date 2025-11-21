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

# Read the CSV file
if os.path.exists('configurations_numa_results.csv'):
    csv_path = 'configurations_numa_results.csv'
elif os.path.exists('results/configurations_numa_results.csv'):
    csv_path = 'results/configurations_numa_results.csv'
else:
    raise FileNotFoundError('configurations_numa_results.csv not found!')
df = pd.read_csv(csv_path)

# Add bind_category column if not present
if 'bind_category' not in df.columns and 'bind_policy' in df.columns:
    def categorize_binding(policy):
        if 'close' in policy.lower():
            return 'Close'
        elif 'spread' in policy.lower():
            return 'Spread'
        elif 'master' in policy.lower():
            return 'Master'
        else:
            return policy
    df['bind_category'] = df['bind_policy'].apply(categorize_binding)

# Create plots directory
os.makedirs('plots', exist_ok=True)

# Define a colormap with enough distinct colors
colors = plt.cm.tab10(np.linspace(0, 1, 10))

# Plot 1: Speedup vs Thread Count by Binding Policy
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
plt.savefig('plots/numa_speedup_vs_threads_binding.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Scaling Efficiency (speedup ratio when doubling threads)
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
                data_t1 = bind_data[(bind_data['threads'] == t1) & \
                                   (bind_data['matrix'] == matrix) & \
                                   (bind_data['configuration'] == config)]
                data_t2 = bind_data[(bind_data['threads'] == t2) & \
                                   (bind_data['matrix'] == matrix) & \
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
x = np.arange(len(pair_labels))
width = 0.2
multiplier = 0
for i, (bind_policy, values) in enumerate(scaling_results.items()):
    offset = width * multiplier
    ax.bar(x + offset, values, width, label=bind_policy, alpha=0.8)
    multiplier += 1
ax.plot(x, ideal_ratios, 'k--', linewidth=2, marker='*', markersize=12, 
        label='Ideal', alpha=0.7)
ax.set_xlabel('Thread Count Doubling', fontsize=13, fontweight='bold')
ax.set_ylabel('Speedup Ratio', fontsize=13, fontweight='bold')
ax.set_title('NUMA: Scaling Efficiency When Doubling Threads', fontsize=15, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(pair_labels, fontsize=12)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('plots/numa_scaling_efficiency_doubling.png', dpi=300, bbox_inches='tight')
plt.close()
