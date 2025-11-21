#!/usr/bin/env python3
"""
Strong Scaling Analysis for OpenMP SpMV Benchmarks

Analyzes parallel efficiency and speedup characteristics from configuration benchmarks.
Generates plots showing how performance scales with increasing thread counts for fixed problem sizes.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import os

def load_data(csv_path):
    """Load benchmark results from CSV file."""
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows from {csv_path}")
        return df
    except FileNotFoundError:
        print(f"ERROR: File not found: {csv_path}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR loading data: {e}")
        sys.exit(1)

def calculate_strong_scaling_metrics(df):
    """
    Calculate strong scaling metrics for each matrix and configuration.
    Strong scaling: Fixed problem size, varying thread count
    - Speedup(p) = T(1) / T(p)
    """
    metrics = []
    for (matrix, config), group in df.groupby(['matrix', 'configuration']):
        group_sorted = group.sort_values('threads')
        baseline = group_sorted[group_sorted['threads'] == 1]
        if baseline.empty:
            continue
        baseline_time = baseline['time_ms'].values[0]
        for _, row in group_sorted.iterrows():
            threads = row['threads']
            time_ms = row['time_ms']
            speedup = baseline_time / time_ms if time_ms > 0 else 0
            metrics.append({
                'matrix': matrix,
                'configuration': config,
                'threads': threads,
                'speedup': speedup
            })
    return pd.DataFrame(metrics)

def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    results_dir = project_root / 'results'
    plots_dir = project_root / 'plots'
    csv_file = results_dir / 'configurations_results.csv'

    print("=" * 80)
    print("STRONG SCALING ANALYSIS: AVERAGE SPEEDUP")
    print("=" * 80)
    print(f"Input: {csv_file}")
    print(f"Output: {plots_dir}")
    print()

    df = load_data(csv_file)
    print("\nCalculating strong scaling metrics...")
    metrics_df = calculate_strong_scaling_metrics(df)
    print(f"Calculated metrics for {len(metrics_df)} test runs")


    avg_speedup = metrics_df.groupby('threads').agg({'speedup': ['mean', 'std']}).reset_index()
    avg_speedup.columns = ['threads', 'mean', 'std']

    # --- Plot 1: Strong Scaling (Speedup vs Threads) ---
    plt.figure(figsize=(10, 7))
    plt.plot(avg_speedup['threads'], avg_speedup['mean'], 'b-o', linewidth=2, label='Mean Speedup')
    plt.fill_between(avg_speedup['threads'],
                     avg_speedup['mean'] - avg_speedup['std'],
                     avg_speedup['mean'] + avg_speedup['std'],
                     alpha=0.3, label='±1 Std Dev')
    plt.plot(avg_speedup['threads'], avg_speedup['threads'], 'k--', linewidth=2, label='Ideal', alpha=0.5)
    plt.xlabel('Number of Threads', fontsize=12, fontweight='bold')
    plt.ylabel('Average Speedup', fontsize=12, fontweight='bold')
    plt.title('Average Speedup Across All Matrices/Configs', fontsize=14, fontweight='bold')
    plt.legend()
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    os.makedirs(plots_dir, exist_ok=True)
    output_file = plots_dir / 'average_speedup_vs_threads.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_file}")

    # --- Plot 2: Parallel Efficiency vs Threads ---
    metrics_df['efficiency'] = metrics_df['speedup'] / metrics_df['threads']
    avg_eff = metrics_df.groupby('threads').agg({'efficiency': ['mean', 'std']}).reset_index()
    avg_eff.columns = ['threads', 'mean', 'std']

    plt.figure(figsize=(10, 7))
    plt.plot(avg_eff['threads'], avg_eff['mean'], 'g-o', linewidth=2, label='Mean Efficiency')
    plt.fill_between(avg_eff['threads'],
                     avg_eff['mean'] - avg_eff['std'],
                     avg_eff['mean'] + avg_eff['std'],
                     alpha=0.3, color='green', label='±1 Std Dev')
    plt.axhline(1.0, color='k', linestyle='--', linewidth=2, alpha=0.5, label='Ideal (100%)')
    plt.xlabel('Number of Threads', fontsize=12, fontweight='bold')
    plt.ylabel('Average Parallel Efficiency', fontsize=12, fontweight='bold')
    plt.title('Average Parallel Efficiency Across All Matrices/Configs', fontsize=14, fontweight='bold')
    plt.legend()
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    output_file_eff = plots_dir / 'average_efficiency_vs_threads.png'
    plt.savefig(output_file_eff, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_file_eff}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Output files: {output_file}, {output_file_eff}")

if __name__ == '__main__':
    main()
