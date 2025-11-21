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
    By default this function computes speedup relative to a serial baseline per matrix.
    The preferred serial baseline is the time at `threads == 1` for the unoptimized
    static configuration (e.g., `Static (default)`). If that is not available, the
    fastest 1-thread time for the matrix is used as a fallback.
    - Speedup(p) = T_serial / T(p)
    """
    metrics = []
    # Build per-matrix serial baseline times (prefer unoptimized Static at 1 thread)
    serial_baseline = {}
    for matrix, group in df.groupby('matrix'):
        one_thread = group[group['threads'] == 1]
        if one_thread.empty:
            continue
        # Prefer 'Static' config without SIMD in the name
        preferred = one_thread[one_thread['configuration'].str.contains('Static', na=False) & ~one_thread['configuration'].str.contains('SIMD', na=False)]
        if not preferred.empty:
            serial_time = preferred['time_ms'].values[0]
        else:
            # Fallback: fastest 1-thread time
            serial_time = one_thread['time_ms'].min()
        serial_baseline[matrix] = serial_time
    for (matrix, config), group in df.groupby(['matrix', 'configuration']):
        group_sorted = group.sort_values('threads')
        # Use serial baseline for this matrix when available
        if matrix not in serial_baseline:
            # If no serial baseline (no 1-thread data), skip this (matrix,config)
            continue
        baseline_time = serial_baseline[matrix]
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
    # Prefer matrices_results.csv if available (contains best-config speedups per matrix)
    if (results_dir / 'matrices_results.csv').exists():
        csv_file = results_dir / 'matrices_results.csv'
    else:
        csv_file = results_dir / 'configurations_results.csv'

    print("=" * 80)
    print("STRONG SCALING ANALYSIS: AVERAGE SPEEDUP")
    print("=" * 80)
    print(f"Input: {csv_file}")
    print(f"Output: {plots_dir}")
    print()

    df = load_data(csv_file)
    # If using matrices_results.csv, it already contains best-config speedups per matrix
    if csv_file.name == 'matrices_results.csv':
        print("\nUsing matrices_results.csv: building metrics from 'speedup_x' column")
        # create a clean matrix name column (avoid touching existing 'matrix' column)
        df['matrix_name'] = df['matrix'].apply(lambda x: os.path.basename(str(x).strip('"')).replace('.mtx', ''))
        # ensure numeric
        df['speedup_x'] = pd.to_numeric(df['speedup_x'], errors='coerce')
        df = df.dropna(subset=['speedup_x', 'threads', 'matrix_name'])
        # Build metrics_df with unique column names to avoid duplicate 'matrix' column issues
        metrics_df = df[['matrix_name', 'threads', 'speedup_x']].copy()
        metrics_df.columns = ['matrix', 'threads', 'speedup']
        print(f"Built metrics for {metrics_df['matrix'].nunique()} matrices, {len(metrics_df)} rows total")
    else:
        print("\nCalculating strong scaling metrics from configurations_results.csv...")
        metrics_df = calculate_strong_scaling_metrics(df)
    print(f"Calculated metrics for {len(metrics_df)} test runs")

    # --- Plot 1: Strong Scaling (Best configuration per matrix OR provided best results) ---
    # If 'configuration' exists in metrics_df (we built metrics from configurations_results.csv),
    # select the best configuration per matrix at the matrix's largest thread count.
    # If metrics_df comes from matrices_results.csv (already per-matrix bests), just use each matrix's series.
    per_matrix_series = []
    if 'configuration' in metrics_df.columns:
        for matrix, group in metrics_df.groupby('matrix'):
            max_t = int(group['threads'].max())
            candidates = group[group['threads'] == max_t]
            if candidates.empty:
                continue
            best_row = candidates.loc[candidates['speedup'].idxmax()]
            best_cfg = best_row['configuration']
            sel = metrics_df[(metrics_df['matrix'] == matrix) & (metrics_df['configuration'] == best_cfg)].copy()
            if not sel.empty:
                per_matrix_series.append(sel)
    else:
        # metrics_df already contains one (best) configuration per matrix across threads
        for matrix, group in metrics_df.groupby('matrix'):
            sel = group.sort_values('threads').copy()
            if not sel.empty:
                per_matrix_series.append(sel)

    output_file = plots_dir / 'strong_scaling_best_configs.png'
    if per_matrix_series:
        best_df = pd.concat(per_matrix_series, ignore_index=True)
        plt.figure(figsize=(12, 8))
        matrices = sorted(best_df['matrix'].unique())
        for m in matrices:
            series = best_df[best_df['matrix'] == m].sort_values('threads')
            plt.plot(series['threads'], series['speedup'], '-o', linewidth=1.5, markersize=6, label=str(m))

        # Plot mean across matrices (best configs)
        mean_best = best_df.groupby('threads')['speedup'].mean().sort_index()
        plt.plot(mean_best.index, mean_best.values, 'k--', linewidth=3, label='Mean (best configs)')

        plt.xlabel('Number of Threads', fontsize=12, fontweight='bold')
        plt.ylabel('Speedup (Best Config per Matrix)', fontsize=12, fontweight='bold')
        plt.title('Strong Scaling: Best Configuration per Matrix', fontsize=16, fontweight='bold')
        plt.legend(fontsize=8, ncol=2)
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        os.makedirs(plots_dir, exist_ok=True)
        output_file = plots_dir / 'strong_scaling_best_configs.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n✓ Plot saved to: {output_file}")
    else:
        print("No per-matrix series found to plot strong scaling.")

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
