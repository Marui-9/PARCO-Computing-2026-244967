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

# Configure matplotlib for better-looking plots
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

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
    - Efficiency(p) = Speedup(p) / p
    - Ideal speedup = p (linear)
    """
    metrics = []
    
    # Group by matrix and configuration
    for (matrix, config), group in df.groupby(['matrix', 'configuration']):
        group_sorted = group.sort_values('threads')
        
        # Get baseline (1 thread) time
        baseline = group_sorted[group_sorted['threads'] == 1]
        if baseline.empty:
            print(f"WARNING: No single-thread baseline for {matrix} / {config}")
            continue
        
        baseline_time = baseline['time_ms'].values[0]
        
        for _, row in group_sorted.iterrows():
            threads = row['threads']
            time_ms = row['time_ms']
            
            # Calculate strong scaling metrics
            speedup = baseline_time / time_ms if time_ms > 0 else 0
            efficiency = (speedup / threads * 100) if threads > 0 else 0
            parallel_overhead = (threads * time_ms - baseline_time) / baseline_time * 100
            
            metrics.append({
                'matrix': matrix,
                'configuration': config,
                'threads': threads,
                'time_ms': time_ms,
                'baseline_time_ms': baseline_time,
                'speedup': speedup,
                'efficiency_pct': efficiency,
                'ideal_speedup': threads,
                'parallel_overhead_pct': parallel_overhead,
                'nnz': row['nnz'],
                'density_pct': row['density_pct']
            })
    
    return pd.DataFrame(metrics)

def plot_strong_scaling_by_matrix(metrics_df, output_dir):
    """Plot strong scaling curves for each matrix (all configurations)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    matrices = metrics_df['matrix'].unique()
    
    for matrix in matrices:
        matrix_data = metrics_df[metrics_df['matrix'] == matrix]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Strong Scaling Analysis: {matrix}', fontsize=16, fontweight='bold')
        
        # Plot 1: Speedup vs Threads
        ax1 = axes[0, 0]
        for config in matrix_data['configuration'].unique()[:10]:  # Limit to top 10 configs for clarity
            config_data = matrix_data[matrix_data['configuration'] == config]
            ax1.plot(config_data['threads'], config_data['speedup'], 
                    marker='o', label=config, alpha=0.7)
        
        # Ideal linear speedup reference
        threads_range = matrix_data['threads'].unique()
        ax1.plot(threads_range, threads_range, 'k--', linewidth=2, label='Ideal (Linear)', alpha=0.5)
        
        ax1.set_xlabel('Number of Threads')
        ax1.set_ylabel('Speedup')
        ax1.set_title('Speedup vs Thread Count')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax1.set_xlim(left=0)
        ax1.set_ylim(bottom=0)
        
        # Plot 2: Efficiency vs Threads
        ax2 = axes[0, 1]
        for config in matrix_data['configuration'].unique()[:10]:
            config_data = matrix_data[matrix_data['configuration'] == config]
            ax2.plot(config_data['threads'], config_data['efficiency_pct'], 
                    marker='s', label=config, alpha=0.7)
        
        ax2.axhline(y=100, color='k', linestyle='--', linewidth=2, label='Ideal (100%)', alpha=0.5)
        ax2.set_xlabel('Number of Threads')
        ax2.set_ylabel('Parallel Efficiency (%)')
        ax2.set_title('Parallel Efficiency vs Thread Count')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax2.set_xlim(left=0)
        ax2.set_ylim(0, 120)
        
        # Plot 3: Execution Time vs Threads (log scale)
        ax3 = axes[1, 0]
        for config in matrix_data['configuration'].unique()[:10]:
            config_data = matrix_data[matrix_data['configuration'] == config]
            ax3.plot(config_data['threads'], config_data['time_ms'], 
                    marker='^', label=config, alpha=0.7)
        
        ax3.set_xlabel('Number of Threads')
        ax3.set_ylabel('Execution Time (ms)')
        ax3.set_title('Execution Time vs Thread Count (log scale)')
        ax3.set_yscale('log')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax3.set_xlim(left=0)
        
        # Plot 4: Parallel Overhead vs Threads
        ax4 = axes[1, 1]
        for config in matrix_data['configuration'].unique()[:10]:
            config_data = matrix_data[matrix_data['configuration'] == config]
            ax4.plot(config_data['threads'], config_data['parallel_overhead_pct'], 
                    marker='d', label=config, alpha=0.7)
        
        ax4.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        ax4.set_xlabel('Number of Threads')
        ax4.set_ylabel('Parallel Overhead (%)')
        ax4.set_title('Parallel Overhead vs Thread Count')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax4.set_xlim(left=0)
        
        plt.tight_layout()
        
        # Sanitize filename
        safe_matrix_name = matrix.replace('/', '_').replace('\\', '_').replace('.mtx', '')
        output_file = output_dir / f'strong_scaling_{safe_matrix_name}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()

def plot_best_configuration_scaling(metrics_df, output_dir):
    """Plot strong scaling for best configuration at 24 threads for each matrix."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find best configuration at 24 threads for each matrix
    best_configs = []
    for matrix in metrics_df['matrix'].unique():
        matrix_24t = metrics_df[(metrics_df['matrix'] == matrix) & (metrics_df['threads'] == 24)]
        if not matrix_24t.empty:
            best = matrix_24t.loc[matrix_24t['speedup'].idxmax()]
            best_configs.append({
                'matrix': matrix,
                'best_config': best['configuration'],
                'speedup_24t': best['speedup'],
                'efficiency_24t': best['efficiency_pct']
            })
    
    best_configs_df = pd.DataFrame(best_configs)
    
    # Plot scaling curves for best configurations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Strong Scaling: Best Configuration per Matrix (at 24 threads)', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Speedup comparison
    ax1 = axes[0, 0]
    for _, row in best_configs_df.iterrows():
        matrix = row['matrix']
        config = row['best_config']
        data = metrics_df[(metrics_df['matrix'] == matrix) & 
                         (metrics_df['configuration'] == config)]
        ax1.plot(data['threads'], data['speedup'], marker='o', label=matrix, alpha=0.7)
    
    threads_range = metrics_df['threads'].unique()
    ax1.plot(threads_range, threads_range, 'k--', linewidth=2, label='Ideal', alpha=0.5)
    ax1.set_xlabel('Number of Threads')
    ax1.set_ylabel('Speedup')
    ax1.set_title('Speedup (Best Configs)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.set_xlim(left=0)
    
    # Plot 2: Efficiency comparison
    ax2 = axes[0, 1]
    for _, row in best_configs_df.iterrows():
        matrix = row['matrix']
        config = row['best_config']
        data = metrics_df[(metrics_df['matrix'] == matrix) & 
                         (metrics_df['configuration'] == config)]
        ax2.plot(data['threads'], data['efficiency_pct'], marker='s', label=matrix, alpha=0.7)
    
    ax2.axhline(y=100, color='k', linestyle='--', linewidth=2, label='Ideal', alpha=0.5)
    ax2.set_xlabel('Number of Threads')
    ax2.set_ylabel('Parallel Efficiency (%)')
    ax2.set_title('Efficiency (Best Configs)')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.set_xlim(left=0)
    ax2.set_ylim(0, 120)
    
    # Plot 3: Speedup at 24 threads
    ax3 = axes[1, 0]
    matrices = best_configs_df['matrix'].values
    speedups = best_configs_df['speedup_24t'].values
    colors = plt.cm.viridis(np.linspace(0, 1, len(matrices)))
    
    bars = ax3.barh(range(len(matrices)), speedups, color=colors, alpha=0.7)
    ax3.set_yticks(range(len(matrices)))
    ax3.set_yticklabels(matrices, fontsize=8)
    ax3.set_xlabel('Speedup at 24 Threads')
    ax3.set_title('Best Configuration Speedup (24 threads)')
    ax3.axvline(x=24, color='k', linestyle='--', linewidth=2, label='Ideal (24x)', alpha=0.5)
    ax3.legend()
    
    # Plot 4: Efficiency at 24 threads
    ax4 = axes[1, 1]
    efficiencies = best_configs_df['efficiency_24t'].values
    bars = ax4.barh(range(len(matrices)), efficiencies, color=colors, alpha=0.7)
    ax4.set_yticks(range(len(matrices)))
    ax4.set_yticklabels(matrices, fontsize=8)
    ax4.set_xlabel('Parallel Efficiency (%) at 24 Threads')
    ax4.set_title('Best Configuration Efficiency (24 threads)')
    ax4.axvline(x=100, color='k', linestyle='--', linewidth=2, label='Ideal (100%)', alpha=0.5)
    ax4.legend()
    ax4.set_xlim(0, 120)
    
    plt.tight_layout()
    output_file = output_dir / 'strong_scaling_best_configs.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()
    
    return best_configs_df

def plot_scalability_summary(metrics_df, output_dir):
    """Create summary plots showing scalability characteristics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Strong Scaling Summary: All Matrices and Configurations', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Average speedup across all matrices/configs
    ax1 = axes[0, 0]
    avg_speedup = metrics_df.groupby('threads').agg({
        'speedup': ['mean', 'std', 'min', 'max']
    }).reset_index()
    avg_speedup.columns = ['threads', 'mean', 'std', 'min', 'max']
    
    ax1.plot(avg_speedup['threads'], avg_speedup['mean'], 'b-o', linewidth=2, label='Mean Speedup')
    ax1.fill_between(avg_speedup['threads'], 
                     avg_speedup['mean'] - avg_speedup['std'],
                     avg_speedup['mean'] + avg_speedup['std'],
                     alpha=0.3, label='±1 Std Dev')
    ax1.plot(avg_speedup['threads'], avg_speedup['threads'], 'k--', linewidth=2, label='Ideal', alpha=0.5)
    ax1.set_xlabel('Number of Threads')
    ax1.set_ylabel('Speedup')
    ax1.set_title('Average Speedup Across All Tests')
    ax1.legend()
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)
    
    # Plot 2: Average efficiency across all matrices/configs
    ax2 = axes[0, 1]
    avg_efficiency = metrics_df.groupby('threads').agg({
        'efficiency_pct': ['mean', 'std', 'min', 'max']
    }).reset_index()
    avg_efficiency.columns = ['threads', 'mean', 'std', 'min', 'max']
    
    ax2.plot(avg_efficiency['threads'], avg_efficiency['mean'], 'g-s', linewidth=2, label='Mean Efficiency')
    ax2.fill_between(avg_efficiency['threads'], 
                     avg_efficiency['mean'] - avg_efficiency['std'],
                     avg_efficiency['mean'] + avg_efficiency['std'],
                     alpha=0.3, label='±1 Std Dev')
    ax2.axhline(y=100, color='k', linestyle='--', linewidth=2, label='Ideal (100%)', alpha=0.5)
    ax2.set_xlabel('Number of Threads')
    ax2.set_ylabel('Parallel Efficiency (%)')
    ax2.set_title('Average Efficiency Across All Tests')
    ax2.legend()
    ax2.set_xlim(left=0)
    ax2.set_ylim(0, 120)
    
    # Plot 3: Efficiency distribution by thread count
    ax3 = axes[1, 0]
    thread_counts = sorted(metrics_df['threads'].unique())
    efficiency_data = [metrics_df[metrics_df['threads'] == t]['efficiency_pct'].values 
                      for t in thread_counts]
    
    bp = ax3.boxplot(efficiency_data, labels=thread_counts, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    ax3.axhline(y=100, color='k', linestyle='--', linewidth=2, alpha=0.5)
    ax3.set_xlabel('Number of Threads')
    ax3.set_ylabel('Parallel Efficiency (%)')
    ax3.set_title('Efficiency Distribution by Thread Count')
    ax3.set_ylim(0, 120)
    
    # Plot 4: Scalability by matrix size (NNZ)
    ax4 = axes[1, 1]
    # Get data at 24 threads
    data_24t = metrics_df[metrics_df['threads'] == 24].copy()
    data_24t_sorted = data_24t.sort_values('nnz')
    
    scatter = ax4.scatter(data_24t_sorted['nnz'], data_24t_sorted['efficiency_pct'],
                         c=data_24t_sorted['density_pct'], cmap='viridis', 
                         alpha=0.6, s=100)
    ax4.axhline(y=100, color='k', linestyle='--', linewidth=2, alpha=0.5)
    ax4.set_xlabel('Problem Size (NNZ)')
    ax4.set_ylabel('Parallel Efficiency (%) at 24 Threads')
    ax4.set_title('Efficiency vs Problem Size (24 threads)')
    ax4.set_xscale('log')
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Density (%)')
    
    plt.tight_layout()
    output_file = output_dir / 'strong_scaling_summary.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def generate_scaling_report(metrics_df, best_configs_df, output_dir):
    """Generate text report with scaling statistics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'strong_scaling_report.txt'
    
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("STRONG SCALING ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total test configurations: {len(metrics_df)}\n")
        f.write(f"Number of matrices: {metrics_df['matrix'].nunique()}\n")
        f.write(f"Number of configurations: {metrics_df['configuration'].nunique()}\n")
        f.write(f"Thread counts tested: {sorted(metrics_df['threads'].unique())}\n\n")
        
        # Scalability at maximum threads (24)
        data_24t = metrics_df[metrics_df['threads'] == 24]
        f.write(f"SCALABILITY AT 24 THREADS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Average speedup: {data_24t['speedup'].mean():.2f}x (ideal: 24x)\n")
        f.write(f"Best speedup: {data_24t['speedup'].max():.2f}x\n")
        f.write(f"Worst speedup: {data_24t['speedup'].min():.2f}x\n")
        f.write(f"Average efficiency: {data_24t['efficiency_pct'].mean():.1f}%\n")
        f.write(f"Best efficiency: {data_24t['efficiency_pct'].max():.1f}%\n")
        f.write(f"Worst efficiency: {data_24t['efficiency_pct'].min():.1f}%\n\n")
        
        # Best configurations per matrix
        f.write("BEST CONFIGURATION PER MATRIX (at 24 threads)\n")
        f.write("-" * 80 + "\n")
        for _, row in best_configs_df.iterrows():
            f.write(f"{row['matrix']:30s} | {row['best_config']:40s} | "
                   f"Speedup: {row['speedup_24t']:5.2f}x | "
                   f"Efficiency: {row['efficiency_24t']:5.1f}%\n")
        f.write("\n")
        
        # Efficiency degradation by thread count
        f.write("EFFICIENCY DEGRADATION BY THREAD COUNT\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Threads':>8s} | {'Avg Efficiency':>15s} | {'Std Dev':>10s} | {'Min':>8s} | {'Max':>8s}\n")
        f.write("-" * 80 + "\n")
        for threads in sorted(metrics_df['threads'].unique()):
            data_t = metrics_df[metrics_df['threads'] == threads]
            f.write(f"{threads:8d} | {data_t['efficiency_pct'].mean():14.2f}% | "
                   f"{data_t['efficiency_pct'].std():9.2f}% | "
                   f"{data_t['efficiency_pct'].min():7.2f}% | "
                   f"{data_t['efficiency_pct'].max():7.2f}%\n")
        f.write("\n")
        
        # Top 10 most scalable configurations (at 24 threads)
        f.write("TOP 10 MOST SCALABLE CONFIGURATIONS (highest efficiency at 24 threads)\n")
        f.write("-" * 80 + "\n")
        top_configs = data_24t.nlargest(10, 'efficiency_pct')
        for i, (_, row) in enumerate(top_configs.iterrows(), 1):
            f.write(f"{i:2d}. {row['matrix']:25s} | {row['configuration']:40s} | "
                   f"Speedup: {row['speedup']:5.2f}x | Efficiency: {row['efficiency_pct']:5.1f}%\n")
        
    print(f"Saved: {output_file}")

def main():
    # Setup paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    results_dir = project_root / 'results'
    plots_dir = project_root / 'plots'
    
    csv_file = results_dir / 'configurations_results.csv'
    
    print("=" * 80)
    print("STRONG SCALING ANALYSIS")
    print("=" * 80)
    print(f"Input: {csv_file}")
    print(f"Output: {plots_dir}")
    print()
    
    # Load data
    df = load_data(csv_file)
    
    # Calculate strong scaling metrics
    print("\nCalculating strong scaling metrics...")
    metrics_df = calculate_strong_scaling_metrics(df)
    print(f"Calculated metrics for {len(metrics_df)} test runs")
    
    # Generate plots
    print("\nGenerating plots...")
    print("1. Per-matrix strong scaling plots...")
    plot_strong_scaling_by_matrix(metrics_df, plots_dir / 'strong_scaling')
    
    print("2. Best configuration scaling comparison...")
    best_configs_df = plot_best_configuration_scaling(metrics_df, plots_dir / 'strong_scaling')
    
    print("3. Scalability summary plots...")
    plot_scalability_summary(metrics_df, plots_dir / 'strong_scaling')
    
    # Generate report
    print("\nGenerating scaling report...")
    generate_scaling_report(metrics_df, best_configs_df, plots_dir / 'strong_scaling')
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Output directory: {plots_dir / 'strong_scaling'}")
    print(f"Report file: {plots_dir / 'strong_scaling' / 'strong_scaling_report.txt'}")
    print()

if __name__ == '__main__':
    main()
