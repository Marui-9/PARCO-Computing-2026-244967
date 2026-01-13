#!/usr/bin/env python3
"""
Generate plots for configurations results: strong scaling, speedup, and parallel efficiency.
Visualizes performance trends across different MPI communication strategies.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import sys

# Set publication-quality plotting parameters
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9
mpl.rcParams['legend.fontsize'] = 9
mpl.rcParams['figure.titlesize'] = 13
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 6

def load_data(csv_path):
    """Load and validate configurations results CSV."""
    if not Path(csv_path).exists():
        print(f"ERROR: CSV file not found: {csv_path}")
        sys.exit(1)
    
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    required = ['num_procs', 'config_name', 'avg_time_ms', 'speedup', 'efficiency_pct']
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        sys.exit(1)
    
    return df

def calculate_averages(df):
    """Calculate averages across all matrices for each config and process count."""
    # Group by configuration and process count, average across matrices
    avg_data = df.groupby(['config_name', 'num_procs']).agg({
        'avg_time_ms': 'mean',
        'speedup': 'mean',
        'efficiency_pct': 'mean'
    }).reset_index()
    
    return avg_data

def plot_strong_scaling(avg_data, output_dir):
    """Plot strong scaling: execution time vs number of processes."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    configs = sorted(avg_data['config_name'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(configs)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    for i, config in enumerate(configs):
        config_data = avg_data[avg_data['config_name'] == config].sort_values('num_procs')
        
        ax.plot(config_data['num_procs'], config_data['avg_time_ms'],
                marker=markers[i % len(markers)], color=colors[i], 
                label=config, linewidth=2, markersize=7)
    
    ax.set_xlabel('Number of Processes', fontweight='bold')
    ax.set_ylabel('Average Execution Time (ms)', fontweight='bold')
    ax.set_title('Strong Scaling: Execution Time vs Process Count\n(Average across all matrices)', 
                 fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.grid(True, which='both', alpha=0.3, linestyle='--')
    ax.legend(loc='best', framealpha=0.9)
    
    plt.tight_layout()
    output_path = output_dir / 'strong_scaling.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_speedup(avg_data, output_dir):
    """Plot speedup vs number of processes."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    configs = sorted(avg_data['config_name'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(configs)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # Plot ideal speedup line
    proc_counts = sorted(avg_data['num_procs'].unique())
    min_procs = min(proc_counts)
    ideal_speedup = [p / min_procs for p in proc_counts]
    ax.plot(proc_counts, ideal_speedup, 'k--', linewidth=2, 
            label='Ideal Speedup', alpha=0.7)
    
    for i, config in enumerate(configs):
        config_data = avg_data[avg_data['config_name'] == config].sort_values('num_procs')
        
        ax.plot(config_data['num_procs'], config_data['speedup'],
                marker=markers[i % len(markers)], color=colors[i], 
                label=config, linewidth=2, markersize=7)
    
    ax.set_xlabel('Number of Processes', fontweight='bold')
    ax.set_ylabel('Speedup', fontweight='bold')
    ax.set_title('Speedup vs Process Count\n(Average across all matrices)', 
                 fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.grid(True, which='both', alpha=0.3, linestyle='--')
    ax.legend(loc='best', framealpha=0.9)
    
    # Add annotation for super-linear regions
    max_speedup = avg_data.groupby('num_procs')['speedup'].max()
    for procs in proc_counts:
        expected = procs / min_procs
        actual = max_speedup.get(procs, 0)
        if actual > expected * 1.1:  # More than 10% above ideal
            ax.annotate('Super-linear', xy=(procs, actual), 
                       xytext=(procs * 1.5, actual * 1.1),
                       arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                       fontsize=8, color='red')
            break
    
    plt.tight_layout()
    output_path = output_dir / 'speedup_vs_procs.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_efficiency(avg_data, output_dir):
    """Plot parallel efficiency vs number of processes."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    configs = sorted(avg_data['config_name'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(configs)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # Plot 100% efficiency line
    proc_counts = sorted(avg_data['num_procs'].unique())
    ax.axhline(y=100, color='k', linestyle='--', linewidth=2, 
               label='Ideal (100%)', alpha=0.7)
    
    # Plot efficiency thresholds
    ax.axhline(y=70, color='green', linestyle=':', linewidth=1.5, 
               label='Good (70%)', alpha=0.5)
    ax.axhline(y=50, color='orange', linestyle=':', linewidth=1.5, 
               label='Acceptable (50%)', alpha=0.5)
    
    for i, config in enumerate(configs):
        config_data = avg_data[avg_data['config_name'] == config].sort_values('num_procs')
        
        ax.plot(config_data['num_procs'], config_data['efficiency_pct'],
                marker=markers[i % len(markers)], color=colors[i], 
                label=config, linewidth=2, markersize=7)
    
    ax.set_xlabel('Number of Processes', fontweight='bold')
    ax.set_ylabel('Parallel Efficiency (%)', fontweight='bold')
    ax.set_title('Parallel Efficiency vs Process Count\n(Average across all matrices)', 
                 fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.set_ylim(bottom=0, top=min(110, avg_data['efficiency_pct'].max() * 1.1))
    ax.grid(True, which='both', alpha=0.3, linestyle='--')
    ax.legend(loc='best', framealpha=0.9, ncol=2 if len(configs) > 4 else 1)
    
    plt.tight_layout()
    output_path = output_dir / 'efficiency_vs_procs.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_combined(avg_data, output_dir):
    """Create a combined figure with all three plots."""
    fig = plt.figure(figsize=(15, 5))
    
    configs = sorted(avg_data['config_name'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(configs)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    proc_counts = sorted(avg_data['num_procs'].unique())
    min_procs = min(proc_counts)
    
    # Plot 1: Strong Scaling
    ax1 = plt.subplot(1, 3, 1)
    for i, config in enumerate(configs):
        config_data = avg_data[avg_data['config_name'] == config].sort_values('num_procs')
        ax1.plot(config_data['num_procs'], config_data['avg_time_ms'],
                marker=markers[i % len(markers)], color=colors[i], 
                label=config, linewidth=2, markersize=6)
    
    ax1.set_xlabel('Number of Processes', fontweight='bold')
    ax1.set_ylabel('Avg Time (ms)', fontweight='bold')
    ax1.set_title('Strong Scaling', fontweight='bold')
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.grid(True, which='both', alpha=0.3, linestyle='--')
    ax1.legend(loc='best', fontsize=8)
    
    # Plot 2: Speedup
    ax2 = plt.subplot(1, 3, 2)
    ideal_speedup = [p / min_procs for p in proc_counts]
    ax2.plot(proc_counts, ideal_speedup, 'k--', linewidth=2, 
            label='Ideal', alpha=0.7)
    
    for i, config in enumerate(configs):
        config_data = avg_data[avg_data['config_name'] == config].sort_values('num_procs')
        ax2.plot(config_data['num_procs'], config_data['speedup'],
                marker=markers[i % len(markers)], color=colors[i], 
                label=config, linewidth=2, markersize=6)
    
    ax2.set_xlabel('Number of Processes', fontweight='bold')
    ax2.set_ylabel('Speedup', fontweight='bold')
    ax2.set_title('Speedup', fontweight='bold')
    ax2.set_xscale('log', base=2)
    ax2.grid(True, which='both', alpha=0.3, linestyle='--')
    ax2.legend(loc='best', fontsize=8)
    
    # Plot 3: Efficiency
    ax3 = plt.subplot(1, 3, 3)
    ax3.axhline(y=100, color='k', linestyle='--', linewidth=1.5, 
               label='Ideal', alpha=0.7)
    ax3.axhline(y=70, color='green', linestyle=':', linewidth=1, alpha=0.4)
    ax3.axhline(y=50, color='orange', linestyle=':', linewidth=1, alpha=0.4)
    
    for i, config in enumerate(configs):
        config_data = avg_data[avg_data['config_name'] == config].sort_values('num_procs')
        ax3.plot(config_data['num_procs'], config_data['efficiency_pct'],
                marker=markers[i % len(markers)], color=colors[i], 
                label=config, linewidth=2, markersize=6)
    
    ax3.set_xlabel('Number of Processes', fontweight='bold')
    ax3.set_ylabel('Efficiency (%)', fontweight='bold')
    ax3.set_title('Parallel Efficiency', fontweight='bold')
    ax3.set_xscale('log', base=2)
    ax3.set_ylim(bottom=0, top=min(110, avg_data['efficiency_pct'].max() * 1.1))
    ax3.grid(True, which='both', alpha=0.3, linestyle='--')
    ax3.legend(loc='best', fontsize=8)
    
    plt.suptitle('MPI Configuration Performance Analysis', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = output_dir / 'configurations_combined.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_config_comparison(avg_data, output_dir):
    """Create bar chart comparing configurations at key process counts."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    configs = sorted(avg_data['config_name'].unique())
    
    # Select key process counts for comparison
    all_procs = sorted(avg_data['num_procs'].unique())
    if len(all_procs) > 6:
        # Select evenly spaced process counts
        step = len(all_procs) // 6
        key_procs = [all_procs[i] for i in range(0, len(all_procs), step)][:6]
    else:
        key_procs = all_procs
    
    # Plot 1: Speedup comparison
    ax1 = axes[0]
    x = np.arange(len(key_procs))
    width = 0.8 / len(configs)
    
    for i, config in enumerate(configs):
        speedups = []
        for procs in key_procs:
            config_proc = avg_data[(avg_data['config_name'] == config) & 
                                   (avg_data['num_procs'] == procs)]
            if len(config_proc) > 0:
                speedups.append(config_proc['speedup'].values[0])
            else:
                speedups.append(0)
        
        ax1.bar(x + i * width, speedups, width, label=config, alpha=0.8)
    
    ax1.set_xlabel('Number of Processes', fontweight='bold')
    ax1.set_ylabel('Speedup', fontweight='bold')
    ax1.set_title('Speedup Comparison at Key Process Counts', fontweight='bold')
    ax1.set_xticks(x + width * (len(configs) - 1) / 2)
    ax1.set_xticklabels([str(p) for p in key_procs])
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Plot 2: Efficiency comparison
    ax2 = axes[1]
    
    for i, config in enumerate(configs):
        efficiencies = []
        for procs in key_procs:
            config_proc = avg_data[(avg_data['config_name'] == config) & 
                                   (avg_data['num_procs'] == procs)]
            if len(config_proc) > 0:
                efficiencies.append(config_proc['efficiency_pct'].values[0])
            else:
                efficiencies.append(0)
        
        ax2.bar(x + i * width, efficiencies, width, label=config, alpha=0.8)
    
    ax2.axhline(y=100, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axhline(y=70, color='green', linestyle=':', linewidth=1, alpha=0.3)
    ax2.axhline(y=50, color='orange', linestyle=':', linewidth=1, alpha=0.3)
    
    ax2.set_xlabel('Number of Processes', fontweight='bold')
    ax2.set_ylabel('Efficiency (%)', fontweight='bold')
    ax2.set_title('Efficiency Comparison at Key Process Counts', fontweight='bold')
    ax2.set_xticks(x + width * (len(configs) - 1) / 2)
    ax2.set_xticklabels([str(p) for p in key_procs])
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'config_comparison_bars.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def generate_summary_stats(avg_data):
    """Generate and print summary statistics."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80 + "\n")
    
    configs = sorted(avg_data['config_name'].unique())
    
    for config in configs:
        config_data = avg_data[avg_data['config_name'] == config]
        
        print(f"**{config}:**")
        print(f"  Average speedup: {config_data['speedup'].mean():.3f}×")
        print(f"  Max speedup: {config_data['speedup'].max():.3f}× at {config_data.loc[config_data['speedup'].idxmax(), 'num_procs']:.0f} processes")
        print(f"  Average efficiency: {config_data['efficiency_pct'].mean():.2f}%")
        print(f"  Best efficiency: {config_data['efficiency_pct'].max():.2f}% at {config_data.loc[config_data['efficiency_pct'].idxmax(), 'num_procs']:.0f} processes")
        print()

def main():
    """Main plotting function."""
    # Determine paths
    script_dir = Path(__file__).parent
    csv_path = script_dir / "../../results/configurations_results.csv"
    output_dir = script_dir / "../../plots"
    
    if not csv_path.exists():
        csv_path = Path("results/configurations_results.csv")
        output_dir = Path("plots")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("CONFIGURATIONS PLOTTING SCRIPT")
    print("="*80)
    print(f"\nInput CSV: {csv_path}")
    print(f"Output directory: {output_dir}")
    
    # Load data
    df = load_data(csv_path)
    print(f"Loaded {len(df)} data points")
    print(f"Configurations: {', '.join(df['config_name'].unique())}")
    print(f"Process counts: {sorted(df['num_procs'].unique())}")
    print(f"Matrices: {len(df['matrix'].unique())}")
    
    # Calculate averages
    print("\nCalculating averages across matrices...")
    avg_data = calculate_averages(df)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_strong_scaling(avg_data, output_dir)
    plot_speedup(avg_data, output_dir)
    plot_efficiency(avg_data, output_dir)
    plot_combined(avg_data, output_dir)
    plot_config_comparison(avg_data, output_dir)
    
    # Generate summary statistics
    generate_summary_stats(avg_data)
    
    print("\n" + "="*80)
    print("PLOTTING COMPLETE")
    print("="*80)
    print(f"\nAll plots saved to: {output_dir}/")
    print("Generated files:")
    print("  - strong_scaling.png")
    print("  - speedup_vs_procs.png")
    print("  - efficiency_vs_procs.png")
    print("  - configurations_combined.png")
    print("  - config_comparison_bars.png")
    print()

if __name__ == "__main__":
    main()
