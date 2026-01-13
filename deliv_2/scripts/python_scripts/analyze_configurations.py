#!/usr/bin/env python3
"""
Analyze configurations CSV results to compare MPI execution modes and process counts.
Generates comprehensive performance analysis including speedup, efficiency, and bottleneck identification.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def load_data(csv_path):
    """Load and validate configurations results CSV."""
    if not Path(csv_path).exists():
        print(f"ERROR: CSV file not found: {csv_path}")
        sys.exit(1)
    
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    required = ['num_procs', 'config_name', 'avg_time_ms', 'comm_time_ms', 
                'compute_time_ms', 'speedup', 'efficiency_pct']
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        sys.exit(1)
    
    return df

def get_baseline_times(df):
    """Extract baseline (single process) times for each configuration."""
    baseline = df[df['num_procs'] == df['num_procs'].min()].copy()
    baselines = {}
    for config in baseline['config_name'].unique():
        config_data = baseline[baseline['config_name'] == config]
        baselines[config] = config_data['avg_time_ms'].mean()
    return baselines

def calculate_metrics(df):
    """Calculate speedup and efficiency metrics for each configuration."""
    configs = df['config_name'].unique()
    proc_counts = sorted(df['num_procs'].unique())
    matrices = df['matrix'].unique()
    
    metrics = {}
    for config in configs:
        config_df = df[df['config_name'] == config]
        
        config_metrics = []
        for procs in proc_counts:
            proc_data = config_df[config_df['num_procs'] == procs]
            if len(proc_data) == 0:
                continue
            
            # Calculate metrics for each matrix, then average
            matrix_speedups = []
            matrix_times = []
            matrix_comm = []
            matrix_comp = []
            
            for matrix in matrices:
                matrix_data = config_df[config_df['matrix'] == matrix]
                if len(matrix_data) == 0:
                    continue
                
                # Get baseline for this matrix
                baseline_data = matrix_data[matrix_data['num_procs'] == proc_counts[0]]
                if len(baseline_data) == 0:
                    continue
                baseline_time = baseline_data['avg_time_ms'].mean()
                
                # Get current process count data
                proc_matrix_data = matrix_data[matrix_data['num_procs'] == procs]
                if len(proc_matrix_data) == 0:
                    continue
                
                current_time = proc_matrix_data['avg_time_ms'].mean()
                speedup = baseline_time / current_time if current_time > 0 else 0
                
                matrix_speedups.append(speedup)
                matrix_times.append(current_time)
                matrix_comm.append(proc_matrix_data['comm_time_ms'].mean())
                matrix_comp.append(proc_matrix_data['compute_time_ms'].mean())
            
            # Average across all matrices
            if matrix_speedups:
                avg_speedup = np.mean(matrix_speedups)
                avg_time = np.mean(matrix_times)
                comm_time = np.mean(matrix_comm)
                comp_time = np.mean(matrix_comp)
                
                efficiency = (avg_speedup / procs) * 100
                comm_pct = (comm_time / avg_time) * 100 if avg_time > 0 else 0
                comp_pct = (comp_time / avg_time) * 100 if avg_time > 0 else 0
                
                config_metrics.append({
                    'procs': procs,
                    'avg_time': avg_time,
                    'comm_time': comm_time,
                    'comp_time': comp_time,
                    'speedup': avg_speedup,
                    'efficiency': efficiency,
                    'comm_pct': comm_pct,
                    'comp_pct': comp_pct
                })
        
        metrics[config] = config_metrics
    
    return metrics

def print_mpi_directives_comparison(metrics, baselines):
    """Print detailed comparison of MPI directives."""
    print("\n" + "="*80)
    print("MPI DIRECTIVES COMPARISON")
    print("="*80 + "\n")
    
    for config, config_metrics in metrics.items():
        if len(config_metrics) < 2:
            continue
        
        baseline = baselines.get(config, config_metrics[0]['avg_time'])
        speedups = [m['speedup'] for m in config_metrics[1:]]
        avg_speedup = np.mean(speedups) if speedups else 0
        max_speedup = max(speedups) if speedups else 0
        max_speedup_idx = speedups.index(max_speedup) if speedups else 0
        max_speedup_procs = config_metrics[max_speedup_idx + 1]['procs'] if speedups else 0
        
        # Determine best characteristics
        best_efficiency_idx = max(range(len(config_metrics)), 
                                   key=lambda i: config_metrics[i]['efficiency'])
        best_eff_procs = config_metrics[best_efficiency_idx]['procs']
        best_eff_val = config_metrics[best_efficiency_idx]['efficiency']
        
        print(f"**{config}**\n")
        print(f"- **Baseline (min processes)**: {baseline:.2f} ms")
        print(f"- **Average speedup (all scales)**: {avg_speedup:.2f}×")
        print(f"- **Peak speedup**: {max_speedup:.2f}× at {max_speedup_procs} processes")
        print(f"- **Peak efficiency**: {best_eff_val:.1f}% at {best_eff_procs} processes")
        
        # Identify characteristics
        all_speedups = [m['speedup'] for c_metrics in metrics.values() 
                       for m in c_metrics[1:] if len(c_metrics) > 1]
        overall_avg = np.mean(all_speedups) if all_speedups else 0
        
        if best_eff_val > 200:
            print(f"- **Best use case**: Super-linear scaling with cache optimization")
        elif avg_speedup > overall_avg:
            print(f"- **Best use case**: Consistently strong scaling ✅")
        else:
            print(f"- **Best use case**: Stable baseline performance")
        
        print()

def print_speedup_table(metrics):
    """Print speedup comparison table across process counts."""
    print("\n" + "="*80)
    print("SPEEDUP COMPARISON")
    print("="*80 + "\n")
    
    configs = list(metrics.keys())
    if not configs:
        return
    
    proc_counts = sorted(set(m['procs'] for config_metrics in metrics.values() 
                             for m in config_metrics))
    
    # Table header
    header = f"| Processes | {' | '.join(configs)} | Winner |"
    separator = "|" + "|".join(["-" * (len(col) + 2) for col in header.split('|')[1:-1]]) + "|"
    
    print(header)
    print(separator)
    
    for procs in proc_counts:
        row_data = [procs]
        speedups = {}
        
        for config in configs:
            config_metrics = metrics[config]
            proc_metric = next((m for m in config_metrics if m['procs'] == procs), None)
            
            if proc_metric:
                speedups[config] = proc_metric['speedup']
                row_data.append(f"{proc_metric['speedup']:.2f}×")
            else:
                speedups[config] = 0
                row_data.append("N/A")
        
        # Determine winner
        if speedups:
            max_speedup = max(speedups.values())
            winners = [config for config, spd in speedups.items() if spd == max_speedup]
            
            if len(winners) == len(configs):
                winner = "Equivalent"
            else:
                winner = winners[0].split('/')[0] if winners else "N/A"
                if max_speedup > 10:
                    winner += " ⭐"
        else:
            winner = "N/A"
        
        row_data.append(winner)
        
        # Format row
        row = f"| {row_data[0]} | " + " | ".join(str(x) for x in row_data[1:-1]) + f" | {row_data[-1]} |"
        print(row)

def print_comm_vs_comp_table(metrics):
    """Print communication vs computation breakdown."""
    print("\n" + "="*80)
    print("COMMUNICATION VS COMPUTATION")
    print("="*80 + "\n")
    
    proc_counts = sorted(set(m['procs'] for config_metrics in metrics.values() 
                             for m in config_metrics))
    
    print("| Processes | Comm % (range) | Comp % (range) | Avg Comm Time | Avg Comp Time |")
    print("|-----------|----------------|----------------|---------------|---------------|")
    
    for procs in proc_counts:
        comm_pcts = []
        comp_pcts = []
        comm_times = []
        comp_times = []
        
        for config_metrics in metrics.values():
            proc_metric = next((m for m in config_metrics if m['procs'] == procs), None)
            if proc_metric:
                comm_pcts.append(proc_metric['comm_pct'])
                comp_pcts.append(proc_metric['comp_pct'])
                comm_times.append(proc_metric['comm_time'])
                comp_times.append(proc_metric['comp_time'])
        
        if comm_pcts:
            comm_range = f"{min(comm_pcts):.1f}-{max(comm_pcts):.1f}%"
            comp_range = f"{min(comp_pcts):.1f}-{max(comp_pcts):.1f}%"
            avg_comm = f"{np.mean(comm_times):.2f} ms"
            avg_comp = f"{np.mean(comp_times):.2f} ms"
            
            # Highlight bottlenecks
            proc_label = f"**{procs}**" if max(comm_pcts) > 60 or procs == proc_counts[0] else str(procs)
            
            print(f"| {proc_label} | {comm_range} | {comp_range} | {avg_comm} | {avg_comp} |")

def print_observations(metrics, proc_counts):
    """Print key observations and recommendations."""
    print("\n" + "="*80)
    print("KEY OBSERVATIONS")
    print("="*80 + "\n")
    
    # 1. Single-process optimization
    print("**1. Single-Process Baseline**\n")
    single_proc_metrics = [m[0] for m in metrics.values() if m]
    if single_proc_metrics:
        avg_comm_pct = np.mean([m['comm_pct'] for m in single_proc_metrics])
        avg_comp_pct = np.mean([m['comp_pct'] for m in single_proc_metrics])
        print(f"- {avg_comp_pct:.1f}% computation time across all strategies")
        print(f"- Minimal MPI overhead ({avg_comm_pct:.2f}%) at baseline")
        print()
    
    # 2. Identify bottlenecks
    print("**2. Communication Bottlenecks**\n")
    high_comm_procs = []
    for procs in proc_counts:
        comm_pcts = []
        for config_metrics in metrics.values():
            proc_metric = next((m for m in config_metrics if m['procs'] == procs), None)
            if proc_metric:
                comm_pcts.append(proc_metric['comm_pct'])
        
        if comm_pcts and max(comm_pcts) > 60:
            high_comm_procs.append((procs, max(comm_pcts)))
    
    if high_comm_procs:
        for procs, comm_pct in high_comm_procs:
            is_power_of_2 = (procs & (procs - 1)) == 0
            if not is_power_of_2:
                print(f"- **{procs} processes**: {comm_pct:.1f}% communication (non-power-of-2 penalty)")
            else:
                print(f"- **{procs} processes**: {comm_pct:.1f}% communication")
        print()
    
    # 3. Super-linear speedup
    print("**3. Super-Linear Speedup Analysis**\n")
    super_linear = []
    for config, config_metrics in metrics.items():
        for m in config_metrics:
            if m['efficiency'] > 150:
                super_linear.append((config, m['procs'], m['efficiency'], m['speedup']))
    
    if super_linear:
        print("- Cache working set reduction improves performance beyond linear scaling")
        for config, procs, eff, speedup in super_linear[:3]:  # Top 3
            print(f"  - {config} @ {procs} processes: {speedup:.2f}× speedup ({eff:.0f}% efficiency)")
        print()
    
    # 4. Best configuration
    print("**4. Optimal Configuration**\n")
    best_config = None
    best_speedup = 0
    
    for config, config_metrics in metrics.items():
        for m in config_metrics:
            if m['speedup'] > best_speedup:
                best_speedup = m['speedup']
                best_config = (config, m['procs'], m['speedup'], m['efficiency'])
    
    if best_config:
        config, procs, speedup, eff = best_config
        print(f"- **{procs} processes with {config}**")
        print(f"  - Speedup: {speedup:.2f}×")
        print(f"  - Efficiency: {eff:.1f}%")
        print(f"  - Fastest execution with best resource utilization ⭐")
        print()

def main():
    """Main analysis function."""
    # Determine CSV path
    script_dir = Path(__file__).parent
    csv_path = script_dir / "../../results/configurations_results.csv"
    
    if not csv_path.exists():
        print(f"Trying alternative path...")
        csv_path = Path("results/configurations_results.csv")
    
    print("="*80)
    print("CONFIGURATIONS PERFORMANCE ANALYSIS")
    print("="*80)
    print(f"\nAnalyzing: {csv_path}")
    
    # Load and process data
    df = load_data(csv_path)
    print(f"Loaded {len(df)} data points")
    print(f"Configurations: {', '.join(df['config_name'].unique())}")
    print(f"Process counts: {sorted(df['num_procs'].unique())}")
    
    # Calculate metrics
    baselines = get_baseline_times(df)
    metrics = calculate_metrics(df)
    proc_counts = sorted(set(m['procs'] for config_metrics in metrics.values() 
                             for m in config_metrics))
    
    # Generate analysis sections
    print_mpi_directives_comparison(metrics, baselines)
    print_speedup_table(metrics)
    print_comm_vs_comp_table(metrics)
    print_observations(metrics, proc_counts)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
