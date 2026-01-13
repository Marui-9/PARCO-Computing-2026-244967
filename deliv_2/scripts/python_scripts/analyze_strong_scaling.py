#!/usr/bin/env python3
"""
Analyze strong scaling behavior from configurations results.
Strong scaling: fixed problem size, increasing number of processes.
Examines how well the application scales as resources increase.
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
    required = ['num_procs', 'matrix', 'config_name', 'avg_time_ms', 
                'speedup', 'efficiency_pct']
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        sys.exit(1)
    
    return df

def calculate_strong_scaling_metrics(df):
    """Calculate strong scaling metrics for each matrix and configuration."""
    matrices = df['matrix'].unique()
    configs = df['config_name'].unique()
    proc_counts = sorted(df['num_procs'].unique())
    
    scaling_data = []
    
    for matrix in matrices:
        for config in configs:
            subset = df[(df['matrix'] == matrix) & (df['config_name'] == config)]
            subset = subset.sort_values('num_procs')
            
            if len(subset) < 2:
                continue
            
            # Use smallest process count as baseline
            baseline = subset.iloc[0]
            baseline_time = baseline['avg_time_ms']
            baseline_procs = baseline['num_procs']
            
            for _, row in subset.iterrows():
                procs = row['num_procs']
                time = row['avg_time_ms']
                
                # Calculate speedup relative to baseline
                speedup = baseline_time / time
                # Ideal speedup if perfectly linear from baseline
                ideal_speedup = procs / baseline_procs
                # Efficiency
                efficiency = (speedup / ideal_speedup) * 100
                # Parallel efficiency (vs single process equivalent)
                parallel_eff = row['efficiency_pct']
                
                scaling_data.append({
                    'matrix': matrix,
                    'config': config,
                    'procs': procs,
                    'time_ms': time,
                    'speedup': speedup,
                    'ideal_speedup': ideal_speedup,
                    'efficiency': efficiency,
                    'parallel_efficiency': parallel_eff,
                    'scaling_factor': speedup / ideal_speedup
                })
    
    return pd.DataFrame(scaling_data)

def analyze_overall_scaling(scaling_df):
    """Analyze overall strong scaling behavior."""
    print("\n" + "="*80)
    print("OVERALL STRONG SCALING ANALYSIS")
    print("="*80 + "\n")
    
    proc_counts = sorted(scaling_df['procs'].unique())
    configs = sorted(scaling_df['config'].unique())
    
    print(f"{'Processes':<12} {'Config':<20} {'Avg Speedup':<15} {'Efficiency':<15} {'Scaling Quality'}")
    print("-" * 80)
    
    for procs in proc_counts:
        for config in configs:
            subset = scaling_df[(scaling_df['procs'] == procs) & (scaling_df['config'] == config)]
            
            if len(subset) > 0:
                avg_speedup = subset['speedup'].mean()
                avg_efficiency = subset['efficiency'].mean()
                scaling_factor = subset['scaling_factor'].mean()
                
                # Classify scaling quality
                if scaling_factor > 0.9:
                    quality = "Excellent ✅"
                elif scaling_factor > 0.7:
                    quality = "Good"
                elif scaling_factor > 0.5:
                    quality = "Fair ⚠️"
                else:
                    quality = "Poor ❌"
                
                print(f"{procs:<12} {config:<20} {avg_speedup:>13.2f}×  {avg_efficiency:>13.1f}%  {quality}")
        print()

def analyze_by_matrix(scaling_df, df_original):
    """Analyze scaling behavior for individual matrices."""
    print("\n" + "="*80)
    print("STRONG SCALING BY MATRIX")
    print("="*80 + "\n")
    
    matrices = scaling_df['matrix'].unique()
    
    # Get matrix characteristics
    matrix_info = df_original.groupby('matrix').agg({
        'rows': 'first',
        'nnz': 'first',
        'density_pct': 'first'
    }).reset_index()
    
    # Calculate average scaling quality for each matrix
    matrix_scaling = []
    for matrix in matrices:
        subset = scaling_df[scaling_df['matrix'] == matrix]
        avg_scaling = subset['scaling_factor'].mean()
        max_speedup = subset['speedup'].max()
        max_procs = subset.loc[subset['speedup'].idxmax(), 'procs']
        
        info = matrix_info[matrix_info['matrix'] == matrix].iloc[0]
        
        matrix_scaling.append({
            'matrix': matrix,
            'rows': info['rows'],
            'nnz': info['nnz'],
            'density': info['density_pct'],
            'avg_scaling': avg_scaling,
            'max_speedup': max_speedup,
            'max_speedup_procs': max_procs
        })
    
    matrix_df = pd.DataFrame(matrix_scaling).sort_values('avg_scaling', ascending=False)
    
    print("**Best Scaling Matrices:**\n")
    print(f"{'Matrix':<25} {'Rows':<10} {'Density':<10} {'Avg Scale':<12} {'Max Speedup':<15}")
    print("-" * 80)
    
    for _, row in matrix_df.head(5).iterrows():
        print(f"{row['matrix']:<25} {row['rows']:<10} {row['density']:>8.4f}%  {row['avg_scaling']:>10.2f}  "
              f"{row['max_speedup']:>6.2f}× @ {int(row['max_speedup_procs'])}p")
    
    print("\n**Worst Scaling Matrices:**\n")
    print(f"{'Matrix':<25} {'Rows':<10} {'Density':<10} {'Avg Scale':<12} {'Max Speedup':<15}")
    print("-" * 80)
    
    for _, row in matrix_df.tail(5).iterrows():
        print(f"{row['matrix']:<25} {row['rows']:<10} {row['density']:>8.4f}%  {row['avg_scaling']:>10.2f}  "
              f"{row['max_speedup']:>6.2f}× @ {int(row['max_speedup_procs'])}p")

def identify_scaling_limits(scaling_df):
    """Identify where strong scaling breaks down."""
    print("\n" + "="*80)
    print("SCALING LIMITS AND BOTTLENECKS")
    print("="*80 + "\n")
    
    configs = sorted(scaling_df['config'].unique())
    
    for config in configs:
        config_data = scaling_df[scaling_df['config'] == config].sort_values('procs')
        
        print(f"**{config}:**\n")
        
        proc_groups = config_data.groupby('procs').agg({
            'efficiency': 'mean',
            'scaling_factor': 'mean'
        }).reset_index()
        
        # Find where efficiency drops below thresholds
        good_threshold = proc_groups[proc_groups['efficiency'] >= 70]
        fair_threshold = proc_groups[proc_groups['efficiency'] >= 50]
        
        if len(good_threshold) > 0:
            max_good_procs = good_threshold['procs'].max()
            print(f"  Strong scaling (≥70% efficiency) up to: {int(max_good_procs)} processes")
        else:
            print(f"  Strong scaling (≥70% efficiency): Never achieved")
        
        if len(fair_threshold) > 0:
            max_fair_procs = fair_threshold['procs'].max()
            print(f"  Acceptable scaling (≥50% efficiency) up to: {int(max_fair_procs)} processes")
        else:
            print(f"  Acceptable scaling (≥50% efficiency): Never achieved")
        
        # Identify steep drop-offs
        for i in range(1, len(proc_groups)):
            prev_eff = proc_groups.iloc[i-1]['efficiency']
            curr_eff = proc_groups.iloc[i]['efficiency']
            drop = prev_eff - curr_eff
            
            if drop > 20:
                prev_procs = int(proc_groups.iloc[i-1]['procs'])
                curr_procs = int(proc_groups.iloc[i]['procs'])
                print(f"  ⚠️  Sharp efficiency drop: {prev_procs}→{curr_procs} processes ({drop:.1f}% drop)")
        
        print()

def compare_config_scaling(scaling_df):
    """Compare scaling behavior across different MPI configurations."""
    print("\n" + "="*80)
    print("CONFIGURATION SCALING COMPARISON")
    print("="*80 + "\n")
    
    proc_counts = sorted(scaling_df['procs'].unique())
    configs = sorted(scaling_df['config'].unique())
    
    print(f"{'Processes':<12} ", end='')
    for config in configs:
        print(f"{config[:18]:<20} ", end='')
    print("Winner")
    print("-" * (12 + 20 * len(configs) + 15))
    
    for procs in proc_counts:
        print(f"{procs:<12} ", end='')
        
        config_effs = {}
        for config in configs:
            subset = scaling_df[(scaling_df['procs'] == procs) & (scaling_df['config'] == config)]
            if len(subset) > 0:
                avg_eff = subset['efficiency'].mean()
                config_effs[config] = avg_eff
                print(f"{avg_eff:>8.1f}%          ", end='')
            else:
                print(f"{'N/A':>8}          ", end='')
        
        if config_effs:
            winner = max(config_effs, key=config_effs.get)
            print(f"{winner[:15]}")
        else:
            print()

def analyze_super_linear_scaling(scaling_df):
    """Identify cases of super-linear speedup."""
    print("\n" + "="*80)
    print("SUPER-LINEAR SCALING ANALYSIS")
    print("="*80 + "\n")
    
    super_linear = scaling_df[scaling_df['scaling_factor'] > 1.05]
    
    if len(super_linear) == 0:
        print("No significant super-linear scaling observed (>5% above ideal).\n")
        return
    
    print("**Cases of Super-Linear Speedup (>5% above ideal):**\n")
    print(f"{'Matrix':<25} {'Config':<20} {'Processes':<12} {'Speedup':<12} {'vs Ideal'}")
    print("-" * 80)
    
    super_linear_sorted = super_linear.sort_values('scaling_factor', ascending=False)
    
    for _, row in super_linear_sorted.head(10).iterrows():
        print(f"{row['matrix']:<25} {row['config']:<20} {int(row['procs']):<12} "
              f"{row['speedup']:>10.2f}×  {row['scaling_factor']:>8.2f}× ideal")
    
    if len(super_linear) > 10:
        print(f"\n... and {len(super_linear) - 10} more cases")
    
    print("\n**Likely Causes:**")
    print("  - Cache effects: Reduced working set per process fits in cache")
    print("  - NUMA effects: Better memory locality with distributed data")
    print("  - Memory bandwidth: Reduced contention with more processes")

def identify_optimal_process_counts(scaling_df):
    """Identify optimal process counts for different objectives."""
    print("\n" + "="*80)
    print("OPTIMAL PROCESS COUNT RECOMMENDATIONS")
    print("="*80 + "\n")
    
    configs = sorted(scaling_df['config'].unique())
    
    for config in configs:
        config_data = scaling_df[scaling_df['config'] == config]
        
        print(f"**{config}:**\n")
        
        proc_groups = config_data.groupby('procs').agg({
            'speedup': 'mean',
            'efficiency': 'mean',
            'time_ms': 'mean',
            'scaling_factor': 'mean'
        }).reset_index()
        
        # Best speedup
        best_speedup_row = proc_groups.loc[proc_groups['speedup'].idxmax()]
        print(f"  Maximum speedup: {int(best_speedup_row['procs'])} processes "
              f"({best_speedup_row['speedup']:.2f}×)")
        
        # Best efficiency (among processes >= 8)
        high_proc = proc_groups[proc_groups['procs'] >= 8]
        if len(high_proc) > 0:
            best_eff_row = high_proc.loc[high_proc['efficiency'].idxmax()]
            print(f"  Best efficiency (≥8p): {int(best_eff_row['procs'])} processes "
                  f"({best_eff_row['efficiency']:.1f}%)")
        
        # Sweet spot: good speedup with acceptable efficiency
        sweet_spot = proc_groups[
            (proc_groups['efficiency'] >= 60) & 
            (proc_groups['speedup'] >= proc_groups['speedup'].quantile(0.6))
        ]
        if len(sweet_spot) > 0:
            # Pick highest speedup among sweet spot candidates
            sweet_row = sweet_spot.loc[sweet_spot['speedup'].idxmax()]
            print(f"  Sweet spot (≥60% eff): {int(sweet_row['procs'])} processes "
                  f"({sweet_row['speedup']:.2f}× speedup, {sweet_row['efficiency']:.1f}% efficiency)")
        
        print()

def generate_recommendations(scaling_df, df_original):
    """Generate actionable recommendations based on scaling analysis."""
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80 + "\n")
    
    proc_counts = sorted(scaling_df['procs'].unique())
    configs = scaling_df['config'].unique()
    
    # Overall scaling quality
    avg_scaling = scaling_df.groupby('procs')['scaling_factor'].mean()
    
    print("**1. Strong Scaling Quality Assessment:**\n")
    
    excellent_procs = [p for p in proc_counts if avg_scaling.get(p, 0) > 0.9]
    good_procs = [p for p in proc_counts if 0.7 < avg_scaling.get(p, 0) <= 0.9]
    fair_procs = [p for p in proc_counts if 0.5 < avg_scaling.get(p, 0) <= 0.7]
    poor_procs = [p for p in proc_counts if avg_scaling.get(p, 0) <= 0.5]
    
    if excellent_procs:
        print(f"  ✅ Excellent scaling (>90% ideal): {excellent_procs}")
    if good_procs:
        print(f"  ✓  Good scaling (70-90% ideal): {good_procs}")
    if fair_procs:
        print(f"  ⚠️  Fair scaling (50-70% ideal): {fair_procs}")
    if poor_procs:
        print(f"  ❌ Poor scaling (<50% ideal): {poor_procs}")
    
    # Best configuration for scaling
    print("\n**2. Best Configuration for Strong Scaling:**\n")
    
    config_avg_scaling = scaling_df.groupby('config')['scaling_factor'].mean().sort_values(ascending=False)
    
    for i, (config, score) in enumerate(config_avg_scaling.items(), 1):
        rank = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"  {i}."
        print(f"  {rank} {config}: {score:.3f} avg scaling factor")
    
    # Problem size considerations
    print("\n**3. Problem Size Considerations:**\n")
    
    # Analyze by matrix size
    matrices_with_size = df_original.groupby('matrix').agg({
        'rows': 'first',
        'nnz': 'first'
    }).reset_index()
    
    scaling_by_size = []
    for _, matrix_info in matrices_with_size.iterrows():
        matrix = matrix_info['matrix']
        matrix_scaling = scaling_df[scaling_df['matrix'] == matrix]['scaling_factor'].mean()
        scaling_by_size.append({
            'rows': matrix_info['rows'],
            'nnz': matrix_info['nnz'],
            'scaling': matrix_scaling
        })
    
    size_df = pd.DataFrame(scaling_by_size)
    
    if len(size_df) > 0:
        # Correlation between size and scaling
        small_matrices = size_df[size_df['rows'] < 500000]
        large_matrices = size_df[size_df['rows'] >= 500000]
        
        if len(small_matrices) > 0 and len(large_matrices) > 0:
            small_avg = small_matrices['scaling'].mean()
            large_avg = large_matrices['scaling'].mean()
            
            print(f"  Small matrices (<500k rows): {small_avg:.3f} avg scaling")
            print(f"  Large matrices (≥500k rows): {large_avg:.3f} avg scaling")
            
            if large_avg > small_avg * 1.1:
                print("  → Larger matrices show better strong scaling ✅")
            elif small_avg > large_avg * 1.1:
                print("  → Smaller matrices show better strong scaling")
                print("  → Consider larger problems for better parallel efficiency")
    
    # Practical recommendations
    print("\n**4. Practical Deployment Recommendations:**\n")
    
    # Find best efficiency process counts
    high_eff = scaling_df[scaling_df['efficiency'] >= 70]
    if len(high_eff) > 0:
        best_procs = high_eff.groupby('procs')['efficiency'].mean().idxmax()
        best_eff = high_eff.groupby('procs')['efficiency'].mean().max()
        print(f"  Production deployment: Use {int(best_procs)} processes")
        print(f"    → {best_eff:.1f}% average efficiency")
        print(f"    → Good balance of performance and resource utilization")
    
    # Maximum throughput
    max_speedup_data = scaling_df.loc[scaling_df.groupby('config')['speedup'].idxmax()]
    best_throughput = max_speedup_data.loc[max_speedup_data['speedup'].idxmax()]
    print(f"\n  Maximum throughput: Use {int(best_throughput['procs'])} processes with {best_throughput['config']}")
    print(f"    → {best_throughput['speedup']:.2f}× speedup")
    print(f"    → {best_throughput['efficiency']:.1f}% efficiency")
    
    # Identify scaling wall
    poor_scaling = scaling_df[scaling_df['efficiency'] < 30]
    if len(poor_scaling) > 0:
        min_poor = poor_scaling['procs'].min()
        print(f"\n  ⚠️  Scaling wall: Beyond {int(min_poor)} processes, efficiency drops significantly")
        print(f"    → Avoid using more processes unless throughput is critical")

def main():
    """Main analysis function."""
    # Determine CSV path
    script_dir = Path(__file__).parent
    csv_path = script_dir / "../../results/configurations_results.csv"
    
    if not csv_path.exists():
        csv_path = Path("results/configurations_results.csv")
    
    print("="*80)
    print("STRONG SCALING ANALYSIS")
    print("="*80)
    print(f"\nAnalyzing: {csv_path}")
    
    # Load data
    df = load_data(csv_path)
    print(f"Loaded {len(df)} data points")
    print(f"Configurations: {', '.join(df['config_name'].unique())}")
    print(f"Process counts: {sorted(df['num_procs'].unique())}")
    print(f"Matrices: {len(df['matrix'].unique())} unique")
    
    # Calculate strong scaling metrics
    print("\nCalculating strong scaling metrics...")
    scaling_df = calculate_strong_scaling_metrics(df)
    
    if len(scaling_df) == 0:
        print("ERROR: No scaling data could be calculated")
        sys.exit(1)
    
    # Run analysis sections
    analyze_overall_scaling(scaling_df)
    analyze_by_matrix(scaling_df, df)
    compare_config_scaling(scaling_df)
    identify_scaling_limits(scaling_df)
    analyze_super_linear_scaling(scaling_df)
    identify_optimal_process_counts(scaling_df)
    generate_recommendations(scaling_df, df)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
