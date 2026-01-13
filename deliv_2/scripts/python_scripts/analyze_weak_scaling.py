#!/usr/bin/env python3
"""
Analyze weak scaling results from weak_scaling_results.csv.

Weak Scaling Analysis:
- Work per process is constant (rows_per_proc fixed)
- Matrix size scales with process count
- Ideal weak scaling: constant execution time as P increases
- Weak efficiency = T(base) / T(P) × 100%

Metrics:
- Weak Efficiency: How well execution time stays constant
- Isospeed Efficiency: Time deviation from baseline
- Communication Overhead: How communication scales with P
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def load_data(csv_path):
    """Load weak scaling results CSV."""
    if not Path(csv_path).exists():
        print(f"ERROR: CSV file not found: {csv_path}")
        print("Run the weak scaling benchmark first:")
        print("  qsub pbs_jobs/weak_scaling.pbs")
        sys.exit(1)
    
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    required = ['num_procs', 'rows_per_proc', 'global_rows', 'config_name', 
                'avg_time_ms', 'comm_time_ms', 'compute_time_ms']
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        sys.exit(1)
    
    return df

def calculate_weak_efficiency(df):
    """
    Calculate weak scaling efficiency.
    Weak efficiency = T(base) / T(P) × 100%
    where T(base) is execution time at smallest process count.
    """
    results = []
    
    for config in df['config_name'].unique():
        config_data = df[df['config_name'] == config].sort_values('num_procs')
        
        # Get baseline time (smallest process count)
        base_time = config_data['avg_time_ms'].iloc[0]
        base_procs = config_data['num_procs'].iloc[0]
        
        for _, row in config_data.iterrows():
            weak_eff = (base_time / row['avg_time_ms']) * 100.0
            
            results.append({
                'num_procs': row['num_procs'],
                'config_name': config,
                'global_rows': row['global_rows'],
                'avg_time_ms': row['avg_time_ms'],
                'comm_time_ms': row['comm_time_ms'],
                'compute_time_ms': row['compute_time_ms'],
                'base_time_ms': base_time,
                'base_procs': base_procs,
                'weak_efficiency': weak_eff,
                'time_increase_factor': row['avg_time_ms'] / base_time,
                'comm_fraction': (row['comm_time_ms'] / row['avg_time_ms']) * 100.0
            })
    
    return pd.DataFrame(results)

def analyze_overall_weak_scaling(eff_data):
    """Analyze overall weak scaling behavior."""
    print("\n" + "="*80)
    print("OVERALL WEAK SCALING ANALYSIS")
    print("="*80)
    
    print("\n**Weak Scaling Summary by Configuration**\n")
    print(f"{'Configuration':<25} | {'Avg Eff':<10} | {'Min Eff':<10} | {'Max Eff':<10} | {'Eff @ Max P':<12}")
    print("-"*80)
    
    for config in sorted(eff_data['config_name'].unique()):
        config_data = eff_data[eff_data['config_name'] == config]
        
        avg_eff = config_data['weak_efficiency'].mean()
        min_eff = config_data['weak_efficiency'].min()
        max_eff = config_data['weak_efficiency'].max()
        max_procs_eff = config_data.loc[config_data['num_procs'].idxmax(), 'weak_efficiency']
        
        print(f"{config:<25} | {avg_eff:>8.2f}% | {min_eff:>8.2f}% | {max_eff:>8.2f}% | {max_procs_eff:>10.2f}%")
    
    # Interpretation
    print("\n**Interpretation:**")
    avg_overall = eff_data['weak_efficiency'].mean()
    if avg_overall >= 80:
        print("  ✅ EXCELLENT weak scaling: Efficiency stays above 80%")
    elif avg_overall >= 60:
        print("  ⚠️  GOOD weak scaling: Some efficiency loss but acceptable")
    elif avg_overall >= 40:
        print("  ⚠️  MODERATE weak scaling: Noticeable efficiency degradation")
    else:
        print("  ❌ POOR weak scaling: Significant communication overhead")

def analyze_by_process_count(eff_data):
    """Analyze weak scaling by process count."""
    print("\n" + "="*80)
    print("WEAK SCALING BY PROCESS COUNT")
    print("="*80)
    
    proc_counts = sorted(eff_data['num_procs'].unique())
    
    print(f"\n{'Procs':<8} | {'Matrix Size':<15} | ", end="")
    configs = sorted(eff_data['config_name'].unique())
    for config in configs:
        short_name = config[:15]
        print(f"{short_name:<18} | ", end="")
    print()
    print("-"*100)
    
    for procs in proc_counts:
        procs_data = eff_data[eff_data['num_procs'] == procs]
        
        if len(procs_data) == 0:
            continue
            
        matrix_size = procs_data['global_rows'].iloc[0]
        
        print(f"{procs:<8} | {matrix_size:>12,} | ", end="")
        
        for config in configs:
            config_procs = procs_data[procs_data['config_name'] == config]
            if len(config_procs) > 0:
                eff = config_procs['weak_efficiency'].iloc[0]
                time = config_procs['avg_time_ms'].iloc[0]
                print(f"{eff:>6.1f}% ({time:>7.2f}ms) | ", end="")
            else:
                print(f"{'N/A':^18} | ", end="")
        print()

def analyze_communication_scaling(eff_data):
    """Analyze how communication scales with process count."""
    print("\n" + "="*80)
    print("COMMUNICATION OVERHEAD SCALING")
    print("="*80)
    
    print("\n**Communication Fraction by Process Count**\n")
    print(f"{'Procs':<8} | ", end="")
    configs = sorted(eff_data['config_name'].unique())
    for config in configs:
        short_name = config[:18]
        print(f"{short_name:<20} | ", end="")
    print()
    print("-"*90)
    
    for procs in sorted(eff_data['num_procs'].unique()):
        procs_data = eff_data[eff_data['num_procs'] == procs]
        
        print(f"{procs:<8} | ", end="")
        
        for config in configs:
            config_procs = procs_data[procs_data['config_name'] == config]
            if len(config_procs) > 0:
                comm_frac = config_procs['comm_fraction'].iloc[0]
                comm_time = config_procs['comm_time_ms'].iloc[0]
                print(f"{comm_frac:>5.1f}% ({comm_time:>8.2f}ms) | ", end="")
            else:
                print(f"{'N/A':^20} | ", end="")
        print()
    
    # Analyze communication growth
    print("\n**Communication Growth Analysis**")
    for config in configs:
        config_data = eff_data[eff_data['config_name'] == config].sort_values('num_procs')
        
        if len(config_data) >= 2:
            base_comm = config_data['comm_time_ms'].iloc[0]
            max_comm = config_data['comm_time_ms'].iloc[-1]
            base_procs = config_data['num_procs'].iloc[0]
            max_procs = config_data['num_procs'].iloc[-1]
            
            comm_growth = max_comm / base_comm if base_comm > 0 else float('inf')
            proc_ratio = max_procs / base_procs
            
            # Ideal: comm grows as O(P) or O(log P)
            ideal_linear = proc_ratio
            ideal_log = np.log2(max_procs) / np.log2(base_procs) if base_procs > 1 else 1
            
            print(f"\n  {config}:")
            print(f"    Comm time at {base_procs:>3} procs: {base_comm:>8.2f} ms")
            print(f"    Comm time at {max_procs:>3} procs: {max_comm:>8.2f} ms")
            print(f"    Actual growth: {comm_growth:.2f}×")
            print(f"    Expected O(P): {ideal_linear:.2f}×")
            print(f"    Expected O(log P): {ideal_log:.2f}×")
            
            if comm_growth <= ideal_log * 1.5:
                print(f"    → ✅ Communication scales as O(log P) - EXCELLENT")
            elif comm_growth <= ideal_linear:
                print(f"    → ⚠️  Communication scales as O(P) - ACCEPTABLE")
            else:
                print(f"    → ❌ Communication scales worse than O(P) - POOR")

def analyze_compute_scaling(eff_data):
    """Analyze how computation time scales (should stay constant in weak scaling)."""
    print("\n" + "="*80)
    print("COMPUTATION TIME SCALING")
    print("="*80)
    
    print("\n**Compute Time by Process Count (should be constant)**\n")
    
    for config in sorted(eff_data['config_name'].unique()):
        config_data = eff_data[eff_data['config_name'] == config].sort_values('num_procs')
        
        base_compute = config_data['compute_time_ms'].iloc[0]
        
        print(f"\n  {config}:")
        print(f"  {'Procs':<8} | {'Compute (ms)':<15} | {'Ratio vs Base':<15} | Status")
        print("  " + "-"*60)
        
        for _, row in config_data.iterrows():
            ratio = row['compute_time_ms'] / base_compute if base_compute > 0 else 0
            
            if 0.9 <= ratio <= 1.1:
                status = "✅ Constant"
            elif ratio < 0.9:
                status = "🎉 Super-linear (cache effects?)"
            elif ratio <= 1.3:
                status = "⚠️  Slight increase"
            else:
                status = "❌ Significant increase"
            
            print(f"  {row['num_procs']:<8} | {row['compute_time_ms']:>13.2f} | {ratio:>13.2f}× | {status}")

def identify_scaling_limit(eff_data):
    """Identify the point where weak scaling breaks down."""
    print("\n" + "="*80)
    print("WEAK SCALING LIMITS")
    print("="*80)
    
    print("\n**Identifying Scaling Breakdown Point**\n")
    
    for config in sorted(eff_data['config_name'].unique()):
        config_data = eff_data[eff_data['config_name'] == config].sort_values('num_procs')
        
        print(f"{config}:")
        
        # Find where efficiency drops below thresholds
        eff_70 = config_data[config_data['weak_efficiency'] < 70]
        eff_50 = config_data[config_data['weak_efficiency'] < 50]
        
        if len(eff_70) > 0:
            limit_70 = eff_70['num_procs'].iloc[0]
            print(f"  ⚠️  Efficiency drops below 70% at: {limit_70} processes")
        else:
            print(f"  ✅ Efficiency stays above 70% for all tested process counts")
        
        if len(eff_50) > 0:
            limit_50 = eff_50['num_procs'].iloc[0]
            print(f"  ❌ Efficiency drops below 50% at: {limit_50} processes")
        else:
            print(f"  ✅ Efficiency stays above 50% for all tested process counts")
        
        # Find optimal process count (best efficiency)
        optimal = config_data.loc[config_data['weak_efficiency'].idxmax()]
        print(f"  🎯 Best efficiency: {optimal['weak_efficiency']:.1f}% at {optimal['num_procs']} processes")
        print()

def generate_recommendations(eff_data):
    """Generate practical recommendations based on weak scaling analysis."""
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    configs = sorted(eff_data['config_name'].unique())
    
    # Find best configuration for weak scaling
    config_avg_eff = {}
    for config in configs:
        config_data = eff_data[eff_data['config_name'] == config]
        config_avg_eff[config] = config_data['weak_efficiency'].mean()
    
    best_config = max(config_avg_eff, key=config_avg_eff.get)
    best_eff = config_avg_eff[best_config]
    
    print(f"\n**1. Best Configuration for Weak Scaling: {best_config}**")
    print(f"   Average weak efficiency: {best_eff:.1f}%")
    
    # Find optimal process count for production
    best_data = eff_data[eff_data['config_name'] == best_config].sort_values('num_procs')
    
    # Production recommendation: highest P with efficiency > 70%
    good_scaling = best_data[best_data['weak_efficiency'] >= 70]
    if len(good_scaling) > 0:
        optimal_procs = good_scaling['num_procs'].max()
        optimal_eff = good_scaling.loc[good_scaling['num_procs'] == optimal_procs, 'weak_efficiency'].iloc[0]
        optimal_rows = good_scaling.loc[good_scaling['num_procs'] == optimal_procs, 'global_rows'].iloc[0]
        print(f"\n**2. Recommended Maximum Process Count: {optimal_procs}**")
        print(f"   Weak efficiency at this scale: {optimal_eff:.1f}%")
        print(f"   Maximum matrix size: {optimal_rows:,} rows")
    else:
        print(f"\n**2. Warning:** All process counts show efficiency < 70%")
        print("   Consider reducing rows_per_proc or optimizing communication")
    
    # Communication overhead warning
    max_procs_data = eff_data[eff_data['num_procs'] == eff_data['num_procs'].max()]
    avg_comm_frac = max_procs_data['comm_fraction'].mean()
    
    if avg_comm_frac > 50:
        print(f"\n**3. Warning: High Communication Overhead**")
        print(f"   At maximum process count, communication is {avg_comm_frac:.1f}% of execution time")
        print("   Consider:")
        print("   - Increasing rows_per_proc for better compute/comm ratio")
        print("   - Using pipelined communication to overlap with computation")
        print("   - Reducing inter-node communication through topology-aware placement")
    else:
        print(f"\n**3. Communication Overhead: Acceptable**")
        print(f"   At maximum process count, communication is {avg_comm_frac:.1f}% of execution time")
    
    # General guidelines
    print("\n**4. General Guidelines for Weak Scaling:**")
    print("   - Weak scaling efficiency > 80%: Excellent for production use")
    print("   - Weak scaling efficiency 60-80%: Acceptable with some overhead")
    print("   - Weak scaling efficiency < 60%: Consider reducing process count")
    print("   - If computation time increases with P: Check for load imbalance")
    print("   - If communication dominates: Increase problem size per process")

def main():
    """Main analysis function."""
    # Determine paths
    script_dir = Path(__file__).parent
    csv_path = script_dir / "../../results/weak_scaling_results.csv"
    
    if not csv_path.exists():
        csv_path = Path("results/weak_scaling_results.csv")
    
    print("="*80)
    print("WEAK SCALING ANALYSIS")
    print("="*80)
    print(f"\nAnalyzing: {csv_path}")
    
    # Load data
    df = load_data(csv_path)
    
    print(f"Loaded {len(df)} data points")
    print(f"Configurations: {', '.join(df['config_name'].unique())}")
    print(f"Process counts: {sorted(df['num_procs'].unique())}")
    print(f"Rows per process: {df['rows_per_proc'].iloc[0]}")
    
    # Calculate weak efficiency metrics
    eff_data = calculate_weak_efficiency(df)
    
    # Run analyses
    analyze_overall_weak_scaling(eff_data)
    analyze_by_process_count(eff_data)
    analyze_communication_scaling(eff_data)
    analyze_compute_scaling(eff_data)
    identify_scaling_limit(eff_data)
    generate_recommendations(eff_data)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
