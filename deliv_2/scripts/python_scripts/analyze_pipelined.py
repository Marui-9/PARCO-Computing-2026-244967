#!/usr/bin/env python3
"""
Analyze pipelined communication mode results and compare against baseline configurations.
Evaluates whether pipelined chunked communication provides benefits over standard MPI collectives.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def load_data(pipelined_path, configs_path):
    """Load both pipelined and configurations CSV files."""
    if not Path(pipelined_path).exists():
        print(f"ERROR: Pipelined results not found: {pipelined_path}")
        sys.exit(1)
    
    if not Path(configs_path).exists():
        print(f"ERROR: Configurations results not found: {configs_path}")
        sys.exit(1)
    
    df_pipelined = pd.read_csv(pipelined_path)
    df_configs = pd.read_csv(configs_path)
    
    # Validate required columns
    required = ['num_procs', 'config_name', 'avg_time_ms', 'comm_time_ms', 
                'compute_time_ms', 'speedup', 'efficiency_pct']
    
    for df, name in [(df_pipelined, "pipelined"), (df_configs, "configurations")]:
        missing = [col for col in required if col not in df.columns]
        if missing:
            print(f"ERROR: Missing columns in {name}: {missing}")
            sys.exit(1)
    
    return df_pipelined, df_configs

def compare_communication_modes(df_pipelined, df_configs):
    """Compare pipelined mode against baseline configurations."""
    print("\n" + "="*80)
    print("PIPELINED VS BASELINE CONFIGURATIONS")
    print("="*80 + "\n")
    
    # Get unique process counts that exist in both datasets
    proc_counts = sorted(set(df_pipelined['num_procs'].unique()) & 
                        set(df_configs['num_procs'].unique()))
    
    print(f"{'Procs':<8} {'Mode':<25} {'Time (ms)':<12} {'Comm %':<10} {'Speedup':<10} {'Efficiency':<12}")
    print("-" * 95)
    
    for procs in proc_counts:
        # Get baseline (MPI_Bcast+Gatherv if available, otherwise first config)
        config_proc = df_configs[df_configs['num_procs'] == procs]
        baseline = config_proc[config_proc['config_name'] == 'MPI_Bcast+Gatherv']
        if len(baseline) == 0:
            baseline = config_proc.groupby('config_name').first().reset_index().iloc[0:1]
        
        baseline_time = baseline['avg_time_ms'].mean()
        baseline_comm_pct = (baseline['comm_time_ms'].mean() / baseline_time) * 100
        
        # Get pipelined mode
        pipelined_proc = df_pipelined[df_pipelined['num_procs'] == procs]
        if len(pipelined_proc) == 0:
            continue
        
        pipelined_time = pipelined_proc['avg_time_ms'].mean()
        pipelined_comm = pipelined_proc['comm_time_ms'].mean()
        pipelined_comm_pct = (pipelined_comm / pipelined_time) * 100
        pipelined_speedup = pipelined_proc['speedup'].mean()
        pipelined_eff = pipelined_proc['efficiency_pct'].mean()
        
        # Calculate improvement over baseline
        improvement = ((baseline_time - pipelined_time) / baseline_time) * 100
        
        # Print baseline
        print(f"{procs:<8} {'Baseline (Bcast+Gatherv)':<25} {baseline_time:>10.2f}  {baseline_comm_pct:>8.1f}%  {'1.00×':<10} {100/procs:>10.2f}%")
        
        # Print pipelined with comparison
        marker = "✅" if improvement > 5 else "⚠️" if improvement > 0 else "❌"
        print(f"{procs:<8} {'Pipelined_Chunked ' + marker:<25} {pipelined_time:>10.2f}  {pipelined_comm_pct:>8.1f}%  {pipelined_speedup:<10.3f}  {pipelined_eff:>10.2f}%")
        print(f"         → Improvement: {improvement:+.2f}% vs baseline\n")

def analyze_communication_overhead(df_pipelined, df_configs):
    """Analyze communication overhead reduction."""
    print("\n" + "="*80)
    print("COMMUNICATION OVERHEAD ANALYSIS")
    print("="*80 + "\n")
    
    proc_counts = sorted(set(df_pipelined['num_procs'].unique()) & 
                        set(df_configs['num_procs'].unique()))
    
    print(f"{'Procs':<8} {'Baseline Comm':<15} {'Pipelined Comm':<18} {'Reduction':<12} {'Status'}")
    print("-" * 80)
    
    total_reduction = 0
    count = 0
    
    for procs in proc_counts:
        # Baseline communication time
        config_proc = df_configs[df_configs['num_procs'] == procs]
        baseline_comm = config_proc['comm_time_ms'].mean()
        baseline_total = config_proc['avg_time_ms'].mean()
        baseline_comm_pct = (baseline_comm / baseline_total) * 100
        
        # Pipelined communication time
        pipelined_proc = df_pipelined[df_pipelined['num_procs'] == procs]
        if len(pipelined_proc) == 0:
            continue
        
        pipelined_comm = pipelined_proc['comm_time_ms'].mean()
        pipelined_total = pipelined_proc['avg_time_ms'].mean()
        pipelined_comm_pct = (pipelined_comm / pipelined_total) * 100
        
        reduction = baseline_comm_pct - pipelined_comm_pct
        total_reduction += reduction
        count += 1
        
        status = "✅ Better" if reduction > 5 else "≈ Similar" if abs(reduction) <= 5 else "❌ Worse"
        
        print(f"{procs:<8} {baseline_comm_pct:>6.1f}%        {pipelined_comm_pct:>8.1f}%         {reduction:>+8.1f}%    {status}")
    
    if count > 0:
        avg_reduction = total_reduction / count
        print("-" * 80)
        print(f"Average communication overhead reduction: {avg_reduction:+.1f}%")

def analyze_by_matrix_size(df_pipelined, df_configs):
    """Analyze performance by matrix size."""
    print("\n" + "="*80)
    print("PERFORMANCE BY MATRIX SIZE")
    print("="*80 + "\n")
    
    # Combine datasets
    df_pipelined_copy = df_pipelined.copy()
    df_pipelined_copy['mode'] = 'Pipelined'
    
    df_configs_copy = df_configs.copy()
    df_configs_copy['mode'] = 'Baseline'
    
    combined = pd.concat([df_pipelined_copy, df_configs_copy])
    
    # Categorize by matrix size (rows)
    combined['size_category'] = pd.cut(combined['rows'], 
                                       bins=[0, 100000, 500000, 1000000, 10000000],
                                       labels=['Small (<100k)', 'Medium (100k-500k)', 
                                              'Large (500k-1M)', 'Very Large (>1M)'])
    
    print(f"{'Size Category':<25} {'Mode':<12} {'Avg Time':<12} {'Avg Speedup':<12} {'Efficiency'}")
    print("-" * 80)
    
    for category in combined['size_category'].unique():
        if pd.isna(category):
            continue
        
        cat_data = combined[combined['size_category'] == category]
        
        for mode in ['Baseline', 'Pipelined']:
            mode_data = cat_data[cat_data['mode'] == mode]
            if len(mode_data) > 0:
                avg_time = mode_data['avg_time_ms'].mean()
                avg_speedup = mode_data['speedup'].mean()
                avg_eff = mode_data['efficiency_pct'].mean()
                
                print(f"{category if mode == 'Baseline' else '':<25} {mode:<12} {avg_time:>10.2f}ms  {avg_speedup:>10.3f}×  {avg_eff:>10.2f}%")
        print()

def analyze_scalability(df_pipelined, df_configs):
    """Analyze scalability of pipelined vs baseline."""
    print("\n" + "="*80)
    print("SCALABILITY ANALYSIS")
    print("="*80 + "\n")
    
    proc_counts = sorted(set(df_pipelined['num_procs'].unique()) & 
                        set(df_configs['num_procs'].unique()))
    
    if len(proc_counts) < 3:
        print("Not enough process counts for scalability analysis")
        return
    
    # Calculate efficiency degradation rate
    baseline_effs = []
    pipelined_effs = []
    
    for procs in proc_counts:
        config_proc = df_configs[df_configs['num_procs'] == procs]
        if len(config_proc) > 0:
            baseline_effs.append(config_proc['efficiency_pct'].mean())
        
        pipelined_proc = df_pipelined[df_pipelined['num_procs'] == procs]
        if len(pipelined_proc) > 0:
            pipelined_effs.append(pipelined_proc['efficiency_pct'].mean())
    
    print("**Efficiency Trend:**\n")
    print(f"{'Processes':<12} {'Baseline Eff':<15} {'Pipelined Eff':<15} {'Difference'}")
    print("-" * 60)
    
    for i, procs in enumerate(proc_counts[:min(len(baseline_effs), len(pipelined_effs))]):
        diff = pipelined_effs[i] - baseline_effs[i]
        marker = "✅" if diff > 0 else "❌" if diff < -1 else "≈"
        print(f"{procs:<12} {baseline_effs[i]:>13.2f}%  {pipelined_effs[i]:>13.2f}%  {diff:>+8.2f}% {marker}")
    
    # Calculate degradation slopes
    if len(baseline_effs) >= 3 and len(pipelined_effs) >= 3:
        baseline_slope = (baseline_effs[-1] - baseline_effs[0]) / (proc_counts[-1] - proc_counts[0])
        pipelined_slope = (pipelined_effs[-1] - pipelined_effs[0]) / (proc_counts[-1] - proc_counts[0])
        
        print(f"\n**Efficiency Degradation Rate:**")
        print(f"  Baseline:  {baseline_slope:.4f}% per process")
        print(f"  Pipelined: {pipelined_slope:.4f}% per process")
        
        if pipelined_slope > baseline_slope:
            print(f"  ✅ Pipelined scales better (slower efficiency loss)")
        else:
            print(f"  ❌ Baseline scales better (slower efficiency loss)")

def generate_recommendations(df_pipelined, df_configs):
    """Generate recommendations based on analysis."""
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80 + "\n")
    
    proc_counts = sorted(set(df_pipelined['num_procs'].unique()) & 
                        set(df_configs['num_procs'].unique()))
    
    wins = 0
    losses = 0
    ties = 0
    
    for procs in proc_counts:
        config_proc = df_configs[df_configs['num_procs'] == procs]
        baseline_time = config_proc['avg_time_ms'].mean()
        
        pipelined_proc = df_pipelined[df_pipelined['num_procs'] == procs]
        if len(pipelined_proc) == 0:
            continue
        
        pipelined_time = pipelined_proc['avg_time_ms'].mean()
        
        improvement = ((baseline_time - pipelined_time) / baseline_time) * 100
        
        if improvement > 5:
            wins += 1
        elif improvement < -5:
            losses += 1
        else:
            ties += 1
    
    total = wins + losses + ties
    
    print("**Overall Performance:**\n")
    print(f"  Pipelined wins:   {wins}/{total} process counts ({wins/total*100:.0f}%)")
    print(f"  Pipelined loses:  {losses}/{total} process counts ({losses/total*100:.0f}%)")
    print(f"  Similar (±5%):    {ties}/{total} process counts ({ties/total*100:.0f}%)")
    
    print("\n**Key Findings:**\n")
    
    if wins > losses:
        print("  ✅ **Pipelined mode shows overall benefit**")
        print("     - Recommended for production use when communication is a bottleneck")
    elif losses > wins:
        print("  ❌ **Pipelined mode underperforms baseline**")
        print("     - Overhead of chunking and selective distribution not justified")
        print("     - Stick with standard MPI collectives (Bcast/Gatherv)")
    else:
        print("  ⚠️  **Pipelined mode shows mixed results**")
        print("     - Performance similar to baseline")
        print("     - Consider complexity vs benefit trade-off")
    
    # Analyze communication bottleneck
    config_avg_comm = df_configs['comm_time_ms'].mean() / df_configs['avg_time_ms'].mean() * 100
    pipelined_avg_comm = df_pipelined['comm_time_ms'].mean() / df_pipelined['avg_time_ms'].mean() * 100
    
    print(f"\n**Communication Overhead:**")
    print(f"  Baseline average:  {config_avg_comm:.1f}%")
    print(f"  Pipelined average: {pipelined_avg_comm:.1f}%")
    
    if config_avg_comm > 50:
        print("\n  ⚠️  **High communication overhead detected (>50%)**")
        if pipelined_avg_comm < config_avg_comm:
            print("     → Pipelined mode successfully reduces communication bottleneck")
            print("     → Recommended for these workloads")
        else:
            print("     → Pipelined mode does NOT reduce communication bottleneck")
            print("     → Consider different approach (larger chunks, different topology)")
    
    # Best use cases
    print("\n**Best Use Cases for Pipelined Mode:**")
    
    # Find where pipelined wins most
    best_improvement = -float('inf')
    best_procs = None
    
    for procs in proc_counts:
        config_proc = df_configs[df_configs['num_procs'] == procs]
        baseline_time = config_proc['avg_time_ms'].mean()
        
        pipelined_proc = df_pipelined[df_pipelined['num_procs'] == procs]
        if len(pipelined_proc) == 0:
            continue
        
        pipelined_time = pipelined_proc['avg_time_ms'].mean()
        improvement = ((baseline_time - pipelined_time) / baseline_time) * 100
        
        if improvement > best_improvement:
            best_improvement = improvement
            best_procs = procs
    
    if best_procs is not None and best_improvement > 5:
        print(f"  - Best performance at {best_procs} processes ({best_improvement:+.1f}% improvement)")
    
    # Matrix size analysis
    if 'rows' in df_pipelined.columns:
        large_matrices = df_pipelined[df_pipelined['rows'] > 500000]
        small_matrices = df_pipelined[df_pipelined['rows'] <= 500000]
        
        if len(large_matrices) > 0 and len(small_matrices) > 0:
            large_speedup = large_matrices['speedup'].mean()
            small_speedup = small_matrices['speedup'].mean()
            
            if large_speedup > small_speedup * 1.1:
                print(f"  - More effective for large matrices (>500k rows)")
            elif small_speedup > large_speedup * 1.1:
                print(f"  - More effective for small matrices (<500k rows)")
    
    print("\n**Implementation Recommendation:**")
    
    if wins > losses and best_improvement > 10:
        print("  ✅ **IMPLEMENT pipelined mode**")
        print("     - Significant performance gains observed")
        print("     - Worth the implementation complexity")
    elif wins > losses and best_improvement > 5:
        print("  ⚠️  **CONSIDER pipelined mode for specific cases**")
        print("     - Moderate improvements in some scenarios")
        print("     - Evaluate complexity vs benefit trade-off")
    else:
        print("  ❌ **DO NOT implement pipelined mode**")
        print("     - No significant benefit over baseline")
        print("     - Added complexity not justified")
        print("     - Focus optimization efforts elsewhere")

def main():
    """Main analysis function."""
    # Determine CSV paths
    script_dir = Path(__file__).parent
    pipelined_path = script_dir / "../../results/pipelined_results.csv"
    configs_path = script_dir / "../../results/configurations_results.csv"
    
    if not pipelined_path.exists():
        pipelined_path = Path("results/pipelined_results.csv")
    if not configs_path.exists():
        configs_path = Path("results/configurations_results.csv")
    
    print("="*80)
    print("PIPELINED MODE PERFORMANCE ANALYSIS")
    print("="*80)
    print(f"\nPipelined results: {pipelined_path}")
    print(f"Baseline results:  {configs_path}")
    
    # Load data
    df_pipelined, df_configs = load_data(pipelined_path, configs_path)
    
    print(f"\nPipelined data points: {len(df_pipelined)}")
    print(f"Baseline data points:  {len(df_configs)}")
    print(f"\nPipelined modes: {', '.join(df_pipelined['config_name'].unique())}")
    print(f"Baseline modes:  {', '.join(df_configs['config_name'].unique())}")
    
    # Run analysis sections
    compare_communication_modes(df_pipelined, df_configs)
    analyze_communication_overhead(df_pipelined, df_configs)
    analyze_by_matrix_size(df_pipelined, df_configs)
    analyze_scalability(df_pipelined, df_configs)
    generate_recommendations(df_pipelined, df_configs)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
