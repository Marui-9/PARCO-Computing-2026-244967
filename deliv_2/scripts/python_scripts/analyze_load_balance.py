#!/usr/bin/env python3
"""
Analyze load balancing strategy results to compare ROW-BASED, NNZ-BASED, and HYBRID approaches.
Provides comprehensive performance analysis, imbalance metrics, and recommendations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def load_data(csv_path):
    """Load and validate load balance results CSV."""
    if not Path(csv_path).exists():
        print(f"ERROR: CSV file not found: {csv_path}")
        sys.exit(1)
    
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    required = ['num_procs', 'config_name', 'avg_time_ms', 'comm_time_ms', 
                'compute_time_ms', 'speedup', 'efficiency_pct', 'imbalance']
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        sys.exit(1)
    
    return df

def calculate_metrics_by_strategy(df):
    """Calculate aggregate metrics for each strategy."""
    strategies = df['config_name'].unique()
    metrics = {}
    
    for strategy in strategies:
        strategy_df = df[df['config_name'] == strategy]
        
        metrics[strategy] = {
            'avg_time': strategy_df['avg_time_ms'].mean(),
            'avg_imbalance': strategy_df['imbalance'].mean(),
            'avg_efficiency': strategy_df['efficiency_pct'].mean(),
            'avg_speedup': strategy_df['speedup'].mean(),
            'comm_pct': (strategy_df['comm_time_ms'].mean() / strategy_df['avg_time_ms'].mean()) * 100,
            'comp_pct': (strategy_df['compute_time_ms'].mean() / strategy_df['avg_time_ms'].mean()) * 100,
            'best_imbalance': strategy_df['imbalance'].min(),
            'worst_imbalance': strategy_df['imbalance'].max()
        }
    
    return metrics

def analyze_by_process_count(df):
    """Analyze performance across different process counts."""
    proc_counts = sorted(df['num_procs'].unique())
    strategies = sorted(df['config_name'].unique())
    
    print("\n" + "="*80)
    print("PERFORMANCE BY PROCESS COUNT")
    print("="*80 + "\n")
    
    print(f"{'Procs':<8} {'Strategy':<15} {'Time (ms)':<12} {'Speedup':<10} {'Efficiency':<12} {'Imbalance':<10}")
    print("-" * 80)
    
    for procs in proc_counts:
        proc_df = df[df['num_procs'] == procs]
        
        for strategy in strategies:
            strat_df = proc_df[proc_df['config_name'] == strategy]
            if len(strat_df) > 0:
                avg_time = strat_df['avg_time_ms'].mean()
                avg_speedup = strat_df['speedup'].mean()
                avg_eff = strat_df['efficiency_pct'].mean()
                avg_imb = strat_df['imbalance'].mean()
                
                print(f"{procs:<8} {strategy:<15} {avg_time:>10.2f}  {avg_speedup:>8.3f}  {avg_eff:>10.2f}%  {avg_imb:>8.3f}")
        print()

def analyze_by_matrix_density(df):
    """Analyze how matrix density affects strategy performance."""
    print("\n" + "="*80)
    print("PERFORMANCE BY MATRIX DENSITY")
    print("="*80 + "\n")
    
    # Bin matrices by density
    df['density_bin'] = pd.cut(df['density_pct'], 
                                bins=[0, 0.01, 0.1, 1.0, 100], 
                                labels=['Very Sparse (<0.01%)', 'Sparse (0.01-0.1%)', 
                                       'Medium (0.1-1%)', 'Dense (>1%)'])
    
    strategies = sorted(df['config_name'].unique())
    
    for density_bin in df['density_bin'].unique():
        if pd.isna(density_bin):
            continue
            
        print(f"\n{density_bin}:")
        print("-" * 80)
        
        density_df = df[df['density_bin'] == density_bin]
        
        for strategy in strategies:
            strat_df = density_df[density_df['config_name'] == strategy]
            if len(strat_df) > 0:
                avg_imb = strat_df['imbalance'].mean()
                avg_speedup = strat_df['speedup'].mean()
                avg_eff = strat_df['efficiency_pct'].mean()
                
                print(f"  {strategy:<15} Imbalance: {avg_imb:.3f}  Speedup: {avg_speedup:.3f}×  Efficiency: {avg_eff:.1f}%")

def compare_strategies(df):
    """Direct comparison of the three strategies."""
    print("\n" + "="*80)
    print("STRATEGY COMPARISON")
    print("="*80 + "\n")
    
    strategies = ['ROW-BASED', 'NNZ-BASED', 'HYBRID-0.5']
    metrics = calculate_metrics_by_strategy(df)
    
    print("**Overall Performance Metrics**\n")
    print(f"{'Strategy':<15} {'Avg Time':<12} {'Avg Imb':<10} {'Efficiency':<12} {'Comm %':<10}")
    print("-" * 80)
    
    for strategy in strategies:
        if strategy in metrics:
            m = metrics[strategy]
            print(f"{strategy:<15} {m['avg_time']:>10.2f}ms  {m['avg_imbalance']:>8.3f}  {m['avg_efficiency']:>10.2f}%  {m['comm_pct']:>8.1f}%")
    
    print("\n**Imbalance Factor Range**\n")
    print(f"{'Strategy':<15} {'Best':<10} {'Worst':<10} {'Range':<10}")
    print("-" * 80)
    
    for strategy in strategies:
        if strategy in metrics:
            m = metrics[strategy]
            range_val = m['worst_imbalance'] - m['best_imbalance']
            print(f"{strategy:<15} {m['best_imbalance']:>8.3f}  {m['worst_imbalance']:>8.3f}  {range_val:>8.3f}")

def identify_best_strategy(df):
    """Identify which strategy performs best under different conditions."""
    print("\n" + "="*80)
    print("BEST STRATEGY BY SCENARIO")
    print("="*80 + "\n")
    
    proc_counts = sorted(df['num_procs'].unique())
    
    # Best by process count
    print("**By Process Count:**\n")
    for procs in proc_counts:
        proc_df = df[df['num_procs'] == procs]
        
        # Find strategy with lowest average time
        best_time = proc_df.groupby('config_name')['avg_time_ms'].mean().idxmin()
        best_time_val = proc_df.groupby('config_name')['avg_time_ms'].mean().min()
        
        # Find strategy with best imbalance
        best_imb = proc_df.groupby('config_name')['imbalance'].mean().idxmin()
        best_imb_val = proc_df.groupby('config_name')['imbalance'].mean().min()
        
        print(f"  {procs} processes:")
        print(f"    Fastest: {best_time} ({best_time_val:.2f} ms)")
        print(f"    Most Balanced: {best_imb} (imbalance: {best_imb_val:.3f})")
    
    # Best by matrix characteristics
    print("\n**By Matrix Characteristics:**\n")
    
    # Very sparse matrices
    sparse_df = df[df['density_pct'] < 0.01]
    if len(sparse_df) > 0:
        best_sparse = sparse_df.groupby('config_name')['speedup'].mean().idxmax()
        best_sparse_speedup = sparse_df.groupby('config_name')['speedup'].mean().max()
        print(f"  Very Sparse Matrices (<0.01% density):")
        print(f"    Best: {best_sparse} (avg speedup: {best_sparse_speedup:.3f}×)")
    
    # Dense matrices
    dense_df = df[df['density_pct'] > 0.1]
    if len(dense_df) > 0:
        best_dense = dense_df.groupby('config_name')['speedup'].mean().idxmax()
        best_dense_speedup = dense_df.groupby('config_name')['speedup'].mean().max()
        print(f"  Dense Matrices (>0.1% density):")
        print(f"    Best: {best_dense} (avg speedup: {best_dense_speedup:.3f}×)")
    
    # Large matrices
    large_df = df[df['rows'] > 500000]
    if len(large_df) > 0:
        best_large = large_df.groupby('config_name')['efficiency_pct'].mean().idxmax()
        best_large_eff = large_df.groupby('config_name')['efficiency_pct'].mean().max()
        print(f"  Large Matrices (>500k rows):")
        print(f"    Best: {best_large} (avg efficiency: {best_large_eff:.1f}%)")

def generate_recommendations(df):
    """Generate recommendations based on analysis."""
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80 + "\n")
    
    strategies = ['ROW-BASED', 'NNZ-BASED', 'HYBRID-0.5']
    metrics = calculate_metrics_by_strategy(df)
    
    # Rank strategies by different criteria
    by_imbalance = sorted(strategies, key=lambda s: metrics.get(s, {}).get('avg_imbalance', float('inf')))
    by_speed = sorted(strategies, key=lambda s: metrics.get(s, {}).get('avg_time', float('inf')))
    by_efficiency = sorted(strategies, key=lambda s: metrics.get(s, {}).get('avg_efficiency', 0), reverse=True)
    
    print("**1. Load Balance Quality**")
    # Filter by_imbalance to only include strategies that exist in metrics
    by_imbalance_valid = [s for s in by_imbalance if s in metrics]
    if by_imbalance_valid:
        print(f"   Best → Worst: {' > '.join(by_imbalance_valid)}")
        if len(by_imbalance_valid) > 1:
            best_imb = metrics[by_imbalance_valid[0]]['avg_imbalance']
            worst_imb = metrics[by_imbalance_valid[-1]]['avg_imbalance']
            improvement = ((worst_imb - best_imb) / worst_imb) * 100
            print(f"   {by_imbalance_valid[0]} reduces imbalance by {improvement:.1f}% vs {by_imbalance_valid[-1]}")
    
    print("\n**2. Raw Performance (Speed)**")
    # Filter by_speed to only include strategies that exist in metrics
    by_speed_valid = [s for s in by_speed if s in metrics]
    if by_speed_valid:
        print(f"   Fastest → Slowest: {' > '.join(by_speed_valid)}")
        if len(by_speed_valid) > 1 and by_speed_valid[0] in metrics and by_speed_valid[-1] in metrics:
            fastest = metrics[by_speed_valid[0]]['avg_time']
            slowest = metrics[by_speed_valid[-1]]['avg_time']
            speedup = slowest / fastest
            print(f"   {by_speed_valid[0]} is {speedup:.2f}× faster than {by_speed_valid[-1]}")
    
    print("\n**3. Parallel Efficiency**")
    print(f"   Best → Worst: {' > '.join(by_efficiency)}")
    if by_efficiency[0] in metrics:
        best_eff = metrics[by_efficiency[0]]['avg_efficiency']
        print(f"   {by_efficiency[0]} achieves {best_eff:.1f}% average efficiency")
    
    print("\n**4. General Guidelines:**")
    
    # Analyze when each strategy wins
    proc_counts = sorted(df['num_procs'].unique())
    nnz_wins = 0
    row_wins = 0
    hyb_wins = 0
    
    for procs in proc_counts:
        proc_df = df[df['num_procs'] == procs]
        winner = proc_df.groupby('config_name')['avg_time_ms'].mean().idxmin()
        if winner == 'NNZ-BASED':
            nnz_wins += 1
        elif winner == 'ROW-BASED':
            row_wins += 1
    
    total = len(proc_counts)
    
    print(f"\n   - NNZ-BASED wins in {nnz_wins}/{total} process counts ({(nnz_wins/total*100):.0f}%)")
    print(f"   - ROW-BASED wins in {row_wins}/{total} process counts ({(row_wins/total*100):.0f}%)")
    
    # Specific recommendations
    print("\n**5. Use Case Recommendations:**")
    
    if nnz_wins > row_wins and nnz_wins > hyb_wins:
        print(f"   ✅ **Primary Choice: NNZ-BASED**")
        print(f"      - Best overall performance in most scenarios")
        print(f"      - Excellent load balance (avg imbalance: {metrics.get('NNZ-BASED', {}).get('avg_imbalance', 'N/A'):.3f})")
    elif hyb_wins > row_wins and hyb_wins > nnz_wins:
        print(f"   ✅ **Primary Choice: HYBRID-0.5**")
        print(f"      - Best balance between simplicity and performance")
    else:
        print(f"   ✅ **Primary Choice: ROW-BASED**")
        print(f"      - Simple and effective for most cases")
    
    # Check if ROW-BASED is ever significantly worse
    if 'ROW-BASED' in metrics and 'NNZ-BASED' in metrics:
        row_imb = metrics['ROW-BASED']['avg_imbalance']
        nnz_imb = metrics['NNZ-BASED']['avg_imbalance']
        
        if row_imb > 1.5:
            print(f"\n   ⚠️  **Warning: ROW-BASED shows high imbalance ({row_imb:.3f})**")
            print(f"      - Consider NNZ-BASED or HYBRID for better load distribution")
        
        if row_imb / nnz_imb > 2.0:
            print(f"\n   ⚠️  **Critical: ROW-BASED imbalance is {row_imb/nnz_imb:.1f}× worse than NNZ-BASED**")
            print(f"      - Strongly recommend NNZ-BASED for production use")

def main():
    """Main analysis function."""
    # Determine CSV path
    script_dir = Path(__file__).parent
    csv_path = script_dir / "../../results/load_balance_results.csv"
    
    if not csv_path.exists():
        csv_path = Path("results/load_balance_results.csv")
    
    print("="*80)
    print("LOAD BALANCING STRATEGY ANALYSIS")
    print("="*80)
    print(f"\nAnalyzing: {csv_path}")
    
    # Load and process data
    df = load_data(csv_path)
    print(f"Loaded {len(df)} data points")
    print(f"Strategies: {', '.join(df['config_name'].unique())}")
    print(f"Process counts: {sorted(df['num_procs'].unique())}")
    print(f"Matrices: {len(df['matrix'].unique())} unique")
    
    # Run analysis sections
    compare_strategies(df)
    analyze_by_process_count(df)
    analyze_by_matrix_density(df)
    identify_best_strategy(df)
    generate_recommendations(df)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
