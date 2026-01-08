#!/usr/bin/env python3
"""
Script: analyze_memory_optimizations.py
Purpose: Analyze Phase 3 memory optimization results
"""

import pandas as pd
import sys

def main():
    import matplotlib.pyplot as plt
    import os

    csv_file = "results/memory_optimizations_results.csv"
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: {csv_file} not found.")
        print("Run ./scripts/bench_memory_optimizations.sh first.")
        sys.exit(1)

    if df.empty:
        print("No data found in results file.")
        sys.exit(1)

    # --- Visualization: Improvement by Optimization Type ---
    plt.figure(figsize=(10, 6))
    opt_summary = df.groupby('Optimization').agg({'Improvement_Percent': 'mean'}).sort_values('Improvement_Percent', ascending=False)
    plt.bar(opt_summary.index, opt_summary['Improvement_Percent'], color='tab:blue', alpha=0.8)
    plt.ylabel('Mean Improvement over Phase 1 (%)', fontsize=12, fontweight='bold')
    plt.xlabel('Optimization', fontsize=12, fontweight='bold')
    plt.title('Mean Improvement by Memory Optimization', fontsize=14, fontweight='bold')
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/memory_optimization_improvement.png', dpi=300, bbox_inches='tight')
    print('✓ Plot saved to: plots/memory_optimization_improvement.png')

    print("=" * 80)
    print("Phase 3: Memory-Level Optimization Analysis")
    print("=" * 80)
    print()
    
    # Summary by optimization type
    print("Summary by Optimization Type:")
    print("-" * 80)
    summary = df.groupby('Optimization').agg({
        'Time_ms': 'mean',
        'Speedup_vs_Serial': 'mean',
        'Speedup_vs_Phase1': 'mean',
        'Improvement_Percent': 'mean'
    }).round(2)
    print(summary)
    print()
    
    # Best configurations
    print("Top 5 Configurations (by improvement over Phase 1):")
    print("-" * 80)
    top5 = df.nlargest(5, 'Improvement_Percent')[['Matrix', 'Threads', 'Config', 'Improvement_Percent']]
    print(top5.to_string(index=False))
    print()
    
    # Analysis by thread count
    print("Performance by Thread Count:")
    print("-" * 80)
    thread_analysis = df.groupby('Threads').agg({
        'Improvement_Percent': ['mean', 'max']
    }).round(2)
    print(thread_analysis)
    print()
    
    # Matrix-specific insights
    print("Performance by Matrix:")
    print("-" * 80)
    matrix_analysis = df[df['Config'] != 'Baseline (Phase 1 winner)'].groupby('Matrix').agg({
        'Improvement_Percent': ['mean', 'max']
    }).round(2)
    print(matrix_analysis)
    print()
    
    # Best overall configuration
    best_row = df[df['Config'] != 'Baseline (Phase 1 winner)'].nlargest(1, 'Improvement_Percent').iloc[0]
    print("=" * 80)
    print("BEST MEMORY OPTIMIZATION:")
    print("-" * 80)
    print(f"Configuration: {best_row['Config']}")
    print(f"Matrix: {best_row['Matrix']}")
    print(f"Threads: {best_row['Threads']}")
    print(f"Improvement over Phase 1: {best_row['Improvement_Percent']:.2f}%")
    print(f"Overall speedup vs serial: {best_row['Speedup_vs_Serial']:.1f}x")
    print("=" * 80)

if __name__ == "__main__":
    main()
