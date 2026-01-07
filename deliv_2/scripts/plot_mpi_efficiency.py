#!/usr/bin/env python3
"""
Plot MPI Communication Efficiency across node counts (2-6 nodes).
Efficiency = (Speedup / Number of Nodes) * 100%
One line per communication mode.
"""

import os
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def read_results(csv_path: str) -> pd.DataFrame:
    """
    Read a CSV results file robustly, handling messy formatting.
    """
    try:
        with open(csv_path, 'r') as f:
            lines = [line.rstrip('\n') for line in f.readlines() if line.strip()]
    except Exception as e:
        print(f"[WARN] Failed to read {csv_path}: {e}", file=sys.stderr)
        return pd.DataFrame()
    
    if not lines:
        return pd.DataFrame()
    
    # Reconstruct header by joining first line(s)
    header_lines = []
    full_header = ""
    for i, line in enumerate(lines):
        header_lines.append(line)
        full_header = ",".join(header_lines)
        if full_header.count(',') >= 16:
            break
    
    headers = [h.strip() for h in full_header.split(',')]
    
    # Fixed column positions
    col_pos = {
        'num_nodes': 0, 'matrix': 1, 'rows': 2, 'cols': 3, 'nnz': 4,
        'density_pct': 5, 'config_name': 6, 'avg_time_ms': 7, 'std_dev_ms': 8,
        'min_time_ms': 9, 'max_time_ms': 10, 'comm_time_ms': 11, 'compute_time_ms': 12,
        'speedup': 13, 'efficiency_pct': 14, 'iterations': 15, 'notes': 16
    }
    
    # Parse data rows
    data_rows = []
    for line in lines[len(header_lines):]:
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(',')]
        
        row = {}
        for col, pos in col_pos.items():
            if pos < len(parts):
                val = parts[pos]
                val = val.replace('"', '').strip()
                row[col] = val
        
        if row:
            data_rows.append(row)
    
    if not data_rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(data_rows)
    
    # Coerce numeric types
    for col in ['num_nodes', 'rows', 'cols', 'nnz', 'avg_time_ms', 'std_dev_ms',
                'min_time_ms', 'max_time_ms', 'comm_time_ms', 'compute_time_ms',
                'speedup', 'efficiency_pct', 'iterations', 'density_pct']:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace('%', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    if 'num_nodes' in df.columns:
        df['num_nodes'] = df['num_nodes'].fillna(0).astype(int)
    if 'config_name' in df.columns:
        df['config_name'] = df['config_name'].fillna('unknown').astype(str).str.strip()
    
    df['source_csv'] = os.path.basename(csv_path)
    return df


def load_all_results(results_dir: str) -> pd.DataFrame:
    """
    Load all MPI result CSVs from the results directory.
    """
    csv_files = glob.glob(os.path.join(results_dir, 'test_config_mpi_results_*nodes.csv'))
    
    if not csv_files:
        print(f"[ERROR] No result files found in {results_dir}", file=sys.stderr)
        return pd.DataFrame()
    
    all_dfs = []
    for csv_file in sorted(csv_files):
        df = read_results(csv_file)
        if not df.empty:
            all_dfs.append(df)
            print(f"[INFO] Loaded {len(df)} rows from {os.path.basename(csv_file)}")
    
    if not all_dfs:
        return pd.DataFrame()
    
    all_df = pd.concat(all_dfs, ignore_index=True)
    all_df = all_df[all_df['efficiency_pct'].notnull()]
    
    return all_df


def plot_efficiency(df: pd.DataFrame, output_png: str):
    """
    Plot efficiency vs. node count with one line per communication mode.
    """
    plt.figure(figsize=(12, 7))
    
    configs = sorted(df['config_name'].unique())
    configs = [c for c in configs if pd.notna(c)]
    
    colors = plt.cm.tab10(range(len(configs)))
    
    for config, color in zip(configs, colors):
        config_data = df[df['config_name'] == config].copy()
        
        # Group by num_nodes and take mean efficiency
        grouped = config_data.groupby('num_nodes').agg({
            'efficiency_pct': ['mean', 'std']
        }).reset_index()
        grouped.columns = ['num_nodes', 'mean_efficiency', 'std_efficiency']
        grouped = grouped.sort_values('num_nodes')
        
        plt.plot(grouped['num_nodes'], grouped['mean_efficiency'],
                marker='o', linewidth=2.5, markersize=8, label=config, color=color)
        
        # Add error bars (optional, can be enabled)
        # plt.errorbar(grouped['num_nodes'], grouped['mean_efficiency'],
        #             yerr=grouped['std_efficiency'], fmt='none', 
        #             color=color, alpha=0.3, capsize=5)
    
    plt.xlabel('Number of Nodes', fontsize=13, fontweight='bold')
    plt.ylabel('Efficiency (%)', fontsize=13, fontweight='bold')
    plt.title('MPI Communication Modes: Efficiency vs. Node Count', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=11, framealpha=0.95)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Set x-axis to show all node counts
    node_counts = sorted(df['num_nodes'].unique())
    node_counts = [n for n in node_counts if pd.notna(n)]
    plt.xticks(node_counts, fontsize=11)
    plt.yticks(fontsize=11)
    
    # Add minor ticks and grid
    ax = plt.gca()
    ax.minorticks_on()
    ax.grid(True, which='minor', alpha=0.15, linestyle=':')
    
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"[INFO] Saved efficiency plot to {output_png}")
    plt.close()


def print_efficiency_summary(df: pd.DataFrame):
    """
    Print a summary table of efficiencies by node count and configuration.
    """
    print("\n" + "="*90)
    print("=== Efficiency Summary (%) ===")
    print("="*90)
    
    for num_nodes in sorted(df['num_nodes'].unique()):
        if pd.isna(num_nodes):
            continue
        
        node_data = df[df['num_nodes'] == num_nodes]
        print(f"\n{int(num_nodes)} Nodes:")
        print("-" * 70)
        
        for config in sorted(node_data['config_name'].unique()):
            if pd.isna(config):
                continue
            
            config_data = node_data[node_data['config_name'] == config]
            mean_eff = config_data['efficiency_pct'].mean()
            std_eff = config_data['efficiency_pct'].std()
            count = len(config_data)
            
            print(f"  {config:25s}: {mean_eff:6.2f}% (±{std_eff:.2f}, n={count})")
    
    print("\n" + "="*90)


def main():
    # Use fixed defaults (no CLI args needed)
    results_dir = 'results'
    output_file = 'plots/mpi_efficiency_vs_nodes.png'
    
    # Load results
    print(f"[INFO] Loading MPI results from: {results_dir}")
    df = load_all_results(results_dir)
    
    if df.empty:
        print("[ERROR] No data to analyze. Exiting.", file=sys.stderr)
        sys.exit(1)
    
    print(f"[INFO] Loaded {len(df)} result rows")
    print(f"[INFO] Node counts: {sorted(df['num_nodes'].unique())}")
    print(f"[INFO] Configurations: {sorted([c for c in df['config_name'].unique() if pd.notna(c)])}")
    
    # Create output directory
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Plot
    plot_efficiency(df, output_file)
    
    # Print summary
    print_efficiency_summary(df)
    
    print("[INFO] Done!")


if __name__ == '__main__':
    main()
