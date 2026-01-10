#!/usr/bin/env python3
"""
Analyze MPI communication configuration results and plot efficiency vs node count.

Reads *nodes.csv files from results directory, aggregates efficiency metrics,
and generates visualization and summary reports.
"""

import argparse
import glob
import os
import sys
from typing import List

import pandas as pd
import matplotlib.pyplot as plt


def read_results(csv_path: str) -> pd.DataFrame:
    """
    Read a CSV results file robustly, handling messy formatting.
    The CSV may have broken headers due to formatting in C code.
    We parse it by reading raw lines and extracting by position.
    """
    try:
        # Read raw lines
        with open(csv_path, 'r') as f:
            lines = [line.rstrip('\n') for line in f.readlines() if line.strip()]
    except Exception as e:
        print(f"[WARN] Failed to read {csv_path}: {e}", file=sys.stderr)
        return pd.DataFrame()
    
    if not lines:
        return pd.DataFrame()
    
    # Reconstruct header by joining first line(s) until we find all expected fields
    header_lines = []
    full_header = ""
    for i, line in enumerate(lines):
        header_lines.append(line)
        full_header = ",".join(header_lines)
        # Check if we have enough commas to likely be complete
        if full_header.count(',') >= 16:
            break
    
    # Split header by comma and find column positions
    headers = [h.strip() for h in full_header.split(',')]
    
    # Build a mapping of expected column names to positions (best effort)
    expected_cols = [
        'num_nodes', 'matrix', 'rows', 'cols', 'nnz', 'density_pct', 
        'config_name', 'avg_time_ms', 'std_dev_ms', 'min_time_ms', 'max_time_ms',
        'comm_time_ms', 'compute_time_ms', 'speedup', 'efficiency_pct', 'iterations', 'notes'
    ]
    
    # Find positions of key columns
    col_pos = {}
    for col in expected_cols:
        for idx, h in enumerate(headers):
            if col in h.lower() or h.lower() in col.lower():
                col_pos[col] = idx
                break
    
    # If we couldn't find all columns by name, use fixed positions
    if len(col_pos) < len(expected_cols):
        # Assume fixed structure: skip header line(s), then data rows
        # Expected (with empty columns): num_nodes,matrix,rows,cols,nnz,density_pct,config_name,,avg_time_ms,std_dev_ms,min_time_ms,max_time_ms,,comm_time_ms,compute_time_ms,speedup,efficiency_pct,iterations,notes
        # Actual positions after parsing (accounting for empty columns):
        col_pos = {
            'num_nodes': 0, 'matrix': 1, 'rows': 2, 'cols': 3, 'nnz': 4,
            'density_pct': 5, 'config_name': 6, 'avg_time_ms': 8, 'std_dev_ms': 9,
            'min_time_ms': 10, 'max_time_ms': 11, 'comm_time_ms': 13, 'compute_time_ms': 14,
            'speedup': 15, 'efficiency_pct': 16, 'iterations': 17, 'notes': 18
        }
    
    # Parse data rows
    data_rows = []
    for line in lines[len(header_lines):]:  # Skip header lines
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(',')]
        
        row = {}
        for col, pos in col_pos.items():
            if pos < len(parts):
                val = parts[pos]
                # Clean up value
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
    
    # Basic cleanup
    if 'num_nodes' in df.columns:
        df['num_nodes'] = df['num_nodes'].fillna(0).astype(int)
    if 'config_name' in df.columns:
        df['config_name'] = df['config_name'].fillna('unknown').astype(str).str.strip()
    
    df['source_csv'] = os.path.basename(csv_path)
    return df


def load_all_results(results_dir: str) -> pd.DataFrame:
    """Load all *nodes.csv files from results directory."""
    # First try the subdirectory structure
    pattern = os.path.join(results_dir, "test_results_Xnodes", "*nodes.csv")
    paths = sorted(glob.glob(pattern))
    
    # Fall back to direct results directory if subdirectory not found
    if not paths:
        pattern = os.path.join(results_dir, "*nodes.csv")
        paths = sorted(glob.glob(pattern))
    
    if not paths:
        print(f"[ERROR] No CSV files matching {pattern}", file=sys.stderr)
        return pd.DataFrame()

    frames: List[pd.DataFrame] = []
    for p in paths:
        df = read_results(p)
        if df.empty:
            print(f"[WARN] No rows in {p} (or parsing failed). Skipping.", file=sys.stderr)
            continue
        frames.append(df)

    if not frames:
        print("[ERROR] No valid CSV rows loaded.", file=sys.stderr)
        return pd.DataFrame()

    all_df = pd.concat(frames, ignore_index=True)
    # Filter rows that actually have efficiency data
    if 'efficiency_pct' in all_df.columns:
        all_df = all_df[all_df['efficiency_pct'].notnull()]
    return all_df


def summarize_efficiency(all_df: pd.DataFrame, per_matrix: bool = False) -> pd.DataFrame:
    """
    Build a summary DataFrame of average efficiency by node count and configuration.
    If per_matrix is True, split lines per matrix; otherwise, average across matrices.
    """
    required = ['num_nodes', 'config_name', 'efficiency_pct']
    for col in required:
        if col not in all_df.columns:
            raise ValueError(f"Missing required column: {col}")

    group_cols = ['num_nodes', 'config_name']
    if per_matrix:
        if 'matrix' not in all_df.columns:
            print("[WARN] per_matrix requested but 'matrix' column missing. Falling back to averaging.", file=sys.stderr)
        else:
            group_cols.append('matrix')

    summary = (all_df
               .groupby(group_cols, dropna=False)['efficiency_pct']
               .agg(['mean', 'count', 'std'])
               .reset_index()
               .rename(columns={'mean': 'avg_efficiency_pct', 'count': 'samples', 'std': 'std_efficiency'}))

    # Sort for plotting
    summary = summary.sort_values(group_cols).reset_index(drop=True)
    return summary


def plot_efficiency(summary: pd.DataFrame, out_path: str, per_matrix: bool = False):
    """
    Plot efficiency vs node count.
    - If per_matrix=False: one line per config_name (averaged across matrices)
    - If per_matrix=True: one line per (config_name, matrix)
    """
    if summary.empty:
        print("[ERROR] Summary is empty; nothing to plot.", file=sys.stderr)
        return

    plt.figure(figsize=(12, 7))
    ax = plt.gca()

    # Build series
    series_key = ['config_name']
    label_fmt = "{config}"
    if per_matrix and 'matrix' in summary.columns:
        series_key = ['config_name', 'matrix']
        label_fmt = "{config} | {matrix}"

    for key_vals, sub in summary.groupby(series_key):
        if isinstance(key_vals, tuple):
            label = label_fmt.format(config=key_vals[0], matrix=key_vals[1] if len(key_vals) > 1 else "")
        else:
            label = label_fmt.format(config=key_vals)

        # Plot avg efficiency vs node count
        sub_sorted = sub.sort_values('num_nodes')
        ax.plot(sub_sorted['num_nodes'], sub_sorted['avg_efficiency_pct'],
                marker='o', linewidth=2, markersize=8, label=label)

    ax.set_title("MPI Communication Efficiency vs Node Count", fontsize=14, fontweight='bold')
    ax.set_xlabel("Number of MPI Ranks (Node Count)", fontsize=12)
    ax.set_ylabel("Efficiency (%)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='best', fontsize=9)
    
    # Set x-axis to integer ticks only
    if 'num_nodes' in summary.columns:
        x_vals = sorted(summary['num_nodes'].unique())
        if len(x_vals) > 0:
            ax.set_xticks(x_vals)
    
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"[INFO] Saved plot to {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze MPI configuration results and plot efficiency vs node count."
    )
    parser.add_argument("--results-dir", default="results",
                        help="Directory containing *nodes.csv files.")
    parser.add_argument("--output", default="plots/mpi_efficiency.png",
                        help="Path to save the efficiency plot.")
    parser.add_argument("--per-matrix", action="store_true",
                        help="Plot separate lines per matrix (default averages across matrices).")
    parser.add_argument("--summary-out", default=None,
                        help="Optional CSV path to save summary table.")
    args = parser.parse_args()

    print("[INFO] Loading MPI results from:", args.results_dir)
    df = load_all_results(args.results_dir)
    if df.empty:
        print("[ERROR] No data to analyze. Exiting.", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Loaded {len(df)} result rows")
    print(f"[INFO] Node counts: {sorted(df['num_nodes'].unique())}")
    print(f"[INFO] Configurations: {sorted(df['config_name'].unique())}")

    summary = summarize_efficiency(df, per_matrix=args.per_matrix)

    # Print concise summary to stdout
    print("\n" + "="*80)
    print("=== Efficiency Summary (averaged across samples per config per node count) ===")
    print("="*80)
    cols_to_print = ['num_nodes', 'config_name', 'avg_efficiency_pct', 'samples']
    if args.per_matrix and 'matrix' in summary.columns:
        cols_to_print = ['num_nodes', 'config_name', 'matrix', 'avg_efficiency_pct', 'samples']
    print(summary[cols_to_print].to_string(index=False))
    print("="*80)

    if args.summary_out:
        os.makedirs(os.path.dirname(args.summary_out) or '.', exist_ok=True)
        summary.to_csv(args.summary_out, index=False)
        print(f"[INFO] Saved summary CSV to {args.summary_out}")

    plot_efficiency(summary, args.output, per_matrix=args.per_matrix)
    print("[INFO] Done!")


if __name__ == "__main__":
    main()
