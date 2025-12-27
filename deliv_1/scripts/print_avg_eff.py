#!/usr/bin/env python3
"""Print per-thread average efficiency (mean/std/count) in percent.

This uses the same baseline selection logic as `analyze_strong_scaling.py`:
- prefer per-matrix `configuration` containing 'Static' at threads==1 (non-SIMD)
- otherwise use the fastest 1-thread time as fallback

Run from project root: `python scripts/print_avg_eff.py`
"""
import pandas as pd
from pathlib import Path
import os


def build_metrics(results_dir):
    conf_csv = results_dir / 'configurations_results.csv'
    if not conf_csv.exists():
        raise SystemExit(f"Missing {conf_csv}; please run benchmarks first")
    conf = pd.read_csv(conf_csv)
    conf['time_ms'] = pd.to_numeric(conf['time_ms'], errors='coerce')
    if 'exit_code' in conf.columns:
        conf = conf[conf['exit_code'] == 0]
    conf = conf.dropna(subset=['time_ms', 'threads', 'matrix'])
    conf['matrix_name'] = conf['matrix'].apply(lambda x: os.path.basename(str(x).strip('"')).replace('.mtx',''))

    serial_baseline = {}
    for m, g in conf.groupby('matrix_name'):
        one = g[g['threads'] == 1]
        if one.empty:
            continue
        # use fastest 1-thread time as baseline (best serial)
        serial_time = one['time_ms'].min()
        serial_baseline[m] = serial_time

    # Print baseline table to help diagnose baseline selection
    print("\nPer-matrix serial baselines (matrix, baseline_ms, fastest_1thread_ms, candidates_count):")
    for m, g in conf.groupby('matrix_name'):
        one = g[g['threads'] == 1]
        if one.empty:
            continue
        fastest = one['time_ms'].min()
        count = len(one)
        baseline = serial_baseline.get(m, float('nan'))
        print(f"{m:30s} {baseline:10.4f} {fastest:10.4f} {count:4d}")

    rows = []
    for (m, t), group in conf.groupby(['matrix_name', 'threads']):
        if m not in serial_baseline:
            continue
        best_time = group['time_ms'].min()
        if best_time <= 0 or pd.isna(best_time):
            continue
        speedup = serial_baseline[m] / best_time
        efficiency = speedup / float(t) if t > 0 else float('nan')
        rows.append({'matrix': m, 'threads': int(t), 'speedup': float(speedup), 'efficiency': float(efficiency)})

    return pd.DataFrame(rows)


def main():
    project_root = Path(__file__).parent.parent
    results_dir = project_root / 'results'
    metrics = build_metrics(results_dir)
    if metrics.empty:
        print('No metrics built; check results CSV')
        return
    agg = metrics.groupby('threads').agg({'efficiency': ['mean', 'std', 'count']}).reset_index()
    agg.columns = ['threads', 'mean', 'std', 'count']
    agg['mean_pct'] = agg['mean'] * 100.0
    agg['std_pct'] = agg['std'] * 100.0
    agg = agg.sort_values('threads')
    print(f"{'threads':>7} {'mean%':>10} {'std%':>10} {'count':>6}")
    for _, r in agg.iterrows():
        print(f"{int(r['threads']):7d} {r['mean_pct']:10.2f} {r['std_pct']:10.2f} {int(r['count']):6d}")


if __name__ == '__main__':
    main()
