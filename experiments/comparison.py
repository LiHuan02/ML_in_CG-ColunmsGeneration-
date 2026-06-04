#!/usr/bin/env python3
"""VCSP Experiments: Compare NO-S vs MILP-S vs GNN-S.

Based on EBSCO paper Table 5.
Uses the original (non-inflated) initial column strategy for fair comparison.
"""

import time
import os
import csv
import json
import argparse
import numpy as np
from core.column_generation import VCSPSolver
from problems.vcsp.instance import VCSPInstance


def _to_builtin(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    return value


def run_instance(num_trips, seed, selection, config=None, max_iterations=300):
    """Run a single CG instance with a given selection strategy."""
    instance = VCSPInstance(num_trips=num_trips, seed=seed)
    solver = VCSPSolver(instance, config)

    start = time.time()
    columns, col_sol, bus_sol, obj = solver.solve(
        selection_strategy=selection,
        max_iterations=max_iterations,
    )
    elapsed = time.time() - start

    return {
        'num_trips': num_trips,
        'seed': seed,
        'selection': selection,
        'total_time': elapsed,
        'rmp_time': solver.stats['rmp_solve_time'],
        'pp_time': solver.stats['pp_solve_time'],
        'sel_time': solver.stats['selection_time'],
        'iterations': solver.stats['iterations'],
        'final_columns': len(columns),
        'objective': obj,
        'buses': bus_sol,
        'columns_generated': sum(solver.stats['columns_generated']),
        'columns_selected': sum(solver.stats['columns_selected']),
        'iteration_logs': solver.stats['iteration_logs'],
    }


def run_comparison_experiment(trip_sizes=(20, 30, 50), num_instances=3,
                               strategies=('no_selection', 'milp', 'gnn'),
                               gnn_model_path='gnn/models/best_model.pt',
                               norm_stats_path='gnn/models/norm_stats.npz',
                               max_iterations=300,
                               config_overrides=None):
    """Run comparison experiments across multiple instance sizes."""
    # Base config shared by all strategies (matching Table 4)
    config = {
        'n_min_cols': 100,
        'n_max_cols': 5000,
        'n_max_blks': 10,
        'epsilon': 0.1,
        'additional_pct': 0.5,
        'model_path': gnn_model_path,
        'norm_stats_path': norm_stats_path,
        'min_select': 5,  # Lower for small instances (paper uses 100 for 400-trip)
    }
    if config_overrides:
        config.update(config_overrides)

    results = []
    for num_trips in trip_sizes:
        for i in range(num_instances):
            seed = 100 * num_trips + i
            print(f"\n--- Instance: {num_trips} trips, seed={seed} ---")

            for strat in strategies:
                print(f"\n[{strat.upper()}]")
                res = run_instance(num_trips, seed, strat, config, max_iterations=max_iterations)
                results.append(res)

            # Print quick comparison for this instance
            print(f"\n  Comparison:")
            parts = []
            for strat in strategies:
                strat_results = [r for r in results if r['num_trips'] == num_trips
                                 and r['seed'] == seed and r['selection'] == strat]
                if strat_results:
                    r = strat_results[-1]
                    parts.append(f"{strat}: {r['total_time']:.1f}s, {r['iterations']} iters, "
                                 f"{r['final_columns']} cols")
            for p in parts:
                print(f"    {p}")

    return results


def summarize_results(results):
    # Group by trip size and selection
    grouped = {}
    for r in results:
        key = (r['num_trips'], r['selection'])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(r)

    summary_rows = []
    for (num_trips, sel) in sorted(grouped.keys()):
        runs = grouped[(num_trips, sel)]
        summary_rows.append({
            'num_trips': num_trips,
            'selection': sel,
            'runs': len(runs),
            'avg_total_time': float(np.mean([r['total_time'] for r in runs])),
            'std_total_time': float(np.std([r['total_time'] for r in runs])),
            'avg_rmp_time': float(np.mean([r['rmp_time'] for r in runs])),
            'avg_pp_time': float(np.mean([r['pp_time'] for r in runs])),
            'avg_selection_time': float(np.mean([r['sel_time'] for r in runs])),
            'avg_iterations': float(np.mean([r['iterations'] for r in runs])),
            'avg_final_columns': float(np.mean([r['final_columns'] for r in runs])),
            'avg_objective': float(np.mean([r['objective'] for r in runs])),
            'avg_buses': float(np.mean([r['buses'] for r in runs])),
            'avg_columns_generated': float(np.mean([r['columns_generated'] for r in runs])),
            'avg_columns_selected': float(np.mean([r['columns_selected'] for r in runs])),
        })

    reductions = []
    no_data = {}
    other_data = {}
    for r in results:
        if r['selection'] == 'no_selection':
            no_data.setdefault(r['num_trips'], []).append(r)
        else:
            key = (r['num_trips'], r['selection'])
            other_data.setdefault(key, []).append(r)

    for n in sorted(no_data.keys()):
        no_avg = np.mean([r['total_time'] for r in no_data[n]])
        for strat in sorted(set(k[1] for k in other_data if k[0] == n)):
            other_avg = np.mean([r['total_time'] for r in other_data.get((n, strat), [])])
            reductions.append({
                'num_trips': n,
                'selection': strat,
                'baseline_total_time': float(no_avg),
                'selection_total_time': float(other_avg),
                'time_reduction_pct': float((no_avg - other_avg) / no_avg * 100) if no_avg > 0 else 0.0,
            })

    return summary_rows, reductions


def print_summary_table(results):
    """Print results formatted as a table matching the paper's Table 5 format."""
    print("\n" + "=" * 110)
    print("SUMMARY TABLE")
    print("=" * 110)

    summary_rows, reductions = summarize_results(results)

    header = (f"{'Trips':>6} {'Strategy':>14} {'Total':>8} {'RMP':>8} {'PP':>8} "
              f"{'Iters':>6} {'Cols':>8} {'Obj':>10} {'Buses':>6}")
    print(header)
    print("-" * 110)

    for row in summary_rows:
        line = (f"{row['num_trips']:>6} {row['selection']:>14} "
                f"{row['avg_total_time']:>8.1f} {row['avg_rmp_time']:>8.1f} "
                f"{row['avg_pp_time']:>8.1f} {row['avg_iterations']:>6.0f} "
                f"{row['avg_final_columns']:>8.0f} {row['avg_objective']:>10.2f} "
                f"{row['avg_buses']:>6.1f}")
        print(line)

    # Compute average reduction relative to NO-S
    print("\n" + "-" * 110)
    print("AVERAGE TIME REDUCTION (vs NO-S)")
    current_n = None
    for row in reductions:
        if row['num_trips'] != current_n:
            current_n = row['num_trips']
            print(f"  {current_n} trips:")
        print(f"    {row['selection']:>14s}: {row['time_reduction_pct']:+.1f}%")


def save_results(results, output_dir, config):
    os.makedirs(output_dir, exist_ok=True)
    summary_rows, reductions = summarize_results(results)

    raw_path = os.path.join(output_dir, 'results.json')
    with open(raw_path, 'w', encoding='utf-8') as f:
        json.dump(_to_builtin({'config': config, 'results': results}), f, indent=2, ensure_ascii=False)

    flat_path = os.path.join(output_dir, 'results.csv')
    flat_fields = [
        'num_trips', 'seed', 'selection', 'total_time', 'rmp_time', 'pp_time',
        'sel_time', 'iterations', 'final_columns', 'objective', 'buses',
        'columns_generated', 'columns_selected',
    ]
    with open(flat_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=flat_fields)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, '') for k in flat_fields})

    summary_path = os.path.join(output_dir, 'summary.csv')
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else [])
        if summary_rows:
            writer.writeheader()
            writer.writerows(summary_rows)

    reduction_path = os.path.join(output_dir, 'reductions.csv')
    with open(reduction_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(reductions[0].keys()) if reductions else [])
        if reductions:
            writer.writeheader()
            writer.writerows(reductions)

    print(f"\nSaved results to {output_dir}")
    print(f"  Raw JSON: {raw_path}")
    print(f"  Runs CSV: {flat_path}")
    print(f"  Summary CSV: {summary_path}")
    print(f"  Reductions CSV: {reduction_path}")


def parse_csv_ints(text):
    return tuple(int(x.strip()) for x in text.split(',') if x.strip())


def parse_csv_strings(text):
    return tuple(x.strip() for x in text.split(',') if x.strip())


def main():
    parser = argparse.ArgumentParser(description='Compare VCSP CG column selection strategies')
    parser.add_argument('--trip-sizes', type=str, default='40,80',
                        help='Comma-separated trip sizes (default: 40,80)')
    parser.add_argument('--instances', type=int, default=2,
                        help='Instances per trip size (default: 2)')
    parser.add_argument('--strategies', type=str, default='no_selection,milp,gnn',
                        help='Comma-separated strategies')
    parser.add_argument('--max-iterations', type=int, default=300,
                        help='Max CG iterations per run')
    parser.add_argument('--output', type=str, default='experiments/results/latest',
                        help='Directory to save comparison data')
    parser.add_argument('--gnn-model', type=str, default='gnn/models/best_model.pt',
                        help='Path to GNN checkpoint')
    parser.add_argument('--norm-stats', type=str, default='gnn/models/norm_stats.npz',
                        help='Path to normalization stats')
    parser.add_argument('--n-min-cols', type=int, default=100)
    parser.add_argument('--n-max-cols', type=int, default=5000)
    parser.add_argument('--n-max-blks', type=int, default=10)
    parser.add_argument('--min-select', type=int, default=5)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--additional-pct', type=float, default=0.5)
    parser.add_argument('--export-readme-table', dest='export_readme_table',
                        action='store_true', help='Read summary.csv and export a Markdown table into README.md')
    parser.add_argument('--no-export-readme-table', dest='export_readme_table',
                        action='store_false', help='Do not export README table (disable automatic export)')
    parser.set_defaults(export_readme_table=True)
    args = parser.parse_args()

    trip_sizes = parse_csv_ints(args.trip_sizes)
    strategies = parse_csv_strings(args.strategies)
    config_overrides = {
        'n_min_cols': args.n_min_cols,
        'n_max_cols': args.n_max_cols,
        'n_max_blks': args.n_max_blks,
        'min_select': args.min_select,
        'epsilon': args.epsilon,
        'additional_pct': args.additional_pct,
    }

    run_config = {
        'trip_sizes': trip_sizes,
        'num_instances': args.instances,
        'strategies': strategies,
        'max_iterations': args.max_iterations,
        'gnn_model': args.gnn_model,
        'norm_stats': args.norm_stats,
        'config_overrides': config_overrides,
    }

    print("=" * 60)
    print("VCSP EXPERIMENTS: Column Selection Comparison")
    print(f"Trip sizes: {trip_sizes}, Instances per size: {args.instances}")
    print(f"Strategies: {strategies}")
    print("=" * 60)

    # If user asks to export README table from an existing summary.csv, do that and exit.
    if args.export_readme_table:
        summary_path = os.path.join(args.output, 'summary.csv')
        readme_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'README.md')
        readme_path = os.path.normpath(readme_path)
        if not os.path.exists(summary_path):
            print(f"Summary file not found: {summary_path}")
            return
        # read summary.csv
        rows = []
        with open(summary_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
        # build markdown table
        headers = ['num_trips', 'selection', 'runs', 'avg_total_time', 'std_total_time',
                   'avg_iterations', 'avg_final_columns', 'avg_columns_generated',
                   'avg_columns_selected', 'avg_objective', 'avg_buses']
        md_lines = []
        md_lines.append('## 实验与对比')
        md_lines.append('')
        md_lines.append('下表使用 `experiments/results/latest/summary.csv` 的聚合统计（示例运行）。字段说明：`avg_total_time` 单位为秒，`avg_objective` 为最终目标值，其他为均值。')
        md_lines.append('')
        # header
        md_lines.append('|' + ' | '.join(headers) + ' |')
        md_lines.append('|' + '---|' * len(headers))
        for r in rows:
            vals = [r.get(h, '') for h in headers]
            md_lines.append('|' + ' | '.join(vals) + ' |')
        md_lines.append('')
        md_content = '\n'.join(md_lines)

        # replace section in README between '## 实验与对比' and next '## '
        with open(readme_path, 'r', encoding='utf-8') as f:
            text = f.read()

        start_idx = text.find('## 实验与对比')
        if start_idx == -1:
            # append at top if not found
            new_text = md_content + '\n' + text
        else:
            # find next H2
            rest = text[start_idx:]
            next_h2 = rest.find('\n## ', 1)
            if next_h2 == -1:
                new_text = text[:start_idx] + md_content
            else:
                new_text = text[:start_idx] + md_content + rest[next_h2:]

        backup_path = readme_path + '.bak'
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(text)
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(new_text)
        print(f'Updated README at {readme_path} (backup at {backup_path})')
        return

    start = time.time()
    results = run_comparison_experiment(
        trip_sizes=trip_sizes,
        num_instances=args.instances,
        strategies=strategies,
        gnn_model_path=args.gnn_model,
        norm_stats_path=args.norm_stats,
        max_iterations=args.max_iterations,
        config_overrides=config_overrides,
    )
    run_config['elapsed'] = time.time() - start
    print_summary_table(results)
    save_results(results, args.output, run_config)


if __name__ == '__main__':
    main()
