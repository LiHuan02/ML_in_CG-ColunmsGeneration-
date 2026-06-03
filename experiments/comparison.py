#!/usr/bin/env python3
"""VCSP Experiments: Compare NO-S vs MILP-S vs GNN-S.

Based on EBSCO paper Table 5.
Uses the original (non-inflated) initial column strategy for fair comparison.
"""

import time
import numpy as np
from core.column_generation import VCSPSolver
from problems.vcsp.instance import VCSPInstance


def run_instance(num_trips, seed, selection, config=None, gnn_config=None):
    """Run a single CG instance with a given selection strategy."""
    instance = VCSPInstance(num_trips=num_trips, seed=seed)
    solver = VCSPSolver(instance, config)

    start = time.time()
    columns, col_sol, bus_sol, obj = solver.solve(
        selection_strategy=selection,
        max_iterations=300,
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
    }


def run_comparison_experiment(trip_sizes=(20, 30, 50), num_instances=3,
                               strategies=('no_selection', 'milp', 'gnn'),
                               gnn_model_path='gnn/models/best_model.pt'):
    """Run comparison experiments across multiple instance sizes."""
    # Base config shared by all strategies (matching Table 4)
    config = {
        'n_min_cols': 100,
        'n_max_cols': 5000,
        'n_max_blks': 10,
        'epsilon': 0.1,
        'additional_pct': 0.5,
        'model_path': gnn_model_path,
        'min_select': 5,  # Lower for small instances (paper uses 100 for 400-trip)
    }

    results = []
    for num_trips in trip_sizes:
        for i in range(num_instances):
            seed = 100 * num_trips + i
            print(f"\n--- Instance: {num_trips} trips, seed={seed} ---")

            for strat in strategies:
                print(f"\n[{strat.upper()}]")
                res = run_instance(num_trips, seed, strat, config)
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


def print_summary_table(results):
    """Print results formatted as a table matching the paper's Table 5 format."""
    print("\n" + "=" * 110)
    print("SUMMARY TABLE")
    print("=" * 110)

    # Group by trip size and selection
    grouped = {}
    for r in results:
        key = (r['num_trips'], r['selection'])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(r)

    header = (f"{'Trips':>6} {'Strategy':>14} {'Total':>8} {'RMP':>8} {'PP':>8} "
              f"{'Iters':>6} {'Cols':>8} {'Obj':>10} {'Buses':>6}")
    print(header)
    print("-" * 110)

    for (num_trips, sel) in sorted(grouped.keys()):
        runs = grouped[(num_trips, sel)]
        avg_time = np.mean([r['total_time'] for r in runs])
        avg_rmp = np.mean([r['rmp_time'] for r in runs])
        avg_pp = np.mean([r['pp_time'] for r in runs])
        avg_iters = np.mean([r['iterations'] for r in runs])
        avg_cols = np.mean([r['final_columns'] for r in runs])
        avg_obj = np.mean([r['objective'] for r in runs])
        avg_buses = np.mean([r['buses'] for r in runs])

        line = (f"{num_trips:>6} {sel:>14} {avg_time:>8.1f} {avg_rmp:>8.1f} {avg_pp:>8.1f} "
                f"{avg_iters:>6.0f} {avg_cols:>8.0f} {avg_obj:>10.2f} {avg_buses:>6.1f}")
        print(line)

    # Compute average reduction relative to NO-S
    print("\n" + "-" * 110)
    print("AVERAGE TIME REDUCTION (vs NO-S)")
    no_data = {}
    other_data = {}
    for r in results:
        if r['selection'] == 'no_selection':
            no_data[r['num_trips']] = no_data.get(r['num_trips'], []) + [r]
        else:
            key = (r['num_trips'], r['selection'])
            other_data[key] = other_data.get(key, []) + [r]

    for n in sorted(no_data.keys()):
        no_avg = np.mean([r['total_time'] for r in no_data[n]])
        print(f"  {n} trips:")
        for strat in sorted(set(k[1] for k in other_data if k[0] == n)):
            other_avg = np.mean([r['total_time'] for r in other_data.get((n, strat), [])])
            if no_avg > 0:
                reduction = (no_avg - other_avg) / no_avg * 100
            else:
                reduction = 0
            print(f"    {strat:>14s}: {reduction:+.1f}%")


if __name__ == '__main__':
    trip_sizes = (40, 80)
    num_instances = 2

    print("=" * 60)
    print("VCSP EXPERIMENTS: NO-S vs MILP-S vs GNN-S")
    print(f"Trip sizes: {trip_sizes}, Instances per size: {num_instances}")
    print("=" * 60)

    results = run_comparison_experiment(trip_sizes, num_instances)
    print_summary_table(results)
