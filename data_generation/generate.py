#!/usr/bin/env python3
"""Generate training data for GNN column selection.

Solves multiple random VCSP instances using MILP column selection,
recording bipartite graph data (features + labels) at each CG iteration.

Usage:
    python -m data_generation.generate                     # default: 20 inst, 50 trips
    python -m data_generation.generate --trips 100 --instances 50
    python -m data_generation.generate --trips 200 --instances 30 --output data/vcsp_200

Paper reference: EBSCO Section 4.3 — 100 instances of 400 trips for VCSP training.
"""

import os
import sys
import time
import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

# Ensure project root is in path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from problems.vcsp.instance import VCSPInstance
from data_generation.data_collector import DataCollector


def generate_single_instance(instance_id, num_trips, seed, config, output_base_dir, overwrite=False):
    """Generate data from a single VCSP instance.

    Returns:
        num_data_points: int, number of iterations recorded
        elapsed: float, wall-clock time
        success: bool
    """
    output_dir = os.path.join(output_base_dir, f'instance_{instance_id:04d}')

    # Skip if already done
    metadata_path = os.path.join(output_dir, 'metadata.npz')
    if os.path.exists(metadata_path) and not overwrite:
        try:
            meta = np.load(metadata_path)
            n = int(meta['num_data_points'])
            print(f"  [SKIP] Instance {instance_id} already exists ({n} data points)")
            return {
                'instance_id': instance_id,
                'num_trips': num_trips,
                'seed': seed,
                'num_data_points': n,
                'elapsed': 0.0,
                'success': True,
                'skipped': True,
                'output_dir': output_dir,
                'error': '',
            }
        except Exception:
            pass  # Corrupted, regenerate

    print(f"\n{'#' * 60}")
    print(f"# Instance {instance_id}: {num_trips} trips, seed={seed}")
    print(f"{'#' * 60}")

    try:
        instance = VCSPInstance(num_trips=num_trips, seed=seed)

        collector = DataCollector(instance, config)
        collector.collect(max_iterations=config.get('max_iterations', 300))

        n_points = collector.save(output_dir)
        elapsed = collector.stats['total_time']

        return {
            'instance_id': instance_id,
            'num_trips': num_trips,
            'seed': seed,
            'num_data_points': n_points,
            'elapsed': elapsed,
            'success': True,
            'skipped': False,
            'output_dir': output_dir,
            'error': '',
        }

    except Exception as e:
        print(f"  [FAIL] Instance {instance_id}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'instance_id': instance_id,
            'num_trips': num_trips,
            'seed': seed,
            'num_data_points': 0,
            'elapsed': 0.0,
            'success': False,
            'skipped': False,
            'output_dir': output_dir,
            'error': str(e),
        }


def _write_json(path, payload):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _write_manifest_csv(path, rows):
    fieldnames = [
        'instance_id', 'num_trips', 'seed', 'num_data_points', 'elapsed',
        'success', 'skipped', 'output_dir', 'error',
    ]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in fieldnames})


def main():
    parser = argparse.ArgumentParser(
        description='Generate VCSP training data for GNN column selection'
    )
    parser.add_argument('--trips', type=int, default=50,
                        help='Number of trips per instance (default: 50)')
    parser.add_argument('--instances', type=int, default=20,
                        help='Number of instances to generate (default: 20)')
    parser.add_argument('--start-id', type=int, default=0,
                        help='Starting instance ID (default: 0)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: data_generation/training_data/vcsp_{trips})')
    parser.add_argument('--max-iterations', type=int, default=300,
                        help='Max CG iterations per instance (default: 300)')
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help='MILP penalty epsilon (default: 0.1)')
    parser.add_argument('--additional-pct', type=float, default=0.5,
                        help='Additional column percentage (default: 0.5)')
    parser.add_argument('--skip-first', type=int, default=0,
                        help='Skip first N iterations of each instance (default: 0)')
    parser.add_argument('--min-generated', type=int, default=5,
                        help='Min new columns to record an iteration (default: 5)')
    parser.add_argument('--cost-inflation', type=float, default=3.0,
                        help='Cost inflation factor for initial columns (default: 3.0)')
    parser.add_argument('--artificial-cost', type=float, default=1e7,
                        help='Cost for artificial f/g trip columns (default: 1e7)')
    parser.add_argument('--seed-base', type=int, default=42,
                        help='Base seed for instance generation (default: 42)')
    parser.add_argument('--workers', type=int, default=1,
                        help='Parallel worker processes (default: 1)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Regenerate instances even if metadata.npz exists')
    parser.add_argument('--early-stop-no-improve', type=int, default=None,
                        help='Optional CG early-stop patience; disabled by default')

    args = parser.parse_args()

    # Output directory
    if args.output is None:
        output_base_dir = os.path.join(
            _project_root, 'data_generation', 'training_data', f'vcsp_{args.trips}'
        )
    else:
        output_base_dir = args.output

    os.makedirs(output_base_dir, exist_ok=True)

    # Config for DataCollector
    config = {
        'epsilon': args.epsilon,
        'additional_pct': args.additional_pct,
        'max_iterations': args.max_iterations,
        'min_generated_for_record': args.min_generated,
        'skip_first_n_iterations': args.skip_first,
        'cost_inflation_factor': args.cost_inflation,
        'artificial_cost': args.artificial_cost,
        'early_stop_no_improve': args.early_stop_no_improve,
    }

    run_config = {
        'trips': args.trips,
        'instances': args.instances,
        'start_id': args.start_id,
        'seed_base': args.seed_base,
        'workers': args.workers,
        'overwrite': args.overwrite,
        'output_base_dir': output_base_dir,
        'collector_config': config,
    }
    _write_json(os.path.join(output_base_dir, 'generation_config.json'), run_config)

    print("=" * 60)
    print("VCSP TRAINING DATA GENERATION")
    print("=" * 60)
    print(f"Trips per instance: {args.trips}")
    print(f"Number of instances: {args.instances}")
    print(f"Output directory: {output_base_dir}")
    print(f"Config: {config}")
    print(f"Estimated constraints per instance: ~{args.trips * 2 + 2 * args.trips} eq + bus count")
    print("=" * 60)

    total_data_points = 0
    total_time = 0.0
    success_count = 0

    start_all = time.time()

    instance_args = []
    for instance_id in range(args.start_id, args.start_id + args.instances):
        seed = args.seed_base * 1000 + instance_id * 137 + args.trips
        instance_args.append((instance_id, args.trips, seed, config, output_base_dir, args.overwrite))

    results = []
    if args.workers <= 1:
        for item in instance_args:
            results.append(generate_single_instance(*item))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(generate_single_instance, *item) for item in instance_args]
            for fut in as_completed(futures):
                results.append(fut.result())

    results.sort(key=lambda r: r['instance_id'])
    for row in results:
        if row['success']:
            total_data_points += row['num_data_points']
            total_time += row['elapsed']
            success_count += 1

    total_wall = time.time() - start_all

    # Print overall summary
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Instances: {success_count}/{args.instances} succeeded")
    print(f"Total data points: {total_data_points}")
    print(f"Total compute time: {total_time:.1f}s (wall: {total_wall:.1f}s)")
    print(f"Avg per instance: {total_time / max(1, success_count):.1f}s")
    if total_data_points > 0:
        print(f"Avg data points per instance: {total_data_points / max(1, success_count):.1f}")
    print(f"Output: {output_base_dir}")

    skipped_count = sum(1 for r in results if r['skipped'])
    failed_count = sum(1 for r in results if not r['success'])
    summary = {
        'success_count': success_count,
        'failed_count': failed_count,
        'skipped_count': skipped_count,
        'requested_instances': args.instances,
        'total_data_points': total_data_points,
        'total_compute_time': total_time,
        'total_wall_time': total_wall,
        'avg_time_per_success': total_time / max(1, success_count),
        'avg_data_points_per_success': total_data_points / max(1, success_count),
        'output_base_dir': output_base_dir,
        'results': results,
        'config': run_config,
    }
    _write_json(os.path.join(output_base_dir, 'generation_summary.json'), summary)
    _write_manifest_csv(os.path.join(output_base_dir, 'manifest.csv'), results)
    print(f"Saved manifest: {os.path.join(output_base_dir, 'manifest.csv')}")
    print(f"Saved summary: {os.path.join(output_base_dir, 'generation_summary.json')}")

    # Estimate: paper reports ~7,000 data points from 100 instances of 400 trips
    if total_data_points > 0:
        target = 7000
        print(f"\nPaper reference: ~7,000 data points from 100 instances of 400 trips")
        print(f"Current: {total_data_points} data points from {success_count} instances of {args.trips} trips")
        if total_data_points < target:
            est_instances = int(success_count * target / total_data_points)
            print(f"Estimated instances needed for ~{target} points: ~{est_instances}")


if __name__ == '__main__':
    main()
