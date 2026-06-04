"""Data collector: runs CG with MILP selection and records bipartite graph data.

At each CG iteration, builds and stores:
  - Column features (12-dim)
  - Constraint features (2-dim)
  - Bipartite edge index (COO format)
  - Labels (from MILP y_p values)

The bipartite graph nodes are:
  - V (columns): basic columns (theta_p > 0 in RMP) + newly generated columns
  - C (constraints): all RMP constraints

Paper reference: EBSCO Section 3.1, Section 4.2
"""

import os
import time
import numpy as np

from problems.vcsp.vcsp_rmp import VCSPRMP
from problems.vcsp.vcsp_pp import VCSPPricingProblem
from problems.vcsp.driver_network import DriverNetwork
from problems.vcsp.column import VCSPColumn
from data_generation.feature_extractor import FeatureExtractor
from data_generation.milp_labeler import MilpLabeler


class DataCollector:
    """Runs CG with MILP column selection and collects training data."""

    def __init__(self, instance, config=None):
        self.instance = instance
        self.config = config or {}

        # Data collection parameters
        self.min_generated_for_record = self.config.get('min_generated_for_record', 5)
        self.skip_first_n_iterations = self.config.get('skip_first_n_iterations', 0)

        # Create driver networks
        self.networks = {
            'I': DriverNetwork('I', instance),
        }

        # Create pricing problems
        self.pricing_problems = {
            dt: VCSPPricingProblem(dt, net, instance)
            for dt, net in self.networks.items()
        }

        # Create RMP
        self.rmp = VCSPRMP(instance, initial_columns=[])

        # Column hash set for duplicate detection
        self.column_hash_set = set()

        # Feature extractor
        self.feature_extractor = FeatureExtractor(instance)

        # MILP labeler
        self.labeler = MilpLabeler(
            instance,
            epsilon=self.config.get('epsilon', 0.1),
            additional_pct=self.config.get('additional_pct', 0.5),
        )

        # Collected data: list of dicts, one per iteration
        self.collected_data = []

        # Stats
        self.stats = {
            'iterations': 0,
            'total_time': 0.0,
            'total_data_points': 0,
        }

    @staticmethod
    def _column_hash(col):
        return (
            tuple(sorted(col.d_trips.keys())),
            tuple(sorted(col.f_trips.keys())),
            tuple(sorted(col.g_trips.keys())),
            tuple(sorted(col.q_times.keys())),
        )

    def _initialize_columns(self):
        """Generate initial columns that leave room for CG improvement.

        Strategy: Create one column per d-trip (at inflated cost) covering ONLY
        the d-trip constraint (no f/g trips). Add separate high-cost artificial
        columns for f_trip and g_trip constraints. This ensures RMP feasibility
        while creating clear opportunities for PP to find cheaper columns that
        combine d_trip + f_trip + g_trip coverage.

        Config keys:
          cost_inflation_factor: multiplier for d-trip column costs (default 3.0)
          artificial_cost: cost for artificial f/g trip columns (default 1e7)
        """
        inflation = self.config.get('cost_inflation_factor', 3.0)
        artificial_cost = self.config.get('artificial_cost', 1e7)
        driver_fixed = self.instance.driver_fixed_cost

        initial_cols = []

        # 1. One column per d-trip: covers d_trip ONLY (no f/g trips), inflated cost
        for trip in self.instance.trips:
            tid = trip['id']
            for d, is_first in [(trip['d_trips'][0], True), (trip['d_trips'][1], False)]:
                drive_to = self.instance.driving_time('depot', d['start_location'])
                drive_back = self.instance.driving_time(d['end_location'], 'depot')
                total_time = (self.instance.sign_on_time + drive_to +
                              d['duration'] + drive_back + self.instance.sign_off_time)
                cost = (total_time * self.instance.cost_per_minute + driver_fixed) * inflation

                col = VCSPColumn(
                    duty_type='I', cost=cost,
                    node_sequence=[], arc_sequence=[],
                    d_trips={d['id']: True},
                    f_trips={}, g_trips={}, q_times={},
                    duty_length=total_time,
                )
                bus_leave = d['start_time'] - drive_to - self.instance.sign_on_time
                bus_return = d['end_time'] + drive_back + self.instance.sign_off_time
                for h in self.instance.departure_times:
                    if bus_leave <= h < bus_return:
                        col.q_times[h] = True
                initial_cols.append(col)

        # 2. Artificial columns for f_trip (bus arrival) constraints: very high cost
        num_d_trips = self.instance.num_d_trips
        for w_id in range(self.instance.num_trips):
            col = VCSPColumn(
                duty_type='I', cost=artificial_cost,
                node_sequence=[], arc_sequence=[],
                d_trips={}, f_trips={w_id: True}, g_trips={}, q_times={},
                duty_length=0,
            )
            initial_cols.append(col)

        # 3. Artificial columns for g_trip (bus departure) constraints: very high cost
        for w_id in range(self.instance.num_trips):
            col = VCSPColumn(
                duty_type='I', cost=artificial_cost,
                node_sequence=[], arc_sequence=[],
                d_trips={}, f_trips={}, g_trips={w_id: True}, q_times={},
                duty_length=0,
            )
            initial_cols.append(col)

        self.rmp.add_columns(initial_cols)
        for col in initial_cols:
            self.column_hash_set.add(self._column_hash(col))
        return initial_cols

    # ------------------------------------------------------------------
    # Build bipartite graph and store data for one iteration
    # ------------------------------------------------------------------
    def _record_iteration(self, iteration, new_columns, labels, rmp_col_sol, duals):
        """Build and record the bipartite graph for this iteration.

        Column nodes = basic columns (theta > 0) + all new unique columns
        Constraint nodes = all RMP constraints

        Args:
            iteration: CG iteration number (0-based)
            new_columns: list of deduplicated new columns
            labels: np.array of shape (len(new_columns),) with 0/1
            rmp_col_sol: list of theta values for all RMP columns
            duals: dual values dict from RMP
        """
        # Identify basic columns (theta_p > small threshold)
        basic_indices = [i for i, v in enumerate(rmp_col_sol) if v > 1e-6]
        basic_columns = [self.rmp.columns[i] for i in basic_indices]

        num_basic = len(basic_columns)
        num_new = len(new_columns)
        num_col_nodes = num_basic + num_new

        if num_new < self.min_generated_for_record:
            return

        # Compute basis constraint support (for incompatibility feature)
        basis_constraint_support = self.feature_extractor.build_basis_constraint_support(
            self.rmp.columns, basic_indices
        )

        # Extract column features
        column_features = np.zeros((num_col_nodes, 12), dtype=np.float32)

        # Basic columns: columnIsNew=0, column_value=theta_p
        for i, col in enumerate(basic_columns):
            column_features[i] = self.feature_extractor.extract_column_features(
                col, is_new=False, theta_value=rmp_col_sol[basic_indices[i]],
                basis_constraint_support=basis_constraint_support,
            )

        # New columns: columnIsNew=1, column_value=0
        for i, col in enumerate(new_columns):
            column_features[num_basic + i] = self.feature_extractor.extract_column_features(
                col, is_new=True, theta_value=0.0,
                basis_constraint_support=basis_constraint_support,
            )

        # New column mask (which column nodes are "new" = prediction targets)
        new_col_mask = np.zeros(num_col_nodes, dtype=bool)
        new_col_mask[num_basic:] = True
        basic_col_mask = np.zeros(num_col_nodes, dtype=bool)
        basic_col_mask[:num_basic] = True

        # Compute constraint degrees (over ALL column nodes)
        all_columns = basic_columns + new_columns
        constraint_degrees = self.feature_extractor.compute_constraint_degrees(all_columns)

        # Extract constraint features
        constraint_features = self.feature_extractor.extract_constraint_features(
            duals, constraint_degrees
        )

        # Build edge index (bipartite: col_idx -> constraint_idx)
        edges = []
        for col_idx, col in enumerate(all_columns):
            coeff_vec = self.feature_extractor.column_coefficient_vector(col)
            constraint_indices = np.nonzero(coeff_vec)[0]
            for c_idx in constraint_indices:
                edges.append([col_idx, int(c_idx)])

        if len(edges) == 0:
            return

        edge_index = np.array(edges, dtype=np.int64).T  # shape (2, n_edges)

        # Store
        data_point = {
            'iteration': iteration,
            'column_features': column_features,
            'constraint_features': constraint_features,
            'edge_index': edge_index,
            'labels': labels.astype(np.int64),
            'new_col_mask': new_col_mask,
            'basic_col_mask': basic_col_mask,
            'num_basic': num_basic,
            'num_new': num_new,
            'num_constraints': self.feature_extractor.num_total_constraints,
            'num_d_trips': self.instance.num_d_trips,
            'num_trips': self.instance.num_trips,
            'num_departures': len(self.instance.departure_times),
            'objective': self.rmp.objective_value,
        }
        self.collected_data.append(data_point)

    # ------------------------------------------------------------------
    # Main CG loop with data collection
    # ------------------------------------------------------------------
    def collect(self, max_iterations=300):
        """Run CG with MILP selection and collect data at each iteration."""
        start_total = time.time()

        # Initialize columns
        self._initialize_columns()

        print(f"\n{'=' * 60}")
        print(f"VCSP Data Collection (MILP-labeled)")
        print(f"Instance: {self.instance.num_trips} trips, {self.instance.num_d_trips} d-trips")
        print(f"Initial columns: {len(self.rmp.columns)}")
        print(f"Constraints: {self.feature_extractor.num_total_constraints}")
        print(f"{'=' * 60}")

        no_improve_count = 0
        last_best_obj = float('inf')
        early_stop_patience = self.config.get('early_stop_no_improve', None)
        total_labels_positive = 0
        total_labels_negative = 0

        for iteration in range(max_iterations):
            # Step 1: Solve RMP
            try:
                col_sol, duals, obj = self.rmp.solve()
            except RuntimeError as e:
                print(f"\nRMP solve failed at iteration {iteration}: {e}")
                break

            print(f"\n--- Iteration {iteration + 1} ---")
            print(f"  RMP: {len(self.rmp.columns)} cols, obj={obj:.2f}, B={self.rmp.bus_solution:.2f}")

            # Check for improvement
            if obj >= last_best_obj - 1e-6:
                no_improve_count += 1
            else:
                no_improve_count = 0
                last_best_obj = obj

            if early_stop_patience and no_improve_count >= early_stop_patience:
                print(f"  Early LP termination (no improvement in {no_improve_count} iters)")
                break

            # Step 2: Solve pricing problems
            all_new_columns = []
            for dt, pp in self.pricing_problems.items():
                pp.set_dual_values(duals)
                new_cols = pp.solve(heuristic=True)
                all_new_columns.extend(new_cols)
                print(f"  PP (Type {dt}): {len(new_cols)} columns")
                if new_cols:
                    print(f"    Best reduced cost: {new_cols[0].reduced_cost:.4f}")

            # Check optimality
            min_rc = min((c.reduced_cost for c in all_new_columns), default=0)
            if min_rc >= -1e-6:
                print(f"\n  *** OPTIMAL reached ***")
                break

            # Deduplicate new columns
            unique_columns = []
            for col in all_new_columns:
                h = self._column_hash(col)
                if h not in self.column_hash_set:
                    self.column_hash_set.add(h)
                    unique_columns.append(col)

            # If no unique columns found but PP generated columns, CG has converged
            if len(unique_columns) == 0:
                print(f"  0 unique columns (all duplicates) - CG converged")
                break

            # Step 3: Solve MILP to get labels
            selected_columns, labels, theta_new = self.labeler.label(
                self.rmp.columns, unique_columns
            )

            n_pos = int(labels.sum())
            n_neg = len(labels) - n_pos
            total_labels_positive += n_pos
            total_labels_negative += n_neg
            print(f"  Labels: {n_pos} positive, {n_neg} negative "
                  f"({100 * n_pos / max(1, len(labels)):.1f}% positive)")

            # Step 4: Record iteration data (before adding columns to RMP)
            if iteration >= self.skip_first_n_iterations:
                self._record_iteration(iteration, unique_columns, labels, col_sol, duals)

            # Step 5: Add selected columns to RMP
            if selected_columns:
                self.rmp.add_columns(selected_columns)
                print(f"  Added: {len(selected_columns)} columns to RMP")

            if len(all_new_columns) == 0:
                break

        else:
            print(f"\nReached max iterations ({max_iterations})")

        self.stats['iterations'] = len(self.collected_data)
        self.stats['total_time'] = time.time() - start_total
        self.stats['total_data_points'] = len(self.collected_data)
        self.stats['total_positive_labels'] = total_labels_positive
        self.stats['total_negative_labels'] = total_labels_negative

        # Print summary
        print(f"\n{'=' * 60}")
        print(f"Data Collection Summary")
        print(f"{'=' * 60}")
        print(f"Total time: {self.stats['total_time']:.2f}s")
        print(f"Data points collected: {self.stats['total_data_points']}")
        print(f"Total labels: {total_labels_positive} positive, "
              f"{total_labels_negative} negative "
              f"({100 * total_labels_positive / max(1, total_labels_positive + total_labels_negative):.1f}% pos)")
        print(f"Final RMP columns: {len(self.rmp.columns)}")
        print(f"Final objective: {self.rmp.objective_value:.2f}")

        return self.collected_data

    # ------------------------------------------------------------------
    # Save collected data to disk
    # ------------------------------------------------------------------
    def save(self, output_dir):
        """Save all collected data to disk as .npz files.

        Directory structure:
          output_dir/
            iter_0000.npz
            iter_0001.npz
            ...
            metadata.npz  (instance info, stats)
        """
        os.makedirs(output_dir, exist_ok=True)

        for i, dp in enumerate(self.collected_data):
            filepath = os.path.join(output_dir, f'iter_{i:04d}.npz')
            np.savez_compressed(
                filepath,
                column_features=dp['column_features'],
                constraint_features=dp['constraint_features'],
                edge_index=dp['edge_index'],
                labels=dp['labels'],
                new_col_mask=dp['new_col_mask'],
                basic_col_mask=dp['basic_col_mask'],
                iteration=dp['iteration'],
                num_basic=dp['num_basic'],
                num_new=dp['num_new'],
                num_constraints=dp['num_constraints'],
                num_d_trips=dp['num_d_trips'],
                num_trips=dp['num_trips'],
                num_departures=dp['num_departures'],
                objective=dp['objective'],
            )

        # Save metadata
        metadata_path = os.path.join(output_dir, 'metadata.npz')
        np.savez(
            metadata_path,
            num_data_points=len(self.collected_data),
            num_d_trips=self.instance.num_d_trips,
            num_trips=self.instance.num_trips,
            num_departures=len(self.instance.departure_times),
            total_time=self.stats['total_time'],
            total_positive_labels=self.stats.get('total_positive_labels', 0),
            total_negative_labels=self.stats.get('total_negative_labels', 0),
        )

        print(f"\nSaved {len(self.collected_data)} data points to {output_dir}/")
        return len(self.collected_data)
