import time
import numpy as np
from problems.vcsp.vcsp_rmp import VCSPRMP
from problems.vcsp.vcsp_pp import VCSPPricingProblem
from problems.vcsp.driver_network import DriverNetwork
from selection.no_selection import NoSelectionCG
from selection.milp_selection import MILPSelectionCG
from selection.gnn_selection import GNNSelectionCG


class VCSPSolver:
    """VCSP Column Generation Solver.

    Orchestrates the column generation loop:
    RMP -> dual values -> PP(s) -> columns -> selection -> RMP
    """

    def __init__(self, instance, config=None):
        self.instance = instance
        self.config = config or {}

        # Create driver networks (one per duty type)
        self.networks = {
            'I': DriverNetwork('I', instance),
            # 'II': DriverNetwork('II', instance),  # Future: duty type II
        }

        # Create pricing problems
        self.pricing_problems = {
            dt: VCSPPricingProblem(dt, net, instance)
            for dt, net in self.networks.items()
        }

        # Create RMP (initially empty, will add initial columns)
        self.rmp = VCSPRMP(instance, initial_columns=[])

        # Column hash set for duplicate detection
        self.column_hash_set = set()

        # Stats tracking
        self.stats = {
            'iterations': 0,
            'rmp_solve_time': 0.0,
            'pp_solve_time': 0.0,
            'selection_time': 0.0,
            'total_time': 0.0,
            'columns_generated': [],
            'columns_selected': [],
            'rmp_column_counts': [],
            'objective_values': [],
            'iteration_logs': [],
        }

        # Column removal parameters
        self.n_min_cols = self.config.get('n_min_cols', 100)
        self.n_max_cols = self.config.get('n_max_cols', 1000)

    def _initialize_columns(self):
        """Generate initial columns (heuristic: one duty per d-trip)."""
        initial_cols = []

        from problems.vcsp.column import VCSPColumn

        for trip in self.instance.trips:
            tid = trip['id']
            d0 = trip['d_trips'][0]  # start -> relief
            d1 = trip['d_trips'][1]  # relief -> end

            # Column for d-trip 0: depot -> start_of_trip -> relief -> depot
            drive_to_0 = self.instance.driving_time('depot', d0['start_location'])
            drive_back_0 = self.instance.driving_time(d0['end_location'], 'depot')
            total_time_0 = (self.instance.sign_on_time + drive_to_0 +
                           d0['duration'] + drive_back_0 + self.instance.sign_off_time)
            cost_0 = total_time_0 * self.instance.cost_per_minute

            driver_fixed = self.instance.driver_fixed_cost

            col0 = VCSPColumn(
                duty_type='I',
                cost=cost_0 + driver_fixed,
                node_sequence=[], arc_sequence=[],
                d_trips={d0['id']: True},
                f_trips={tid: True},
                g_trips={},
                q_times={},
            )
            bus_leave_0 = d0['start_time'] - drive_to_0 - self.instance.sign_on_time
            bus_return_0 = d0['end_time'] + drive_back_0 + self.instance.sign_off_time
            for h in self.instance.departure_times:
                if bus_leave_0 <= h < bus_return_0:
                    col0.q_times[h] = True
            initial_cols.append(col0)

            drive_to_1 = self.instance.driving_time('depot', d1['start_location'])
            drive_back_1 = self.instance.driving_time(d1['end_location'], 'depot')
            total_time_1 = (self.instance.sign_on_time + drive_to_1 +
                           d1['duration'] + drive_back_1 + self.instance.sign_off_time)
            cost_1 = total_time_1 * self.instance.cost_per_minute

            col1 = VCSPColumn(
                duty_type='I',
                cost=cost_1 + driver_fixed,
                node_sequence=[], arc_sequence=[],
                d_trips={d1['id']: True},
                f_trips={},
                g_trips={tid: True},
                q_times={},
            )
            bus_leave_1 = d1['start_time'] - drive_to_1 - self.instance.sign_on_time
            bus_return_1 = d1['end_time'] + drive_back_1 + self.instance.sign_off_time
            for h in self.instance.departure_times:
                if bus_leave_1 <= h < bus_return_1:
                    col1.q_times[h] = True
            initial_cols.append(col1)

        self.rmp.add_columns(initial_cols)
        for col in initial_cols:
            self.column_hash_set.add(self._column_hash(col))
        return initial_cols

    def _remove_columns(self):
        """Remove columns from RMP when it exceeds max size.
        Keep the n_min_cols most promising columns.
        """
        if len(self.rmp.columns) <= self.n_max_cols:
            return

        # Sort by reduced cost (using last known dual values)
        # Keep only n_min_cols
        n_remove = len(self.rmp.columns) - self.n_min_cols
        # Simple strategy: keep the first n_min_cols (added earlier)
        self.rmp.columns = self.rmp.columns[n_remove:]

    @staticmethod
    def _column_hash(col):
        """Hash a column by its full master-problem coefficient signature."""
        return (
            tuple(sorted(col.d_trips.keys())),
            tuple(sorted(col.f_trips.keys())),
            tuple(sorted(col.g_trips.keys())),
            tuple(sorted(col.q_times.keys())),
        )

    def solve(self, selection_strategy='no_selection', max_iterations=500, **kwargs):
        """Run the column generation algorithm."""
        start_total = time.time()

        # Initialize columns
        self._initialize_columns()

        # Setup selection strategy
        if selection_strategy == 'no_selection':
            selector = NoSelectionCG(self.rmp, None, self.config)
        elif selection_strategy == 'milp':
            selector = MILPSelectionCG(self.rmp, None, self.config)
        elif selection_strategy == 'gnn':
            selector = GNNSelectionCG(self.rmp, None, self.config)
        else:
            raise ValueError(f"Unknown selection strategy: {selection_strategy}")

        print(f"\n{'=' * 60}")
        print(f"VCSP Column Generation ({selection_strategy})")
        print(f"Instance: {self.instance.num_trips} trips, {self.instance.num_d_trips} d-trips")
        print(f"Initial columns: {len(self.rmp.columns)}")
        print(f"{'=' * 60}")

        no_improve_count = 0
        last_best_obj = float('inf')
        early_stop_patience = self.config.get('early_stop_no_improve', None)

        for iteration in range(max_iterations):
            iter_start = time.time()

            # Step 1: Solve RMP
            t0 = time.time()
            try:
                col_sol, duals, obj = self.rmp.solve()
            except RuntimeError as e:
                print(f"\nRMP solve failed at iteration {iteration}: {e}")
                break
            rmp_time = time.time() - t0

            self.stats['rmp_column_counts'].append(len(self.rmp.columns))
            self.stats['objective_values'].append(obj)

            print(f"\n--- Iteration {iteration + 1} ---")
            print(f"  RMP: {len(self.rmp.columns)} columns, obj={obj:.4f}, B={self.rmp.bus_solution:.2f}")
            print(f"  RMP solve time: {rmp_time:.3f}s")

            # Check for improvement
            if obj >= last_best_obj - 1e-6:
                no_improve_count += 1
            else:
                no_improve_count = 0
                last_best_obj = obj
            if early_stop_patience and no_improve_count >= early_stop_patience:
                print(f"  ! No improvement in {no_improve_count} iterations - early LP termination")
                break

            # Step 2: Solve pricing problems
            t0 = time.time()
            all_new_columns = []
            for dt, pp in self.pricing_problems.items():
                pp.set_dual_values(duals)
                new_cols = pp.solve(heuristic=True)
                all_new_columns.extend(new_cols)
                print(f"  PP (Type {dt}): {len(new_cols)} columns")
                if new_cols:
                    print(f"    Best reduced cost: {new_cols[0].reduced_cost:.4f}")
            pp_time = time.time() - t0

            self.stats['columns_generated'].append(len(all_new_columns))

            # Check for optimality (use all_new_columns before dedup)
            min_rc = min((c.reduced_cost for c in all_new_columns), default=0)
            if min_rc >= -1e-6:
                print(f"\n  *** OPTIMAL reached ***")
                print(f"  Final objective: {obj:.4f}")
                break

            # Remove columns already in RMP by signature
            unique_columns = []
            for col in all_new_columns:
                h = self._column_hash(col)
                if h not in self.column_hash_set:
                    self.column_hash_set.add(h)
                    unique_columns.append(col)

            if not unique_columns:
                print("  0 unique columns (all duplicates) - CG converged")
                break

            # Select columns
            t0 = time.time()
            selected_columns = selector.select(unique_columns)
            sel_time = time.time() - t0

            self.stats['columns_selected'].append(len(selected_columns))
            print(f"  Selection: {len(selected_columns)} / {len(unique_columns)} unique columns")

            # Step 5: Add selected columns to RMP
            if selected_columns:
                self.rmp.add_columns(selected_columns)

            # Step 5: Remove columns if too large
            self._remove_columns()

            # Accumulate stats
            iter_time = time.time() - iter_start
            self.stats['rmp_solve_time'] += rmp_time
            self.stats['pp_solve_time'] += pp_time
            self.stats['selection_time'] += sel_time

            # Log
            self.stats['iteration_logs'].append({
                'iteration': iteration,
                'obj': obj,
                'bus_solution': self.rmp.bus_solution,
                'num_rmp_columns': len(self.rmp.columns),
                'columns_generated': len(all_new_columns),
                'columns_selected': len(selected_columns),
                'min_reduced_cost': min_rc,
                'rmp_time': rmp_time,
                'pp_time': pp_time,
                'sel_time': sel_time,
                'total_time': iter_time,
            })

            # Only stop when no negative reduced cost columns exist
            if len(all_new_columns) == 0:
                break

        else:
            print(f"\nReached max iterations ({max_iterations})")

        self.stats['total_time'] = time.time() - start_total
        self.stats['iterations'] = len(self.stats['iteration_logs'])

        # Print summary
        print(f"\n{'=' * 60}")
        print(f"CG Summary ({selection_strategy})")
        print(f"{'=' * 60}")
        print(f"Iterations: {self.stats['iterations']}")
        print(f"Total time: {self.stats['total_time']:.2f}s")
        print(f"  RMP solve: {self.stats['rmp_solve_time']:.2f}s")
        print(f"  PP solve: {self.stats['pp_solve_time']:.2f}s")
        print(f"  Selection: {self.stats['selection_time']:.2f}s")
        print(f"Final columns in RMP: {len(self.rmp.columns)}")
        print(f"Final objective: {self.rmp.objective_value:.4f}")
        print(f"Final B: {self.rmp.bus_solution:.2f}")

        return (
            self.rmp.columns,
            self.rmp.column_solution,
            self.rmp.bus_solution,
            self.rmp.objective_value,
        )
