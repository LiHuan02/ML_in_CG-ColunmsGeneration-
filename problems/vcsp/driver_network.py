import numpy as np


class DriverNetwork:
    """Driver network G_u for a specific duty type.

    Simplified: implements duty type I (one-piece, no break).
    Network structure based on Haase, Desaulniers, Desrosiers (2001) Section 2.1.

    Node types:
    - source (0), sink (1)
    - start_of_trip (one per trip)
    - end_of_trip (one per trip)
    - relief (intermediate relief points per trip)

    Arc types:
    - start_of_duty: source -> trip start/relief/end
    - end_of_duty: trip start/relief/end -> sink
    - d-trip: consecutive nodes on same trip
    - inter-trip: between end_of_trip and start_of_trip (driving or walking)
    """

    NODE_SOURCE = 0
    NODE_SINK = 1
    NODE_START_OF_TRIP = 2  # base index, actual = base + trip_id
    NODE_RELIEF = 2 + 10000  # base index
    NODE_END_OF_TRIP = 2 + 20000  # base index

    ARC_START_OF_DUTY = 'start_of_duty'
    ARC_END_OF_DUTY = 'end_of_duty'
    ARC_D_TRIP = 'd_trip'
    ARC_INTER_TRIP_DRIVING = 'inter_trip_driving'
    ARC_INTER_TRIP_WALKING = 'inter_trip_walking'

    def __init__(self, duty_type, instance):
        self.duty_type = duty_type  # 'I' or 'II'
        self.instance = instance
        self.nodes = {}  # node_id -> node_info
        self.arcs = {}  # (from, to) -> arc_info
        self._build_network()

    def _node_id(self, node_type, trip_id=0, sub_type='start'):
        """Generate unique node ID from node type and trip info."""
        if node_type == 'source':
            return self.NODE_SOURCE
        elif node_type == 'sink':
            return self.NODE_SINK
        elif node_type == 'start_of_trip':
            return self.NODE_START_OF_TRIP + trip_id
        elif node_type == 'relief':
            return self.NODE_RELIEF + trip_id
        elif node_type == 'end_of_trip':
            return self.NODE_END_OF_TRIP + trip_id
        else:
            raise ValueError(f"Unknown node type: {node_type}")

    def _build_network(self):
        """Build the complete driver network for this duty type."""
        instance = self.instance

        # --- Create nodes ---

        for trip in instance.trips:
            tid = trip['id']
            start_nid = self._node_id('start_of_trip', tid)
            relief_nid = self._node_id('relief', tid)
            end_nid = self._node_id('end_of_trip', tid)

            self.nodes[start_nid] = {
                'type': 'start_of_trip',
                'trip_id': tid,
                'time': trip['start_time'],
                'location': trip['start_location'],
            }
            self.nodes[relief_nid] = {
                'type': 'relief',
                'trip_id': tid,
                'time': trip['d_trips'][0]['end_time'],  # end of first d-trip
                'location': trip['mid_location'],
            }
            self.nodes[end_nid] = {
                'type': 'end_of_trip',
                'trip_id': tid,
                'time': trip['end_time'],
                'location': trip['end_location'],
            }

        self.nodes[self.NODE_SOURCE] = {'type': 'source', 'time': 0, 'location': 'depot'}
        self.nodes[self.NODE_SINK] = {'type': 'sink', 'time': float('inf'), 'location': 'depot'}

        # --- Create arcs ---

        # D-trip arcs (driving segments within a trip)
        for trip in instance.trips:
            tid = trip['id']
            d1 = trip['d_trips'][0]
            d2 = trip['d_trips'][1]

            # First d-trip: start_of_trip -> relief
            start_nid = self._node_id('start_of_trip', tid)
            relief_nid = self._node_id('relief', tid)
            self._add_d_trip_arc(start_nid, relief_nid, d1, instance.cost_per_minute)

            # Second d-trip: relief -> end_of_trip
            end_nid = self._node_id('end_of_trip', tid)
            self._add_d_trip_arc(relief_nid, end_nid, d2, instance.cost_per_minute)

        # Start of duty arcs: source -> feasible first work nodes.
        # Starting at an end_of_trip node would create duties with no d-trip service.
        for trip in instance.trips:
            tid = trip['id']
            for node_type in ['start_of_trip', 'relief']:
                nid = self._node_id(node_type, tid)
                self._add_start_of_duty_arc(nid, instance)

        # End of duty arcs: feasible last work nodes.
        # Ending at a start_of_trip node would create duties with no d-trip service.
        for trip in instance.trips:
            tid = trip['id']
            for node_type in ['relief', 'end_of_trip']:
                nid = self._node_id(node_type, tid)
                self._add_end_of_duty_arc(nid, instance)

        # Inter-trip arcs: end_of_trip / relief -> start_of_trip / relief
        for trip_i in instance.trips:
            for trip_j in instance.trips:
                if trip_i is trip_j:
                    continue
                self._add_inter_trip_arcs(trip_i, trip_j, instance)

        # Also: relief -> start_of_trip (need for piece of work continuity)
        for trip_i in instance.trips:
            for trip_j in instance.trips:
                if trip_i is trip_j:
                    continue
                self._add_inter_trip_from_relief(trip_i, trip_j, instance)

    def _add_d_trip_arc(self, from_nid, to_nid, d_trip, cost_per_min):
        """Add a d-trip arc."""
        cost = d_trip['duration'] * cost_per_min
        self.arcs[(from_nid, to_nid)] = {
            'type': self.ARC_D_TRIP,
            'd_trip_id': d_trip['id'],
            'trip_id': d_trip['trip_id'],
            'cost': cost,
            'reduced_cost': cost,
            'time': d_trip['duration'],
            'from_location': d_trip['start_location'],
            'to_location': d_trip['end_location'],
            'is_driving': True,
        }

    def _add_start_of_duty_arc(self, to_nid, instance):
        """Add start_of_duty arc from source to a node.
        Includes sign-on time."""
        to_node = self.nodes[to_nid]
        to_time = to_node['time']
        from_loc = instance.depot_location
        to_loc = instance.relief_points[to_node['location']]

        # Time to get from depot to relief point (driving)
        travel_time = instance.driving_time('depot', to_node['location'])
        total_time = instance.sign_on_time + travel_time

        cost = total_time * instance.cost_per_minute + instance.driver_fixed_cost

        self.arcs[(self.NODE_SOURCE, to_nid)] = {
            'type': self.ARC_START_OF_DUTY,
            'cost': cost,
            'reduced_cost': cost,
            'time': total_time,
            'from_location': 'depot',
            'to_location': to_node['location'],
            'is_driving': True,
        }

    def _add_end_of_duty_arc(self, from_nid, instance):
        """Add end_of_duty arc from a node to sink.
        Includes sign-off time."""
        from_node = self.nodes[from_nid]
        from_time = from_node['time']
        from_loc = instance.relief_points[from_node['location']]
        depot = instance.depot_location

        travel_time = instance.driving_time(from_node['location'], 'depot')
        total_time = travel_time + instance.sign_off_time

        cost = total_time * instance.cost_per_minute

        self.arcs[(from_nid, self.NODE_SINK)] = {
            'type': self.ARC_END_OF_DUTY,
            'cost': cost,
            'reduced_cost': cost,
            'time': total_time,
            'from_location': from_node['location'],
            'to_location': 'depot',
            'is_driving': True,
        }

    def _add_inter_trip_arcs(self, trip_i, trip_j, instance):
        """Add inter-trip arcs from trip_i's end to trip_j's start."""
        i_end_nid = self._node_id('end_of_trip', trip_i['id'])
        j_start_nid = self._node_id('start_of_trip', trip_j['id'])

        end_time = trip_i['end_time']
        start_time = trip_j['start_time']

        i_loc = trip_i['end_location']
        j_loc = trip_j['start_location']

        # Driving arc
        drive_time = instance.driving_time(i_loc, j_loc)
        if end_time + drive_time <= start_time:
            elapsed_time = start_time - end_time
            cost = elapsed_time * instance.cost_per_minute
            self.arcs[(i_end_nid, j_start_nid)] = {
                'type': self.ARC_INTER_TRIP_DRIVING,
                'trip_id': trip_j['id'],
                'cost': cost,
                'reduced_cost': cost,
                'time': elapsed_time,
                'travel_time': drive_time,
                'from_location': i_loc,
                'to_location': j_loc,
                'is_driving': True,
            }

        # Walking arc
        walk_time = instance.walking_time(i_loc, j_loc)
        if end_time + walk_time <= start_time:
            elapsed_time = start_time - end_time
            cost = elapsed_time * instance.cost_per_minute
            self.arcs[(i_end_nid, j_start_nid, 'walk')] = {
                'type': self.ARC_INTER_TRIP_WALKING,
                'trip_id': trip_j['id'],
                'cost': cost,
                'reduced_cost': cost,
                'time': elapsed_time,
                'travel_time': walk_time,
                'from_location': i_loc,
                'to_location': j_loc,
                'is_driving': False,
            }

    def _add_inter_trip_from_relief(self, trip_i, trip_j, instance):
        """Add inter-trip arcs from trip_i's relief point to trip_j's start / relief."""
        i_relief_nid = self._node_id('relief', trip_i['id'])
        i_relief_time = trip_i['d_trips'][0]['end_time']
        i_loc = trip_i['mid_location']

        for target_type, target_trip_id, target_time, target_loc in [
            ('start_of_trip', trip_j['id'], trip_j['start_time'], trip_j['start_location']),
            ('relief', trip_j['id'], trip_j['d_trips'][0]['end_time'], trip_j['mid_location']),
        ]:
            target_nid = self._node_id(target_type, target_trip_id)

            # Driving
            drive_time = instance.driving_time(i_loc, target_loc)
            if i_relief_time + drive_time <= target_time:
                elapsed_time = target_time - i_relief_time
                cost = elapsed_time * instance.cost_per_minute
                self.arcs[(i_relief_nid, target_nid)] = {
                    'type': self.ARC_INTER_TRIP_DRIVING,
                    'trip_id': target_trip_id,
                    'cost': cost,
                    'reduced_cost': cost,
                    'time': elapsed_time,
                    'travel_time': drive_time,
                    'from_location': i_loc,
                    'to_location': target_loc,
                    'is_driving': True,
                }

            # Walking
            walk_time = instance.walking_time(i_loc, target_loc)
            if i_relief_time + walk_time <= target_time:
                elapsed_time = target_time - i_relief_time
                cost = elapsed_time * instance.cost_per_minute
                self.arcs[(i_relief_nid, target_nid, 'walk')] = {
                    'type': self.ARC_INTER_TRIP_WALKING,
                    'trip_id': target_trip_id,
                    'cost': cost,
                    'reduced_cost': cost,
                    'time': elapsed_time,
                    'travel_time': walk_time,
                    'from_location': i_loc,
                    'to_location': target_loc,
                    'is_driving': False,
                }

    def get_outgoing_arcs(self, node_id):
        """Get all outgoing arcs from a node."""
        result = []
        for (f, t, *rest), info in self.arcs.items():
            if f == node_id:
                result.append(((f, t, tuple(rest)), info))
        return result

    def get_node_ids(self):
        return list(self.nodes.keys())

    def get_num_nodes(self):
        return len(self.nodes)

    def get_num_arcs(self):
        return len(self.arcs)

    def print_summary(self):
        print(f"Driver Network (Type {self.duty_type}):")
        print(f"  Nodes: {self.get_num_nodes()}")
        print(f"  Arcs: {self.get_num_arcs()}")
        # Count by type
        type_counts = {}
        for info in self.arcs.values():
            t = info['type']
            type_counts[t] = type_counts.get(t, 0) + 1
        for t, c in sorted(type_counts.items()):
            print(f"    {t}: {c}")
