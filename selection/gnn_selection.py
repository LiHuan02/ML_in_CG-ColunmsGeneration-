"""GNN-S: Column selection using a trained bipartite GNN.

At each CG iteration:
  1. Extract column features (basic + new) and constraint features
  2. Build bipartite graph (edges between columns and constraints)
  3. Run GNN inference to get selection probabilities
  4. Select columns with probability > 0.5

Paper reference: EBSCO Section 4.4.3, Table 4 parameter values.
"""

import os
import numpy as np
import torch

from data_generation.feature_extractor import FeatureExtractor


class GNNSelectionCG:
    """GNN-S: Column selection using a learned bipartite GNN model."""

    def __init__(self, rmp, pp, config=None):
        self.rmp = rmp
        self.pp = pp
        self.config = config or {}

        # Model loading
        self.model_path = self.config.get('model_path', 'gnn/models/best_model.pt')
        self.norm_stats_path = self.config.get('norm_stats_path', 'gnn/models/norm_stats.npz')
        self.device = self.config.get('device', 'cpu')

        # Selection parameters (Table 4)
        self.n_max_blks = self.config.get('n_max_blks', 14)
        self.min_select = self.config.get('min_select', 5)
        self.select_ratio = self.config.get('select_ratio', 0.3)  # Top-K ratio

        # Feature extractor
        self.feature_extractor = FeatureExtractor(rmp.instance)

        # Load model (lazy)
        self._model = None
        self._norm_stats = None

    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model

    def _load_model(self):
        """Load the trained GNN model and normalization stats."""
        from gnn.bipartite_gnn import BipartiteGNN

        self._model = BipartiteGNN(
            col_feat_dim=12, constr_feat_dim=2,
            hidden_dim=self.config.get('hidden_dim', 32),
            num_iterations=self.config.get('num_iterations', 1),
        ).to(torch.device(self.device))
        self._model.eval()

        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            self._model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  GNN-S: Loaded model from {self.model_path}")
        else:
            print(f"  GNN-S: WARNING - Model not found at {self.model_path}, using random predictions")

        # Load normalization stats
        if os.path.exists(self.norm_stats_path):
            self._norm_stats = np.load(self.norm_stats_path)
        else:
            self._norm_stats = None

    def _normalize(self, col_feat, constr_feat):
        """Normalize features using stored statistics."""
        if self._norm_stats is None:
            return col_feat, constr_feat

        col_feat = (col_feat - self._norm_stats['col_mean']) / np.maximum(
            self._norm_stats['col_std'], 1e-8
        )
        constr_feat = (constr_feat - self._norm_stats['constr_mean']) / np.maximum(
            self._norm_stats['constr_std'], 1e-8
        )
        return col_feat, constr_feat

    def select(self, columns):
        """Select columns using the GNN model.

        Args:
            columns: list of newly generated VCSPColumn

        Returns:
            list of selected VCSPColumn
        """
        if not columns:
            return []

        # Apply disjoint block preselection first (Table 4: n_max_blks = 14)
        if self.n_max_blks > 0:
            sorted_cols = sorted(columns, key=lambda c: c.reduced_cost)
            blocks = []
            for col in sorted_cols:
                placed = False
                for block in blocks:
                    disjoint = True
                    for existing in block:
                        if set(col.d_trips.keys()) & set(existing.d_trips.keys()):
                            disjoint = False
                            break
                    if disjoint:
                        block.append(col)
                        placed = True
                        break
                if not placed:
                    blocks.append([col])
            candidate_cols = []
            for block in blocks[:self.n_max_blks]:
                candidate_cols.extend(block)
        else:
            candidate_cols = list(columns)

        if len(candidate_cols) < self.min_select:
            return list(columns)

        # Get basic columns
        try:
            col_sol = self.rmp.column_solution
        except AttributeError:
            col_sol = [0.0] * len(self.rmp.columns)

        basic_indices = [i for i, v in enumerate(col_sol) if v > 1e-6]
        basic_columns = [self.rmp.columns[i] for i in basic_indices]

        # Build column features
        basis_support = self.feature_extractor.build_basis_constraint_support(
            self.rmp.columns, basic_indices
        )

        col_feat_list = []
        for i, col in enumerate(basic_columns):
            feat = self.feature_extractor.extract_column_features(
                col, is_new=False, theta_value=col_sol[basic_indices[i]],
                basis_constraint_support=basis_support,
            )
            col_feat_list.append(feat)

        new_col_start = len(col_feat_list)
        for col in candidate_cols:
            feat = self.feature_extractor.extract_column_features(
                col, is_new=True, theta_value=0.0,
                basis_constraint_support=basis_support,
            )
            col_feat_list.append(feat)

        col_features = np.stack(col_feat_list, axis=0)

        # Build constraint features
        all_columns = basic_columns + candidate_cols
        constraint_degrees = self.feature_extractor.compute_constraint_degrees(all_columns)

        # Get dual values from RMP
        try:
            duals = self.rmp.dual_values
        except AttributeError:
            # Reconstruct from last solve (approximate)
            duals = {
                'alpha': np.zeros(self.rmp.instance.num_d_trips),
                'beta': np.zeros(self.rmp.instance.num_trips),
                'gamma': np.zeros(self.rmp.instance.num_trips),
                'delta': np.zeros(len(self.rmp.instance.departure_times)),
            }

        constr_features = self.feature_extractor.extract_constraint_features(
            duals, constraint_degrees
        )

        # Build edge index
        edges = []
        for col_idx, col in enumerate(all_columns):
            coeff_vec = self.feature_extractor.column_coefficient_vector(col)
            c_indices = np.nonzero(coeff_vec)[0]
            for c_idx in c_indices:
                edges.append([col_idx, int(c_idx)])

        if len(edges) == 0:
            return list(columns)

        edge_index = np.array(edges, dtype=np.int64).T

        # Normalize
        col_features, constr_features = self._normalize(col_features, constr_features)

        # Run GNN inference
        with torch.no_grad():
            cf_t = torch.from_numpy(col_features).float().to(self.device)
            ctf_t = torch.from_numpy(constr_features).float().to(self.device)
            ei_t = torch.from_numpy(edge_index).long().to(self.device)

            logits = self.model(cf_t, ctf_t, ei_t)
            probs = torch.sigmoid(logits).cpu().numpy()

        # Select top-K new columns by GNN probability (instead of fixed threshold)
        new_probs = probs[new_col_start:]
        n_candidates = len(candidate_cols)
        k = max(self.min_select, int(n_candidates * self.select_ratio))
        k = min(k, n_candidates)

        top_indices = np.argsort(new_probs)[::-1][:k]
        selected = [candidate_cols[i] for i in range(n_candidates) if i in top_indices]

        n_selected = len(selected)
        print(f"  GNN-S: {n_selected}/{len(candidate_cols)} columns selected "
              f"({100 * n_selected / max(1, len(candidate_cols)):.1f}%)")

        return selected
