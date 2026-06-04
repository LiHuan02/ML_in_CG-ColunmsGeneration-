class NoSelectionCG:
    """NO-S: Standard CG with no column selection.

    Adds ALL generated negative reduced cost columns to the RMP.
    Applies disjoint block preselection for diversity.
    """

    def __init__(self, rmp, pp, config=None):
        self.rmp = rmp
        self.pp = pp
        self.config = config or {}
        self.n_max_blks = self.config.get('n_max_blks', 10)

    def select(self, columns):
        """Select ALL columns (with optional disjoint block preselection)."""
        if not columns:
            return []

        # Sort by reduced cost ascending (most negative first)
        sorted_cols = sorted(columns, key=lambda c: c.reduced_cost)

        if self.n_max_blks <= 0:
            return sorted_cols

        # Disjoint block preselection
        # Blocks of mutually disjoint columns (no common d-trips)
        blocks = []
        for col in sorted_cols:
            placed = False
            for block in blocks:
                # Check if column is disjoint from all in this block
                disjoint = True
                for existing in block:
                    # Check if they share any d-trip
                    if set(col.d_trips.keys()) & set(existing.d_trips.keys()):
                        disjoint = False
                        break
                if disjoint:
                    block.append(col)
                    placed = True
                    break
            if not placed:
                blocks.append([col])

        # Take columns from first n_max_blks blocks
        selected = []
        for block in blocks[:self.n_max_blks]:
            selected.extend(block)

        return selected


class SortSelectionCG:
    """Sort-S: Select columns by sorted reduced cost (baseline)."""

    def __init__(self, rmp, pp, config=None):
        self.rmp = rmp
        self.pp = pp
        self.config = config or {}
        self.n_select = self.config.get('n_select', 50)

    def select(self, columns):
        sorted_cols = sorted(columns, key=lambda c: c.reduced_cost)
        return sorted_cols[:self.n_select]


class RandomSelectionCG:
    """Rand-S: Random column selection (baseline)."""

    def __init__(self, rmp, pp, config=None):
        self.rmp = rmp
        self.pp = pp
        self.config = config or {}
        self.n_select = self.config.get('n_select', 50)

    def select(self, columns):
        import random
        return random.sample(columns, min(self.n_select, len(columns)))
