"""Reference-grid binning + QD metrics.

Binning is quantile-based and computed jointly from all entries across
all strategies — so the same grid is used to score every method.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Grid:
    bin_edges: list[np.ndarray]   # one (n_bins+1,) array per axis
    n_bins_per_axis: int

    @property
    def n_cells(self) -> int:
        return self.n_bins_per_axis ** len(self.bin_edges)

    def cell_of(self, coord: tuple[float, float, float]) -> tuple[int, int, int]:
        idxs = []
        for i, edges in enumerate(self.bin_edges):
            # np.digitize: bin index in [1, n_bins] for edges of length n_bins+1
            # we want index in [0, n_bins-1]
            idx = int(np.clip(np.digitize([coord[i]], edges) - 1, 0, self.n_bins_per_axis - 1)[0])
            idxs.append(idx)
        return tuple(idxs)


def build_grid(coords: list[tuple[float, float, float]], n_bins: int = 5) -> Grid:
    """Quantile-based edges per axis from `coords`."""
    arr = np.asarray(coords, dtype=np.float64)
    d = arr.shape[1]
    edges: list[np.ndarray] = []
    qs = np.linspace(0, 1, n_bins + 1)
    for axis in range(d):
        col = arr[:, axis]
        if col.std() < 1e-9:
            # degenerate axis: just use tiny epsilon range so digitize works
            mid = col.mean()
            e = np.linspace(mid - 1e-3, mid + 1e-3, n_bins + 1)
        else:
            e = np.quantile(col, qs)
            # nudge edges to be strictly increasing
            for k in range(1, len(e)):
                if e[k] <= e[k - 1]:
                    e[k] = e[k - 1] + 1e-9
        edges.append(e)
    return Grid(bin_edges=edges, n_bins_per_axis=n_bins)


def coverage(grid: Grid, coords: list[tuple[float, float, float]]) -> float:
    """Fraction of cells with >=1 entry."""
    if not coords:
        return 0.0
    cells = {grid.cell_of(c) for c in coords}
    return len(cells) / grid.n_cells


def qd_score(grid: Grid,
             coords: list[tuple[float, float, float]],
             fitnesses: list[float]) -> float:
    """Sum over filled cells of normalized max-fitness-in-cell.

    Normalization uses the global (across all coords) min/max so the score
    is comparable across strategies.
    """
    if not coords:
        return 0.0
    fits = np.asarray(fitnesses, dtype=np.float64)
    lo, hi = float(fits.min()), float(fits.max())
    span = max(hi - lo, 1e-9)

    # per-cell max
    per_cell: dict[tuple, float] = {}
    for c, f in zip(coords, fitnesses):
        cell = grid.cell_of(c)
        per_cell[cell] = max(per_cell.get(cell, -np.inf), float(f))
    total = 0.0
    for f in per_cell.values():
        total += (f - lo) / span
    return float(total)


def diversity_at_k(coords: list[tuple[float, float, float]], k: int | None = None) -> float:
    """Mean pairwise euclidean distance among (top-k by index) coords in normalized space."""
    if len(coords) < 2:
        return 0.0
    arr = np.asarray(coords, dtype=np.float64)
    # min-max normalize axes
    lo = arr.min(axis=0)
    hi = arr.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    norm = (arr - lo) / span
    if k is not None:
        norm = norm[:k]
    n = norm.shape[0]
    if n < 2:
        return 0.0
    # pairwise
    diff = norm[:, None, :] - norm[None, :, :]
    pairwise = np.linalg.norm(diff, axis=-1)
    iu = np.triu_indices(n, k=1)
    return float(pairwise[iu].mean())
