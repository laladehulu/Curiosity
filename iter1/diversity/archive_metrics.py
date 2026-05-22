"""MAP-Elites archive coverage metrics. Per protocol §"Archive coverage metrics"."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class ArchiveSummary:
    filled_cells: int
    total_cells: int
    coverage: float
    mean_quality: float
    qd_score: float
    quality_std: float
    best_quality: float


def summarize_archive(archive: dict, total_cells: int, normalize_min: float = 0.0,
                      normalize_max: float = 1.0) -> ArchiveSummary:
    """archive: mapping cell -> Candidate-like object with .mean_fitness float.

    qd_score sums (mean_fitness - normalize_min)/(normalize_max - normalize_min) across
    filled cells; pass appropriate bounds per task if you want a comparable score.
    """
    filled = len(archive)
    if filled == 0:
        return ArchiveSummary(0, total_cells, 0.0, 0.0, 0.0, 0.0, 0.0)
    qs = np.array([c.mean_fitness for c in archive.values()])
    span = (normalize_max - normalize_min) if (normalize_max > normalize_min) else 1.0
    qd = float(np.sum((qs - normalize_min) / span))
    return ArchiveSummary(
        filled_cells=filled,
        total_cells=total_cells,
        coverage=filled / max(1, total_cells),
        mean_quality=float(np.mean(qs)),
        qd_score=qd,
        quality_std=float(np.std(qs)),
        best_quality=float(np.max(qs)),
    )
