"""Phase 5 — statistical tests. Per protocol §"Required tests".

Bootstrap CI + Mann-Whitney U + Holm correction across the family of comparisons.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.stats import mannwhitneyu


@dataclass
class BootstrapCI:
    mean: float
    lo: float
    hi: float


def bootstrap_mean_ci(samples: Sequence[float], n_resamples: int = 10_000,
                      alpha: float = 0.05, rng_seed: int = 0) -> BootstrapCI:
    rng = np.random.default_rng(rng_seed)
    arr = np.asarray(samples, dtype=np.float64)
    if arr.size == 0:
        return BootstrapCI(0.0, 0.0, 0.0)
    boots = np.array([rng.choice(arr, size=arr.size, replace=True).mean() for _ in range(n_resamples)])
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return BootstrapCI(mean=float(arr.mean()), lo=lo, hi=hi)


@dataclass
class GroupComparison:
    name: str
    n_a: int
    n_b: int
    mean_a: float
    mean_b: float
    u_stat: float
    p_value: float
    p_adj: float = float("nan")


def compare_groups(a: Sequence[float], b: Sequence[float], name: str) -> GroupComparison:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    u, p = mannwhitneyu(a, b, alternative="two-sided") if (a.size and b.size) else (0.0, 1.0)
    return GroupComparison(
        name=name, n_a=int(a.size), n_b=int(b.size),
        mean_a=float(a.mean()) if a.size else 0.0,
        mean_b=float(b.mean()) if b.size else 0.0,
        u_stat=float(u), p_value=float(p),
    )


def holm_adjust(comparisons: list[GroupComparison]) -> list[GroupComparison]:
    """Holm-Bonferroni correction. Operates in place on .p_adj."""
    ordered = sorted(range(len(comparisons)), key=lambda i: comparisons[i].p_value)
    n = len(comparisons)
    last = 0.0
    for rank, i in enumerate(ordered):
        adj = min(1.0, (n - rank) * comparisons[i].p_value)
        adj = max(adj, last)
        comparisons[i].p_adj = adj
        last = adj
    return comparisons
