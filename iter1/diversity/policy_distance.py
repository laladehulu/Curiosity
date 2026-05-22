"""Phase 3 — policy-space distances between trained policies.

Per protocol §"Policy-space distance":
1. Trajectory DTW distance over 100 rollouts from fixed initial states
2. Action-distribution KL divergence at 1000 fixed states
3. Behavioral-descriptor distance
4. State-occupancy total-variation estimated from rollouts
"""
from __future__ import annotations

import numpy as np
from typing import Callable, Sequence


def _dtw(a: np.ndarray, b: np.ndarray) -> float:
    """Plain DTW on 1-D / k-D sequences (length T_a x feature, T_b x feature)."""
    Ta, Tb = a.shape[0], b.shape[0]
    INF = float("inf")
    dp = np.full((Ta + 1, Tb + 1), INF)
    dp[0, 0] = 0.0
    for i in range(1, Ta + 1):
        for j in range(1, Tb + 1):
            cost = float(np.linalg.norm(a[i - 1] - b[j - 1]))
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[Ta, Tb])


def trajectory_dtw_distance(rollouts_a: Sequence[np.ndarray], rollouts_b: Sequence[np.ndarray]) -> float:
    """Mean DTW distance between paired rollouts. rollouts are arrays (T, D) of observations."""
    if not rollouts_a or not rollouts_b:
        return 0.0
    n = min(len(rollouts_a), len(rollouts_b))
    dists = [_dtw(rollouts_a[i], rollouts_b[i]) for i in range(n)]
    return float(np.mean(dists))


def action_distribution_kl(
    policy_a: Callable[[np.ndarray], np.ndarray],
    policy_b: Callable[[np.ndarray], np.ndarray],
    states: np.ndarray,
    action_dim: int,
    bins: int = 10,
    action_low: float = -1.0,
    action_high: float = 1.0,
) -> float:
    """Estimate symmetric KL between action distributions over a fixed state bank.
    Both policies are deterministic in this codebase, so we discretize the per-state
    action choice and compute KL of the marginal histogram."""
    actions_a = np.array([np.atleast_1d(policy_a(s)).ravel() for s in states])
    actions_b = np.array([np.atleast_1d(policy_b(s)).ravel() for s in states])
    # marginal histogram over first action dim only — workshop-scope simplification
    ha, _ = np.histogram(actions_a[:, 0], bins=bins, range=(action_low, action_high), density=True)
    hb, _ = np.histogram(actions_b[:, 0], bins=bins, range=(action_low, action_high), density=True)
    ha = ha + 1e-9
    hb = hb + 1e-9
    ha /= ha.sum()
    hb /= hb.sum()
    kl_ab = float(np.sum(ha * np.log(ha / hb)))
    kl_ba = float(np.sum(hb * np.log(hb / ha)))
    return 0.5 * (kl_ab + kl_ba)


def descriptor_distance(desc_a: np.ndarray, desc_b: np.ndarray) -> float:
    """Euclidean distance in descriptor space."""
    return float(np.linalg.norm(np.asarray(desc_a) - np.asarray(desc_b)))


def state_occupancy_tv(rollouts_a, rollouts_b, bins: int = 10) -> float:
    """Total variation between empirical state-occupancy histograms (first 2 dims).
    Workshop scope: 2D histogram is enough for the classic-control tasks here."""
    states_a = np.concatenate([r for r in rollouts_a], axis=0) if rollouts_a else np.zeros((1, 2))
    states_b = np.concatenate([r for r in rollouts_b], axis=0) if rollouts_b else np.zeros((1, 2))
    if states_a.shape[1] < 2:
        states_a = np.column_stack([states_a, np.zeros(states_a.shape[0])])
        states_b = np.column_stack([states_b, np.zeros(states_b.shape[0])])
    states_a = states_a[:, :2]
    states_b = states_b[:, :2]
    lo = np.minimum(states_a.min(axis=0), states_b.min(axis=0))
    hi = np.maximum(states_a.max(axis=0), states_b.max(axis=0))
    rng = [[lo[0], hi[0] + 1e-9], [lo[1], hi[1] + 1e-9]]
    ha, _, _ = np.histogram2d(states_a[:, 0], states_a[:, 1], bins=bins, range=rng, density=True)
    hb, _, _ = np.histogram2d(states_b[:, 0], states_b[:, 1], bins=bins, range=rng, density=True)
    ha /= ha.sum() + 1e-12
    hb /= hb.sum() + 1e-12
    return float(0.5 * np.sum(np.abs(ha - hb)))
