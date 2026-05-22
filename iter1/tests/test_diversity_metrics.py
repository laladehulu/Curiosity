"""Tests for diversity/policy_distance.py — ensure diversity metrics
don't produce degenerate values that would mask true differences."""
from __future__ import annotations

import numpy as np
import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diversity.policy_distance import (
    state_occupancy_tv,
    trajectory_dtw_distance,
    action_distribution_kl,
)


class TestStateOccupancyTV:
    def test_identical_rollouts_zero_tv(self):
        """Same rollouts → TV = 0."""
        traj = [np.random.randn(50, 3) for _ in range(5)]
        tv = state_occupancy_tv(traj, traj)
        assert abs(tv) < 1e-6

    def test_disjoint_rollouts_high_tv(self):
        """Rollouts in opposite corners of state space → TV close to 1."""
        traj_a = [np.full((50, 2), -10.0) for _ in range(5)]
        traj_b = [np.full((50, 2), 10.0) for _ in range(5)]
        tv = state_occupancy_tv(traj_a, traj_b)
        assert tv > 0.3, f"Disjoint rollouts should have high TV, got {tv}"

    def test_tv_bounded_0_1(self):
        """TV must be in [0, 1]."""
        rng = np.random.default_rng(0)
        for _ in range(5):
            a = [rng.standard_normal((30, 3)) for _ in range(3)]
            b = [rng.standard_normal((30, 3)) for _ in range(3)]
            tv = state_occupancy_tv(a, b)
            assert 0.0 <= tv <= 1.0 + 1e-6, f"TV out of bounds: {tv}"

    def test_symmetry(self):
        """TV(A,B) == TV(B,A)."""
        rng = np.random.default_rng(7)
        a = [rng.standard_normal((40, 2)) for _ in range(4)]
        b = [rng.standard_normal((40, 2)) + 2 for _ in range(4)]
        assert abs(state_occupancy_tv(a, b) - state_occupancy_tv(b, a)) < 1e-10


class TestTrajectoryDTW:
    def test_identical_trajectories_zero(self):
        traj = [np.random.randn(20, 3) for _ in range(3)]
        assert trajectory_dtw_distance(traj, traj) < 1e-10

    def test_different_trajectories_positive(self):
        a = [np.zeros((20, 2)) for _ in range(3)]
        b = [np.ones((20, 2)) * 5 for _ in range(3)]
        d = trajectory_dtw_distance(a, b)
        assert d > 0

    def test_empty_returns_zero(self):
        assert trajectory_dtw_distance([], []) == 0.0


class TestActionKL:
    def test_identical_policies_low_kl(self):
        """Same policy → KL ≈ 0."""
        def policy(s):
            return np.array([0.5])
        states = np.random.randn(100, 3)
        kl = action_distribution_kl(policy, policy, states, action_dim=1)
        assert kl < 0.1, f"Identical policies should have near-zero KL, got {kl}"

    def test_different_policies_positive_kl(self):
        """Policies with different constant actions → positive KL."""
        def pol_a(s):
            return np.array([-0.9])
        def pol_b(s):
            return np.array([0.9])
        states = np.random.randn(200, 3)
        kl = action_distribution_kl(pol_a, pol_b, states, action_dim=1)
        assert kl > 0.0
