"""Tests for Pareto front computation — guards against selection bugs
that could produce false-positive diversity claims."""
from __future__ import annotations

import numpy as np
import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reward_gen.gepa_pareto import pareto_front


class TestParetoFront:
    def test_single_point(self):
        """Single candidate is always on the front."""
        m = np.array([[1.0, 2.0]])
        assert pareto_front(m).tolist() == [True]

    def test_identical_points(self):
        """Identical points: none dominates, all on front."""
        m = np.array([[1.0, 2.0], [1.0, 2.0]])
        assert pareto_front(m).tolist() == [True, True]

    def test_clear_domination(self):
        """Point B dominates A: only B on front."""
        m = np.array([[1.0, 1.0], [2.0, 2.0]])
        front = pareto_front(m)
        assert front.tolist() == [False, True]

    def test_tradeoff_both_on_front(self):
        """Two points with tradeoff: both on front."""
        m = np.array([[3.0, 1.0], [1.0, 3.0]])
        front = pareto_front(m)
        assert front.tolist() == [True, True]

    def test_3d_front(self):
        """Pareto front works for >2 objectives."""
        m = np.array([
            [3, 1, 1],
            [1, 3, 1],
            [1, 1, 3],
            [0, 0, 0],  # dominated by all above
        ], dtype=float)
        front = pareto_front(m)
        assert front.tolist() == [True, True, True, False]

    def test_curiosity_2d_scenario(self):
        """Simulate a realistic curiosity-Pareto scenario:
        high-fitness/low-entropy and low-fitness/high-entropy both survive."""
        m = np.array([
            [-150.0, 4.0],   # good fitness, low entropy
            [-300.0, 6.5],   # bad fitness, high entropy
            [-250.0, 5.0],   # middle
            [-400.0, 3.0],   # dominated by first
        ])
        front = pareto_front(m)
        # First: not dominated (best fitness)
        # Second: not dominated (best entropy)
        # Third: dominated by first (first has better fitness and >=5 entropy? let's check)
        # Actually: [-150, 4] vs [-250, 5]: -150 > -250 but 4 < 5, so neither dominates
        # [-150, 4] vs [-300, 6.5]: -150 > -300 but 4 < 6.5, neither dominates
        # [-250, 5] vs [-300, 6.5]: -250 > -300 but 5 < 6.5, neither dominates
        # [-400, 3] vs [-150, 4]: dominated (worse on both)
        assert front.tolist() == [True, True, True, False]

    def test_empty_input_handled(self):
        """Edge case: empty matrix."""
        m = np.array([]).reshape(0, 2)
        front = pareto_front(m)
        assert len(front) == 0


class TestParetoFrontNoFalsePositives:
    """Ensure the Pareto front doesn't spuriously include dominated points,
    which would inflate diversity metrics."""

    def test_strictly_dominated_never_included(self):
        """Generate N random points, add one that's strictly worse on all axes.
        It must never appear on the front."""
        rng = np.random.default_rng(42)
        for trial in range(10):
            n = rng.integers(3, 15)
            d = rng.integers(2, 5)
            m = rng.uniform(0, 10, (n, d))
            # Add a point dominated by the maximum
            dominated = m.min(axis=0) - 1.0
            m_ext = np.vstack([m, dominated.reshape(1, -1)])
            front = pareto_front(m_ext)
            assert not front[-1], f"Dominated point was on front in trial {trial}"

    def test_all_equal_all_on_front(self):
        """If all candidates are identical, they're all non-dominated."""
        m = np.ones((5, 3))
        front = pareto_front(m)
        assert all(front), "Identical points should all be on front"
