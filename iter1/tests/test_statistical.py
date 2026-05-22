"""Tests for statistical tests — ensure bootstrap CI and Mann-Whitney
don't produce nonsensical results that would hide real effects."""
from __future__ import annotations

import numpy as np
import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.statistical_tests import bootstrap_mean_ci, compare_groups, holm_adjust, GroupComparison


class TestBootstrapCI:
    def test_ci_contains_mean(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        ci = bootstrap_mean_ci(vals)
        assert ci.lo <= ci.mean <= ci.hi

    def test_constant_values_tight_ci(self):
        vals = [3.0] * 20
        ci = bootstrap_mean_ci(vals)
        assert abs(ci.lo - 3.0) < 0.01
        assert abs(ci.hi - 3.0) < 0.01

    def test_empty_returns_zeros(self):
        ci = bootstrap_mean_ci([])
        assert ci.mean == 0.0

    def test_wider_distribution_wider_ci(self):
        narrow = bootstrap_mean_ci([5.0, 5.1, 4.9, 5.0, 5.1])
        wide = bootstrap_mean_ci([0.0, 10.0, 5.0, 2.0, 8.0])
        narrow_width = narrow.hi - narrow.lo
        wide_width = wide.hi - wide.lo
        assert wide_width > narrow_width


class TestMannWhitney:
    def test_identical_groups_high_p(self):
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        comp = compare_groups(a, a, name="test")
        assert comp.p_value > 0.05

    def test_clearly_different_groups_low_p(self):
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [100.0, 200.0, 300.0, 400.0, 500.0]
        comp = compare_groups(a, b, name="test")
        assert comp.p_value < 0.05

    def test_holm_correction_increases_p(self):
        comps = [
            GroupComparison("a", 5, 5, 1.0, 2.0, 10.0, 0.01),
            GroupComparison("b", 5, 5, 1.0, 2.0, 10.0, 0.03),
            GroupComparison("c", 5, 5, 1.0, 2.0, 10.0, 0.05),
        ]
        holm_adjust(comps)
        # Adjusted p-values should be >= raw p-values
        for c in comps:
            assert c.p_adj >= c.p_value - 1e-10

    def test_holm_monotone(self):
        """Holm-adjusted p-values should be monotonically non-decreasing
        when sorted by raw p-value."""
        comps = [
            GroupComparison("a", 5, 5, 1.0, 2.0, 10.0, 0.01),
            GroupComparison("b", 5, 5, 1.0, 2.0, 10.0, 0.02),
            GroupComparison("c", 5, 5, 1.0, 2.0, 10.0, 0.04),
        ]
        holm_adjust(comps)
        sorted_by_raw = sorted(comps, key=lambda c: c.p_value)
        for i in range(len(sorted_by_raw) - 1):
            assert sorted_by_raw[i].p_adj <= sorted_by_raw[i + 1].p_adj + 1e-10
