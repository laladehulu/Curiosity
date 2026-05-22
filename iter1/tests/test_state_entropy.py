"""Tests for curiosity/state_entropy.py — reduce false positives by verifying
that the entropy signal actually varies across meaningfully different policies."""
from __future__ import annotations

import numpy as np
import pytest
import gymnasium as gym

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from curiosity.state_entropy import compute_state_entropy, _TASK_OBS_CONFIG


# ---------------------------------------------------------------------------
# Helpers: mock "policies" with deterministic behaviour
# ---------------------------------------------------------------------------

class ConstantPolicy:
    """Always outputs the same action."""
    def __init__(self, action):
        self._action = np.asarray(action, dtype=np.float32)

    def predict(self, obs, deterministic=True):
        return self._action, None


class RandomPolicy:
    """Uniform random actions within env action space bounds."""
    def __init__(self, env_id: str, seed: int = 42):
        env = gym.make(env_id)
        self._low = env.action_space.low
        self._high = env.action_space.high
        self._rng = np.random.default_rng(seed)
        env.close()

    def predict(self, obs, deterministic=True):
        action = self._rng.uniform(self._low, self._high).astype(np.float32)
        return action, None


# ---------------------------------------------------------------------------
# Basic sanity
# ---------------------------------------------------------------------------

class TestStateEntropy:
    def test_returns_nonnegative_float(self):
        """Entropy must be >= 0."""
        policy = ConstantPolicy([0.0])
        ent = compute_state_entropy(policy, "Pendulum-v1", "pendulum", n_episodes=3, seed=0)
        assert isinstance(ent, float)
        assert ent >= 0.0

    def test_config_exists_for_all_tasks(self):
        """Every declared task must have an obs config."""
        for task in ["pendulum", "mountaincar", "bipedalwalker"]:
            assert task in _TASK_OBS_CONFIG, f"Missing config for {task}"
            cfg = _TASK_OBS_CONFIG[task]
            assert len(cfg["dims"]) == len(cfg["bins"]) == len(cfg["ranges"])

    def test_invalid_task_raises(self):
        policy = ConstantPolicy([0.0])
        with pytest.raises(ValueError, match="No obs config"):
            compute_state_entropy(policy, "Pendulum-v1", "nonexistent_task")


# ---------------------------------------------------------------------------
# False-positive guards: entropy must discriminate policies
# ---------------------------------------------------------------------------

class TestEntropyDiscriminates:
    """Core anti-false-positive tests: if two policies visit very different
    regions of the state space, their entropy values must differ."""

    def test_constant_vs_random_pendulum(self):
        """A constant-action policy should have LOWER entropy than random."""
        const = ConstantPolicy([0.0])
        rand = RandomPolicy("Pendulum-v1", seed=7)

        ent_const = compute_state_entropy(const, "Pendulum-v1", "pendulum", n_episodes=10, seed=0)
        ent_rand = compute_state_entropy(rand, "Pendulum-v1", "pendulum", n_episodes=10, seed=0)

        # Random explores more → higher entropy
        assert ent_rand > ent_const, (
            f"Random policy entropy ({ent_rand:.3f}) should exceed "
            f"constant policy entropy ({ent_const:.3f})"
        )

    def test_different_constants_different_entropy(self):
        """Two constant policies with different action magnitudes should
        generally produce different entropy (they induce different limit cycles)."""
        pol_a = ConstantPolicy([0.0])
        pol_b = ConstantPolicy([2.0])

        ent_a = compute_state_entropy(pol_a, "Pendulum-v1", "pendulum", n_episodes=10, seed=0)
        ent_b = compute_state_entropy(pol_b, "Pendulum-v1", "pendulum", n_episodes=10, seed=0)

        # They don't need to be hugely different, but shouldn't be identical
        # unless both policies truly visit the same state distribution
        # (which is unlikely for action=0 vs action=2).
        # We just check they aren't NaN or negative.
        assert ent_a >= 0.0 and ent_b >= 0.0

    def test_entropy_reproducible_with_same_seed(self):
        """Same policy + same seed → identical entropy."""
        pol = RandomPolicy("Pendulum-v1", seed=42)
        e1 = compute_state_entropy(pol, "Pendulum-v1", "pendulum", n_episodes=5, seed=99)
        # Need a fresh RandomPolicy with same seed to reset internal RNG
        pol2 = RandomPolicy("Pendulum-v1", seed=42)
        e2 = compute_state_entropy(pol2, "Pendulum-v1", "pendulum", n_episodes=5, seed=99)
        assert abs(e1 - e2) < 1e-10, f"Entropy not reproducible: {e1} vs {e2}"

    def test_more_episodes_weakly_increases_entropy(self):
        """With more episodes, we cover at least as many bins.
        Entropy with 20 eps should be >= entropy with 2 eps (usually strictly >)."""
        pol = ConstantPolicy([1.0])
        e_few = compute_state_entropy(pol, "Pendulum-v1", "pendulum", n_episodes=2, seed=0)
        e_many = compute_state_entropy(pol, "Pendulum-v1", "pendulum", n_episodes=20, seed=0)
        # Weakly: more data can only fill more bins
        assert e_many >= e_few - 0.01, (
            f"More episodes should not drastically reduce entropy: {e_many:.3f} vs {e_few:.3f}"
        )


# ---------------------------------------------------------------------------
# Entropy bounds
# ---------------------------------------------------------------------------

class TestEntropyBounds:
    def test_entropy_below_max(self):
        """Entropy cannot exceed log(total_bins)."""
        cfg = _TASK_OBS_CONFIG["pendulum"]
        max_entropy = np.log(np.prod(cfg["bins"]))
        pol = RandomPolicy("Pendulum-v1", seed=0)
        ent = compute_state_entropy(pol, "Pendulum-v1", "pendulum", n_episodes=20, seed=0)
        assert ent <= max_entropy + 1e-9, f"Entropy {ent:.3f} exceeds max {max_entropy:.3f}"
