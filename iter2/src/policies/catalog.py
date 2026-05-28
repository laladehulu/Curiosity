"""Registry of policies to train + index.

Each PolicySpec is a (name, env_id, total_steps, checkpoint_at) tuple.
checkpoint_at is a list of step counts where we'll snapshot the model —
early checkpoints show emergent/clumsy behavior, late ones show converged
gaits. This gives behavioral diversity for free without writing multiple
reward functions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class PolicySpec:
    name: str
    env_id: str
    total_steps: int
    checkpoint_at: tuple  # checkpoint step counts (must be <= total_steps)
    seed: int = 0


# Defaults tuned for "runs on a MacBook in under ~30 minutes total".
# Adjust total_steps up for cleaner gaits; down for faster iteration.
DEFAULT_CATALOG: List[PolicySpec] = [
    PolicySpec(
        name="walker2d",
        env_id="Walker2d-v5",
        total_steps=300_000,
        checkpoint_at=(50_000, 150_000, 300_000),
    ),
    PolicySpec(
        name="hopper",
        env_id="Hopper-v5",
        total_steps=200_000,
        checkpoint_at=(40_000, 100_000, 200_000),
    ),
    PolicySpec(
        name="halfcheetah",
        env_id="HalfCheetah-v5",
        total_steps=300_000,
        checkpoint_at=(50_000, 150_000, 300_000),
    ),
]


def checkpoint_id(spec: PolicySpec, step: int) -> str:
    """Stable, filesystem-safe id for a (policy, checkpoint) pair."""
    return f"{spec.name}__s{spec.seed}__step{step}"
