"""Per-task PPO hyperparameters. Frozen — do not tune in code.

Defaults follow stable-baselines3 + Hyperparameter Tuner Zoo conventions for
classic control. Workshop-scope: we are not claiming hyperparameter novelty.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PPOConfig:
    total_timesteps: int
    n_steps: int
    batch_size: int
    n_epochs: int
    gamma: float
    gae_lambda: float
    clip_range: float
    ent_coef: float
    vf_coef: float
    learning_rate: float
    policy_kwargs: dict


PHASE2_CONFIGS: dict[str, PPOConfig] = {
    "pendulum": PPOConfig(
        total_timesteps=200_000,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.9,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        learning_rate=1e-3,
        policy_kwargs={"net_arch": [64, 64]},
    ),
    "mountaincar": PPOConfig(
        total_timesteps=500_000,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.9999,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # nonzero entropy bonus helps exploration on sparse reward
        vf_coef=0.5,
        learning_rate=7e-4,
        policy_kwargs={"net_arch": [64, 64]},
    ),
    "bipedalwalker": PPOConfig(
        total_timesteps=2_000_000,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.18,
        ent_coef=0.0,
        vf_coef=0.5,
        learning_rate=3e-4,
        policy_kwargs={"net_arch": [128, 128]},
    ),
}
