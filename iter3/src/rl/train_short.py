"""Short PPO training under a custom reward function."""
from __future__ import annotations

from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO

from src.rl.env import RewardOverrideEnv
from src.rl.reward_template import CompiledReward


def train_short(
    reward: CompiledReward,
    env_id: str,
    out_ckpt: Path,
    total_timesteps: int = 30_000,
    seed: int = 0,
    verbose: int = 0,
) -> Path:
    """Train PPO with `reward` for `total_timesteps`, save to `out_ckpt`."""
    base_env = gym.make(env_id)
    env = RewardOverrideEnv(base_env, reward)
    model = PPO("MlpPolicy", env, seed=seed, verbose=verbose, device="cpu")
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_ckpt))
    env.close()
    return out_ckpt
