"""Gym wrapper that swaps env reward with an LLM-generated callable."""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from src.rl.reward_template import CompiledReward


class RewardOverrideEnv(gym.Wrapper):
    """Replace env reward with reward_fn(obs, action, next_obs, done)."""

    def __init__(self, env: gym.Env, reward_fn: CompiledReward):
        super().__init__(env)
        self._reward_fn = reward_fn
        self._last_obs: np.ndarray | None = None

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        self._last_obs = np.asarray(obs, dtype=np.float64)
        return obs, info

    def step(self, action):
        next_obs, _orig_reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)
        r = self._reward_fn(self._last_obs, np.asarray(action, dtype=np.float64),
                            np.asarray(next_obs, dtype=np.float64), done)
        self._last_obs = np.asarray(next_obs, dtype=np.float64)
        return next_obs, float(r), terminated, truncated, info
