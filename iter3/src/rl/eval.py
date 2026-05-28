"""Rollout a trained policy under the DEFAULT env reward.

Returns:
  - fitness scalar (sum of default rewards / n_episodes)
  - video (mp4) and state log (npz) for the rollout
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import PPO


@dataclass
class EvalArtifact:
    fitness: float                  # mean episode return under DEFAULT reward
    n_episodes: int
    video_path: Path
    state_path: Path
    n_steps: int
    fps: int


def rollout_eval(
    ckpt_path: Path,
    env_id: str,
    video_path: Path,
    state_path: Path,
    n_steps: int = 600,
    fps: int = 30,
    seed: int = 0,
    deterministic: bool = True,
) -> EvalArtifact:
    env = gym.make(env_id, render_mode="rgb_array")
    model = PPO.load(str(ckpt_path), device="cpu")

    obs, _ = env.reset(seed=seed)
    frames: list[np.ndarray] = []
    obs_log, act_log, rew_log = [], [], []
    term_log, trunc_log = [], []
    episode_returns: list[float] = []
    cur_return = 0.0

    for _ in range(n_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs_log.append(np.asarray(obs, dtype=np.float32))
        act_log.append(np.asarray(action, dtype=np.float32))
        obs, reward, terminated, truncated, _ = env.step(action)
        rew_log.append(float(reward))
        term_log.append(bool(terminated))
        trunc_log.append(bool(truncated))
        cur_return += float(reward)
        frames.append(env.render())
        if terminated or truncated:
            episode_returns.append(cur_return)
            cur_return = 0.0
            obs, _ = env.reset()
    if cur_return != 0.0 and not episode_returns:
        # rollout never ended — count partial as one episode
        episode_returns.append(cur_return)

    env.close()

    video_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    if frames:
        with imageio.get_writer(str(video_path), fps=fps, codec="libx264", quality=7) as w:
            for f in frames:
                w.append_data(f)

    np.savez_compressed(
        state_path,
        obs=np.stack(obs_log),
        action=np.stack(act_log),
        reward=np.asarray(rew_log, dtype=np.float32),
        terminated=np.asarray(term_log, dtype=bool),
        truncated=np.asarray(trunc_log, dtype=bool),
    )

    fitness = float(np.mean(episode_returns)) if episode_returns else float(sum(rew_log))
    return EvalArtifact(
        fitness=fitness,
        n_episodes=len(episode_returns) or 1,
        video_path=video_path,
        state_path=state_path,
        n_steps=len(frames),
        fps=fps,
    )
