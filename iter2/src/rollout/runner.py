"""Roll out a trained policy, save video + per-step state log."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import PPO

from src import paths


@dataclass
class RolloutArtifact:
    ckpt_id: str
    env_id: str
    video_path: Path
    state_path: Path
    n_steps: int
    fps: int

    def to_dict(self):
        return {
            "ckpt_id": self.ckpt_id,
            "env_id": self.env_id,
            "video_path": str(self.video_path),
            "state_path": str(self.state_path),
            "n_steps": self.n_steps,
            "fps": self.fps,
        }


def rollout(
    ckpt_path: Path,
    env_id: str,
    out_dir: Path | None = None,
    n_steps: int = 1000,
    fps: int = 30,
    seed: int = 0,
    deterministic: bool = True,
) -> RolloutArtifact:
    """Run a single rollout and save artifacts.

    State log columns vary by env, but we always store:
      obs[t]        (T, obs_dim)
      action[t]     (T, act_dim)
      reward[t]     (T,)
      terminated    (T,) bool
      truncated     (T,) bool
    """
    out_dir = out_dir or paths.ROLLOUTS
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_id = ckpt_path.stem
    video_path = out_dir / f"{ckpt_id}.mp4"
    state_path = out_dir / f"{ckpt_id}.npz"
    meta_path = out_dir / f"{ckpt_id}.rollout.json"

    env = gym.make(env_id, render_mode="rgb_array")
    model = PPO.load(str(ckpt_path), device="cpu")

    obs, _info = env.reset(seed=seed)
    frames: list[np.ndarray] = []
    obs_log, act_log, rew_log = [], [], []
    term_log, trunc_log = [], []

    for _ in range(n_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs_log.append(np.asarray(obs, dtype=np.float32))
        act_log.append(np.asarray(action, dtype=np.float32))
        obs, reward, terminated, truncated, _info = env.step(action)
        rew_log.append(float(reward))
        term_log.append(bool(terminated))
        trunc_log.append(bool(truncated))
        frames.append(env.render())
        if terminated or truncated:
            obs, _info = env.reset()

    env.close()

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

    artifact = RolloutArtifact(
        ckpt_id=ckpt_id,
        env_id=env_id,
        video_path=video_path,
        state_path=state_path,
        n_steps=len(frames),
        fps=fps,
    )
    meta_path.write_text(json.dumps(artifact.to_dict(), indent=2))
    return artifact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path, help="Path to .zip checkpoint")
    ap.add_argument("--env", required=True, help="Gymnasium env id (e.g. Walker2d-v4)")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    art = rollout(args.ckpt, args.env, n_steps=args.steps, fps=args.fps, seed=args.seed)
    print(json.dumps(art.to_dict(), indent=2))


if __name__ == "__main__":
    main()
