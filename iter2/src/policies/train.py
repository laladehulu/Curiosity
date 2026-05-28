"""Train a PPO policy and snapshot it at multiple checkpoints.

Usage:
    python -m src.policies.train --name walker2d
    python -m src.policies.train --all
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from src import paths
from src.policies.catalog import DEFAULT_CATALOG, PolicySpec, checkpoint_id


class CheckpointCallback(BaseCallback):
    """Save the model whenever num_timesteps crosses a target."""

    def __init__(self, spec: PolicySpec, out_dir: Path):
        super().__init__()
        self.spec = spec
        self.out_dir = out_dir
        self.targets = sorted(spec.checkpoint_at)
        self._next_idx = 0

    def _on_step(self) -> bool:
        while (
            self._next_idx < len(self.targets)
            and self.num_timesteps >= self.targets[self._next_idx]
        ):
            step = self.targets[self._next_idx]
            ckpt_id = checkpoint_id(self.spec, step)
            path = self.out_dir / f"{ckpt_id}.zip"
            self.model.save(str(path))
            meta = {
                "ckpt_id": ckpt_id,
                "spec": self.spec.__dict__ | {"checkpoint_at": list(self.spec.checkpoint_at)},
                "step_recorded": int(self.num_timesteps),
            }
            (self.out_dir / f"{ckpt_id}.json").write_text(json.dumps(meta, indent=2))
            if self.verbose:
                print(f"[checkpoint] saved {ckpt_id} at step={self.num_timesteps}")
            self._next_idx += 1
        return True


def train_spec(spec: PolicySpec, out_dir: Path | None = None, verbose: int = 1) -> list[Path]:
    out_dir = out_dir or paths.POLICIES
    out_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(spec.env_id)
    model = PPO("MlpPolicy", env, seed=spec.seed, verbose=verbose, device="cpu")

    cb = CheckpointCallback(spec, out_dir)
    t0 = time.time()
    model.learn(total_timesteps=spec.total_steps, callback=cb, progress_bar=False)
    dt = time.time() - t0
    if verbose:
        print(f"[train] {spec.name} done in {dt:.1f}s ({spec.total_steps} steps)")

    env.close()
    return [out_dir / f"{checkpoint_id(spec, s)}.zip" for s in spec.checkpoint_at]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", help="Name from catalog (omit to use --all)")
    ap.add_argument("--all", action="store_true", help="Train every spec in the catalog")
    ap.add_argument("--quick", action="store_true", help="Override to ~20k steps for smoke test")
    args = ap.parse_args()

    specs = DEFAULT_CATALOG if args.all else [s for s in DEFAULT_CATALOG if s.name == args.name]
    if not specs:
        raise SystemExit(f"no spec found for name={args.name!r}")

    if args.quick:
        specs = [
            PolicySpec(
                name=s.name,
                env_id=s.env_id,
                total_steps=20_000,
                checkpoint_at=(10_000, 20_000),
                seed=s.seed,
            )
            for s in specs
        ]

    for s in specs:
        print(f"\n=== training {s.name} ({s.env_id}, {s.total_steps} steps) ===")
        train_spec(s)


if __name__ == "__main__":
    main()
