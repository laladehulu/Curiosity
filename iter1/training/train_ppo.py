"""Single PPO training run for Phase 2 (full training of a chosen reward).

Use after Phase 1. Typical invocation:
    python -m training.train_ppo --task pendulum --method pareto --reward-idx 7 --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reward_gen._loop_common import RewardOverrideEnv, TASKS  # noqa: E402
from reward_gen.reward_template import compile_reward  # noqa: E402
from training.ppo_configs import PHASE2_CONFIGS  # noqa: E402


def make_full_env(env_id: str, compiled):
    from stable_baselines3.common.monitor import Monitor
    return Monitor(RewardOverrideEnv(gym.make(env_id), compiled))


def train(task: str, method: str, reward_idx: int, seed: int, save_dir: Path) -> dict:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    spec = TASKS[task]
    cfg = PHASE2_CONFIGS[task]
    pool = ROOT / "reward_gen" / "pool" / f"{task}_{method}"
    src = (pool / f"reward_{reward_idx:03d}.py").read_text()
    compiled = compile_reward(src)

    vec = DummyVecEnv([lambda: make_full_env(spec.env_id, compiled)])
    model = PPO(
        "MlpPolicy",
        vec,
        verbose=0,
        device="cpu",
        seed=seed,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        learning_rate=cfg.learning_rate,
        policy_kwargs=cfg.policy_kwargs,
    )
    model.learn(total_timesteps=cfg.total_timesteps)
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / f"{task}_{method}_r{reward_idx:03d}_s{seed}.zip"
    model.save(out)
    meta = {
        "task": task, "method": method, "reward_idx": reward_idx, "seed": seed,
        "total_timesteps": cfg.total_timesteps, "checkpoint": str(out),
    }
    (save_dir / f"{task}_{method}_r{reward_idx:03d}_s{seed}.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS.keys()))
    ap.add_argument("--method", required=True, choices=["pareto", "archive"])
    ap.add_argument("--reward-idx", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-dir", type=Path, default=HERE / "checkpoints")
    args = ap.parse_args()
    meta = train(args.task, args.method, args.reward_idx, args.seed, args.save_dir)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
