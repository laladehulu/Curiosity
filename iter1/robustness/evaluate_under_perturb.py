"""Phase 4 — evaluate a policy pool's robustness under environment perturbations.

Pool robustness score = sum over perturbations of (max over pool of success rate).
This matches protocol §"Perturbation protocol".
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import gymnasium as gym
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reward_gen._loop_common import TASKS  # noqa: E402
from robustness.perturbations import PERTURBATIONS, task_success_threshold  # noqa: E402


def evaluate_policy_on_env(model, env_id: str, perturbation, n_eps: int = 50, seed: int = 0):
    """Run n_eps episodes, return (success_rate, mean_return)."""
    successes = 0
    returns = []
    for i in range(n_eps):
        env = gym.make(env_id)
        env = perturbation.apply(env)
        obs, _ = env.reset(seed=seed * 1_000 + i)
        env = perturbation.apply(env)  # re-apply after reset for box2d-style envs
        total = 0.0
        done = False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(a)
            total += float(r)
            done = bool(term or trunc)
        env.close()
        returns.append(total)
    return float(np.mean(returns)), returns


def pool_robustness(task: str, checkpoint_paths: Sequence[Path], n_eps: int = 50, seed: int = 0) -> dict:
    """Compute pool robustness = sum over perturbations of (max over pool of success rate)."""
    from stable_baselines3 import PPO
    spec = TASKS[task]
    success_thresh = task_success_threshold(task)

    per_perturb: dict[str, dict] = {}
    pool_summed_max_success = 0.0

    for p in PERTURBATIONS[task]:
        per_policy: list[dict] = []
        for ckpt in checkpoint_paths:
            model = PPO.load(str(ckpt), device="cpu")
            mean_ret, all_rets = evaluate_policy_on_env(model, spec.env_id, p, n_eps=n_eps, seed=seed)
            success_rate = float(np.mean([1.0 if r >= success_thresh else 0.0 for r in all_rets]))
            per_policy.append(
                {"checkpoint": str(ckpt), "mean_return": mean_ret, "success_rate": success_rate}
            )
        max_success = max((r["success_rate"] for r in per_policy), default=0.0)
        pool_summed_max_success += max_success
        per_perturb[p.name] = {"per_policy": per_policy, "max_success_in_pool": max_success}

    return {
        "task": task,
        "pool_size": len(checkpoint_paths),
        "perturbations": per_perturb,
        "pool_robustness_score": pool_summed_max_success,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS.keys()))
    ap.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    ap.add_argument("--n-eps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    result = pool_robustness(args.task, args.checkpoints, n_eps=args.n_eps, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "perturbations"}, indent=2))


if __name__ == "__main__":
    main()
