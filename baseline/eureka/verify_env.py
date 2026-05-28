"""Verify the Eureka baseline environment is ready to run.

By default, runs *offline* checks only: imports, MuJoCo HalfCheetah-v4,
PPO smoke (1k steps), reward compile. No Anthropic key required.

Pass `--with-anthropic` to additionally exercise the Claude API path
(spends ~$0.01).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def check_imports() -> None:
    print("[1/5] imports...", flush=True)
    import gymnasium  # noqa
    import mujoco  # noqa
    import stable_baselines3  # noqa
    import torch  # noqa
    import numpy  # noqa
    try:
        import anthropic  # noqa
    except ImportError:
        print("      (anthropic not installed; mock-llm mode will still work)")
    print("      OK")


def check_env() -> None:
    print("[2/5] HalfCheetah-v4 makes + steps...", flush=True)
    import gymnasium as gym
    env = gym.make("HalfCheetah-v4")
    obs, info = env.reset(seed=0)
    assert obs.shape == (17,), f"unexpected obs shape: {obs.shape}"
    for _ in range(3):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
    env.close()
    print(f"      OK (obs shape {obs.shape}, info keys: {sorted(info.keys())})")


def check_mock_llm() -> None:
    print("[3/5] mock LLM bank (offline)...", flush=True)
    from mock_llm import MockLLMClient
    m = MockLLMClient()
    seen = set()
    for _ in range(4):
        resp = m.chat("sys", "user")
        assert "compute_reward" in resp.text or "```" in resp.text
        seen.add(resp.text[:30])
    assert len(seen) >= 3, "mock LLM should rotate through variants"
    print(f"      OK (rotated through {len(seen)} distinct variants)")


def check_anthropic_optional() -> None:
    print("[extra] Anthropic key + tiny chat call...", flush=True)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("      SKIPPED — ANTHROPIC_API_KEY not set")
        return
    from components import LLMClient
    llm = LLMClient(model="claude-haiku-4-5", max_tokens=64)
    resp = llm.chat("Reply with exactly 'OK'.", "say OK")
    print(f"        OK (got: {resp.text!r}, ~${llm.cost_estimate_usd():.4f})")


def check_ppo_smoke() -> None:
    print("[4/5] PPO smoke (1k steps under env-true reward)...", flush=True)
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    t0 = time.time()
    vec = DummyVecEnv([lambda: Monitor(gym.make("HalfCheetah-v4"))])
    model = PPO("MlpPolicy", vec, verbose=0, device="cpu", seed=0, n_steps=512)
    model.learn(total_timesteps=1024)
    dt = time.time() - t0
    print(f"      OK ({dt:.1f}s)")


def check_compile_reward() -> None:
    print("[5/5] compile + safe_call of a hand-written reward...", flush=True)
    from components import compile_reward, safe_call
    src = (
        "def compute_reward(obs, action, next_obs, done):\n"
        "    return float(next_obs[8] - 0.1 * (action ** 2).sum())\n"
    )
    c = compile_reward(src)
    import numpy as np
    val = safe_call(
        c, np.zeros(17), np.zeros(6), np.array([0.0] * 8 + [1.0] + [0.0] * 8), False
    )
    assert val > 0.99 and val < 1.01, f"unexpected reward: {val}"
    print(f"      OK (got {val:.3f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-anthropic", action="store_true",
                    help="Also exercise the Anthropic API path (spends ~$0.01).")
    args = ap.parse_args()

    check_imports()
    check_env()
    check_mock_llm()
    check_ppo_smoke()
    check_compile_reward()
    if args.with_anthropic:
        check_anthropic_optional()

    print("\nAll offline checks passed. Proof-of-concept run:")
    print("    python eureka.py --mock-llm --smoke")


if __name__ == "__main__":
    main()
