"""Environment verification per protocol §"Verification".

Exit nonzero with a clear error on any failure. Do not silently proceed.
"""
from __future__ import annotations

import os
import platform
import sys
import time
import traceback


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def step(label: str) -> None:
    print(f"[verify] {label} ...", flush=True)


def main() -> None:
    print(f"Python {sys.version.split()[0]} on {platform.platform()}")
    print(f"CPU count: {os.cpu_count()}")

    # 1. Imports
    step("imports")
    try:
        import numpy  # noqa: F401
        import scipy  # noqa: F401
        import sklearn  # noqa: F401
        import pandas  # noqa: F401
        import matplotlib  # noqa: F401
        import seaborn  # noqa: F401
        import torch
        import gymnasium as gym
        import stable_baselines3
        import anthropic  # noqa: F401
        import zss  # noqa: F401
    except Exception as e:
        traceback.print_exc()
        fail(f"import error: {e}")
    print(f"  torch={torch.__version__} sb3={stable_baselines3.__version__} gym={gym.__version__}")
    if torch.cuda.is_available():
        print("  NOTE: CUDA detected; protocol assumes CPU-only. Set CUDA_VISIBLE_DEVICES= to force CPU.")

    # 2. Task smoke — each env can step 100 random actions
    step("env smoke (Pendulum, MountainCar, BipedalWalker)")
    tasks_ok: list[str] = []
    tasks_failed: list[tuple[str, str]] = []
    for env_id in ["Pendulum-v1", "MountainCarContinuous-v0", "BipedalWalker-v3"]:
        try:
            env = gym.make(env_id)
            env.reset(seed=0)
            for _ in range(100):
                obs, r, term, trunc, info = env.step(env.action_space.sample())
                if term or trunc:
                    env.reset()
            env.close()
            tasks_ok.append(env_id)
        except Exception as e:
            tasks_failed.append((env_id, str(e)))
    print(f"  OK: {tasks_ok}")
    if tasks_failed:
        print(f"  FAILED: {tasks_failed}")
        if not tasks_ok or all(t.startswith("Pendulum") or t.startswith("MountainCar") for t, _ in tasks_failed):
            # if at least Pendulum + MountainCar are ok, the protocol allows reduced scope
            if "Pendulum-v1" not in tasks_ok or "MountainCarContinuous-v0" not in tasks_ok:
                fail("Pendulum or MountainCar broken — cannot proceed.")
            print("  WARN: BipedalWalker not available; protocol allows reduced scope.")

    # 3. Anthropic key + small call (claude-haiku-4-5 — Anthropic equivalent of
    #    protocol §Verification step 3's gpt-4o-mini smoke test)
    step("anthropic smoke (claude-haiku-4-5)")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        fail("ANTHROPIC_API_KEY not set. export ANTHROPIC_API_KEY=sk-ant-... before running.")
    try:
        from anthropic import Anthropic
        client = Anthropic()
        resp = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=8,
            messages=[{"role": "user", "content": "Reply with exactly the token: ok"}],
        )
        body = next((b.text for b in resp.content if b.type == "text"), "").strip().lower()
        if "ok" not in body:
            fail(f"unexpected response from claude-haiku-4-5: {body!r}")
        print(f"  response: {body!r}")
    except Exception as e:
        traceback.print_exc()
        fail(f"anthropic call failed: {e}")

    # 4. PPO smoke — 10k steps on Pendulum, ~10s on Mac
    step("ppo smoke (Pendulum, 10k steps)")
    try:
        from stable_baselines3 import PPO
        env = gym.make("Pendulum-v1")
        model = PPO("MlpPolicy", env, verbose=0, device="cpu", n_steps=512, seed=0)
        t0 = time.time()
        model.learn(total_timesteps=10_000)
        dt = time.time() - t0
        env.close()
        print(f"  PPO 10k steps in {dt:.1f}s")
    except Exception as e:
        traceback.print_exc()
        fail(f"PPO smoke failed: {e}")

    print("\nVerification PASSED. You may proceed to Phase 1 preflight.")


if __name__ == "__main__":
    main()
