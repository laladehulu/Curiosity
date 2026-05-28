"""Smoke test for iter2 deps + a minimal end-to-end run (no VLM by default)."""
from __future__ import annotations

import importlib
import os
import sys
import traceback

CHECKS = [
    ("gymnasium", "0.29"),
    ("stable_baselines3", "2.3"),
    ("torch", "2.0"),
    ("mujoco", "3.0"),
    ("imageio", "2.30"),
    ("PIL", None),
    ("ruptures", None),
    ("anthropic", None),
    ("chromadb", None),
    ("sentence_transformers", None),
]


def _check_import():
    ok = True
    for name, _min in CHECKS:
        try:
            mod = importlib.import_module(name)
            v = getattr(mod, "__version__", "?")
            print(f"  ok  {name:25s} {v}")
        except Exception as e:
            print(f"  FAIL {name:25s} {e}")
            ok = False
    return ok


def _check_env():
    import gymnasium as gym
    env = gym.make("Walker2d-v5", render_mode="rgb_array")
    obs, _ = env.reset(seed=0)
    for _ in range(10):
        obs, *_ = env.step(env.action_space.sample())
    frame = env.render()
    env.close()
    assert frame is not None and frame.shape[-1] == 3
    print(f"  ok  Walker2d-v5 stepped + rendered frame {frame.shape}")


def _check_pipeline_dry_run():
    """Train a tiny PPO, roll it out, segment, dry-run describe, index."""
    from pathlib import Path

    from src import paths
    from src.policies.catalog import PolicySpec
    from src.policies.train import train_spec
    import importlib
    pipeline = importlib.import_module("pipeline")

    spec = PolicySpec(
        name="_smoke",
        env_id="Hopper-v5",
        total_steps=2_000,
        checkpoint_at=(2_000,),
    )
    saved = train_spec(spec, verbose=0)
    ckpt = saved[0]
    print(f"  ok  trained smoke ckpt: {ckpt.name}")

    n = pipeline.process_checkpoint(
        ckpt, env_id="Hopper-v5",
        n_steps=120, fps=30,
        max_seg_seconds=3.0, min_seg_seconds=1.0,
        force_rollout=True, force_describe=True,
        n_frames_per_seg=2,
        dry_run_describe=True,
    )
    print(f"  ok  pipeline dry-run indexed {n} segments")

    # Clean up the smoke artifacts so they don't pollute the real index.
    from src.index.store import get_collection
    col = get_collection()
    got = col.get(where={"ckpt_id": ckpt.stem})
    if got["ids"]:
        col.delete(ids=got["ids"])
    for p in [
        paths.POLICIES / f"{ckpt.stem}.zip",
        paths.POLICIES / f"{ckpt.stem}.json",
        paths.ROLLOUTS / f"{ckpt.stem}.mp4",
        paths.ROLLOUTS / f"{ckpt.stem}.npz",
        paths.ROLLOUTS / f"{ckpt.stem}.rollout.json",
        paths.DESCRIPTIONS / f"{ckpt.stem}.jsonl",
    ]:
        if p.exists():
            p.unlink()


def main() -> int:
    print(f"python={sys.version.split()[0]}  platform={sys.platform}")
    print("[1/3] imports")
    if not _check_import():
        return 1

    print("[2/3] gym + mujoco round-trip")
    try:
        _check_env()
    except Exception:
        traceback.print_exc()
        return 1

    if os.environ.get("ITER2_VERIFY_PIPELINE") == "1":
        print("[3/3] pipeline dry-run")
        try:
            _check_pipeline_dry_run()
        except Exception:
            traceback.print_exc()
            return 1
    else:
        print("[3/3] pipeline dry-run skipped (set ITER2_VERIFY_PIPELINE=1 to enable)")

    print("\n[verify] all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
