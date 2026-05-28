"""Smoke test: deps + sandbox + reference grid + AE + (optional) tiny e2e."""
from __future__ import annotations

import importlib
import os
import sys
import traceback

CHECKS = [
    "gymnasium", "stable_baselines3", "torch", "mujoco", "imageio", "PIL",
    "anthropic", "sentence_transformers", "sklearn", "matplotlib",
]


def _check_imports():
    ok = True
    for name in CHECKS:
        try:
            mod = importlib.import_module(name)
            print(f"  ok  {name:25s} {getattr(mod, '__version__', '?')}")
        except Exception as e:
            print(f"  FAIL {name:25s} {e}")
            ok = False
    return ok


def _check_reward_sandbox():
    from src.rl.reward_template import CompiledReward, RewardCodeError
    good = """
def compute_reward(obs, action, next_obs, done):
    return float(next_obs[5]) - 0.001 * float(np.sum(action ** 2))
"""
    fn = CompiledReward.from_code(good)
    import numpy as np
    r = fn(np.zeros(11), np.zeros(3), np.ones(11), False)
    assert isinstance(r, float)
    print(f"  ok  reward sandbox: legal compute_reward returned {r}")

    evil = """
import os
def compute_reward(obs, action, next_obs, done):
    return os.system('echo hax')
"""
    try:
        CompiledReward.from_code(evil)
        print("  FAIL reward sandbox didn't reject import")
        return False
    except RewardCodeError as e:
        print(f"  ok  reward sandbox rejected evil code: {e}")
    return True


def _check_env_wrapper():
    import gymnasium as gym
    from src.rl.env import RewardOverrideEnv
    from src.rl.reward_template import CompiledReward
    fn = CompiledReward.from_code(
        "def compute_reward(obs, action, next_obs, done):\n    return float(next_obs[5])\n"
    )
    env = RewardOverrideEnv(gym.make("Hopper-v5"), fn)
    obs, _ = env.reset(seed=0)
    for _ in range(5):
        obs, r, term, trunc, _ = env.step(env.action_space.sample())
    env.close()
    print(f"  ok  RewardOverrideEnv stepped Hopper-v5; last r={r:.4f}")


def _check_archive():
    import numpy as np
    from src.archive.entry import Entry
    from src.archive.knn import KNNArchive
    arc = KNNArchive(embed_key="vlm", rng_seed=0, tau_novel=0.3)

    def mk(_id, vec, fit):
        return Entry(
            id=_id, iteration=0, parent_id=None, reward_path="",
            fitness=fit, description=f"t{_id}", ckpt_path="",
            video_path="", state_path="", embed_vlm=np.asarray(vec, dtype=np.float32),
        )

    # 3 clearly different embeddings
    s1 = arc.insert(mk("a", [1, 0, 0, 0], 1.0))
    s2 = arc.insert(mk("b", [0, 1, 0, 0], 2.0))
    s3 = arc.insert(mk("c", [1, 0, 0, 0.05], 0.5))   # near 'a', worse → dominated
    print(f"  ok  insert sequence: {s1} {s2} {s3} (expect seeded novel dominated)")
    assert (s1, s2, s3) == ("seeded", "novel", "dominated"), (s1, s2, s3)

    # density sample (with only 2 entries)
    p = arc.sample_parent()
    assert p is not None
    print(f"  ok  density-sampled parent: {p.id}")


def _check_ref_grid():
    import numpy as np
    from src.eval_grid.descriptor import descriptor
    from src.eval_grid.grid import build_grid, coverage, qd_score

    fake_state = {
        "obs": np.stack([np.array([1.2 + 0.1 * i, 0, 0.3 * np.sin(i * 0.5), 0, 0,
                                   1.0 + 0.05 * i, 0, 0, 0, 0, 0]) for i in range(60)])
                 .astype(np.float32),
        "action": np.zeros((60, 3), dtype=np.float32),
    }
    coord = descriptor(fake_state)
    print(f"  ok  ref descriptor: {coord}")

    coords = [coord, (0.5, 1.0, 1.0), (1.5, 1.3, 2.0), (1.7, 1.1, 3.0), (0.6, 1.2, 1.5)]
    grid = build_grid(coords, n_bins=3)
    print(f"  ok  grid: cells={grid.n_cells} coverage={coverage(grid, coords)*100:.0f}%  "
          f"qd={qd_score(grid, coords, [1.0]*len(coords)):.2f}")


def _check_autoencoder():
    import numpy as np
    from src.embed_aurora.pool import POOLED_DIM
    from src.embed_aurora.train import fit_autoencoder
    feats = np.random.RandomState(0).randn(8, POOLED_DIM).astype(np.float32)
    model = fit_autoencoder(feats, epochs=10, lr=1e-2, seed=0)
    z = model.embed(feats)
    print(f"  ok  AE embed shape: {z.shape}")


def main() -> int:
    print(f"python={sys.version.split()[0]}  platform={sys.platform}")
    print("[1] imports")
    if not _check_imports():
        return 1
    print("[2] reward sandbox")
    if not _check_reward_sandbox():
        return 1
    print("[3] env wrapper")
    try:
        _check_env_wrapper()
    except Exception:
        traceback.print_exc(); return 1
    print("[4] k-NN archive")
    try:
        _check_archive()
    except Exception:
        traceback.print_exc(); return 1
    print("[5] reference grid")
    try:
        _check_ref_grid()
    except Exception:
        traceback.print_exc(); return 1
    print("[6] autoencoder")
    try:
        _check_autoencoder()
    except Exception:
        traceback.print_exc(); return 1
    print("\n[verify] all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
