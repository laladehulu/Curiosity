"""State-visitation entropy as a curiosity signal.

Rolls out a trained policy for N episodes, bins the visited observations,
and returns the Shannon entropy of the resulting histogram.  Higher entropy
means the policy visits a more diverse set of states.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

# Per-task observation config: which dims to use, how many bins, value ranges.
_TASK_OBS_CONFIG: dict[str, dict] = {
    "pendulum": {
        "dims": [0, 1, 2],
        "bins": [10, 10, 10],
        "ranges": [(-1.0, 1.0), (-1.0, 1.0), (-8.0, 8.0)],
    },
    "mountaincar": {
        "dims": [0, 1],
        "bins": [20, 20],
        "ranges": [(-1.2, 0.6), (-0.07, 0.07)],
    },
    "bipedalwalker": {
        "dims": [2, 3, 4],
        "bins": [10, 10, 10],
        "ranges": [(-1.0, 1.0), (-2.0, 2.0), (-2.0, 2.0)],
    },
}


def compute_state_entropy(
    model,
    env_id: str,
    task_name: str,
    n_episodes: int = 20,
    seed: int = 0,
) -> float:
    """Roll out *model* for *n_episodes* and return Shannon entropy of binned obs.

    Returns 0.0 if no observations are collected (e.g. immediate termination).
    """
    cfg = _TASK_OBS_CONFIG.get(task_name)
    if cfg is None:
        raise ValueError(f"No obs config for task {task_name!r}; known: {list(_TASK_OBS_CONFIG)}")

    dims = cfg["dims"]
    bins = cfg["bins"]
    ranges = cfg["ranges"]

    all_obs: list[np.ndarray] = []
    for ep in range(n_episodes):
        env = gym.make(env_id)
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            all_obs.append(np.asarray(obs, dtype=np.float64)[dims])
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(action)
            done = bool(term or trunc)
        env.close()

    if not all_obs:
        return 0.0

    sample = np.stack(all_obs)  # (N, D)
    # Clip to ranges to avoid out-of-range bins
    for i, (lo, hi) in enumerate(ranges):
        sample[:, i] = np.clip(sample[:, i], lo, hi)

    hist, _ = np.histogramdd(sample, bins=bins, range=ranges)
    # Normalize to probability distribution
    total = hist.sum()
    if total == 0:
        return 0.0
    p = hist.ravel() / total
    p = p[p > 0]
    return -float(np.sum(p * np.log(p)))
