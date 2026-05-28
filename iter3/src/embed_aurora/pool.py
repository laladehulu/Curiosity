"""Trajectory -> fixed-size pooled feature vector for the AURORA-style AE."""
from __future__ import annotations

import numpy as np


def pool_trajectory(state_npz_or_dict) -> np.ndarray:
    """Take an npz-loaded state log and return a 28-d feature.

    Layout: [mean(obs)(11) | std(obs)(11) | mean(action)(3) | std(action)(3)]
    """
    if hasattr(state_npz_or_dict, "files"):
        # NpzFile
        obs = state_npz_or_dict["obs"]
        action = state_npz_or_dict["action"]
    else:
        obs = state_npz_or_dict["obs"]
        action = state_npz_or_dict["action"]
    obs = np.asarray(obs, dtype=np.float32)
    action = np.asarray(action, dtype=np.float32)
    feat = np.concatenate([
        obs.mean(axis=0),
        obs.std(axis=0),
        action.mean(axis=0),
        action.std(axis=0),
    ]).astype(np.float32)
    # guard against NaNs (zero-length rollout shouldn't happen but)
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    return feat


POOLED_DIM = 11 + 11 + 3 + 3  # 28
