"""BipedalWalker-v3 behavioral descriptor.

Per protocol §"BipedalWalker-v3":
    (gait frequency from hip joint, mean forward velocity, mean vertical body oscillation)
    3D, 4×4×4 = 64 cells.

Indices in obs (gymnasium BipedalWalker-v3):
    obs[2]  x velocity
    obs[3]  y velocity
    obs[4]  hip1 angle (used for gait frequency estimation via zero-crossings)
"""
from __future__ import annotations

import numpy as np

BINS: tuple[int, ...] = (4, 4, 4)
_GAITFREQ_RANGE = (0.0, 3.0)    # Hz; episodes ~ 1600 steps at 50Hz = 32s
_VX_RANGE = (-0.5, 1.5)
_OSC_RANGE = (0.0, 0.5)


def _gait_freq(hip_angle: np.ndarray, dt_per_step: float = 1.0 / 50.0) -> float:
    """Estimate hip-joint oscillation frequency via zero-crossings of detrended signal."""
    if hip_angle.size < 4:
        return 0.0
    centered = hip_angle - np.mean(hip_angle)
    signs = np.sign(centered)
    # count sign changes
    crossings = int(np.sum(signs[1:] != signs[:-1]))
    duration = hip_angle.size * dt_per_step
    # each cycle = 2 crossings
    return float(crossings / 2.0 / max(duration, 1e-6))


def compute_descriptor(trajs: list[dict]) -> np.ndarray:
    freqs, vxs, oscs = [], [], []
    for ep in trajs:
        obs = ep["obs"]
        if obs.shape[0] < 2:
            continue
        hip = obs[:, 4]
        freqs.append(_gait_freq(hip))
        vxs.append(float(np.mean(obs[:, 2])))
        oscs.append(float(np.std(obs[:, 3])))  # vertical velocity std as oscillation proxy
    if not freqs:
        return np.array([0.0, 0.0, 0.0])
    return np.array([np.mean(freqs), np.mean(vxs), np.mean(oscs)], dtype=np.float64)


def cell_index(desc: np.ndarray) -> tuple[int, ...]:
    nf, nv, no = BINS
    f_norm = (desc[0] - _GAITFREQ_RANGE[0]) / (_GAITFREQ_RANGE[1] - _GAITFREQ_RANGE[0])
    v_norm = (desc[1] - _VX_RANGE[0]) / (_VX_RANGE[1] - _VX_RANGE[0])
    o_norm = (desc[2] - _OSC_RANGE[0]) / (_OSC_RANGE[1] - _OSC_RANGE[0])
    i = int(np.clip(f_norm * nf, 0, nf - 1))
    j = int(np.clip(v_norm * nv, 0, nv - 1))
    k = int(np.clip(o_norm * no, 0, no - 1))
    return (i, j, k)
