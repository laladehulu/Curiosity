"""Pendulum-v1 behavioral descriptor.

Per protocol §"Pendulum-v1":
    (mean angular velocity during episode, fraction of episode within ±0.2 rad of upright)
    2D, each axis binned into 5 cells → 25 cells total.
"""
from __future__ import annotations

import math
import numpy as np

BINS: tuple[int, ...] = (5, 5)
# axis bounds chosen from env spec
_OMEGA_RANGE = (-8.0, 8.0)
_FRAC_RANGE = (0.0, 1.0)


def compute_descriptor(trajs: list[dict]) -> np.ndarray:
    """trajs is a list of dicts {"obs": np.ndarray (T+1, 3), "action": (T,1), ...}.
    Pendulum obs = [cos(theta), sin(theta), theta_dot]."""
    omegas = []
    frac_uprights = []
    for ep in trajs:
        obs = ep["obs"]  # (T+1, 3)
        cos_th = obs[:, 0]
        sin_th = obs[:, 1]
        th_dot = obs[:, 2]
        # angle from upright: theta where 0 = upright
        theta = np.arctan2(sin_th, cos_th)
        omegas.append(float(np.mean(np.abs(th_dot))))
        frac_uprights.append(float(np.mean(np.abs(theta) < 0.2)))
    return np.array([np.mean(omegas), np.mean(frac_uprights)], dtype=np.float64)


def cell_index(desc: np.ndarray) -> tuple[int, ...]:
    nb_omega, nb_frac = BINS
    omega_norm = (desc[0] - _OMEGA_RANGE[0]) / (_OMEGA_RANGE[1] - _OMEGA_RANGE[0])
    frac_norm = (desc[1] - _FRAC_RANGE[0]) / (_FRAC_RANGE[1] - _FRAC_RANGE[0])
    i = int(np.clip(omega_norm * nb_omega, 0, nb_omega - 1))
    j = int(np.clip(frac_norm * nb_frac, 0, nb_frac - 1))
    return (i, j)
