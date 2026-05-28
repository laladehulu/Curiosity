"""Hand-designed Hopper-v5 reference descriptor — 3-D.

Used ONLY for evaluation, never for selection. Computed once per candidate
from the (default-reward) rollout state log.

Axes:
  0: mean_forward_velocity   = mean(obs[:, 5])
  1: mean_body_height        = mean(obs[:, 0])
  2: gait_frequency          = dominant FFT freq of obs[:, 2] (thigh angle),
                               normalized to control-rate. We report it in
                               Hz, assuming 30 fps control rate, but the
                               grid bins are quantile-based so units cancel.
"""
from __future__ import annotations

import numpy as np


def descriptor(state: dict, fps: int = 30) -> tuple[float, float, float]:
    obs = np.asarray(state["obs"], dtype=np.float32)
    if obs.shape[0] < 4:
        return (0.0, 0.0, 0.0)
    mean_fwd_vel = float(obs[:, 5].mean())
    mean_height = float(obs[:, 0].mean())
    # gait frequency via FFT on thigh joint angle
    thigh = obs[:, 2] - obs[:, 2].mean()
    if thigh.std() < 1e-6:
        gait_freq = 0.0
    else:
        # rfft → power spectrum
        spec = np.abs(np.fft.rfft(thigh))
        freqs = np.fft.rfftfreq(thigh.shape[0], d=1.0 / fps)
        # skip DC and very-low-freq bin
        mask = freqs > 0.5
        if not mask.any():
            gait_freq = 0.0
        else:
            idx = int(np.argmax(spec[mask]))
            gait_freq = float(freqs[mask][idx])
    return (mean_fwd_vel, mean_height, gait_freq)
