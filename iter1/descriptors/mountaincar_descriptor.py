"""MountainCarContinuous-v0 behavioral descriptor.

Per protocol §"MountainCarContinuous-v0":
    (max position reached before first success, mean absolute action)
    2D, 5×5 = 25 cells.

"Max position before first success" is hard to define for unsuccessful policies;
we use max position over the episode regardless. This is a conservative deviation
from the protocol; document in critique.md.
"""
from __future__ import annotations

import numpy as np

BINS: tuple[int, ...] = (5, 5)
_POS_RANGE = (-1.2, 0.6)
_ABSACT_RANGE = (0.0, 1.0)


def compute_descriptor(trajs: list[dict]) -> np.ndarray:
    max_positions = []
    abs_actions = []
    for ep in trajs:
        positions = ep["obs"][:, 0]
        max_positions.append(float(np.max(positions)))
        if len(ep["action"]) > 0:
            abs_actions.append(float(np.mean(np.abs(ep["action"]))))
        else:
            abs_actions.append(0.0)
    return np.array([np.mean(max_positions), np.mean(abs_actions)], dtype=np.float64)


def cell_index(desc: np.ndarray) -> tuple[int, ...]:
    nb_pos, nb_act = BINS
    pos_norm = (desc[0] - _POS_RANGE[0]) / (_POS_RANGE[1] - _POS_RANGE[0])
    act_norm = (desc[1] - _ABSACT_RANGE[0]) / (_ABSACT_RANGE[1] - _ABSACT_RANGE[0])
    i = int(np.clip(pos_norm * nb_pos, 0, nb_pos - 1))
    j = int(np.clip(act_norm * nb_act, 0, nb_act - 1))
    return (i, j)
