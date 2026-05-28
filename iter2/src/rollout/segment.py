"""Segment a rollout into behavioral phases.

We combine three signals:
  1. Episode resets (terminated/truncated) — hard boundaries.
  2. Change points in the smoothed observation stream (via ruptures PELT).
  3. Max-length split — if a phase is longer than `max_seconds`, halve it.

Min phase length enforces that we don't emit one-frame segments after a chain
of changepoints fires; max phase length keeps each VLM call cheap.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import numpy as np


@dataclass
class Segment:
    idx: int
    start_step: int
    end_step: int          # exclusive
    start_frame: int
    end_frame: int         # exclusive
    duration_s: float
    why: str               # 'changepoint' | 'episode_end' | 'min_length' | 'max_split'


def _smooth(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    out = np.empty_like(x)
    for d in range(x.shape[1]):
        out[:, d] = np.convolve(x[:, d], kernel, mode="same")
    return out


def _detect_changepoints(obs: np.ndarray, penalty_mult: float = 6.0) -> List[int]:
    """Return change-point indices (excluding 0 and T) on smoothed obs."""
    import ruptures as rpt

    T, D = obs.shape
    smoothed = _smooth(obs, window=max(5, T // 100))
    # Reduce dimensionality cheaply: keep top-variance components.
    if D > 6:
        variances = smoothed.var(axis=0)
        keep = np.argsort(variances)[-6:]
        feat = smoothed[:, keep]
    else:
        feat = smoothed
    algo = rpt.Pelt(model="rbf", min_size=max(15, T // 30)).fit(feat)
    pen = penalty_mult * np.log(max(T, 2)) * feat.shape[1]
    bkps = algo.predict(pen=pen)
    return [b for b in bkps if 0 < b < T]


def segment_rollout(
    state_path: Path,
    fps: int = 30,
    min_seconds: float = 1.0,
    max_seconds: float = 6.0,
    penalty_mult: float = 6.0,
) -> List[Segment]:
    npz = np.load(state_path)
    obs = npz["obs"]
    terminated = npz["terminated"]
    truncated = npz["truncated"]
    T = obs.shape[0]
    min_len = max(2, int(min_seconds * fps))
    max_len = max(min_len + 1, int(max_seconds * fps))

    # 1. Hard boundaries from episode resets.
    resets = np.where(terminated | truncated)[0].tolist()
    hard = sorted({i + 1 for i in resets} | {T})

    # 2. Change-point boundaries.
    try:
        cps = _detect_changepoints(obs, penalty_mult=penalty_mult)
    except Exception as e:
        print(f"[segment] changepoint failed ({e}); falling back to fixed windows")
        cps = []

    # Boundary set (sorted, ends with T).
    boundaries = sorted(set(hard) | set(cps))
    if T not in boundaries:
        boundaries.append(T)

    # 3. Build segments, enforce min/max length.
    segments: list[tuple[int, int, str]] = []
    last = 0
    for b in boundaries:
        if b <= last:
            continue
        why = "episode_end" if b in hard else "changepoint"
        # Merge if too short — extend the previous one if we can.
        if b - last < min_len and segments:
            prev_start, _, prev_why = segments[-1]
            segments[-1] = (prev_start, b, prev_why if prev_why == "episode_end" else why)
        else:
            segments.append((last, b, why))
        last = b

    # Split any segment that's too long.
    split: list[tuple[int, int, str]] = []
    for s, e, w in segments:
        length = e - s
        if length <= max_len:
            split.append((s, e, w))
            continue
        n = int(np.ceil(length / max_len))
        edges = np.linspace(s, e, n + 1, dtype=int)
        for i in range(n):
            split.append((int(edges[i]), int(edges[i + 1]), "max_split" if i > 0 else w))

    out: list[Segment] = []
    for i, (s, e, w) in enumerate(split):
        out.append(
            Segment(
                idx=i,
                start_step=s,
                end_step=e,
                start_frame=s,
                end_frame=e,
                duration_s=(e - s) / fps,
                why=w,
            )
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state_npz", type=Path)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--min-seconds", type=float, default=1.0)
    ap.add_argument("--max-seconds", type=float, default=6.0)
    args = ap.parse_args()
    segs = segment_rollout(args.state_npz, fps=args.fps,
                           min_seconds=args.min_seconds, max_seconds=args.max_seconds)
    print(json.dumps([asdict(s) for s in segs], indent=2))


if __name__ == "__main__":
    main()
