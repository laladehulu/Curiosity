"""Call Claude vision API on sampled frames of a segment, then condense."""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from src.describe.prompts import (
    CONDENSE_SYSTEM,
    CONDENSE_USER,
    DESCRIBE_SYSTEM,
    DESCRIBE_USER,
)

VISION_MODEL = os.environ.get("ITER2_VISION_MODEL", "claude-sonnet-4-6")
CONDENSE_MODEL = os.environ.get("ITER2_CONDENSE_MODEL", "claude-haiku-4-5-20251001")


def _client():
    import anthropic
    return anthropic.Anthropic()


def sample_frames(video_path: Path, start_frame: int, end_frame: int, n: int) -> list[np.ndarray]:
    """Read `n` evenly spaced frames from a video over [start_frame, end_frame)."""
    end_frame = max(start_frame + 1, end_frame)
    indices = np.linspace(start_frame, end_frame - 1, n, dtype=int).tolist()
    frames: list[np.ndarray] = []
    reader = imageio.get_reader(str(video_path))
    try:
        for i in indices:
            frames.append(reader.get_data(int(i)))
    finally:
        reader.close()
    return frames


def _frame_to_block(arr: np.ndarray, max_side: int = 512) -> dict:
    """Convert an HWC uint8 frame to an Anthropic image content block."""
    img = Image.fromarray(arr)
    # Downscale for faster API + cheaper tokens. VLMs don't need full-res for gait.
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": b64},
    }


@dataclass
class Description:
    detailed: str
    concise: str
    n_frames: int
    vision_model: str
    condense_model: str


def describe_segment(
    video_path: Path,
    start_frame: int,
    end_frame: int,
    env_id: str,
    fps: int = 30,
    n_frames: int = 6,
    max_side: int = 512,
) -> Description:
    client = _client()
    frames = sample_frames(video_path, start_frame, end_frame, n_frames)
    duration = (end_frame - start_frame) / max(fps, 1)

    blocks: list[dict] = [_frame_to_block(f, max_side=max_side) for f in frames]
    blocks.append(
        {
            "type": "text",
            "text": DESCRIBE_USER.format(
                n_frames=len(frames), duration=duration, env_id=env_id
            ),
        }
    )

    resp = client.messages.create(
        model=VISION_MODEL,
        max_tokens=400,
        system=DESCRIBE_SYSTEM,
        messages=[{"role": "user", "content": blocks}],
    )
    detailed = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()

    cond_resp = client.messages.create(
        model=CONDENSE_MODEL,
        max_tokens=80,
        system=CONDENSE_SYSTEM,
        messages=[{"role": "user", "content": CONDENSE_USER.format(detailed=detailed)}],
    )
    concise = "".join(
        b.text for b in cond_resp.content if getattr(b, "type", None) == "text"
    ).strip().strip('"').strip("'")

    return Description(
        detailed=detailed,
        concise=concise,
        n_frames=len(frames),
        vision_model=VISION_MODEL,
        condense_model=CONDENSE_MODEL,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--env", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=90)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--n-frames", type=int, default=6)
    args = ap.parse_args()
    d = describe_segment(args.video, args.start, args.end, args.env, args.fps, args.n_frames)
    print(json.dumps(d.__dict__, indent=2))


if __name__ == "__main__":
    main()
