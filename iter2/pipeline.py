"""End-to-end orchestrator.

Pipeline:
    discover checkpoints in data/policies/
      -> rollout each (video + state log)
        -> segment into behavioral phases
          -> VLM describes each segment (detailed)
            -> condense to short phrase
              -> embed + upsert into Chroma

Idempotent: skips work already on disk / already in the index.

Usage:
    python pipeline.py                                   # process every checkpoint
    python pipeline.py --ckpt walker2d__s0__step300000   # process one
    python pipeline.py --steps 600 --fps 30              # shorter rollouts
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src import paths
from src.describe.vlm import describe_segment
from src.index.store import Entry, get_collection, upsert
from src.rollout.runner import RolloutArtifact, rollout
from src.rollout.segment import segment_rollout


def _load_rollout_meta(meta_path: Path) -> RolloutArtifact:
    d = json.loads(meta_path.read_text())
    return RolloutArtifact(
        ckpt_id=d["ckpt_id"],
        env_id=d["env_id"],
        video_path=Path(d["video_path"]),
        state_path=Path(d["state_path"]),
        n_steps=d["n_steps"],
        fps=d["fps"],
    )


def ensure_rollout(
    ckpt_path: Path, env_id: str, n_steps: int, fps: int, force: bool
) -> RolloutArtifact:
    meta = paths.ROLLOUTS / f"{ckpt_path.stem}.rollout.json"
    if meta.exists() and not force:
        art = _load_rollout_meta(meta)
        if art.video_path.exists() and art.state_path.exists():
            return art
    return rollout(ckpt_path, env_id, n_steps=n_steps, fps=fps)


def existing_segment_ids(ckpt_id: str) -> set[str]:
    col = get_collection()
    got = col.get(where={"ckpt_id": ckpt_id})
    return set(got["ids"]) if got["ids"] else set()


def discover_checkpoints() -> list[tuple[Path, str]]:
    """Return [(ckpt_path, env_id), ...] from json sidecars next to .zip files."""
    out = []
    for meta_path in sorted(paths.POLICIES.glob("*.json")):
        meta = json.loads(meta_path.read_text())
        env_id = meta["spec"]["env_id"]
        ckpt_path = meta_path.with_suffix(".zip")
        if ckpt_path.exists():
            out.append((ckpt_path, env_id))
    return out


def process_checkpoint(
    ckpt_path: Path,
    env_id: str,
    n_steps: int,
    fps: int,
    max_seg_seconds: float,
    min_seg_seconds: float,
    force_rollout: bool,
    force_describe: bool,
    n_frames_per_seg: int,
    dry_run_describe: bool,
) -> int:
    """Process one checkpoint end-to-end. Returns count of new indexed segments."""
    print(f"\n=== {ckpt_path.stem}  ({env_id}) ===")
    art = ensure_rollout(ckpt_path, env_id, n_steps, fps, force=force_rollout)
    print(f"[rollout] {art.video_path.name}  steps={art.n_steps}  fps={art.fps}")

    segments = segment_rollout(
        art.state_path, fps=fps,
        min_seconds=min_seg_seconds, max_seconds=max_seg_seconds,
    )
    print(f"[segment] {len(segments)} phases")

    have = set() if force_describe else existing_segment_ids(art.ckpt_id)

    desc_log = paths.DESCRIPTIONS / f"{art.ckpt_id}.jsonl"
    new_entries: list[Entry] = []
    with desc_log.open("a") as fh:
        for seg in segments:
            entry_id = f"{art.ckpt_id}__seg{seg.idx}"
            if entry_id in have:
                continue
            if dry_run_describe:
                concise = f"[dry-run] {art.ckpt_id} seg {seg.idx} ({seg.duration_s:.1f}s, {seg.why})"
                detailed = "[dry-run]"
            else:
                t0 = time.time()
                d = describe_segment(
                    art.video_path, seg.start_frame, seg.end_frame,
                    env_id=env_id, fps=fps, n_frames=n_frames_per_seg,
                )
                detailed, concise = d.detailed, d.concise
                print(f"[vlm] seg{seg.idx} ({seg.duration_s:.1f}s) "
                      f"in {time.time() - t0:.1f}s -> {concise!r}")
            entry = Entry(
                ckpt_id=art.ckpt_id,
                env_id=env_id,
                segment_idx=seg.idx,
                start_step=seg.start_step,
                end_step=seg.end_step,
                duration_s=seg.duration_s,
                why=seg.why,
                video_path=str(art.video_path),
                concise=concise,
                detailed=detailed,
            )
            new_entries.append(entry)
            fh.write(json.dumps({**entry.metadata(), "id": entry.id, "concise": concise}) + "\n")

    upsert(new_entries)
    print(f"[index] +{len(new_entries)} entries  (collection now has {get_collection().count()})")
    return len(new_entries)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="Process only this checkpoint stem (e.g. walker2d__s0__step300000)")
    ap.add_argument("--steps", type=int, default=900, help="Rollout length per checkpoint")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--max-seg-seconds", type=float, default=5.0)
    ap.add_argument("--min-seg-seconds", type=float, default=1.5)
    ap.add_argument("--n-frames-per-seg", type=int, default=6)
    ap.add_argument("--force-rollout", action="store_true")
    ap.add_argument("--force-describe", action="store_true")
    ap.add_argument("--dry-run-describe", action="store_true",
                    help="Skip VLM calls; useful for plumbing tests")
    args = ap.parse_args()

    discovered = discover_checkpoints()
    if args.ckpt:
        discovered = [(p, e) for p, e in discovered if p.stem == args.ckpt]
    if not discovered:
        raise SystemExit("no checkpoints found — train some via src.policies.train first")

    total = 0
    for ckpt_path, env_id in discovered:
        total += process_checkpoint(
            ckpt_path, env_id,
            n_steps=args.steps, fps=args.fps,
            max_seg_seconds=args.max_seg_seconds,
            min_seg_seconds=args.min_seg_seconds,
            force_rollout=args.force_rollout,
            force_describe=args.force_describe,
            n_frames_per_seg=args.n_frames_per_seg,
            dry_run_describe=args.dry_run_describe,
        )
    print(f"\n[done] processed {len(discovered)} checkpoint(s), +{total} new index entries")


if __name__ == "__main__":
    main()
