"""Archive entry dataclass — one per candidate, serializable to JSON."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np


@dataclass
class Entry:
    id: str
    iteration: int
    parent_id: str | None
    reward_path: str
    fitness: float
    description: str | None
    ckpt_path: str
    video_path: str
    state_path: str
    # behavior descriptors
    embed_vlm: np.ndarray | None = None        # 384-d, L2-normalized
    embed_aurora: np.ndarray | None = None     # 8-d (set by aurora strategy)
    ref_coord: tuple[float, float, float] | None = None
    # outcome
    insert_status: str | None = None
    prompt_kind: str | None = None             # cold / parent_only / with_neighbors

    def get_embed(self, key: str) -> np.ndarray:
        if key == "vlm":
            if self.embed_vlm is None:
                raise KeyError(f"entry {self.id} has no VLM embedding")
            return self.embed_vlm
        if key == "aurora":
            if self.embed_aurora is None:
                raise KeyError(f"entry {self.id} has no AURORA embedding")
            return self.embed_aurora
        raise KeyError(key)

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.embed_vlm is not None:
            d["embed_vlm"] = self.embed_vlm.tolist()
        if self.embed_aurora is not None:
            d["embed_aurora"] = self.embed_aurora.tolist()
        if self.ref_coord is not None:
            d["ref_coord"] = list(self.ref_coord)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Entry":
        d = dict(d)
        if d.get("embed_vlm") is not None:
            d["embed_vlm"] = np.asarray(d["embed_vlm"], dtype=np.float32)
        if d.get("embed_aurora") is not None:
            d["embed_aurora"] = np.asarray(d["embed_aurora"], dtype=np.float32)
        if d.get("ref_coord") is not None:
            d["ref_coord"] = tuple(d["ref_coord"])
        return cls(**d)


def save_entries(entries: list[Entry], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([e.to_dict() for e in entries], indent=2))


def load_entries(path: Path) -> list[Entry]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text())
    return [Entry.from_dict(d) for d in raw]
