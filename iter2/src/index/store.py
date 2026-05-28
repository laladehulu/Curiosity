"""ChromaDB-backed behavior index.

Collection schema:
  id        := f"{ckpt_id}__seg{idx}"
  document  := concise string (what we embed + show)
  metadata  := {
      ckpt_id, env_id, segment_idx, start_step, end_step,
      duration_s, why, video_path, detailed
  }
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from src import paths
from src.index.embed import ChromaEmbeddingFunction

COLLECTION = "behaviors"


@dataclass
class Entry:
    ckpt_id: str
    env_id: str
    segment_idx: int
    start_step: int
    end_step: int
    duration_s: float
    why: str
    video_path: str
    concise: str
    detailed: str

    @property
    def id(self) -> str:
        return f"{self.ckpt_id}__seg{self.segment_idx}"

    def metadata(self) -> dict:
        return {
            "ckpt_id": self.ckpt_id,
            "env_id": self.env_id,
            "segment_idx": self.segment_idx,
            "start_step": self.start_step,
            "end_step": self.end_step,
            "duration_s": self.duration_s,
            "why": self.why,
            "video_path": self.video_path,
            "detailed": self.detailed,
        }


def _client():
    import chromadb
    return chromadb.PersistentClient(path=str(paths.CHROMA))


def get_collection():
    return _client().get_or_create_collection(
        name=COLLECTION,
        embedding_function=ChromaEmbeddingFunction(),
        metadata={"hnsw:space": "cosine"},
    )


def upsert(entries: Iterable[Entry]) -> int:
    entries = list(entries)
    if not entries:
        return 0
    col = get_collection()
    col.upsert(
        ids=[e.id for e in entries],
        documents=[e.concise for e in entries],
        metadatas=[e.metadata() for e in entries],
    )
    return len(entries)


@dataclass
class Hit:
    id: str
    score: float          # similarity, higher = closer (1 - cosine_distance)
    concise: str
    metadata: dict


def query(text: str, k: int = 5, where: Optional[dict] = None) -> list[Hit]:
    col = get_collection()
    res = col.query(query_texts=[text], n_results=k, where=where)
    out: list[Hit] = []
    if not res["ids"] or not res["ids"][0]:
        return out
    for i, _id in enumerate(res["ids"][0]):
        dist = float(res["distances"][0][i])
        out.append(
            Hit(
                id=_id,
                score=1.0 - dist,
                concise=res["documents"][0][i],
                metadata=res["metadatas"][0][i],
            )
        )
    return out


def neighbors_of(entry_id: str, k: int = 5) -> list[Hit]:
    """Find behaviors near a specific indexed entry (e.g. a known policy segment)."""
    col = get_collection()
    got = col.get(ids=[entry_id])
    if not got["documents"]:
        raise KeyError(entry_id)
    # Self will likely come back first; oversample then drop it.
    res = col.query(query_texts=got["documents"], n_results=k + 1)
    out: list[Hit] = []
    for i, _id in enumerate(res["ids"][0]):
        if _id == entry_id:
            continue
        out.append(
            Hit(
                id=_id,
                score=1.0 - float(res["distances"][0][i]),
                concise=res["documents"][0][i],
                metadata=res["metadatas"][0][i],
            )
        )
        if len(out) >= k:
            break
    return out


def stats() -> dict:
    col = get_collection()
    return {"count": col.count(), "path": str(paths.CHROMA)}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s_stats = sub.add_parser("stats")
    s_query = sub.add_parser("query")
    s_query.add_argument("text")
    s_query.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    if args.cmd == "stats":
        print(json.dumps(stats(), indent=2))
    elif args.cmd == "query":
        hits = query(args.text, k=args.k)
        for h in hits:
            print(f"[{h.score:+.3f}] {h.id}")
            print(f"        {h.concise}")
            print(f"        {h.metadata['video_path']}")


if __name__ == "__main__":
    main()
