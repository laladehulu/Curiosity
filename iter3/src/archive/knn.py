"""k-NN dominance archive with density-corrected sampling.

Parameterized over embedding key ('vlm' | 'aurora') so the same class runs
both the `archive` and `aurora` strategies — they differ only in which
embedding drives selection.

Insertion rule:
  - first entry: seeded
  - new entry's nearest-neighbor cosine distance > tau_novel  -> add (novel niche)
  - else if new entry beats nearest neighbor on fitness        -> replace
  - else                                                       -> dominated

Sampling rule:
  - per entry: mean cosine distance to its k=3 nearest neighbors (larger
    = more isolated)
  - sample parent ∝ that quantity (over-sample sparse regions)
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from src.archive.entry import Entry, load_entries, save_entries

EmbedKey = Literal["vlm", "aurora"]


def _cosine_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (d,)  b: (n, d).  Returns (n,) cosine distances in [0, 2]."""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return 1.0 - b @ a


@dataclass
class KNNArchive:
    embed_key: EmbedKey
    tau_novel: float = 0.30
    k_neighbors: int = 3
    sampling_temperature: float = 1.0
    rng_seed: int = 0
    entries: list[Entry] = field(default_factory=list)
    _rng: random.Random = field(init=False, default=None)  # type: ignore

    def __post_init__(self):
        self._rng = random.Random(self.rng_seed)

    # ---- core ----

    def insert(self, entry: Entry) -> str:
        emb = entry.get_embed(self.embed_key)
        if not self.entries:
            self.entries.append(entry)
            entry.insert_status = "seeded"
            return "seeded"
        mat = np.stack([e.get_embed(self.embed_key) for e in self.entries])
        d = _cosine_dist(emb, mat)
        nn = int(np.argmin(d))
        if d[nn] > self.tau_novel:
            self.entries.append(entry)
            entry.insert_status = "novel"
            return "novel"
        if entry.fitness > self.entries[nn].fitness:
            entry.insert_status = "replaced"
            self.entries[nn] = entry
            return "replaced"
        entry.insert_status = "dominated"
        return "dominated"

    # ---- sampling ----

    def sample_parent(self) -> Entry | None:
        if not self.entries:
            return None
        if len(self.entries) == 1:
            return self.entries[0]
        weights = self._density_weights()
        # temperature: higher -> closer to uniform
        if self.sampling_temperature != 1.0:
            weights = weights ** (1.0 / max(self.sampling_temperature, 1e-3))
        total = weights.sum()
        if total <= 0:
            return self._rng.choice(self.entries)
        probs = weights / total
        idx = int(np.random.default_rng(self._rng.randint(0, 1 << 31))
                   .choice(len(self.entries), p=probs))
        return self.entries[idx]

    def _density_weights(self) -> np.ndarray:
        """Per-entry: mean cosine distance to k nearest neighbors.

        Larger -> entry is in a sparser region -> sample more often.
        """
        n = len(self.entries)
        mat = np.stack([e.get_embed(self.embed_key) for e in self.entries])
        # normalize once
        normed = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
        sims = normed @ normed.T
        dists = 1.0 - sims
        np.fill_diagonal(dists, np.inf)
        k = min(self.k_neighbors, n - 1)
        sorted_d = np.sort(dists, axis=1)[:, :k]
        return sorted_d.mean(axis=1)

    # ---- nearest neighbors for mutation prompt ----

    def k_nearest(self, entry: Entry, k: int = 3) -> list[Entry]:
        if not self.entries:
            return []
        emb = entry.get_embed(self.embed_key)
        mat = np.stack([e.get_embed(self.embed_key) for e in self.entries])
        d = _cosine_dist(emb, mat)
        idx = np.argsort(d)[:k]
        return [self.entries[int(i)] for i in idx]

    # ---- calibration ----

    def calibrate_tau(self) -> None:
        """Set tau_novel = median pairwise distance / 2 over current archive."""
        n = len(self.entries)
        if n < 3:
            return
        mat = np.stack([e.get_embed(self.embed_key) for e in self.entries])
        normed = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
        sims = normed @ normed.T
        # upper triangle, excluding diagonal
        iu = np.triu_indices(n, k=1)
        pairwise = 1.0 - sims[iu]
        self.tau_novel = float(np.median(pairwise) / 2.0)

    # ---- persistence ----

    def save(self, path: Path) -> None:
        save_entries(self.entries, path)

    @classmethod
    def load(cls, path: Path, embed_key: EmbedKey, **kwargs) -> "KNNArchive":
        arc = cls(embed_key=embed_key, **kwargs)
        arc.entries = load_entries(path)
        return arc

    # ---- stats ----

    def stats(self) -> dict:
        return {
            "n_entries": len(self.entries),
            "tau_novel": self.tau_novel,
            "embed_key": self.embed_key,
            "best_fitness": (max(e.fitness for e in self.entries)
                             if self.entries else None),
        }
