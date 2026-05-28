"""Argmax baseline (vanilla Eureka).

Same Entry / insert / save interface as KNNArchive so run.py can swap.
Selection: always return the running-best-fitness entry. Insertion: append
every entry (no dominance check); insert_status is "kept".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.archive.entry import Entry, load_entries, save_entries


@dataclass
class ArgmaxArchive:
    entries: list[Entry] = field(default_factory=list)

    def insert(self, entry: Entry) -> str:
        entry.insert_status = "kept"
        self.entries.append(entry)
        return "kept"

    def sample_parent(self) -> Entry | None:
        if not self.entries:
            return None
        return max(self.entries, key=lambda e: e.fitness)

    def k_nearest(self, entry: Entry, k: int = 3) -> list[Entry]:
        # baseline doesn't use neighbors for mutation prompts
        return []

    def calibrate_tau(self) -> None:
        return

    def save(self, path: Path) -> None:
        save_entries(self.entries, path)

    @classmethod
    def load(cls, path: Path) -> "ArgmaxArchive":
        arc = cls()
        arc.entries = load_entries(path)
        return arc

    def stats(self) -> dict:
        return {
            "n_entries": len(self.entries),
            "best_fitness": (max(e.fitness for e in self.entries)
                             if self.entries else None),
        }
