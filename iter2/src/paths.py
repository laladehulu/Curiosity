"""Centralized path resolution. Everything writes under iter2/data/."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
POLICIES = DATA / "policies"
ROLLOUTS = DATA / "rollouts"
DESCRIPTIONS = DATA / "descriptions"
CHROMA = DATA / "chroma"

for _p in (POLICIES, ROLLOUTS, DESCRIPTIONS, CHROMA):
    _p.mkdir(parents=True, exist_ok=True)
