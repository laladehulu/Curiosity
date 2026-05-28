"""Centralized path resolution for iter3."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REWARDS = DATA / "rewards"
POLICIES = DATA / "policies"
ROLLOUTS = DATA / "rollouts"
ARCHIVES = DATA / "archives"
LOGS = DATA / "logs"
REPORT = DATA / "report"

for _p in (REWARDS, POLICIES, ROLLOUTS, ARCHIVES, LOGS, REPORT):
    _p.mkdir(parents=True, exist_ok=True)


def run_dir(base: Path, run_id: str) -> Path:
    """Subdirectory for a single run_id (e.g. data/rewards/<run_id>/)."""
    p = base / run_id
    p.mkdir(parents=True, exist_ok=True)
    return p
