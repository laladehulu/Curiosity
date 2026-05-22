"""Live monitoring dashboard for experiment runs.

Reads JSONL logs and pool dirs to display progress. Run alongside experiments:
    python iter1/monitor.py --task pendulum --watch

Or one-shot:
    python iter1/monitor.py --task pendulum
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE

ALL_METHODS = ["curiosity", "pareto", "eureka", "archive"]


def read_logs(task: str, method: str) -> list[dict]:
    log_path = ROOT / "logs" / f"{task}_{method}.jsonl"
    if not log_path.exists():
        return []
    events = []
    for line in log_path.read_text().strip().split("\n"):
        if line.strip():
            events.append(json.loads(line))
    return events


def pool_stats(task: str, method: str) -> dict:
    """Summary stats from pool directory."""
    d = ROOT / "reward_gen" / "pool" / f"{task}_{method}"
    if not d.exists():
        return {"exists": False}
    jsons = list(d.glob("reward_*.json"))
    compiled = 0
    fitnesses = []
    entropies = []
    for jp in jsons:
        data = json.loads(jp.read_text())
        if data.get("failed_compile") or not data.get("fitness_vec"):
            continue
        compiled += 1
        fitnesses.append(data["mean_fitness"])
        entropies.append(data.get("curiosity", 0.0))

    return {
        "exists": True,
        "total": len(jsons),
        "compiled": compiled,
        "mean_f": float(np.mean(fitnesses)) if fitnesses else None,
        "max_f": float(np.max(fitnesses)) if fitnesses else None,
        "min_f": float(np.min(fitnesses)) if fitnesses else None,
        "mean_entropy": float(np.mean(entropies)) if entropies else None,
        "entropy_range": (float(np.min(entropies)), float(np.max(entropies))) if entropies else None,
    }


def checkpoints_count(task: str, method: str) -> int:
    ckpt_dir = ROOT / "training" / "checkpoints"
    if not ckpt_dir.exists():
        return 0
    return len(list(ckpt_dir.glob(f"{task}_{method}_*.zip")))


def display_status(task: str) -> None:
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT STATUS: {task}  ({time.strftime('%H:%M:%S')})")
    print(f"{'='*70}")

    for method in ALL_METHODS:
        events = read_logs(task, method)
        stats = pool_stats(task, method)
        ckpts = checkpoints_count(task, method)

        # Phase detection
        done_event = next((e for e in events if e.get("event") == "done"), None)
        ok_events = [e for e in events if e.get("event") == "ok"]
        fail_events = [e for e in events if e.get("event") in ("compile_fail", "train_fail")]

        if not stats["exists"] and not events:
            phase = "NOT STARTED"
        elif done_event:
            phase = "PHASE 1 DONE"
        elif events:
            last_iter = max((e.get("iter", 0) for e in events), default=0)
            phase = f"PHASE 1 IN PROGRESS (iter {last_iter})"
        else:
            phase = "UNKNOWN"

        if ckpts > 0:
            phase += f" | PHASE 2: {ckpts} checkpoints"

        print(f"\n  [{method.upper()}] {phase}")
        if stats["exists"]:
            print(f"    Pool: {stats['compiled']}/{stats['total']} compiled")
            if stats["mean_f"] is not None:
                print(f"    Fitness: mean={stats['mean_f']:.2f}, max={stats['max_f']:.2f}, min={stats['min_f']:.2f}")
            if stats.get("mean_entropy") is not None and method == "curiosity":
                lo, hi = stats["entropy_range"]
                print(f"    Entropy: mean={stats['mean_entropy']:.3f}, range=[{lo:.3f}, {hi:.3f}]")
            if ok_events:
                print(f"    Success rate: {len(ok_events)}/{len(ok_events)+len(fail_events)} "
                      f"({100*len(ok_events)/(len(ok_events)+len(fail_events)):.0f}%)")
            if done_event:
                cost = done_event.get("llm_cost_usd", 0)
                print(f"    LLM cost: ${cost:.4f}")

    # Check for tables
    tables = ROOT / "tables"
    if tables.exists():
        csvs = list(tables.glob(f"{task}_*.csv"))
        jsons_t = list(tables.glob(f"{task}_*.json"))
        if csvs or jsons_t:
            print(f"\n  TABLES: {len(csvs)} CSV, {len(jsons_t)} JSON")

    # Check for figures
    figs = ROOT / "figures"
    if figs.exists():
        pngs = list(figs.glob("*.png"))
        if pngs:
            print(f"  FIGURES: {len(pngs)} generated")

    # Warnings / anomalies
    warnings = []
    for method in ALL_METHODS:
        events = read_logs(task, method)
        fail_events = [e for e in events if e.get("event") in ("compile_fail", "train_fail")]
        ok_events = [e for e in events if e.get("event") == "ok"]
        total = len(fail_events) + len(ok_events)
        if total > 5 and len(fail_events) / total > 0.5:
            warnings.append(f"{method}: >50% failure rate ({len(fail_events)}/{total})")

        stats = pool_stats(task, method)
        if stats["exists"] and stats["compiled"] > 3:
            if stats["max_f"] is not None and stats["min_f"] is not None:
                if abs(stats["max_f"] - stats["min_f"]) < 1e-3:
                    warnings.append(f"{method}: all candidates have ~identical fitness (possible bug)")

    if warnings:
        print(f"\n  WARNINGS:")
        for w in warnings:
            print(f"    ! {w}")

    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--watch", action="store_true", help="Refresh every 30s")
    ap.add_argument("--interval", type=int, default=30, help="Refresh interval in seconds")
    args = ap.parse_args()

    if args.watch:
        print(f"Watching {args.task} (Ctrl+C to stop)...")
        try:
            while True:
                display_status(args.task)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        display_status(args.task)


if __name__ == "__main__":
    main()
