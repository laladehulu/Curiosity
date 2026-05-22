"""Generate results_curiosity.md from experiment outputs.

Reads pool JSONs, diversity CSVs, entropy CSVs, and robustness JSONs
to produce a self-contained results markdown.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.statistical_tests import bootstrap_mean_ci  # noqa: E402
from reward_gen._loop_common import pool_dir  # noqa: E402

ALL_METHODS = ["curiosity", "pareto", "eureka", "archive"]


def _load_pool_stats(task: str, method: str) -> dict:
    """Load summary stats from a method's pool directory."""
    d = pool_dir(task, method)
    candidates = []
    for jp in sorted(d.glob("reward_*.json")):
        data = json.loads(jp.read_text())
        if data.get("failed_compile") or not data.get("fitness_vec"):
            continue
        candidates.append(data)

    if not candidates:
        return {"n": 0, "mean_f": 0, "max_f": 0, "ci": None, "entropies": []}

    fitnesses = [c["mean_fitness"] for c in candidates]
    entropies = [c.get("curiosity", 0.0) for c in candidates]
    ci = bootstrap_mean_ci(fitnesses)
    return {
        "n": len(candidates),
        "mean_f": float(np.mean(fitnesses)),
        "max_f": float(np.max(fitnesses)),
        "ci": ci,
        "entropies": entropies,
        "fitnesses": fitnesses,
    }


def generate(task: str) -> None:
    tables_dir = ROOT / "tables"

    # Gather stats
    stats = {m: _load_pool_stats(task, m) for m in ALL_METHODS}

    lines = []
    lines.append("# Curiosity-as-Pareto-Axis: Experiment Results\n")

    # Abstract
    lines.append("## Abstract\n")
    lines.append(
        "We test whether adding state-visitation entropy (a curiosity signal) as an explicit "
        "Pareto selection axis in LLM-based reward evolution preserves behaviorally distinct "
        "policies without hand-designed descriptors. We compare four variants: (1) Curiosity-Pareto "
        "(proposed), (2) GEPA-Pareto (fitness-only Pareto front), (3) Eureka best-of-K (greedy), "
        "and (4) MAP-Elites behavior archive (descriptor-based). All variants receive identical "
        "compute budgets.\n"
    )

    # Method
    lines.append("## Method\n")
    lines.append(
        "The curiosity variant maintains a Pareto frontier over two axes: mean env-true return "
        "(fitness) and state-visitation entropy computed from 20 rollout episodes. Shannon entropy "
        "is estimated by binning observations into a task-specific grid via `np.histogramdd`. "
        "Parents are sampled uniformly from the frontier and mutated by an LLM.\n"
    )

    # Setup
    lines.append("## Setup\n")
    lines.append(f"- **Task:** {task}\n")
    lines.append(f"- **Variants:** {', '.join(ALL_METHODS)}\n")
    lines.append("- **Hardware:** Mac CPU\n")
    lines.append("- **Phase 1:** LLM-guided reward search with short PPO training\n")
    lines.append("- **Phase 2:** Full PPO training on top-3 rewards x 3 seeds\n")

    # Phase 1 Results
    lines.append("\n## Phase 1: Reward Search Results\n")
    lines.append("| Method | Compiled | Mean F | Max F | 95% CI |\n")
    lines.append("|--------|----------|--------|-------|--------|\n")
    for m in ALL_METHODS:
        s = stats[m]
        if s["n"] == 0:
            lines.append(f"| {m} | 0 | - | - | - |\n")
        else:
            ci = s["ci"]
            lines.append(
                f"| {m} | {s['n']} | {s['mean_f']:.2f} | {s['max_f']:.2f} | "
                f"[{ci.lo:.2f}, {ci.hi:.2f}] |\n"
            )

    # Entropy distributions (curiosity variant)
    cs = stats.get("curiosity", {})
    if cs.get("entropies") and any(e > 0 for e in cs["entropies"]):
        lines.append("\n### Curiosity variant entropy distribution\n")
        ents = cs["entropies"]
        lines.append(f"- Range: [{min(ents):.3f}, {max(ents):.3f}]\n")
        lines.append(f"- Mean: {np.mean(ents):.3f}, Std: {np.std(ents):.3f}\n")

    # Diversity comparison
    div_path = tables_dir / f"{task}_diversity.csv"
    if div_path.exists():
        import pandas as pd
        df = pd.read_csv(div_path)
        lines.append("\n## Diversity Comparison\n")
        lines.append("Mean pairwise state-occupancy total variation within each method's trained policy pool.\n\n")
        lines.append("| Method | Mean TV | 95% CI | N pairs |\n")
        lines.append("|--------|---------|--------|--------|\n")
        for m in ALL_METHODS:
            sub = df[df["method"] == m]
            if sub.empty:
                lines.append(f"| {m} | - | - | 0 |\n")
            else:
                vals = sub["state_occ_tv"].tolist()
                ci = bootstrap_mean_ci(vals)
                lines.append(f"| {m} | {ci.mean:.4f} | [{ci.lo:.4f}, {ci.hi:.4f}] | {len(vals)} |\n")

    # Performance comparison
    lines.append("\n## Performance Comparison\n")
    lines.append("| Method | Mean F | Max F | 95% CI |\n")
    lines.append("|--------|--------|-------|--------|\n")
    for m in ALL_METHODS:
        s = stats[m]
        if s["n"] == 0:
            lines.append(f"| {m} | - | - | - |\n")
        else:
            ci = s["ci"]
            lines.append(f"| {m} | {s['mean_f']:.2f} | {s['max_f']:.2f} | [{ci.lo:.2f}, {ci.hi:.2f}] |\n")

    # Robustness
    lines.append("\n## Robustness\n")
    rob_found = False
    for m in ALL_METHODS:
        rp = tables_dir / f"{task}_{m}_robustness.json"
        if rp.exists():
            rob_found = True
            data = json.loads(rp.read_text())
            lines.append(f"- **{m}**: pool_robustness = {data.get('pool_robustness_score', 'N/A')}\n")
    if not rob_found:
        lines.append("No robustness data available yet.\n")

    # Discussion
    lines.append("\n## Discussion\n")
    lines.append(
        "The curiosity-Pareto variant adds state-visitation entropy as a second Pareto axis, "
        "requiring no hand-designed behavioral descriptors. This is the key advantage over the "
        "MAP-Elites archive variant, which requires task-specific descriptor functions.\n\n"
    )
    lines.append(
        "**Limitations:** (1) Shannon entropy over binned observations is a coarse diversity "
        "signal and may miss fine-grained behavioral differences. (2) The bin configuration is "
        "still somewhat task-specific (obs dims, ranges). (3) Results are on simple classic control "
        "tasks; scaling to high-dimensional environments needs further work.\n"
    )

    out_path = ROOT / "results_curiosity.md"
    out_path.write_text("".join(lines))
    print(f"Results written to {out_path}")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    args = ap.parse_args()
    generate(args.task)


if __name__ == "__main__":
    main()
