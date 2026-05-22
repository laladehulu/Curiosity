"""Phase 5 — headline comparison: pool robustness for all variants across all tasks.

Reads robustness JSONs produced by robustness/evaluate_under_perturb.py and
emits a CSV table + a JSON blob with bootstrap CIs and Mann-Whitney + Holm.

Supports 4 methods: curiosity, pareto, eureka, archive.
All 6 pairwise comparisons are tested with Holm correction.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.statistical_tests import bootstrap_mean_ci, compare_groups, holm_adjust  # noqa: E402

ALL_METHODS = ["curiosity", "pareto", "eureka", "archive"]


def _infer_method(d: dict, path: Path) -> str:
    """Infer method from JSON content or filename."""
    if d.get("method"):
        return d["method"]
    stem = path.stem.lower()
    for m in ALL_METHODS:
        if m in stem:
            return m
    return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robustness-jsons", nargs="+", type=Path, required=True,
                    help="JSON files emitted by robustness/evaluate_under_perturb.py — one per (task, method, seed).")
    ap.add_argument("--out-csv", type=Path, default=ROOT / "tables" / "headline.csv")
    ap.add_argument("--out-json", type=Path, default=ROOT / "tables" / "headline.json")
    args = ap.parse_args()

    rows = []
    for p in args.robustness_jsons:
        d = json.loads(p.read_text())
        method = _infer_method(d, p)
        seed = d.get("seed") or 0
        rows.append({
            "task": d["task"],
            "method": method,
            "seed": seed,
            "pool_robustness": d["pool_robustness_score"],
            "pool_size": d["pool_size"],
            "source": str(p),
        })
    df = pd.DataFrame(rows)

    # Aggregate per (task, method)
    agg = (df.groupby(["task", "method"])["pool_robustness"]
             .agg(["mean", "std", "count"]).reset_index())

    # Bootstrap CIs
    ci_rows = []
    for (task, method), sub in df.groupby(["task", "method"]):
        ci = bootstrap_mean_ci(sub["pool_robustness"].tolist())
        ci_rows.append({"task": task, "method": method, "ci_mean": ci.mean, "ci_lo": ci.lo, "ci_hi": ci.hi})
    cis = pd.DataFrame(ci_rows)
    out = agg.merge(cis, on=["task", "method"])

    # All 6 pairwise Mann-Whitney comparisons per task, with Holm correction
    comps = []
    for task, sub in df.groupby("task"):
        methods_present = [m for m in ALL_METHODS if m in sub["method"].values]
        for m_a, m_b in itertools.combinations(methods_present, 2):
            a = sub[sub["method"] == m_a]["pool_robustness"].tolist()
            b = sub[sub["method"] == m_b]["pool_robustness"].tolist()
            if a and b:
                comps.append(compare_groups(a, b, name=f"{task}__{m_a}_vs_{m_b}"))
    holm_adjust(comps)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    payload = {
        "summary": out.to_dict(orient="records"),
        "comparisons": [c.__dict__ for c in comps],
    }
    args.out_json.write_text(json.dumps(payload, indent=2))
    print(out.to_string(index=False))
    print()
    print("Pairwise comparisons (Mann-Whitney, Holm-adjusted):")
    for c in comps:
        print(f"  {c.name}: mean_a={c.mean_a:.3f}, mean_b={c.mean_b:.3f}, "
              f"U={c.u_stat:.1f}, p={c.p_value:.4f}, p_adj={c.p_adj:.4f}")


if __name__ == "__main__":
    main()
