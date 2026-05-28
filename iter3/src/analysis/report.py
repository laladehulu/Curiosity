"""3-way comparison report across strategy run_ids.

Auto-discovers the most recent run_id per strategy under data/archives/ and
data/logs/ unless --runs is passed explicitly. Computes the reference grid
jointly across all runs, then emits:

  data/report/
    headline.json     summary numbers
    headline.md       paste-into-result.md table
    per_strategy.json detailed per-strategy summary
    fitness_curve.png max-fitness-so-far vs iteration (3 lines)
    coverage_qd.png   coverage + QD score bar chart per strategy
    cross_pca.png     all candidates in VLM-MiniLM PCA (2-d), colored by strategy
    top3.md           top-3 reward functions per strategy with VLM descriptions
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src import paths
from src.archive.entry import Entry, load_entries
from src.eval_grid.grid import build_grid, coverage, diversity_at_k, qd_score

STRATEGIES = ["archive", "argmax", "aurora"]


# ---- discovery -------------------------------------------------------------

def discover_runs(explicit: list[str] | None = None) -> dict[str, str]:
    """Return {strategy: run_id} — most recent archive per strategy."""
    if explicit:
        out = {}
        for r in explicit:
            for s in STRATEGIES:
                if r.startswith(s + "__"):
                    out[s] = r
                    break
        return out
    out = {}
    for s in STRATEGIES:
        candidates = sorted(paths.ARCHIVES.glob(f"{s}__*.json"))
        if candidates:
            out[s] = candidates[-1].stem
    return out


def load_log(run_id: str) -> list[dict]:
    p = paths.LOGS / f"{run_id}.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def load_archive(run_id: str) -> list[Entry]:
    p = paths.ARCHIVES / f"{run_id}.json"
    return load_entries(p)


# ---- core ------------------------------------------------------------------

def build_shared_grid(all_log_lines: list[dict], n_bins: int = 5):
    coords = []
    for ln in all_log_lines:
        if ln.get("skipped"):
            continue
        if "ref_coord" in ln:
            coords.append(tuple(ln["ref_coord"]))
    return build_grid(coords, n_bins=n_bins)


def per_strategy_summary(strategy: str, run_id: str, grid) -> dict:
    archive = load_archive(run_id)
    log = load_log(run_id)
    coords = [tuple(ln["ref_coord"]) for ln in log
              if not ln.get("skipped") and "ref_coord" in ln]
    fits = [ln["fitness"] for ln in log if not ln.get("skipped") and "fitness" in ln]
    return {
        "strategy": strategy,
        "run_id": run_id,
        "n_candidates_evaluated": len(coords),
        "n_archive_final": len(archive),
        "best_fitness": float(max(fits)) if fits else 0.0,
        "mean_fitness": float(np.mean(fits)) if fits else 0.0,
        "coverage": coverage(grid, coords),                         # 0..1
        "qd_score": qd_score(grid, coords, fits),                   # ~0..n_cells
        "diversity_at_all": diversity_at_k(coords),                 # in normalized ref space
    }


# ---- figures ---------------------------------------------------------------

def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def fig_fitness_curve(per_run_logs: dict[str, list[dict]], out: Path):
    plt = _setup_mpl()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for s, log in per_run_logs.items():
        if not log:
            continue
        fits = [ln.get("fitness", float("nan")) for ln in log if not ln.get("skipped")]
        if not fits:
            continue
        best_so_far = np.maximum.accumulate(fits)
        ax.plot(range(1, len(best_so_far) + 1), best_so_far, marker="o",
                markersize=4, label=s)
    ax.set_xlabel("iteration")
    ax.set_ylabel("best fitness so far (default Hopper reward)")
    ax.set_title("Fitness curve — best-so-far vs iteration")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_coverage_qd(per_run_summary: dict[str, dict], n_cells: int, out: Path):
    plt = _setup_mpl()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    strategies = list(per_run_summary.keys())
    covs = [per_run_summary[s]["coverage"] * 100 for s in strategies]
    qds = [per_run_summary[s]["qd_score"] for s in strategies]
    ax1.bar(strategies, covs, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax1.set_ylabel(f"reference-grid coverage  (%; {n_cells} cells)")
    ax1.set_title("Coverage")
    ax1.grid(alpha=0.3, axis="y")
    ax2.bar(strategies, qds, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax2.set_ylabel("QD score  (sum of normalized fitness over filled cells)")
    ax2.set_title("QD score")
    ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_cross_pca(per_run_archive: dict[str, list[Entry]], out: Path):
    plt = _setup_mpl()
    from sklearn.decomposition import PCA

    all_emb, all_strat = [], []
    for s, entries in per_run_archive.items():
        for e in entries:
            if e.embed_vlm is not None:
                all_emb.append(e.embed_vlm)
                all_strat.append(s)
    if not all_emb:
        return
    X = np.stack(all_emb)
    if X.shape[0] < 2:
        return
    pca = PCA(n_components=2, random_state=0).fit(X)
    Z = pca.transform(X)
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"archive": "#1f77b4", "argmax": "#ff7f0e", "aurora": "#2ca02c"}
    for s in colors:
        mask = [x == s for x in all_strat]
        if any(mask):
            pts = Z[mask]
            ax.scatter(pts[:, 0], pts[:, 1], s=40, alpha=0.7,
                       color=colors[s], label=f"{s} (n={int(sum(mask))})")
    ax.set_xlabel(f"VLM-PCA-1  (var={pca.explained_variance_ratio_[0]:.2f})")
    ax.set_ylabel(f"VLM-PCA-2  (var={pca.explained_variance_ratio_[1]:.2f})")
    ax.set_title("Behavior coverage in VLM-MiniLM space (PCA-2d)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


# ---- emit ------------------------------------------------------------------

def write_headline_md(per_run_summary: dict[str, dict], n_cells: int, out: Path):
    lines = [
        "# Headline comparison",
        "",
        f"Reference grid: {n_cells} cells (5 bins × 3 axes).",
        "",
        "| strategy | best fit | mean fit | coverage | QD score | diversity | archive size | candidates |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s, sm in per_run_summary.items():
        lines.append(
            f"| **{s}** | {sm['best_fitness']:.2f} | {sm['mean_fitness']:.2f} | "
            f"{sm['coverage']*100:.1f}% | {sm['qd_score']:.2f} | "
            f"{sm['diversity_at_all']:.3f} | {sm['n_archive_final']} | "
            f"{sm['n_candidates_evaluated']} |"
        )
    out.write_text("\n".join(lines) + "\n")


def write_top3_md(per_run_archive: dict[str, list[Entry]], out: Path):
    parts = ["# Top-3 reward functions per strategy", ""]
    for s, entries in per_run_archive.items():
        parts.append(f"## {s}")
        top = sorted(entries, key=lambda e: e.fitness, reverse=True)[:3]
        for rank, e in enumerate(top, 1):
            code = Path(e.reward_path).read_text()
            parts.append(f"\n### {rank}.  fitness = {e.fitness:.2f}  (`{e.id}`)")
            parts.append(f"\n> {e.description}\n")
            parts.append(f"ref_coord = {e.ref_coord}\n")
            parts.append("```python")
            parts.append(code.rstrip())
            parts.append("```")
        parts.append("")
    out.write_text("\n".join(parts) + "\n")


# ---- driver ----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", help="Explicit run_ids (default: most recent per strategy)")
    ap.add_argument("--n-bins", type=int, default=5)
    args = ap.parse_args()

    runs = discover_runs(args.runs)
    if not runs:
        raise SystemExit("no runs found under data/archives/")
    print("[report] using runs:")
    for s, r in runs.items():
        print(f"  {s}: {r}")

    # Pool all log lines to build the shared grid
    all_logs = []
    per_run_logs = {}
    for s, r in runs.items():
        log = load_log(r)
        per_run_logs[s] = log
        all_logs.extend(log)
    grid = build_shared_grid(all_logs, n_bins=args.n_bins)

    summary = {s: per_strategy_summary(s, r, grid) for s, r in runs.items()}
    per_run_archive = {s: load_archive(r) for s, r in runs.items()}

    # Write JSON
    (paths.REPORT / "headline.json").write_text(json.dumps({
        "runs": runs,
        "n_cells": grid.n_cells,
        "summary": summary,
    }, indent=2))
    write_headline_md(summary, grid.n_cells, paths.REPORT / "headline.md")
    (paths.REPORT / "per_strategy.json").write_text(json.dumps(summary, indent=2))

    # Figures
    fig_fitness_curve(per_run_logs, paths.REPORT / "fitness_curve.png")
    fig_coverage_qd(summary, grid.n_cells, paths.REPORT / "coverage_qd.png")
    fig_cross_pca(per_run_archive, paths.REPORT / "cross_pca.png")
    write_top3_md(per_run_archive, paths.REPORT / "top3.md")

    print("\n[report] wrote:")
    for p in sorted(paths.REPORT.iterdir()):
        print(f"  {p.relative_to(paths.ROOT)}")
    print("\n[report] headline:")
    print((paths.REPORT / "headline.md").read_text())


if __name__ == "__main__":
    main()
