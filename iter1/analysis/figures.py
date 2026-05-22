"""Phase 6 — figures. Per protocol §"Figures Claude Code must produce".

Each figure function reads from JSON/CSV outputs produced by earlier phases and
writes BOTH .png (preview) and .pdf (submission) into figures/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"


def _save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_1_headline(headline_csv: Path) -> None:
    """Bar chart: pool robustness per (task, method) with CI bars."""
    df = pd.read_csv(headline_csv)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=df, x="task", y="ci_mean", hue="method", ax=ax)
    # CI bars
    for i, row in df.reset_index(drop=True).iterrows():
        x = ax.patches[i].get_x() + ax.patches[i].get_width() / 2
        ax.errorbar(x, row["ci_mean"],
                    yerr=[[row["ci_mean"] - row["ci_lo"]], [row["ci_hi"] - row["ci_mean"]]],
                    fmt="none", color="black", capsize=4)
    ax.set_ylabel("Pool robustness (summed max-success across perturbations)")
    ax.set_title("Figure 1 — Headline: pool robustness by method")
    _save(fig, "figure1_headline")


def figure_2_diversity_scatter(pairs_csv: Path) -> None:
    """Scatter: reward-space distance vs policy-space distance, colored by variant."""
    df = pd.read_csv(pairs_csv)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(data=df, x="reward_emb_dist", y="policy_dtw_dist", hue="method", ax=ax, alpha=0.7)
    ax.set_xlabel("Reward-code embedding distance")
    ax.set_ylabel("Policy trajectory DTW distance")
    ax.set_title("Figure 2 — Reward-space vs policy-space diversity")
    _save(fig, "figure2_diversity_scatter")


def figure_3_archive_coverage(archive_summary_json: Path) -> None:
    """Bar plot of archive coverage per task."""
    data = json.loads(archive_summary_json.read_text())
    tasks = [d["task"] for d in data]
    coverages = [d["coverage"] for d in data]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(tasks, coverages)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Archive coverage (filled / total cells)")
    ax.set_title("Figure 3 — Behavior-archive coverage")
    _save(fig, "figure3_archive_coverage")


def figure_4_noise_floor(noise_json: Path) -> None:
    """Noise floor vs mechanism effect, per task."""
    data = json.loads(noise_json.read_text())
    tasks = [d["task"] for d in data]
    noise = [d["seed_noise_std"] for d in data]
    effect = [d["mechanism_effect"] for d in data]
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(tasks))
    width = 0.35
    ax.bar(x - width / 2, noise, width, label="Seed-noise floor")
    ax.bar(x + width / 2, effect, width, label="Mechanism effect")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylabel("Pool-robustness std (within reward) vs effect (across methods)")
    ax.set_title("Figure 4 — Noise floor vs mechanism effect")
    ax.legend()
    _save(fig, "figure4_noise_floor")


def figure_5_perturbation_breakdown(robustness_json: Path) -> None:
    """Per-perturbation success-rate heatmap, per task."""
    data = json.loads(robustness_json.read_text())
    perturbs = data["perturbations"]
    rows = []
    for p_name, info in perturbs.items():
        for pp in info["per_policy"]:
            rows.append({"perturbation": p_name,
                         "policy": Path(pp["checkpoint"]).stem,
                         "success_rate": pp["success_rate"]})
    df = pd.DataFrame(rows)
    if df.empty:
        return
    pivot = df.pivot_table(index="policy", columns="perturbation", values="success_rate")
    fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(pivot))))
    sns.heatmap(pivot, vmin=0, vmax=1, cmap="viridis", cbar_kws={"label": "Success rate"})
    ax.set_title(f"Figure 5 — Per-perturbation breakdown ({data['task']})")
    _save(fig, f"figure5_perturbation_{data['task']}")


def figure_6_pareto_front_2d(curiosity_pool_dir: Path) -> None:
    """Scatter of (mean_fitness, curiosity) for the curiosity variant, with Pareto front highlighted."""
    jsons = sorted(curiosity_pool_dir.glob("reward_*.json"))
    if not jsons:
        return
    fitnesses, entropies, on_front_list = [], [], []
    candidates = []
    for jp in jsons:
        d = json.loads(jp.read_text())
        if d.get("failed_compile") or not d.get("fitness_vec"):
            continue
        candidates.append(d)
        fitnesses.append(d["mean_fitness"])
        entropies.append(d.get("curiosity", 0.0))

    if not candidates:
        return

    # Compute Pareto front
    score_mat = np.array(list(zip(fitnesses, entropies)))
    n = score_mat.shape[0]
    on_front = np.ones(n, dtype=bool)
    for i in range(n):
        if not on_front[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if np.all(score_mat[j] >= score_mat[i]) and np.any(score_mat[j] > score_mat[i]):
                on_front[i] = False
                break

    fig, ax = plt.subplots(figsize=(7, 5))
    fitnesses = np.array(fitnesses)
    entropies = np.array(entropies)
    ax.scatter(fitnesses[~on_front], entropies[~on_front], c="gray", alpha=0.5, label="Dominated", s=30)
    ax.scatter(fitnesses[on_front], entropies[on_front], c="red", marker="*", s=120, label="Pareto front", zorder=5)
    # Connect front points
    front_f = fitnesses[on_front]
    front_e = entropies[on_front]
    order = np.argsort(front_f)
    ax.plot(front_f[order], front_e[order], "r--", alpha=0.5)
    ax.set_xlabel("Mean env-true return (fitness)")
    ax.set_ylabel("State-visitation entropy (curiosity)")
    ax.set_title("Figure 6 — Curiosity-Pareto front: (Fitness, Entropy)")
    ax.legend()
    _save(fig, "figure6_pareto_front_2d")


def figure_7_diversity_by_method(diversity_csv: Path) -> None:
    """Box plot of pairwise state-occupancy TV divergence within each method's pool."""
    df = pd.read_csv(diversity_csv)
    fig, ax = plt.subplots(figsize=(7, 5))
    order = ["curiosity", "pareto", "eureka", "archive"]
    present = [m for m in order if m in df["method"].values]
    sns.boxplot(data=df[df["method"].isin(present)], x="method", y="state_occ_tv",
                order=present, ax=ax)
    ax.set_ylabel("Pairwise state-occupancy TV distance")
    ax.set_title("Figure 7 — Within-pool behavioral diversity by method")
    _save(fig, "figure7_diversity_by_method")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--headline-csv", type=Path)
    ap.add_argument("--pairs-csv", type=Path)
    ap.add_argument("--archive-json", type=Path)
    ap.add_argument("--noise-json", type=Path)
    ap.add_argument("--robustness-json", type=Path)
    ap.add_argument("--curiosity-pool-dir", type=Path, help="Pool dir for curiosity variant (for figure 6)")
    ap.add_argument("--diversity-csv", type=Path, help="Pairwise diversity CSV (for figure 7)")
    args = ap.parse_args()
    if args.headline_csv:
        figure_1_headline(args.headline_csv)
    if args.pairs_csv:
        figure_2_diversity_scatter(args.pairs_csv)
    if args.archive_json:
        figure_3_archive_coverage(args.archive_json)
    if args.noise_json:
        figure_4_noise_floor(args.noise_json)
    if args.robustness_json:
        figure_5_perturbation_breakdown(args.robustness_json)
    if args.curiosity_pool_dir:
        figure_6_pareto_front_2d(args.curiosity_pool_dir)
    if args.diversity_csv:
        figure_7_diversity_by_method(args.diversity_csv)


if __name__ == "__main__":
    main()
