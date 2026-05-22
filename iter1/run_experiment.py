"""Experiment orchestrator: runs all 4 variants and produces results.

Usage:
    python iter1/run_experiment.py --task pendulum --phase1-iters 30
    python iter1/run_experiment.py --task pendulum --phase1-iters 10 --preflight
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from reward_gen._loop_common import TASKS, pool_dir, sample_fixed_init_states  # noqa: E402
from curiosity.state_entropy import compute_state_entropy  # noqa: E402

ALL_METHODS = ["curiosity", "pareto", "eureka", "archive"]


# ---------------------------------------------------------------------------
# Phase 1: Run all 4 reward-search variants
# ---------------------------------------------------------------------------

def phase1(task: str, iters: int, model: str, seed: int, preflight: bool) -> None:
    print(f"\n{'='*60}")
    print(f"PHASE 1: Reward search — {task}, {iters} iters, 4 variants")
    print(f"{'='*60}\n")

    from reward_gen.gepa_curiosity import run as run_curiosity
    from reward_gen.gepa_pareto import run as run_pareto
    from reward_gen.eureka_bestofk import run as run_eureka
    from reward_gen.gepa_archive import run as run_archive

    runners = {
        "curiosity": run_curiosity,
        "pareto": run_pareto,
        "eureka": run_eureka,
        "archive": run_archive,
    }
    for method, runner in runners.items():
        t0 = time.time()
        print(f"\n--- Starting {method} ---")
        runner(task, iters, model=model, seed=seed, preflight=preflight)
        dt = time.time() - t0
        print(f"--- {method} done in {dt/60:.1f} min ---\n")


# ---------------------------------------------------------------------------
# Phase 2: Full PPO training for top-K rewards per variant
# ---------------------------------------------------------------------------

def _select_top_k(task: str, method: str, k: int) -> list[int]:
    """Return reward indices of top-k candidates by mean_fitness."""
    d = pool_dir(task, method)
    candidates = []
    for jp in sorted(d.glob("reward_*.json")):
        data = json.loads(jp.read_text())
        if data.get("failed_compile") or not data.get("fitness_vec"):
            continue
        candidates.append((data["mean_fitness"], data["idx"]))
    candidates.sort(reverse=True)
    return [idx for _, idx in candidates[:k]]


def phase2(task: str, top_k: int, seeds: list[int]) -> None:
    print(f"\n{'='*60}")
    print(f"PHASE 2: Full PPO training — top-{top_k} x {len(seeds)} seeds x 4 methods")
    print(f"{'='*60}\n")

    from training.train_ppo import train

    save = HERE / "training" / "checkpoints"
    for method in ALL_METHODS:
        idxs = _select_top_k(task, method, top_k)
        if not idxs:
            print(f"  [{method}] no scored candidates, skipping")
            continue
        print(f"  [{method}] top-{top_k} reward indices: {idxs}")
        for ridx, sd in itertools.product(idxs, seeds):
            t0 = time.time()
            print(f"    training {method} r{ridx:03d} seed={sd} ...", end=" ", flush=True)
            try:
                meta = train(task, method, ridx, sd, save)
                dt = time.time() - t0
                print(f"done ({dt/60:.1f} min) -> {meta['checkpoint']}")
            except Exception as e:
                print(f"FAIL: {e}")


# ---------------------------------------------------------------------------
# Phase 3: Diversity metrics on trained policies
# ---------------------------------------------------------------------------

def phase3(task: str) -> None:
    print(f"\n{'='*60}")
    print(f"PHASE 3: Diversity metrics")
    print(f"{'='*60}\n")

    import gymnasium as gym
    from stable_baselines3 import PPO
    from diversity.policy_distance import state_occupancy_tv

    spec = TASKS[task]
    ckpt_dir = HERE / "training" / "checkpoints"
    init_seeds = sample_fixed_init_states(spec.env_id, 20, seed=0)

    # Collect rollouts per method per checkpoint
    method_rollouts: dict[str, list[tuple[str, list[np.ndarray]]]] = {m: [] for m in ALL_METHODS}

    for ckpt in sorted(ckpt_dir.glob(f"{task}_*.zip")):
        stem = ckpt.stem
        method = None
        for m in ALL_METHODS:
            if f"_{m}_" in stem:
                method = m
                break
        if method is None:
            continue

        model = PPO.load(ckpt, device="cpu")
        rollouts = []
        for s in init_seeds[:5]:
            env = gym.make(spec.env_id)
            obs, _ = env.reset(seed=int(s))
            traj = [obs.copy()]
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, term, trunc, _ = env.step(action)
                traj.append(obs.copy())
                done = bool(term or trunc)
            rollouts.append(np.array(traj))
            env.close()
        method_rollouts[method].append((stem, rollouts))

    # Compute pairwise state-occupancy TV within each method
    rows = []
    for method, entries in method_rollouts.items():
        if len(entries) < 2:
            continue
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                tv = state_occupancy_tv(entries[i][1], entries[j][1])
                rows.append({
                    "method": method,
                    "policy_a": entries[i][0],
                    "policy_b": entries[j][0],
                    "state_occ_tv": tv,
                })

    out_dir = HERE / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_csv(out_dir / f"{task}_diversity.csv", index=False)
        print(f"  Saved {len(rows)} pairwise diversity measurements")
        for method in ALL_METHODS:
            sub = df[df["method"] == method]
            if not sub.empty:
                print(f"    {method}: mean TV = {sub['state_occ_tv'].mean():.4f} "
                      f"(n={len(sub)} pairs)")
    else:
        print("  No pairwise comparisons possible (need >=2 checkpoints per method)")

    # Also compute state_entropy on ALL methods' policies for fair comparison
    entropy_rows = []
    for method, entries in method_rollouts.items():
        for stem, _ in entries:
            ckpt = ckpt_dir / f"{stem}.zip"
            model = PPO.load(ckpt, device="cpu")
            ent = compute_state_entropy(model, spec.env_id, spec.name, n_episodes=20, seed=0)
            entropy_rows.append({"method": method, "policy": stem, "state_entropy": ent})

    if entropy_rows:
        import pandas as pd
        edf = pd.DataFrame(entropy_rows)
        edf.to_csv(out_dir / f"{task}_entropy.csv", index=False)
        print(f"\n  State entropy across all methods:")
        for method in ALL_METHODS:
            sub = edf[edf["method"] == method]
            if not sub.empty:
                print(f"    {method}: mean entropy = {sub['state_entropy'].mean():.3f} "
                      f"(std={sub['state_entropy'].std():.3f})")


# ---------------------------------------------------------------------------
# Phase 4: Perturbation robustness
# ---------------------------------------------------------------------------

def phase4(task: str) -> None:
    print(f"\n{'='*60}")
    print(f"PHASE 4: Perturbation robustness")
    print(f"{'='*60}\n")

    from robustness.evaluate_under_perturb import pool_robustness as eval_pool_robustness

    for method in ALL_METHODS:
        ckpts = list((HERE / "training" / "checkpoints").glob(f"{task}_{method}_*.zip"))
        if not ckpts:
            print(f"  [{method}] no checkpoints, skipping robustness eval")
            continue
        print(f"  [{method}] evaluating robustness on {len(ckpts)} checkpoints ...")
        try:
            result = eval_pool_robustness(task, ckpts)
            out_dir = HERE / "tables"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{task}_{method}_robustness.json"
            out_path.write_text(json.dumps(result, indent=2))
            print(f"    pool_robustness = {result.get('pool_robustness_score', 'N/A')}")
        except Exception as e:
            print(f"    FAIL: {e}")


# ---------------------------------------------------------------------------
# Phase 5: Statistical comparison
# ---------------------------------------------------------------------------

def phase5(task: str) -> None:
    print(f"\n{'='*60}")
    print(f"PHASE 5: Statistical comparison")
    print(f"{'='*60}\n")

    from analysis.statistical_tests import bootstrap_mean_ci, compare_groups, holm_adjust

    # Load diversity data
    div_path = HERE / "tables" / f"{task}_diversity.csv"
    if div_path.exists():
        import pandas as pd
        df = pd.read_csv(div_path)
        print("Diversity (state-occupancy TV):")
        comps = []
        methods_present = [m for m in ALL_METHODS if m in df["method"].values]
        for m in methods_present:
            vals = df[df["method"] == m]["state_occ_tv"].tolist()
            ci = bootstrap_mean_ci(vals)
            print(f"  {m}: mean={ci.mean:.4f} [{ci.lo:.4f}, {ci.hi:.4f}] (n={len(vals)})")

        for m_a, m_b in itertools.combinations(methods_present, 2):
            a = df[df["method"] == m_a]["state_occ_tv"].tolist()
            b = df[df["method"] == m_b]["state_occ_tv"].tolist()
            if a and b:
                comps.append(compare_groups(a, b, name=f"{m_a}_vs_{m_b}"))
        if comps:
            holm_adjust(comps)
            print("\n  Pairwise (Mann-Whitney, Holm-adjusted):")
            for c in comps:
                print(f"    {c.name}: U={c.u_stat:.1f}, p={c.p_value:.4f}, p_adj={c.p_adj:.4f}")
    else:
        print("  No diversity data found")

    # Load fitness from pool JSONs
    print("\nFitness (mean env-true return):")
    for method in ALL_METHODS:
        d = pool_dir(task, method)
        fitnesses = []
        for jp in sorted(d.glob("reward_*.json")):
            data = json.loads(jp.read_text())
            if data.get("failed_compile") or not data.get("fitness_vec"):
                continue
            fitnesses.append(data["mean_fitness"])
        if fitnesses:
            ci = bootstrap_mean_ci(fitnesses)
            print(f"  {method}: mean={ci.mean:.2f} [{ci.lo:.2f}, {ci.hi:.2f}], "
                  f"max={max(fitnesses):.2f} (n={len(fitnesses)})")
        else:
            print(f"  {method}: no data")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Run full curiosity-as-Pareto-axis experiment")
    ap.add_argument("--task", required=True, choices=list(TASKS.keys()))
    ap.add_argument("--phase1-iters", type=int, default=30)
    ap.add_argument("--phase2-seeds", type=int, default=3)
    ap.add_argument("--phase2-top-k", type=int, default=3)
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preflight", action="store_true", help="Short preflight check (10 iters, no phase 2+)")
    ap.add_argument("--skip-phase1", action="store_true", help="Skip phase 1 (use existing pool)")
    ap.add_argument("--skip-phase2", action="store_true", help="Skip phase 2 (use existing checkpoints)")
    args = ap.parse_args()

    t_start = time.time()
    seeds = list(range(args.phase2_seeds))

    if args.preflight:
        args.phase1_iters = min(args.phase1_iters, 10)

    if not args.skip_phase1:
        phase1(args.task, args.phase1_iters, args.model, args.seed, args.preflight)

    if args.preflight:
        print(f"\nPreflight complete in {(time.time()-t_start)/60:.1f} min. Skipping phases 2-5.")
        return

    if not args.skip_phase2:
        phase2(args.task, args.phase2_top_k, seeds)

    phase3(args.task)
    phase4(args.task)
    phase5(args.task)

    dt = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"ALL PHASES COMPLETE for {args.task} in {dt/60:.1f} min ({dt/3600:.1f} hrs)")
    print(f"{'='*60}")

    # Generate results markdown
    try:
        from analysis.generate_results_md import generate
        generate(args.task)
        print(f"\nResults written to {HERE / 'results_curiosity.md'}")
    except Exception as e:
        print(f"\nResults markdown generation failed: {e}")


if __name__ == "__main__":
    main()
