"""Curiosity-Pareto variant of Phase 1 reward search.

Selection: maintain a Pareto frontier over TWO axes —
  (1) mean env-true return (fitness)
  (2) state-visitation entropy (curiosity)
Mutation: sample a parent uniformly from the current frontier.

This is the thesis's core claim: curiosity-derived behavioral novelty as an
explicit Pareto selection axis preserves behaviorally distinct policies
WITHOUT hand-designed descriptors.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from curiosity.state_entropy import compute_state_entropy  # noqa: E402
from reward_gen._loop_common import (  # noqa: E402
    Candidate,
    TASKS,
    build_mutation_suffix,
    build_user_blocks,
    evaluate_per_init_state,
    log_event,
    sample_fixed_init_states,
    save_candidate,
    train_short_ppo,
)
from reward_gen.gepa_pareto import pareto_front  # noqa: E402
from reward_gen.llm_client import LLMClient  # noqa: E402
from reward_gen.reward_template import (  # noqa: E402
    RewardCompileError,
    compile_reward,
    extract_code,
)

SYSTEM_PROMPT = (
    "You are an expert reinforcement-learning engineer designing reward functions. "
    "Output only a single Python code block. Follow the signature exactly."
)


def run(task_name: str, iters: int, model: str = "claude-haiku-4-5", seed: int = 0, preflight: bool = False) -> None:
    spec = TASKS[task_name]
    random.seed(seed)
    np.random.seed(seed)

    llm = LLMClient(model=model, temperature=0.8)
    task_prompt = spec.prompt()
    init_seeds = sample_fixed_init_states(spec.env_id, spec.n_init_states, seed=seed)

    pool: list[Candidate] = []

    print(f"[gepa_curiosity] task={spec.name} iters={iters} model={model} preflight={preflight}", flush=True)

    for it in range(iters):
        if it == 0:
            volatile = "\n\nProduce the initial reward function now."
            parent_idx: Optional[int] = None
        else:
            scored = [c for c in pool if c.fitness_vec]
            if not scored:
                parent_idx = None
                volatile = "\n\nAll previous candidates failed; produce a fresh initial reward."
            else:
                # 2D Pareto: [mean_fitness, curiosity]
                score_mat = np.array([[c.mean_fitness, c.curiosity] for c in scored])
                front_mask = pareto_front(score_mat)
                front_cands = [scored[k] for k, on in enumerate(front_mask) if on]
                parent = random.choice(front_cands)
                parent_idx = parent.idx
                fit_summary = (
                    f"mean env return = {parent.mean_fitness:.2f}; "
                    f"state-visitation entropy = {parent.curiosity:.3f}; "
                    f"per-init-state min={min(parent.fitness_vec):.2f}, "
                    f"max={max(parent.fitness_vec):.2f}. "
                    f"Curiosity-Pareto frontier size = {len(front_cands)}."
                )
                hint = (
                    "This candidate is on the Pareto frontier of (env-true return, state-visitation entropy). "
                    "Try to improve the task return OR produce more diverse state coverage (visiting more "
                    "distinct regions of the observation space). Both axes matter."
                )
                volatile = build_mutation_suffix(parent.source, parent.curve, fit_summary, hint)

        resp = llm.chat(SYSTEM_PROMPT, build_user_blocks(task_prompt, volatile))
        source = extract_code(resp.text)

        c = Candidate(idx=it, source=source, parent_idx=parent_idx, train_seed=seed * 1000 + it)
        try:
            compiled = compile_reward(source)
        except RewardCompileError as e:
            c.failed_compile = True
            c.error = str(e)
            pool.append(c)
            save_candidate(spec.name, "curiosity", c)
            log_event(spec.name, "curiosity", {"iter": it, "event": "compile_fail", "error": str(e)})
            print(f"  iter {it}: compile FAIL — {e}", flush=True)
            continue

        try:
            model_, curve = train_short_ppo(spec.env_id, compiled, spec.short_steps, c.train_seed)
            fitness_vec, _ = evaluate_per_init_state(model_, spec.env_id, init_seeds, use_env_reward=True)
            curiosity_val = compute_state_entropy(model_, spec.env_id, spec.name, n_episodes=20, seed=c.train_seed)
            c.curve = curve
            c.fitness_vec = [float(x) for x in fitness_vec]
            c.mean_fitness = float(np.mean(fitness_vec))
            c.curiosity = curiosity_val
        except Exception as e:
            c.error = f"train/eval: {e}"
            pool.append(c)
            save_candidate(spec.name, "curiosity", c)
            log_event(spec.name, "curiosity", {"iter": it, "event": "train_fail", "error": str(e)})
            print(f"  iter {it}: train/eval FAIL — {e}", flush=True)
            continue

        pool.append(c)
        save_candidate(spec.name, "curiosity", c)
        log_event(
            spec.name,
            "curiosity",
            {
                "iter": it,
                "event": "ok",
                "mean_fitness": c.mean_fitness,
                "curiosity": c.curiosity,
                "parent_idx": parent_idx,
            },
        )
        print(
            f"  iter {it}: mean_fitness={c.mean_fitness:.2f} curiosity={c.curiosity:.3f} "
            f"(per-state min={min(c.fitness_vec):.2f}, max={max(c.fitness_vec):.2f})",
            flush=True,
        )

    # Final report
    scored = [c for c in pool if c.fitness_vec]
    if not scored:
        print("\n[gepa_curiosity] no scored candidates — abort.")
        return
    score_mat = np.array([[c.mean_fitness, c.curiosity] for c in scored])
    front = pareto_front(score_mat)
    n_front = int(front.sum())
    max_mean = max(c.mean_fitness for c in scored)
    entropies = [c.curiosity for c in scored]
    print(
        f"\n[gepa_curiosity] DONE. compiled={len(scored)}/{len(pool)}, "
        f"frontier_size={n_front}, best_mean_fitness={max_mean:.2f}, "
        f"entropy range=[{min(entropies):.3f}, {max(entropies):.3f}], "
        f"llm_cost=${llm.cost_estimate_usd():.4f}"
    )
    log_event(
        spec.name,
        "curiosity",
        {
            "event": "done",
            "compiled": len(scored),
            "frontier_size": n_front,
            "best_mean_fitness": max_mean,
            "entropy_min": min(entropies),
            "entropy_max": max(entropies),
            "llm_cost_usd": llm.cost_estimate_usd(),
        },
    )

    if preflight:
        entropy_varies = (max(entropies) - min(entropies)) > 0.01
        nonzero = any(c.mean_fitness > -1e5 for c in scored)
        if n_front < 2 or not nonzero or not entropy_varies:
            print(
                f"PREFLIGHT WARN: frontier_size={n_front}, nonzero_return={nonzero}, "
                f"entropy_varies={entropy_varies}. Per protocol, stop and report to human author."
            )
        else:
            print(f"PREFLIGHT OK: frontier={n_front}, entropy varies, nonzero returns present.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS.keys()))
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preflight", action="store_true",
                    help="Reduce to short check; print curiosity-frontier diagnostics")
    args = ap.parse_args()
    if args.preflight:
        args.iters = min(args.iters, 10)
    run(args.task, args.iters, model=args.model, seed=args.seed, preflight=args.preflight)


if __name__ == "__main__":
    main()
