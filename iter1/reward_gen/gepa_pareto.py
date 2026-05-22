"""GEPA-Pareto variant of Phase 1 reward search.

Selection: maintain a Pareto frontier over per-init-state env-true return vectors.
Mutation: sample a parent uniformly from the current frontier; ask the LLM for a
reflective variant.

Per protocol §"Variant A: GEPA-Pareto (baseline)".
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
from reward_gen.llm_client import LLMClient  # noqa: E402
from reward_gen.reward_template import (  # noqa: E402
    RewardCompileError,
    compile_reward,
    extract_code,
)


def pareto_front(score_matrix: np.ndarray) -> np.ndarray:
    """Return boolean mask of non-dominated rows. score_matrix[i] = i-th candidate's score vector.
    Higher is better on every axis."""
    n = score_matrix.shape[0]
    on_front = np.ones(n, dtype=bool)
    for i in range(n):
        if not on_front[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j >= i everywhere and j > i somewhere
            if np.all(score_matrix[j] >= score_matrix[i]) and np.any(score_matrix[j] > score_matrix[i]):
                on_front[i] = False
                break
    return on_front


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

    print(f"[gepa_pareto] task={spec.name} iters={iters} model={model} preflight={preflight}", flush=True)

    for it in range(iters):
        if it == 0:
            volatile = "\n\nProduce the initial reward function now."
            parent_idx: Optional[int] = None
        else:
            # Pick parent from Pareto front
            score_mat = np.array([c.fitness_vec for c in pool if c.fitness_vec])
            if score_mat.size == 0:
                parent_idx = None
                volatile = "\n\nAll previous candidates failed; produce a fresh initial reward."
            else:
                idxs = [c.idx for c in pool if c.fitness_vec]
                front = pareto_front(score_mat)
                front_idxs = [idxs[k] for k, on in enumerate(front) if on]
                parent_idx = random.choice(front_idxs)
                parent = next(c for c in pool if c.idx == parent_idx)
                fit_summary = (
                    f"mean env return = {parent.mean_fitness:.2f}; "
                    f"per-init-state min={min(parent.fitness_vec):.2f}, "
                    f"max={max(parent.fitness_vec):.2f}, "
                    f"std={float(np.std(parent.fitness_vec)):.2f}. "
                    f"Frontier size = {len(front_idxs)}."
                )
                hint = (
                    "This candidate is on the Pareto frontier over per-init-state env-true returns. "
                    "Try to dominate it on the init states where it scored lowest while not regressing where it scored highest."
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
            save_candidate(spec.name, "pareto", c)
            log_event(spec.name, "pareto", {"iter": it, "event": "compile_fail", "error": str(e)})
            print(f"  iter {it}: compile FAIL — {e}", flush=True)
            continue

        try:
            model_, curve = train_short_ppo(spec.env_id, compiled, spec.short_steps, c.train_seed)
            fitness_vec, _ = evaluate_per_init_state(model_, spec.env_id, init_seeds, use_env_reward=True)
            c.curve = curve
            c.fitness_vec = [float(x) for x in fitness_vec]
            c.mean_fitness = float(np.mean(fitness_vec))
        except Exception as e:
            c.error = f"train/eval: {e}"
            pool.append(c)
            save_candidate(spec.name, "pareto", c)
            log_event(spec.name, "pareto", {"iter": it, "event": "train_fail", "error": str(e)})
            print(f"  iter {it}: train/eval FAIL — {e}", flush=True)
            continue

        pool.append(c)
        save_candidate(spec.name, "pareto", c)
        log_event(
            spec.name,
            "pareto",
            {
                "iter": it,
                "event": "ok",
                "mean_fitness": c.mean_fitness,
                "parent_idx": parent_idx,
            },
        )
        print(
            f"  iter {it}: mean_fitness={c.mean_fitness:.2f} "
            f"(per-state min={min(c.fitness_vec):.2f}, max={max(c.fitness_vec):.2f})",
            flush=True,
        )

    # Final report
    scored = [c for c in pool if c.fitness_vec]
    if not scored:
        print("\n[gepa_pareto] no scored candidates — abort.")
        return
    score_mat = np.array([c.fitness_vec for c in scored])
    front = pareto_front(score_mat)
    n_front = int(front.sum())
    max_mean = max(c.mean_fitness for c in scored)
    print(
        f"\n[gepa_pareto] DONE. compiled={len(scored)}/{len(pool)}, "
        f"frontier_size={n_front}, best_mean_fitness={max_mean:.2f}, "
        f"llm_cost=${llm.cost_estimate_usd():.4f}"
    )
    log_event(
        spec.name,
        "pareto",
        {
            "event": "done",
            "compiled": len(scored),
            "frontier_size": n_front,
            "best_mean_fitness": max_mean,
            "llm_cost_usd": llm.cost_estimate_usd(),
        },
    )

    if preflight:
        # Per protocol §"Pre-flight checks": confirm Pareto frontier updates sensibly
        # and at least one nonzero-return reward exists.
        threshold = spec.success_threshold
        nonzero = any(c.mean_fitness > -1e5 for c in scored)
        if n_front < 2 or not nonzero:
            print(
                f"PREFLIGHT WARN: frontier_size={n_front}, nonzero_return={nonzero}. "
                f"Per protocol, stop and report to human author."
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS.keys()))
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preflight", action="store_true",
                    help="Reduce to short check; print Pareto-frontier diagnostics")
    args = ap.parse_args()
    if args.preflight:
        args.iters = min(args.iters, 10)
    run(args.task, args.iters, model=args.model, seed=args.seed, preflight=args.preflight)


if __name__ == "__main__":
    main()
