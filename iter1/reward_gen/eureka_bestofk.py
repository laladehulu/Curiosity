"""Eureka best-of-K baseline: greedy selection of highest-fitness candidate.

No Pareto front, no curiosity, no descriptors.  Parent is always the single
best candidate seen so far.  This isolates the effect of LLM-guided mutation
without any diversity-preserving selection mechanism.
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
    best: Optional[Candidate] = None

    print(f"[eureka_bestofk] task={spec.name} iters={iters} model={model} preflight={preflight}", flush=True)

    for it in range(iters):
        if it == 0 or best is None:
            volatile = "\n\nProduce the initial reward function now."
            parent_idx: Optional[int] = None
        else:
            parent_idx = best.idx
            fit_summary = (
                f"mean env return = {best.mean_fitness:.2f} (BEST so far); "
                f"per-init-state min={min(best.fitness_vec):.2f}, "
                f"max={max(best.fitness_vec):.2f}, "
                f"std={float(np.std(best.fitness_vec)):.2f}."
            )
            hint = (
                "This is the best reward so far. Improve its env-true return. "
                "Focus on maximizing the mean return across initial states."
            )
            volatile = build_mutation_suffix(best.source, best.curve, fit_summary, hint)

        resp = llm.chat(SYSTEM_PROMPT, build_user_blocks(task_prompt, volatile))
        source = extract_code(resp.text)

        c = Candidate(idx=it, source=source, parent_idx=parent_idx, train_seed=seed * 1000 + it)
        try:
            compiled = compile_reward(source)
        except RewardCompileError as e:
            c.failed_compile = True
            c.error = str(e)
            pool.append(c)
            save_candidate(spec.name, "eureka", c)
            log_event(spec.name, "eureka", {"iter": it, "event": "compile_fail", "error": str(e)})
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
            save_candidate(spec.name, "eureka", c)
            log_event(spec.name, "eureka", {"iter": it, "event": "train_fail", "error": str(e)})
            print(f"  iter {it}: train/eval FAIL — {e}", flush=True)
            continue

        pool.append(c)
        save_candidate(spec.name, "eureka", c)

        if best is None or c.mean_fitness > best.mean_fitness:
            best = c
            improved = "NEW BEST"
        else:
            improved = f"no improvement (best={best.mean_fitness:.2f})"

        log_event(
            spec.name,
            "eureka",
            {
                "iter": it,
                "event": "ok",
                "mean_fitness": c.mean_fitness,
                "is_new_best": (best is c),
                "parent_idx": parent_idx,
            },
        )
        print(
            f"  iter {it}: mean_fitness={c.mean_fitness:.2f} {improved}",
            flush=True,
        )

    # Final report
    scored = [c for c in pool if c.fitness_vec]
    if not scored:
        print("\n[eureka_bestofk] no scored candidates — abort.")
        return
    max_mean = max(c.mean_fitness for c in scored)
    print(
        f"\n[eureka_bestofk] DONE. compiled={len(scored)}/{len(pool)}, "
        f"best_mean_fitness={max_mean:.2f}, "
        f"llm_cost=${llm.cost_estimate_usd():.4f}"
    )
    log_event(
        spec.name,
        "eureka",
        {
            "event": "done",
            "compiled": len(scored),
            "best_mean_fitness": max_mean,
            "llm_cost_usd": llm.cost_estimate_usd(),
        },
    )

    if preflight:
        nonzero = any(c.mean_fitness > -1e5 for c in scored)
        improving = len(scored) >= 2 and scored[-1].mean_fitness > scored[0].mean_fitness
        if not nonzero:
            print(f"PREFLIGHT WARN: nonzero_return={nonzero}. Per protocol, stop and report to human author.")
        else:
            print(f"PREFLIGHT OK: best={max_mean:.2f}, improving={improving}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS.keys()))
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preflight", action="store_true")
    args = ap.parse_args()
    if args.preflight:
        args.iters = min(args.iters, 10)
    run(args.task, args.iters, model=args.model, seed=args.seed, preflight=args.preflight)


if __name__ == "__main__":
    main()
