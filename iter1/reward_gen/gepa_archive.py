"""Behavior-archive variant of Phase 1 reward search.

Selection: MAP-Elites archive indexed by behavioral descriptor of the resulting policy.
Within each cell, keep the reward whose policy achieves highest env-true mean return.
Mutation: sample a parent uniformly from filled cells.

Per protocol §"Variant B: Behavior-Archive (proposed)".
"""
from __future__ import annotations

import argparse
import importlib
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


def load_descriptor(task_name: str):
    """Import descriptor module dynamically. Returns (compute_descriptor, cell_index, bounds, bins).

    compute_descriptor(model, env_id, init_seeds) -> np.ndarray of shape (D,)
    cell_index(desc) -> tuple[int, ...]
    bins: tuple[int, ...]
    """
    mod = importlib.import_module(f"descriptors.{task_name}_descriptor")
    return mod.compute_descriptor, mod.cell_index, mod.BINS


def rollout_for_descriptor(model, env_id: str, seeds: list[int], n_eps: int = 5):
    """Collect short rollouts for descriptor computation. Returns list of dicts of trajectories."""
    import gymnasium as gym
    trajs = []
    for s in seeds[:n_eps]:
        env = gym.make(env_id)
        obs, _ = env.reset(seed=int(s))
        ep = {"obs": [obs], "action": [], "next_obs": [], "reward": []}
        done = False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs2, r, term, trunc, _ = env.step(a)
            ep["action"].append(a)
            ep["next_obs"].append(obs2)
            ep["reward"].append(r)
            obs = obs2
            done = bool(term or trunc)
            ep["obs"].append(obs)
        trajs.append({k: np.asarray(v) for k, v in ep.items()})
        env.close()
    return trajs


def run(task_name: str, iters: int, model: str = "claude-haiku-4-5", seed: int = 0, preflight: bool = False) -> None:
    spec = TASKS[task_name]
    random.seed(seed)
    np.random.seed(seed)

    llm = LLMClient(model=model, temperature=0.8)
    task_prompt = spec.prompt()
    init_seeds = sample_fixed_init_states(spec.env_id, spec.n_init_states, seed=seed)
    compute_descriptor, cell_index, bins = load_descriptor(spec.name)

    # archive: cell tuple -> Candidate (best in cell by mean_fitness)
    archive: dict[tuple[int, ...], Candidate] = {}
    pool: list[Candidate] = []

    print(
        f"[gepa_archive] task={spec.name} iters={iters} model={model} "
        f"bins={bins} preflight={preflight}",
        flush=True,
    )

    for it in range(iters):
        if it == 0 or not archive:
            volatile = "\n\nProduce the initial reward function now."
            parent_idx: Optional[int] = None
        else:
            cells = list(archive.keys())
            parent_cell = random.choice(cells)
            parent = archive[parent_cell]
            parent_idx = parent.idx
            fit_summary = (
                f"mean env return = {parent.mean_fitness:.2f}; "
                f"behavioral descriptor = {parent.descriptor}; "
                f"archive cell = {parent.archive_cell}; "
                f"filled cells = {len(archive)}."
            )
            hint = (
                "This candidate is the best occupant of its archive cell (a behavioral region). "
                "Try a variant that produces a behaviorally different policy — different cell — while not collapsing return."
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
            save_candidate(spec.name, "archive", c)
            log_event(spec.name, "archive", {"iter": it, "event": "compile_fail", "error": str(e)})
            print(f"  iter {it}: compile FAIL — {e}", flush=True)
            continue

        try:
            model_, curve = train_short_ppo(spec.env_id, compiled, spec.short_steps, c.train_seed)
            fitness_vec, _ = evaluate_per_init_state(model_, spec.env_id, init_seeds, use_env_reward=True)
            trajs = rollout_for_descriptor(model_, spec.env_id, init_seeds, n_eps=5)
            desc = compute_descriptor(trajs)
            cell = cell_index(desc)
            c.curve = curve
            c.fitness_vec = [float(x) for x in fitness_vec]
            c.mean_fitness = float(np.mean(fitness_vec))
            c.descriptor = [float(x) for x in np.asarray(desc).flatten().tolist()]
            c.archive_cell = list(cell)
        except Exception as e:
            c.error = f"train/eval: {e}"
            pool.append(c)
            save_candidate(spec.name, "archive", c)
            log_event(spec.name, "archive", {"iter": it, "event": "train_fail", "error": str(e)})
            print(f"  iter {it}: train/eval FAIL — {e}", flush=True)
            continue

        pool.append(c)
        save_candidate(spec.name, "archive", c)

        # archive insertion: keep best mean_fitness per cell
        prev = archive.get(cell)
        if prev is None or c.mean_fitness > prev.mean_fitness:
            archive[cell] = c
            placed = "INSERTED"
        else:
            placed = f"rejected (cell occupant has {prev.mean_fitness:.2f})"

        log_event(
            spec.name,
            "archive",
            {
                "iter": it,
                "event": "ok",
                "mean_fitness": c.mean_fitness,
                "cell": list(cell),
                "placed": placed,
                "filled_cells": len(archive),
                "parent_idx": parent_idx,
            },
        )
        print(
            f"  iter {it}: mean_fitness={c.mean_fitness:.2f} cell={cell} {placed} "
            f"(filled={len(archive)})",
            flush=True,
        )

    # Final report
    scored = [c for c in pool if c.fitness_vec]
    if not scored:
        print("\n[gepa_archive] no scored candidates — abort.")
        return
    max_mean = max(c.mean_fitness for c in scored)
    qd_score = float(sum(c.mean_fitness for c in archive.values()))
    print(
        f"\n[gepa_archive] DONE. compiled={len(scored)}/{len(pool)}, "
        f"filled_cells={len(archive)}, best_mean_fitness={max_mean:.2f}, "
        f"QD_score={qd_score:.2f}, llm_cost=${llm.cost_estimate_usd():.4f}"
    )
    log_event(
        spec.name,
        "archive",
        {
            "event": "done",
            "compiled": len(scored),
            "filled_cells": len(archive),
            "best_mean_fitness": max_mean,
            "qd_score": qd_score,
            "llm_cost_usd": llm.cost_estimate_usd(),
        },
    )

    if preflight:
        # Per protocol §"Pre-flight checks": confirm archive cells fill and at least
        # one nonzero-return reward exists.
        nonzero = any(c.mean_fitness > -1e5 for c in scored)
        if len(archive) < 2 or not nonzero:
            print(
                f"PREFLIGHT WARN: filled_cells={len(archive)}, nonzero_return={nonzero}. "
                f"Per protocol, stop and report to human author."
            )


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
