"""Eureka (Ma et al., 2023) baseline — Mac-CPU port for HalfCheetah-v4.

Faithful to Eureka's algorithm at workshop-pilot scale:

    repeat N iterations:
        sample K candidate reward functions from the LLM
        train PPO under each candidate for short_steps
        evaluate each under the ENV-TRUE reward (the fitness function)
        keep the best candidate; build a Reward Reflection containing
          per-step fitness-component statistics (Eureka §3.3)
        use (best source + reflection) as the parent for next iteration's
          K samples

Differences vs the original paper:
- Original Eureka uses Isaac Gym for parallel GPU sims and trains for ~3e8
  steps per candidate. This is a CPU pilot — K is small, short_steps is
  small. Defaults aim to fit in ~30 min total on M-series CPU.
- Original Eureka feeds the env source code into the LLM. We feed a prompt
  describing the observation/action layout and fitness components instead,
  because the Gymnasium HalfCheetah-v4 env is not a single readable file.

CLI:
    python eureka.py --iters 3 --samples 3 --steps 50000
    python eureka.py --smoke              # 1 iter, 2 samples, 5k steps
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from components import (  # noqa: E402
    EpisodeStats,
    LLMClient,
    RewardCompileError,
    build_reflection_block,
    compile_reward,
    evaluate_policy,
    extract_code,
    train_short_ppo,
)
from mock_llm import MockLLMClient  # noqa: E402


ENV_ID = "HalfCheetah-v4"
PROMPT_PATH = HERE / "prompts" / "halfcheetah.txt"
POOL_DIR = HERE / "pool"
LOG_DIR = HERE / "logs"

SYSTEM_PROMPT = (
    "You are an expert reinforcement-learning engineer designing reward functions "
    "for robot locomotion. Output only a single Python code block. Follow the "
    "signature exactly. Be willing to try qualitatively different shaping ideas."
)


@dataclass
class Candidate:
    gen: int
    sample_idx: int
    source: str
    parent_gen: Optional[int]
    train_seed: int
    failed_compile: bool = False
    error: Optional[str] = None
    mean_env_return: float = float("-inf")
    eval_returns: list[float] = field(default_factory=list)
    eval_lengths: list[int] = field(default_factory=list)

    def label(self) -> str:
        return f"gen{self.gen:02d}_s{self.sample_idx}"

    def to_dict(self) -> dict:
        return {
            "gen": self.gen,
            "sample_idx": self.sample_idx,
            "parent_gen": self.parent_gen,
            "train_seed": self.train_seed,
            "failed_compile": self.failed_compile,
            "error": self.error,
            "mean_env_return": self.mean_env_return,
            "eval_returns": self.eval_returns,
            "eval_lengths": self.eval_lengths,
        }


# ---------------------------------------------------------------------------
# Sampling K candidates from the LLM
# ---------------------------------------------------------------------------

def build_user_prompt(task_prompt: str, reflection: Optional[str]) -> str:
    if reflection is None:
        return task_prompt + "\n\nProduce the initial reward function now."
    return task_prompt + reflection


def sample_candidates(
    llm,
    task_prompt: str,
    reflection: Optional[str],
    k: int,
    parallel: bool,
) -> list[str]:
    """Sample K LLM outputs. Anthropic API does not return multiple choices
    per call; we issue K independent requests at temperature > 0."""
    user = build_user_prompt(task_prompt, reflection)
    sources: list[str] = [""] * k

    def one(i: int) -> tuple[int, str]:
        resp = llm.chat(SYSTEM_PROMPT, user)
        return i, extract_code(resp.text)

    if parallel and k > 1:
        with ThreadPoolExecutor(max_workers=min(k, 4)) as ex:
            for fut in as_completed([ex.submit(one, i) for i in range(k)]):
                i, src = fut.result()
                sources[i] = src
    else:
        for i in range(k):
            _, sources[i] = one(i)
    return sources


# ---------------------------------------------------------------------------
# Training + evaluation for one candidate
# ---------------------------------------------------------------------------

def train_and_eval(
    cand: Candidate,
    short_steps: int,
    eval_seeds: list[int],
) -> tuple[Candidate, Optional[list[EpisodeStats]]]:
    try:
        compiled = compile_reward(cand.source)
    except RewardCompileError as e:
        cand.failed_compile = True
        cand.error = str(e)
        return cand, None

    try:
        model = train_short_ppo(ENV_ID, compiled, short_steps, cand.train_seed)
        mean_ret, stats = evaluate_policy(model, ENV_ID, eval_seeds)
        cand.mean_env_return = mean_ret
        cand.eval_returns = [float(s.env_return) for s in stats]
        cand.eval_lengths = [int(s.length) for s in stats]
        return cand, stats
    except Exception as e:
        cand.error = f"train/eval: {e}"
        return cand, None


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_candidate(run_id: str, cand: Candidate) -> None:
    d = POOL_DIR / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{cand.label()}.py").write_text(cand.source or "# compile failed\n")
    (d / f"{cand.label()}.json").write_text(json.dumps(cand.to_dict(), indent=2))


def log_event(run_id: str, event: dict) -> None:
    LOG_DIR.mkdir(exist_ok=True)
    line = json.dumps({"ts": time.time(), "run_id": run_id, **event})
    with (LOG_DIR / f"{run_id}.jsonl").open("a") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(
    iters: int,
    samples: int,
    short_steps: int,
    eval_episodes: int,
    model_name: str,
    seed: int,
    parallel_sampling: bool,
    mock_llm: bool,
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    run_id = f"halfcheetah_{int(time.time())}"
    task_prompt = PROMPT_PATH.read_text()
    if mock_llm:
        llm = MockLLMClient()
        effective_model = "mock-llm"
        parallel_sampling = False  # mock is in-process; threads add nothing
    else:
        llm = LLMClient(model=model_name, temperature=1.0)
        effective_model = model_name

    print(
        f"[eureka] run_id={run_id} env={ENV_ID} model={effective_model}\n"
        f"         iters={iters} samples_per_iter={samples} "
        f"short_steps={short_steps} eval_episodes={eval_episodes}",
        flush=True,
    )

    best_so_far: Optional[Candidate] = None
    best_stats: Optional[list[EpisodeStats]] = None
    reflection: Optional[str] = None
    eval_seeds = [seed * 1000 + i for i in range(eval_episodes)]

    for gen in range(iters):
        t0 = time.time()
        print(f"\n[eureka] gen {gen}: sampling {samples} candidates...", flush=True)
        sources = sample_candidates(
            llm, task_prompt, reflection, samples, parallel_sampling
        )
        gen_cands: list[Candidate] = []
        gen_stats: list[Optional[list[EpisodeStats]]] = []
        for s_idx, src in enumerate(sources):
            cand = Candidate(
                gen=gen,
                sample_idx=s_idx,
                source=src,
                parent_gen=(gen - 1) if best_so_far else None,
                train_seed=seed * 10000 + gen * 100 + s_idx,
            )
            print(f"  [{cand.label()}] train+eval ...", flush=True)
            cand, stats = train_and_eval(cand, short_steps, eval_seeds)
            if cand.failed_compile:
                print(f"    compile FAIL: {cand.error}", flush=True)
            elif cand.error:
                print(f"    train/eval FAIL: {cand.error}", flush=True)
            else:
                print(
                    f"    mean_env_return={cand.mean_env_return:.2f} "
                    f"(seeds={cand.eval_returns})",
                    flush=True,
                )
            save_candidate(run_id, cand)
            gen_cands.append(cand)
            gen_stats.append(stats)
            log_event(
                run_id,
                {
                    "gen": gen,
                    "sample_idx": s_idx,
                    "event": "candidate",
                    "mean_env_return": cand.mean_env_return,
                    "failed_compile": cand.failed_compile,
                    "error": cand.error,
                },
            )

        # Pick best of this generation (Eureka: best by env-true return)
        scored = [(c, st) for c, st in zip(gen_cands, gen_stats) if st is not None]
        if not scored:
            print(f"[eureka] gen {gen}: all candidates failed; keeping prior best", flush=True)
            continue

        scored.sort(key=lambda cs: cs[0].mean_env_return, reverse=True)
        gen_best, gen_best_stats = scored[0]

        # Survivor = better of (gen_best, best_so_far)
        if best_so_far is None or gen_best.mean_env_return > best_so_far.mean_env_return:
            best_so_far = gen_best
            best_stats = gen_best_stats

        dt = time.time() - t0
        print(
            f"[eureka] gen {gen} done in {dt:.1f}s. "
            f"gen_best={gen_best.label()} ({gen_best.mean_env_return:.2f}); "
            f"running_best={best_so_far.label()} ({best_so_far.mean_env_return:.2f}); "
            f"llm_spend=${llm.cost_estimate_usd():.4f}",
            flush=True,
        )
        log_event(
            run_id,
            {
                "gen": gen,
                "event": "gen_summary",
                "gen_best_label": gen_best.label(),
                "gen_best_return": gen_best.mean_env_return,
                "running_best_label": best_so_far.label(),
                "running_best_return": best_so_far.mean_env_return,
                "elapsed_s": dt,
                "llm_spend_usd": llm.cost_estimate_usd(),
            },
        )

        # Build the reflection block for next iteration's prompt
        reflection = build_reflection_block(
            best_so_far.source, best_so_far.mean_env_return, best_stats or []
        )

    print("\n[eureka] DONE.", flush=True)
    if best_so_far is not None:
        print(
            f"  best={best_so_far.label()}  "
            f"mean_env_return={best_so_far.mean_env_return:.2f}\n"
            f"  source saved at: {POOL_DIR / run_id / (best_so_far.label() + '.py')}"
        )
    print(f"  total LLM spend ~${llm.cost_estimate_usd():.4f}")
    print(f"  log: {LOG_DIR / (run_id + '.jsonl')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=3,
                    help="Eureka outer iterations (generations).")
    ap.add_argument("--samples", type=int, default=3,
                    help="Reward candidates sampled per iteration (K).")
    ap.add_argument("--steps", type=int, default=50_000,
                    help="PPO steps per candidate (Eureka 'short' training).")
    ap.add_argument("--eval-episodes", type=int, default=5,
                    help="Env-true eval episodes per candidate.")
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="Tiny config (1 iter, 2 samples, 5k steps, 2 eval eps).")
    ap.add_argument("--no-parallel-sampling", action="store_true",
                    help="Issue LLM calls serially instead of in threads.")
    ap.add_argument("--mock-llm", action="store_true",
                    help="Bypass Anthropic; rotate through a built-in bank of "
                         "hand-written reward variants. No API key needed.")
    args = ap.parse_args()

    if args.smoke:
        args.iters = 1
        args.samples = 2
        args.steps = 5_000
        args.eval_episodes = 2

    run(
        iters=args.iters,
        samples=args.samples,
        short_steps=args.steps,
        eval_episodes=args.eval_episodes,
        model_name=args.model,
        seed=args.seed,
        parallel_sampling=not args.no_parallel_sampling,
        mock_llm=args.mock_llm,
    )


if __name__ == "__main__":
    main()
