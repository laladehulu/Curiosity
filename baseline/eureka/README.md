# Eureka baseline — HalfCheetah-v4 on Mac CPU

A workshop-pilot-scale replication of Eureka (Ma et al., 2023, "Eureka:
Human-Level Reward Design via Coding Large Language Models") on a single
easy-to-train locomotion task that runs on M-series Apple Silicon CPU.

## Why HalfCheetah-v4?

- **Locomotion** — the agent learns to run forward, the canonical setting
  Eureka targets.
- **Easy to train** on CPU: no early termination, dense env reward, PPO
  reaches non-trivial behavior in ~50k steps (~3-5 min on M-series CPU).
- **Rich info dict** — `x_velocity`, `reward_run`, `reward_ctrl` come back
  per step, so Eureka's "Reward Reflection" has real components to surface.

## What this is (and isn't)

This **is** a faithful pilot replication of Eureka's outer loop:

1. Sample `K` candidate `compute_reward` functions from an LLM (Claude).
2. Train PPO under each candidate for `short_steps`.
3. Evaluate each policy under the *env-true* reward (the fitness function,
   distinct from the LLM's reward — Eureka §3.1).
4. Keep the best. Build a **Reward Reflection** (Eureka §3.3) — per-step
   env fitness components aggregated across eval episodes — and prepend it
   to the prompt for the next iteration's `K` samples.

This **is not** the original Isaac Gym Eureka. Differences:

| | Original Eureka | This baseline |
|---|---|---|
| Sim | Isaac Gym, GPU, 4096 parallel envs | Gymnasium HalfCheetah-v4, CPU, 1 env |
| Steps / candidate | ~3e8 | 5e4 (smoke: 5e3) |
| `K` per iter | 16 | 3 default |
| Iterations | 5 | 3 default |
| Env in prompt | Full env source code | Hand-written obs/action description |
| Backend LLM | GPT-4 | Claude Haiku 4.5 / Opus 4.7 |

These reductions are intentional — the goal is a clean, runnable reference
implementation of the *algorithm*, not the original's compute scale.

## Quick start

```bash
cd baseline/eureka
bash setup.sh
export ANTHROPIC_API_KEY=sk-ant-...
python verify_env.py              # ~30s, spends a few cents
python eureka.py --smoke          # 1 iter, 2 samples, 5k steps, ~3 min
python eureka.py                  # 3 iters x 3 samples x 50k steps, ~30-45 min
```

Run with a different model:

```bash
python eureka.py --model claude-opus-4-7 --iters 5 --samples 4
```

## Layout

```
baseline/eureka/
├── README.md
├── requirements.txt
├── setup.sh
├── verify_env.py        smoke tests
├── eureka.py            main outer loop (CLI entry)
├── components.py        LLM client, reward compile, env wrapper, PPO, reflection
├── prompts/halfcheetah.txt
├── pool/<run_id>/       saved candidates (source + json metadata)
└── logs/<run_id>.jsonl  one line per candidate + per-generation summary
```

## Output

Each candidate is written to `pool/<run_id>/gen{NN}_s{i}.py` (the LLM
source) and `.json` (eval returns, train seed, parent gen). The final
`[eureka] DONE.` line points to the best one. `logs/<run_id>.jsonl` is
machine-readable progress.

## Cost notes

`claude-haiku-4-5` at default `K=3, iters=3`: ~9 LLM calls × ~1.5k input
tokens × ~600 output tokens ≈ **< $0.05 per run**. Opus 4.7 is ~5× more.
The dominant cost is wall-clock PPO time, not LLM tokens.

## Honest caveats

- 50k PPO steps is **far below convergence** on HalfCheetah (which typically
  needs ~1M+ for strong forward velocity). The Eureka loop here is fair
  *across reward candidates* at this budget, but the absolute returns are
  modest. Increase `--steps` for stronger results.
- Single-seed eval per candidate. The original Eureka averages many.
- LLM sampling is K independent API calls at `temperature=1.0` — Anthropic
  doesn't return multiple completions per call.
- AST validation blocks file IO / subprocess / network but is not
  adversarial-grade. The reward is `exec()`-ed in-process.
