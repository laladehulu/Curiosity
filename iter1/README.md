# iter1 — GEPA-Pareto vs Behavior-Archive ablation

Workshop-scope pilot ablation. Mac CPU only. See `protocol.md` for the full
protocol this code implements.

## Status

Code scaffold only. **No experiments have been run.** Files under
`results.md`, `critique.md`, `figures/`, `tables/`, `logs/`, and
`reward_gen/pool/` are placeholders or empty until experiments execute.

## Quick start

```bash
# 1. Install dependencies (Mac M-series CPU)
bash env/setup.sh

# 2. Verify environment — imports, env smoke tests, Anthropic key, PPO smoke
export ANTHROPIC_API_KEY=sk-ant-...
python env/verify_env.py

# 3. If verification passes, run pre-flight on Pendulum (Protocol Phase 1 check)
python -m reward_gen.gepa_pareto --task pendulum --iters 10 --preflight
python -m reward_gen.gepa_archive --task pendulum --iters 10 --preflight

# 4. Full Phase 1 (per protocol budgets — ~12hr wall-clock)
python -m reward_gen.gepa_pareto --task pendulum --iters 50
# ... see protocol.md §"Phase 1" for the full sequence
```

## Layout

Matches `protocol.md` §"Repository layout" exactly, rooted at this directory
instead of `gepa_archive_ablation/`.

## What Claude Code built vs. did not

Claude Code produced **scaffolding code only** in the turn that created this
directory:

| Component | Status |
|---|---|
| Directory structure | done |
| `protocol.md` (copy of `iter1.md`) | done |
| `env/setup.sh`, `env/requirements.txt`, `env/verify_env.py` | done, runnable |
| `reward_gen/llm_client.py`, `reward_template.py`, prompts/* | done |
| `reward_gen/gepa_pareto.py`, `gepa_archive.py` | done, **untested** — needs preflight run to validate |
| `training/train_ppo.py`, `ppo_configs.py` | done |
| `descriptors/*` | done |
| `diversity/*`, `robustness/*`, `analysis/*` | done |
| Phase 1 generation runs | **not started** |
| Phase 2 full PPO training | **not started** |
| Phase 3-5 measurement & analysis | **not started** |
| `results.md`, `critique.md` final contents | **placeholders only** |

## Honesty notes

- The GEPA library at https://github.com/gepa-ai/gepa is designed for
  *prompt* optimization. The protocol asks us to adapt it for reward search
  by "treating reward code as the prompt being optimized." Rather than
  fragile dependence on private GEPA internals, `gepa_pareto.py` implements
  the Pareto-frontier + reflective-mutation loop self-contained, following
  the same conceptual recipe. If reviewers ask "is this really GEPA?", the
  honest answer is "it's GEPA's selection and mutation principles, applied
  to reward code; we did not use the GEPA library directly because its
  prompt-optimization adapter does not fit reward-function search without
  custom subclassing."
- All hyperparameters (PPO configs, archive cell counts, iteration budgets)
  follow `protocol.md`. They are frozen there for a reason — do not tune.
- **LLM swapped from OpenAI to Anthropic** per user request (2026-05-21).
  Default model is `claude-haiku-4-5` (testing slot; ~$1/$5 per 1M tokens),
  with `claude-opus-4-7` available for finals via `--model`. The protocol's
  $50 total project spend cap still applies. Requires `ANTHROPIC_API_KEY`,
  not `OPENAI_API_KEY`. `verify_env.py` spends a few cents on its smoke test.
