"""Shared helpers for gepa_pareto.py and gepa_archive.py.

Not in protocol §"Repository layout" — this is an internal organization file to
keep gepa_pareto.py and gepa_archive.py thin. Both loops share:
- TaskSpec (env id, fixed initial states, success criterion, prompt path)
- Short-PPO training with a candidate reward
- Per-initial-state return evaluation
- Reflective-mutation user prompt construction
- Reward pool persistence
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import gymnasium as gym
import numpy as np

# Local imports
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reward_gen.llm_client import LLMClient  # noqa: E402
from reward_gen.reward_template import (  # noqa: E402
    CompiledReward,
    RewardCompileError,
    compile_reward,
    extract_code,
    safe_call,
)

# Per-task config -----------------------------------------------------------

@dataclass
class TaskSpec:
    name: str            # "pendulum" | "mountaincar" | "bipedalwalker"
    env_id: str          # gymnasium id
    n_init_states: int   # number of fixed init states for Pareto axes
    short_steps: int     # PPO steps for Phase 1 short training
    success_threshold: float  # task-defined return threshold for "success"
    prompt_path: str     # path under prompts/

    def prompt(self) -> str:
        return (HERE / "prompts" / Path(self.prompt_path).name).read_text()


TASKS: dict[str, TaskSpec] = {
    "pendulum": TaskSpec(
        name="pendulum",
        env_id="Pendulum-v1",
        n_init_states=20,
        short_steps=50_000,
        success_threshold=-200.0,  # default reward; raw env return
        prompt_path="pendulum.txt",
    ),
    "mountaincar": TaskSpec(
        name="mountaincar",
        env_id="MountainCarContinuous-v0",
        n_init_states=20,
        short_steps=50_000,
        success_threshold=90.0,  # +100 on goal minus action cost
        prompt_path="mountaincar.txt",
    ),
    "bipedalwalker": TaskSpec(
        name="bipedalwalker",
        env_id="BipedalWalker-v3",
        n_init_states=20,
        short_steps=50_000,
        success_threshold=100.0,  # forward progress; not full solve
        prompt_path="bipedalwalker.txt",
    ),
}


# Reward-injecting env wrapper ----------------------------------------------

class RewardOverrideEnv(gym.Wrapper):
    """Replaces the env reward with a candidate compute_reward(obs, action, next_obs, done)."""

    def __init__(self, env: gym.Env, compiled: CompiledReward) -> None:
        super().__init__(env)
        self._compiled = compiled
        self._last_obs: Optional[np.ndarray] = None

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self._last_obs = np.asarray(obs, dtype=np.float64)
        return obs, info

    def step(self, action):
        next_obs, env_r, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)
        try:
            r = safe_call(
                self._compiled,
                self._last_obs,
                np.asarray(action, dtype=np.float64),
                np.asarray(next_obs, dtype=np.float64),
                done,
            )
        except Exception:
            r = -1e6
        self._last_obs = np.asarray(next_obs, dtype=np.float64)
        info = dict(info)
        info["env_reward"] = float(env_r)
        return next_obs, r, terminated, truncated, info


# Fixed initial states ------------------------------------------------------

def sample_fixed_init_states(env_id: str, n: int, seed: int = 0) -> list[int]:
    """Return n distinct integer seeds. The seeds reproduce initial states via env.reset(seed=...)."""
    rng = np.random.default_rng(seed)
    return [int(s) for s in rng.integers(0, 2**31 - 1, size=n)]


def evaluate_per_init_state(model, env_id: str, init_seeds: list[int], use_env_reward: bool = True,
                            compiled: Optional[CompiledReward] = None) -> tuple[np.ndarray, float]:
    """Run one episode from each init seed. Returns (per-seed-return vector, success_rate).

    use_env_reward=True scores by raw env return (this is the *fitness* axis, separate
    from the LLM reward, per Eureka §"Fitness function").
    """
    returns = np.zeros(len(init_seeds), dtype=np.float64)
    successes = 0
    for i, s in enumerate(init_seeds):
        base = gym.make(env_id)
        env = base if use_env_reward else RewardOverrideEnv(base, compiled)
        obs, _ = env.reset(seed=int(s))
        total = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            r_for_fitness = info.get("env_reward", r) if not use_env_reward else r
            total += float(r_for_fitness)
            done = bool(term or trunc)
        returns[i] = total
        env.close()
    # success defined per task by threshold lookup at the caller
    return returns, float(successes) / max(1, len(init_seeds))


# Short PPO training under a candidate reward ------------------------------

def train_short_ppo(env_id: str, compiled: CompiledReward, total_steps: int, seed: int) -> tuple[object, list[tuple[int, float]]]:
    """Train PPO for total_steps under the candidate reward. Returns (model, training_curve).

    training_curve is a list of (timestep, ep_rew_mean_under_candidate_reward).
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv

    class CurveCb(BaseCallback):
        def __init__(self):
            super().__init__()
            self.curve: list[tuple[int, float]] = []

        def _on_step(self) -> bool:
            # ep_info_buffer is populated by SB3 monitor wrapper; we use rollout collector via on_rollout_end
            return True

        def _on_rollout_end(self) -> None:
            buf = self.model.ep_info_buffer
            if buf:
                mean_r = float(np.mean([ep["r"] for ep in buf]))
                self.curve.append((int(self.num_timesteps), mean_r))

    def make_env():
        from stable_baselines3.common.monitor import Monitor
        base = gym.make(env_id)
        wrapped = RewardOverrideEnv(base, compiled)
        return Monitor(wrapped)

    vec = DummyVecEnv([make_env])
    model = PPO("MlpPolicy", vec, verbose=0, device="cpu", seed=seed, n_steps=512)
    cb = CurveCb()
    model.learn(total_timesteps=total_steps, callback=cb)
    return model, cb.curve


# Reflective mutation prompt -----------------------------------------------

def build_mutation_suffix(
    parent_source: str,
    parent_curve: list[tuple[int, float]],
    parent_fitness_summary: str,
    method_hint: str,
) -> str:
    """Volatile part of the mutation user message — everything that differs per
    call. The stable task prompt is sent as a separate, cacheable content block;
    see build_user_blocks() below."""
    curve_lines = "\n".join(f"  step={s}: ep_rew_mean={r:.3f}" for s, r in parent_curve[-8:])
    return (
        "\n\n## Reflection on the previous candidate\n\n"
        "Previous reward function:\n"
        f"```python\n{parent_source}\n```\n\n"
        "Recent training curve (last 8 logged points, ep_rew_mean is mean of episode "
        "returns under THIS candidate reward, not the env's true reward):\n"
        f"{curve_lines}\n\n"
        f"Fitness summary (using the env's TRUE return as ground truth, not your reward):\n"
        f"{parent_fitness_summary}\n\n"
        f"Method-specific selection note: {method_hint}\n\n"
        "Propose a NEW reward function that is meaningfully different from the previous one "
        "and likely to produce higher env-true return. Keep the same signature and constraints."
    )


def build_user_blocks(task_prompt: str, volatile_suffix: str) -> list[dict]:
    """Two content blocks: stable task prompt (cache-marked) + volatile per-call suffix.

    The cache marker is harmless if the combined prefix is below the model's
    cacheable minimum (4096 tokens on haiku-4-5 / opus-4-7); Anthropic silently
    skips the cache write. If reflection contexts grow, caching will start firing
    automatically without code changes.
    """
    return [
        {"type": "text", "text": task_prompt, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": volatile_suffix},
    ]


# Persistence ---------------------------------------------------------------

@dataclass
class Candidate:
    idx: int
    source: str
    parent_idx: Optional[int]
    train_seed: int
    curve: list[tuple[int, float]] = field(default_factory=list)
    fitness_vec: list[float] = field(default_factory=list)  # per init state
    mean_fitness: float = 0.0
    descriptor: Optional[list[float]] = None
    archive_cell: Optional[list[int]] = None
    failed_compile: bool = False
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "idx": self.idx,
            "parent_idx": self.parent_idx,
            "train_seed": self.train_seed,
            "curve": self.curve,
            "fitness_vec": self.fitness_vec,
            "mean_fitness": self.mean_fitness,
            "descriptor": self.descriptor,
            "archive_cell": self.archive_cell,
            "failed_compile": self.failed_compile,
            "error": self.error,
        }


def pool_dir(task: str, method: str) -> Path:
    d = ROOT / "reward_gen" / "pool" / f"{task}_{method}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_candidate(task: str, method: str, c: Candidate) -> None:
    d = pool_dir(task, method)
    (d / f"reward_{c.idx:03d}.py").write_text(c.source if c.source else "# compile failed\n")
    (d / f"reward_{c.idx:03d}.json").write_text(json.dumps(c.to_dict(), indent=2))


def log_event(task: str, method: str, event: dict) -> None:
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    line = json.dumps({"ts": time.time(), "task": task, "method": method, **event})
    with (logs / f"{task}_{method}.jsonl").open("a") as f:
        f.write(line + "\n")
