"""Reusable building blocks for the Eureka baseline.

- LLMClient: minimal Anthropic Messages API wrapper with cost accounting.
- compile_reward / safe_call: AST-validated compile of an LLM-generated
  `compute_reward(obs, action, next_obs, done) -> float`.
- RewardOverrideEnv: gym.Wrapper that replaces the env reward with the
  candidate while preserving the env's true reward + info components for
  fitness scoring and Eureka-style reward reflection.
- train_short_ppo / evaluate_policy: PPO short-training + env-true fitness.
"""
from __future__ import annotations

import ast
import math
import os
import re
import textwrap
import time
from dataclasses import dataclass
from typing import Callable, Optional

import gymnasium as gym
import numpy as np


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float


_PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-7": (5.00, 25.00),
}


class LLMClient:
    def __init__(
        self,
        model: str = "claude-haiku-4-5",
        temperature: float = 1.0,
        max_tokens: int = 2048,
    ) -> None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Pass --mock-llm to eureka.py for "
                "a no-key proof of concept."
            )
        from anthropic import Anthropic  # lazy: keeps mock-llm path import-free
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = Anthropic()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0

    def chat(self, system: str, user: str) -> LLMResponse:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if not self.model.startswith("claude-opus-4-7"):
            kwargs["temperature"] = self.temperature

        t0 = time.time()
        resp = self._client.messages.create(**kwargs)
        dt = time.time() - t0

        text = next((b.text for b in resp.content if b.type == "text"), "")
        in_toks = int(getattr(resp.usage, "input_tokens", 0) or 0)
        out_toks = int(getattr(resp.usage, "output_tokens", 0) or 0)
        self.total_input_tokens += in_toks
        self.total_output_tokens += out_toks
        self.total_calls += 1
        return LLMResponse(text, self.model, in_toks, out_toks, dt)

    def cost_estimate_usd(self) -> float:
        in_price, out_price = _PRICES.get(self.model, (5.0, 25.0))
        return (
            self.total_input_tokens * in_price / 1_000_000
            + self.total_output_tokens * out_price / 1_000_000
        )


# ---------------------------------------------------------------------------
# Reward compile + validate
# ---------------------------------------------------------------------------

REQUIRED_FN = "compute_reward"
ALLOWED_TOP_LEVEL_IMPORTS = {"numpy", "math", "scipy"}
CODE_BLOCK_RE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)


@dataclass
class CompiledReward:
    source: str
    fn: Callable[..., float]


class RewardCompileError(ValueError):
    pass


def extract_code(llm_output: str) -> str:
    m = CODE_BLOCK_RE.search(llm_output)
    if m:
        return textwrap.dedent(m.group(1)).strip()
    return textwrap.dedent(llm_output).strip()


def _attr_chain(node: ast.Attribute) -> str:
    parts = [node.attr]
    cur = node.value
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    return ".".join(reversed(parts))


def _validate_ast(source: str) -> None:
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise RewardCompileError(f"syntax error: {e}") from e

    if not any(isinstance(n, ast.FunctionDef) and n.name == REQUIRED_FN for n in tree.body):
        raise RewardCompileError(f"missing required function: {REQUIRED_FN}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_TOP_LEVEL_IMPORTS:
                    raise RewardCompileError(f"disallowed import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root and root not in ALLOWED_TOP_LEVEL_IMPORTS:
                raise RewardCompileError(f"disallowed import: {node.module}")
        elif isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id in {"open", "exec", "eval", "compile", "__import__"}:
                raise RewardCompileError(f"disallowed call: {fn.id}")
            if isinstance(fn, ast.Attribute):
                full = _attr_chain(fn)
                if full and full.split(".")[0] in {"os", "sys", "subprocess", "socket"}:
                    raise RewardCompileError(f"disallowed call: {full}")


def compile_reward(source: str) -> CompiledReward:
    _validate_ast(source)
    namespace: dict = {"np": np, "numpy": np, "math": math}
    try:
        import scipy  # noqa: F401
        namespace["scipy"] = scipy
    except ImportError:
        pass
    try:
        exec(compile(source, "<reward>", "exec"), namespace)
    except Exception as e:
        raise RewardCompileError(f"exec failed: {e}") from e
    fn = namespace.get(REQUIRED_FN)
    if not callable(fn):
        raise RewardCompileError(f"{REQUIRED_FN} is not callable after exec")
    return CompiledReward(source=source, fn=fn)


def safe_call(compiled: CompiledReward, obs, action, next_obs, done) -> float:
    try:
        val = float(compiled.fn(obs, action, next_obs, done))
        if not math.isfinite(val):
            return -1e6
        return float(np.clip(val, -1e4, 1e4))
    except Exception:
        return -1e6


# ---------------------------------------------------------------------------
# Env wrapper: replace env reward, keep env-true return + info components
# ---------------------------------------------------------------------------

class RewardOverrideEnv(gym.Wrapper):
    """Replace step reward with `compiled` while exposing the env-true reward
    in info['env_reward'] and the env's own info components (e.g. reward_run,
    reward_ctrl, x_velocity) verbatim for fitness reflection."""

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
        r = safe_call(
            self._compiled,
            self._last_obs,
            np.asarray(action, dtype=np.float64),
            np.asarray(next_obs, dtype=np.float64),
            done,
        )
        self._last_obs = np.asarray(next_obs, dtype=np.float64)
        info = dict(info)
        info["env_reward"] = float(env_r)
        return next_obs, r, terminated, truncated, info


# ---------------------------------------------------------------------------
# Short PPO training under candidate reward
# ---------------------------------------------------------------------------

def train_short_ppo(
    env_id: str,
    compiled: CompiledReward,
    total_steps: int,
    seed: int,
) -> object:
    """Train PPO with candidate reward; return the model."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    def make_env():
        base = gym.make(env_id)
        wrapped = RewardOverrideEnv(base, compiled)
        return Monitor(wrapped)

    vec = DummyVecEnv([make_env])
    model = PPO(
        "MlpPolicy", vec, verbose=0, device="cpu", seed=seed,
        n_steps=2048, batch_size=64, gae_lambda=0.95, gamma=0.99,
        learning_rate=3e-4, ent_coef=0.0,
    )
    model.learn(total_timesteps=total_steps)
    return model


# ---------------------------------------------------------------------------
# Env-true fitness + component collection (for Eureka reward reflection)
# ---------------------------------------------------------------------------

# Per-step info keys we track for HalfCheetah-v4 and surface in the
# reflection text. Other envs simply contribute fewer components.
_TRACKED_INFO_KEYS = (
    "x_velocity", "reward_run", "reward_ctrl", "x_position",
)


@dataclass
class EpisodeStats:
    env_return: float
    length: int
    components: dict[str, list[float]]


def rollout_episode(model, env_id: str, seed: int) -> EpisodeStats:
    env = gym.make(env_id)
    obs, _ = env.reset(seed=seed)
    components: dict[str, list[float]] = {k: [] for k in _TRACKED_INFO_KEYS}
    env_return = 0.0
    length = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action)
        env_return += float(r)
        length += 1
        for k in _TRACKED_INFO_KEYS:
            if k in info:
                components[k].append(float(info[k]))
        done = bool(term or trunc)
    env.close()
    return EpisodeStats(env_return=env_return, length=length, components=components)


def evaluate_policy(
    model, env_id: str, seeds: list[int]
) -> tuple[float, list[EpisodeStats]]:
    """Run one episode per seed; return (mean env_return, per-episode stats)."""
    stats = [rollout_episode(model, env_id, s) for s in seeds]
    mean_ret = float(np.mean([s.env_return for s in stats]))
    return mean_ret, stats


# ---------------------------------------------------------------------------
# Reward Reflection: Eureka-style textual summary of fitness components
# ---------------------------------------------------------------------------

def build_reflection_block(
    parent_source: str,
    parent_mean_env_return: float,
    parent_stats: list[EpisodeStats],
) -> str:
    """Eureka §3.3-style "Reward Reflection": surface the env's task fitness
    components as text so the LLM can see *which terms* drove env-true return
    up or down, beyond a single scalar."""
    # Aggregate components across episodes.
    agg: dict[str, dict[str, float]] = {}
    for stat in parent_stats:
        for k, vals in stat.components.items():
            if not vals:
                continue
            arr = np.asarray(vals, dtype=np.float64)
            d = agg.setdefault(k, {"mean": 0.0, "min": float("inf"), "max": float("-inf"), "n": 0})
            d["mean"] += float(arr.mean()) * len(arr)
            d["min"] = min(d["min"], float(arr.min()))
            d["max"] = max(d["max"], float(arr.max()))
            d["n"] += len(arr)
    component_lines = []
    for k, d in agg.items():
        mean = d["mean"] / max(1, d["n"])
        component_lines.append(
            f"  {k}: mean={mean:.3f}, min={d['min']:.3f}, max={d['max']:.3f}"
        )
    components_block = "\n".join(component_lines) if component_lines else "  (none collected)"
    mean_len = float(np.mean([s.length for s in parent_stats])) if parent_stats else 0.0

    return (
        "\n\n## Reflection on the previous best candidate\n\n"
        "Previous reward function:\n"
        f"```python\n{parent_source}\n```\n\n"
        f"Mean env-TRUE return across {len(parent_stats)} eval episodes: "
        f"{parent_mean_env_return:.2f}\n"
        f"Mean episode length: {mean_len:.1f} steps\n\n"
        "Per-step environment fitness components (env's own info dict, "
        "aggregated across all eval-episode steps):\n"
        f"{components_block}\n\n"
        "Use this signal to diagnose what the policy is doing under your reward.\n"
        "For HalfCheetah, high mean x_velocity is the goal; high reward_ctrl (which is\n"
        "negated control cost; less negative is better) means the agent is using\n"
        "smooth actions; episode length is fixed at 1000.\n\n"
        "Propose a NEW reward function that is meaningfully different and likely "
        "to produce HIGHER env-true return. Keep the same signature and constraints."
    )
