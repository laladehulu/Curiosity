"""Mock LLM for the no-API-key proof of concept.

Stands in for `components.LLMClient` so the entire Eureka loop — sample K
candidates → train → eval → reflect → keep best — runs without spending a
penny on Anthropic.

The "samples" are a hand-written bank of HalfCheetah reward variants
authored by Claude Code itself; they span:
- a near-default reward (forward velocity − action cost)
- a velocity-only variant (no shaping)
- an aggressively-shaped variant (velocity + uprightness + smoothness)
- a deliberately-bad variant (constant zero — should train to nothing)
- a magnitude-blown-up variant (huge coefficients — tests safe_call clamp)
- a syntactically-broken variant (tests the compile-fail path)
- a quadratic-velocity variant
- a velocity + alive-time bonus variant

Each `chat()` call rotates through the bank deterministically. This gives
the outer loop something to actually rank, so "select best" is meaningful.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _MockResponse:
    text: str
    model: str = "mock"
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0


_BANK: list[str] = [
    # 0. Near-default: forward velocity − small action cost. Should train well.
    """```python
def compute_reward(obs, action, next_obs, done):
    forward = float(next_obs[8])
    ctrl_cost = 0.1 * float((action ** 2).sum())
    return forward - ctrl_cost
```""",

    # 1. Velocity only.
    """```python
def compute_reward(obs, action, next_obs, done):
    return float(next_obs[8])
```""",

    # 2. Aggressively shaped: velocity + uprightness + smoothness.
    """```python
import numpy as np
def compute_reward(obs, action, next_obs, done):
    forward = float(next_obs[8])
    pitch = float(next_obs[1])
    uprightness = -0.5 * pitch * pitch
    ctrl_cost = 0.05 * float((action ** 2).sum())
    return forward + uprightness - ctrl_cost
```""",

    # 3. Constant zero — trains to nothing useful. Demonstrates "bad reward" handling.
    """```python
def compute_reward(obs, action, next_obs, done):
    return 0.0
```""",

    # 4. Huge coefficients — exercises safe_call's clip to [-1e4, 1e4].
    """```python
def compute_reward(obs, action, next_obs, done):
    return 1e8 * float(next_obs[8]) - 1e7 * float((action ** 2).sum())
```""",

    # 5. Syntactically broken — exercises the compile-fail path.
    "```python\ndef compute_reward(obs, action, next_obs, done)\n    return next_obs[8]\n```",

    # 6. Quadratic velocity (super-linear bonus for going fast).
    """```python
import numpy as np
def compute_reward(obs, action, next_obs, done):
    v = float(next_obs[8])
    return np.sign(v) * v * v - 0.05 * float((action ** 2).sum())
```""",

    # 7. Velocity + small constant alive-time bonus + control cost.
    """```python
def compute_reward(obs, action, next_obs, done):
    return float(next_obs[8]) + 0.01 - 0.08 * float((action ** 2).sum())
```""",
]


class MockLLMClient:
    """Drop-in replacement for `components.LLMClient`.

    Returns reward sources from `_BANK` in deterministic rotating order
    starting from an offset that depends on whether the user message
    contains a "Reflection" block — so iteration 0 starts at offset 0,
    later iterations at offset (banks_used_so_far) % len(_BANK). This
    crudely simulates "the LLM is reacting to the reflection."
    """

    def __init__(self, *_, **__) -> None:
        self._calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def chat(self, system: str, user: str) -> _MockResponse:  # noqa: ARG002
        src = _BANK[self._calls % len(_BANK)]
        self._calls += 1
        return _MockResponse(text=src)

    def cost_estimate_usd(self) -> float:
        return 0.0
