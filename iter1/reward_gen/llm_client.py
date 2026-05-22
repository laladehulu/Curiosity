"""Minimal Anthropic wrapper. Single Messages-API call, returns text + usage.

Accepts either a plain user string or a list of content blocks; use the list
form with `cache_control: {"type": "ephemeral"}` on the stable prefix to enable
prompt caching. Note: per shared/prompt-caching, the minimum cacheable prefix
is 4096 tokens on haiku-4-5 / opus-4-7 — our reward-design prompts are well
below that, so the markers will most often silently no-op (no cost penalty).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

from anthropic import Anthropic


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    latency_s: float


# USD per 1M tokens, (input, output). April 2026 list pricing.
_PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
}


class LLMClient:
    """Thread-unsafe minimal client. One per process is fine for Phase 1 budgets."""

    def __init__(
        self,
        model: str = "claude-haiku-4-5",
        temperature: float = 0.8,
        max_tokens: int = 2048,
    ) -> None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = Anthropic()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cache_read_tokens = 0
        self.total_cache_write_tokens = 0
        self.total_calls = 0

    def chat(self, system: str, user: list[dict] | str) -> LLMResponse:
        """Call Claude. `user` is either a plain string (no caching) or a list
        of content blocks. Use the list form with `cache_control` on stable
        prefix blocks to enable prompt caching when prefixes are large enough.
        """
        if isinstance(user, str):
            user_blocks: list[dict] = [{"type": "text", "text": user}]
        else:
            user_blocks = user

        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user_blocks}],
        }
        # Opus 4.7 removed sampling params (returns 400 if sent); other models still accept temperature.
        if not self.model.startswith("claude-opus-4-7"):
            kwargs["temperature"] = self.temperature

        t0 = time.time()
        resp = self._client.messages.create(**kwargs)
        dt = time.time() - t0

        text = next((b.text for b in resp.content if b.type == "text"), "")
        u = resp.usage
        in_toks = int(getattr(u, "input_tokens", 0) or 0)
        out_toks = int(getattr(u, "output_tokens", 0) or 0)
        cache_read = int(getattr(u, "cache_read_input_tokens", 0) or 0)
        cache_write = int(getattr(u, "cache_creation_input_tokens", 0) or 0)
        self.total_input_tokens += in_toks
        self.total_output_tokens += out_toks
        self.total_cache_read_tokens += cache_read
        self.total_cache_write_tokens += cache_write
        self.total_calls += 1

        return LLMResponse(
            text=text,
            model=self.model,
            input_tokens=in_toks,
            output_tokens=out_toks,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            latency_s=dt,
        )

    def cost_estimate_usd(self) -> float:
        """Approximate USD spend so far. Cache reads ~0.1×, cache writes ~1.25× of input price."""
        in_price, out_price = _PRICES.get(self.model, (5.00, 25.00))
        return (
            self.total_input_tokens * in_price / 1_000_000
            + self.total_output_tokens * out_price / 1_000_000
            + self.total_cache_read_tokens * (in_price * 0.1) / 1_000_000
            + self.total_cache_write_tokens * (in_price * 1.25) / 1_000_000
        )
