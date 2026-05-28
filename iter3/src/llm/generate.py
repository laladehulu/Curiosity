"""Call Claude to cold-generate or mutate a reward function."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

from src.llm.prompts import (
    COLD_GEN,
    MUTATE_PARENT_ONLY,
    MUTATE_WITH_NEIGHBORS,
    SYSTEM,
    format_neighbor_list,
)

GEN_MODEL = os.environ.get("ITER3_GEN_MODEL", "claude-sonnet-4-6")


def _client():
    import anthropic
    return anthropic.Anthropic()


_FENCED_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
_OPEN_FENCE_RE = re.compile(r"```(?:python)?\s*\n", re.DOTALL)


def _extract_code(text: str) -> str:
    """Strip ```python fences. Handle truncated responses where the closing
    fence is missing (the LLM hit max_tokens mid-code-block)."""
    m = _FENCED_RE.search(text)
    if m:
        return m.group(1).strip()
    # truncated mid-block: take everything after the first opening fence
    m = _OPEN_FENCE_RE.search(text)
    if m:
        return text[m.end():].strip()
    return text.strip()


def _call(prompt: str, max_tokens: int = 2000) -> str:
    client = _client()
    resp = client.messages.create(
        model=GEN_MODEL,
        max_tokens=max_tokens,
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    return _extract_code(text)


@dataclass
class GenResult:
    code: str
    prompt_kind: str       # "cold" | "parent_only" | "with_neighbors"
    parent_id: str | None


def cold_generate() -> GenResult:
    code = _call(COLD_GEN)
    return GenResult(code=code, prompt_kind="cold", parent_id=None)


def mutate(
    parent_code: str,
    parent_desc: str,
    parent_fit: float,
    parent_id: str,
    neighbors: list[tuple[str, float, str]] | None = None,
) -> GenResult:
    """If `neighbors` provided (and non-empty), include them in the prompt for
    divergence pressure. Otherwise mutate from parent alone."""
    if neighbors:
        prompt = MUTATE_WITH_NEIGHBORS.format(
            parent_fit=parent_fit,
            parent_desc=parent_desc,
            parent_code=parent_code,
            neighbor_list=format_neighbor_list(neighbors),
        )
        kind = "with_neighbors"
    else:
        prompt = MUTATE_PARENT_ONLY.format(
            parent_fit=parent_fit,
            parent_desc=parent_desc,
            parent_code=parent_code,
        )
        kind = "parent_only"
    code = _call(prompt)
    return GenResult(code=code, prompt_kind=kind, parent_id=parent_id)
