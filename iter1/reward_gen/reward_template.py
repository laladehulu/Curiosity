"""Compile, validate, and execute LLM-generated reward code.

Signature contract (must match prompts/*.txt):
    def compute_reward(obs, action, next_obs, done) -> float

Returned value must be a finite scalar. We do NOT sandbox imports here — the
LLM is instructed to use only numpy/scipy and is checked. Workshop scope.
"""
from __future__ import annotations

import ast
import math
import re
import textwrap
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

REQUIRED_FN = "compute_reward"
ALLOWED_TOP_LEVEL_IMPORTS = {"numpy", "math", "scipy"}

CODE_BLOCK_RE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)


@dataclass
class CompiledReward:
    source: str
    fn: Callable[..., float]
    name: str = REQUIRED_FN


class RewardCompileError(ValueError):
    pass


def extract_code(llm_output: str) -> str:
    """Pull the first ```python fenced block; if none, treat the whole output as code."""
    m = CODE_BLOCK_RE.search(llm_output)
    if m:
        return textwrap.dedent(m.group(1)).strip()
    return textwrap.dedent(llm_output).strip()


def _validate_ast(source: str) -> None:
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise RewardCompileError(f"syntax error: {e}") from e

    # Must define compute_reward at top level
    fn_defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == REQUIRED_FN]
    if not fn_defs:
        raise RewardCompileError(f"missing required function: {REQUIRED_FN}")

    # Restrict imports — block file IO, subprocess, network
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


def _attr_chain(node: ast.Attribute) -> str:
    parts = [node.attr]
    cur = node.value
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    return ".".join(reversed(parts))


def compile_reward(source: str) -> CompiledReward:
    """Validate and compile the source into a callable. Raises RewardCompileError on any issue."""
    _validate_ast(source)
    # Restricted but not sandboxed exec; protocol scope = workshop pilot, not adversarial.
    namespace: dict = {"np": np, "numpy": np, "math": math}
    try:
        exec(compile(source, "<reward>", "exec"), namespace)
    except Exception as e:
        raise RewardCompileError(f"exec failed: {e}") from e
    fn = namespace.get(REQUIRED_FN)
    if not callable(fn):
        raise RewardCompileError(f"{REQUIRED_FN} is not callable after exec")
    return CompiledReward(source=source, fn=fn)


def safe_call(compiled: CompiledReward, obs, action, next_obs, done) -> float:
    """Call the compiled reward; clamp/replace NaN, inf, and exception-throwing returns."""
    try:
        val = compiled.fn(obs, action, next_obs, done)
        val = float(val)
        if not math.isfinite(val):
            return -1e6
        # Clip extreme magnitudes — LLM rewards sometimes return huge values that break PPO.
        return float(np.clip(val, -1e4, 1e4))
    except Exception:
        return -1e6
