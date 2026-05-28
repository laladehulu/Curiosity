"""AST whitelist + safe compile for LLM-generated compute_reward.

LLM is asked to produce a Python function of the signature:

    def compute_reward(obs, action, next_obs, done):
        # ... return scalar
        return r

We sandbox by:
  - AST parse, reject unknown nodes (Import, Attribute on dunders, etc.)
  - allow-list a small set of names (np / math / basic builtins)
  - compile to a function object via exec with a constrained __builtins__
  - safe-call wrapper clamps output, swallows exceptions to a finite fallback
"""
from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np


ALLOWED_NAMES = {
    "compute_reward",
    "obs", "action", "next_obs", "done",
    "np", "math",
    "min", "max", "abs", "sum", "len", "float", "int", "bool",
    "True", "False", "None",
    "range", "enumerate", "zip",
}
ALLOWED_DUNDERS = {"__name__"}  # nothing else


class RewardCodeError(ValueError):
    pass


def validate(code: str) -> None:
    """Raise RewardCodeError if the snippet uses anything outside our whitelist."""
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        raise RewardCodeError(f"syntax error: {e}") from e

    found_fn = False

    for node in ast.walk(tree):
        # block imports outright
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise RewardCodeError("imports are not allowed")
        # block exec/eval/open/etc by name
        if isinstance(node, ast.Name) and node.id not in ALLOWED_NAMES:
            # variable bindings (local assignments) are fine — but Name nodes in
            # Load context that aren't whitelisted *and* aren't locally assigned
            # are the problem. We don't track scope, so we approximate: any
            # Name in a Load context not in the whitelist is suspect.
            if isinstance(node.ctx, ast.Load):
                # allow common locals via heuristic: short identifier from inside fn
                # but flag dangerous names explicitly
                if node.id in {"exec", "eval", "open", "compile", "globals",
                               "locals", "vars", "getattr", "setattr",
                               "delattr", "hasattr", "__import__"}:
                    raise RewardCodeError(f"name '{node.id}' is not allowed")
        # block attribute access starting with underscore (e.g. obj.__class__)
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_") and node.attr not in ALLOWED_DUNDERS:
                raise RewardCodeError(f"attribute '{node.attr}' is not allowed")
        # block lambdas with weird tricks (allow them in general though)
        # block class defs
        if isinstance(node, ast.ClassDef):
            raise RewardCodeError("class definitions are not allowed")
        if isinstance(node, ast.FunctionDef) and node.name == "compute_reward":
            found_fn = True

    if not found_fn:
        raise RewardCodeError("must define `def compute_reward(obs, action, next_obs, done)`")


def compile_reward(code: str) -> Callable:
    """Compile + return the compute_reward callable."""
    validate(code)
    # Restricted exec env
    safe_globals = {
        "__builtins__": {
            "min": min, "max": max, "abs": abs, "sum": sum, "len": len,
            "float": float, "int": int, "bool": bool, "True": True, "False": False,
            "None": None, "range": range, "enumerate": enumerate, "zip": zip,
        },
        "np": np,
        "math": math,
    }
    local_ns: dict = {}
    exec(code, safe_globals, local_ns)
    fn = local_ns.get("compute_reward")
    if not callable(fn):
        raise RewardCodeError("compute_reward not callable after compile")
    return fn


def safe_call(fn: Callable, obs, action, next_obs, done,
              clamp: tuple[float, float] = (-100.0, 100.0)) -> float:
    """Call fn defensively: catch exceptions, clamp non-finite to 0."""
    try:
        r = fn(obs, action, next_obs, done)
        r = float(r)
        if not math.isfinite(r):
            return 0.0
        lo, hi = clamp
        return max(lo, min(hi, r))
    except Exception:
        return 0.0


@dataclass
class CompiledReward:
    code: str
    fn: Callable

    @classmethod
    def from_code(cls, code: str) -> "CompiledReward":
        return cls(code=code, fn=compile_reward(code))

    def __call__(self, obs, action, next_obs, done) -> float:
        return safe_call(self.fn, obs, action, next_obs, done)
