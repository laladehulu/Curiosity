"""Phase 3 — reward-space distances between candidate reward code.

Three metrics per protocol §"Reward-space distance":
1. AST tree edit distance (zss)
2. Embedding distance (mean-pooled sentence-transformer over code text)
3. Behavioral distance of the reward (correlation across 1000 random-policy trajectories)

All three are returned in a dataclass. Caller picks how to summarize.
"""
from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

# Lazy import — sentence-transformers is heavy.
_ST_MODEL = None


def _ast_to_zss_tree(source: str):
    """Convert Python AST to a zss-compatible tree (node label = ast class name)."""
    import zss
    py_tree = ast.parse(source)

    def build(n):
        node = zss.Node(type(n).__name__)
        for c in ast.iter_child_nodes(n):
            node.addkid(build(c))
        return node

    return build(py_tree)


def ast_distance(src_a: str, src_b: str) -> float:
    """zss tree edit distance with unit insert/remove/update costs."""
    import zss
    a = _ast_to_zss_tree(src_a)
    b = _ast_to_zss_tree(src_b)
    return float(zss.simple_distance(a, b))


def embedding_distance(src_a: str, src_b: str) -> float:
    """Cosine distance of sentence-transformer embeddings (mean-pooled)."""
    global _ST_MODEL
    if _ST_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    embs = _ST_MODEL.encode([src_a, src_b], normalize_embeddings=True)
    cos_sim = float(np.dot(embs[0], embs[1]))
    return float(1.0 - cos_sim)


def behavioral_distance(
    eval_a: Callable[[np.ndarray, np.ndarray, np.ndarray, bool], float],
    eval_b: Callable[[np.ndarray, np.ndarray, np.ndarray, bool], float],
    trajectories: Sequence[dict],
) -> float:
    """1 - |Pearson correlation| of per-step reward across a fixed trajectory bank."""
    rewards_a, rewards_b = [], []
    for traj in trajectories:
        for t in range(len(traj["action"])):
            obs = traj["obs"][t]
            act = traj["action"][t]
            nxt = traj["next_obs"][t]
            done = bool(t == len(traj["action"]) - 1)
            try:
                rewards_a.append(float(eval_a(obs, act, nxt, done)))
            except Exception:
                rewards_a.append(0.0)
            try:
                rewards_b.append(float(eval_b(obs, act, nxt, done)))
            except Exception:
                rewards_b.append(0.0)
    a = np.array(rewards_a)
    b = np.array(rewards_b)
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 1.0  # one of them is constant — treat as fully different
    corr = float(np.corrcoef(a, b)[0, 1])
    return float(1.0 - abs(corr))


@dataclass
class PairDistance:
    idx_a: int
    idx_b: int
    ast: float
    emb: float
    behav: float
