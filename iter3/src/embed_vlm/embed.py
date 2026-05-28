"""Sentence-transformers embedder wrapped for ChromaDB."""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Sequence

EMBED_MODEL = os.environ.get("ITER2_EMBED_MODEL", "all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def get_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)


def embed(texts: Sequence[str]):
    model = get_model()
    arr = model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
    return arr.tolist()


class ChromaEmbeddingFunction:
    """Chroma's EmbeddingFunction protocol: callable taking list[str] -> list[list[float]].

    Chroma >= 1.x calls ``embed_query`` / ``embed_documents`` for queries vs.
    inserts. Sentence-transformers doesn't differentiate so both route to the
    same path.
    """

    def __init__(self, model_name: str | None = None):
        global EMBED_MODEL
        if model_name:
            EMBED_MODEL = model_name
            get_model.cache_clear()

    def __call__(self, input):  # noqa: A002 — Chroma's signature
        return embed(input)

    def embed_query(self, input):  # noqa: A002
        return embed(input)

    def embed_documents(self, input):  # noqa: A002
        return embed(input)

    def name(self) -> str:
        return f"sentence-transformers/{EMBED_MODEL}"
