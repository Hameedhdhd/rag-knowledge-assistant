"""A tiny, dependency-light vector store.

Uses OpenAI embeddings when a key is available, otherwise a deterministic
hashing-based embedding so the pipeline is fully runnable offline. Similarity
search is plain cosine similarity over NumPy arrays — small enough to read,
big enough to show the RAG pattern end to end.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from typing import List

import numpy as np

_DIM = 512


@dataclass
class Chunk:
    text: str
    source: str
    embedding: np.ndarray = field(default=None, repr=False)


def _normalize(vec: np.ndarray) -> np.ndarray:
    """Scale a vector to unit length so a dot product equals cosine similarity."""
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


def _hash_embed(text: str, dim: int = _DIM) -> np.ndarray:
    """Deterministic bag-of-words hashing embedding (offline fallback)."""
    vec = np.zeros(dim, dtype=np.float32)
    for token in re.findall(r"\w+", text.lower()):
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    return _normalize(vec)


class VectorStore:
    def __init__(self) -> None:
        self.chunks: List[Chunk] = []
        self._api_key = os.getenv("OPENAI_API_KEY")

    # -- embedding ------------------------------------------------------
    def embed(self, text: str) -> np.ndarray:
        if self._api_key:
            try:
                from openai import OpenAI

                client = OpenAI(api_key=self._api_key)
                resp = client.embeddings.create(
                    model=os.getenv("EMBED_MODEL", "text-embedding-3-small"),
                    input=text[:8000],
                )
                return _normalize(np.array(resp.data[0].embedding, dtype=np.float32))
            except Exception:  # pragma: no cover
                pass
        return _hash_embed(text)

    # -- ingestion ------------------------------------------------------
    def add_document(self, text: str, source: str, chunk_size: int = 500) -> int:
        added = 0
        for chunk in self._split(text, chunk_size):
            self.chunks.append(Chunk(text=chunk, source=source, embedding=self.embed(chunk)))
            added += 1
        return added

    @staticmethod
    def _split(text: str, chunk_size: int) -> List[str]:
        words, chunks, current = text.split(), [], []
        for word in words:
            current.append(word)
            if len(" ".join(current)) >= chunk_size:
                chunks.append(" ".join(current))
                current = []
        if current:
            chunks.append(" ".join(current))
        return chunks

    # -- retrieval ------------------------------------------------------
    def search(self, query: str, k: int = 3) -> List[Chunk]:
        if not self.chunks:
            return []
        q = self.embed(query)
        scored = sorted(
            self.chunks,
            key=lambda c: float(np.dot(q, c.embedding)),
            reverse=True,
        )
        return scored[:k]
