"""Retrieval-Augmented Generation pipeline."""
from __future__ import annotations

import os
from typing import List

from .store import Chunk, VectorStore

_PROMPT = (
    "Answer the question using ONLY the context below. If the answer is not in "
    "the context, say you don't know.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
)


class RAGAssistant:
    def __init__(self) -> None:
        self.store = VectorStore()
        self._api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    def ingest(self, text: str, source: str) -> int:
        return self.store.add_document(text, source)

    def answer(self, question: str, k: int = 3) -> dict:
        hits: List[Chunk] = self.store.search(question, k=k)
        context = "\n---\n".join(c.text for c in hits)
        sources = sorted({c.source for c in hits})

        if self._api_key:
            try:
                answer = self._generate(question, context)
            except Exception:  # pragma: no cover
                answer = self._extractive(question, hits)
        else:
            answer = self._extractive(question, hits)

        return {"answer": answer, "sources": sources, "chunks_used": len(hits)}

    def _generate(self, question: str, context: str) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self._api_key)
        resp = client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[{"role": "user", "content": _PROMPT.format(context=context, question=question)}],
        )
        return resp.choices[0].message.content.strip()

    @staticmethod
    def _extractive(question: str, hits: List[Chunk]) -> str:
        """Offline fallback: return the most relevant retrieved passage."""
        if not hits:
            return "I don't know — no documents have been ingested yet."
        return f"(offline mode) Most relevant passage:\n{hits[0].text}"
