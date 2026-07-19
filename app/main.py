"""FastAPI + CLI entrypoint for the RAG Knowledge Assistant."""
from __future__ import annotations

import sys

from fastapi import FastAPI
from pydantic import BaseModel

from .rag import RAGAssistant

app = FastAPI(title="RAG Knowledge Assistant", version="1.0.0")
assistant = RAGAssistant()


class IngestRequest(BaseModel):
    text: str
    source: str = "manual"


class AskRequest(BaseModel):
    question: str
    k: int = 3


@app.post("/ingest")
def ingest(req: IngestRequest) -> dict:
    n = assistant.ingest(req.text, req.source)
    return {"chunks_added": n, "total_chunks": len(assistant.store.chunks)}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    return assistant.answer(req.question, k=req.k)


def _cli() -> None:
    """Minimal terminal demo: python -m app.main path/to/file.txt"""
    if len(sys.argv) < 2:
        print("Usage: python -m app.main <file.txt>")
        return
    with open(sys.argv[1], "r", encoding="utf-8") as fh:
        assistant.ingest(fh.read(), source=sys.argv[1])
    print("Document ingested. Ask questions (Ctrl-C to quit).")
    try:
        while True:
            q = input("\n> ")
            result = assistant.answer(q)
            print(f"\n{result['answer']}\nSources: {result['sources']}")
    except (KeyboardInterrupt, EOFError):
        print("\nBye.")


if __name__ == "__main__":
    _cli()
