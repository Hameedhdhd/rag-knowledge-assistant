# RAG Knowledge Assistant

[![CI](https://github.com/Hameedhdhd/rag-knowledge-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/Hameedhdhd/rag-knowledge-assistant/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Code style: ruff](https://img.shields.io/badge/lint-ruff-orange)

A compact **Retrieval-Augmented Generation** system: ingest documents, embed and
index them in a vector store, retrieve the most relevant passages, and generate
grounded answers with an LLM — cutting hallucination by answering only from
retrieved context.

Runs both as a **REST API** and an interactive **CLI**. Works fully offline with a
deterministic fallback embedder, or with OpenAI when a key is supplied.

## Why it's interesting

RAG is the core pattern behind most production LLM apps (chatbots over private
docs, internal knowledge search, support assistants). This repo implements the
full loop — chunking → embedding → cosine retrieval → grounded generation — in
readable, dependency-light Python.

## Tech stack

Python · FastAPI · NumPy · OpenAI embeddings & chat · pytest

## Quickstart (API)

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

```bash
# ingest a document
curl -X POST localhost:8000/ingest -H "Content-Type: application/json" \
  -d '{"text": "Hamburg is a European tech hub.", "source": "notes"}'

# ask a question
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "What is Hamburg?"}'
```

## Quickstart (CLI)

```bash
python -m app.main data/sample.txt
> What does the assistant do?
```

Add an `OPENAI_API_KEY` in `.env` (see `.env.example`) to enable real embeddings
and LLM generation.

## Run with Docker

```bash
docker build -t rag-knowledge-assistant .
docker run -p 8000:8000 --env-file .env rag-knowledge-assistant
```

## How it works

```
ingest → chunk → embed → store            ask → embed query → cosine retrieve top-k → ground LLM
```

Embeddings are unit-normalized, so retrieval scoring is exact cosine similarity.
With no API key, a deterministic hashing embedder and extractive answerer keep the
full pipeline runnable offline.

## Development

```bash
pip install -r requirements.txt ruff
ruff check .   # lint
pytest -q      # tests (run fully offline — no API key needed)
```

CI runs lint + tests on Python 3.10–3.12 via GitHub Actions on every push.

## Tests

```bash
pytest
```

## Author

**Hamid Abdullah** — AI / IT Engineer, Hamburg
[LinkedIn](https://www.linkedin.com/in/hamidabdullah-41b91319)
