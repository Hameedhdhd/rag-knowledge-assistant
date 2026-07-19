"""API-level tests using FastAPI's TestClient (offline, no API key required)."""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_ingest_then_ask():
    ingest = client.post(
        "/ingest",
        json={"text": "Hamburg is a major European tech hub in Germany.", "source": "doc1"},
    )
    assert ingest.status_code == 200
    assert ingest.json()["chunks_added"] >= 1

    ask = client.post("/ask", json={"question": "Where is Hamburg?"})
    assert ask.status_code == 200
    body = ask.json()
    assert body["chunks_used"] >= 1
    assert "doc1" in body["sources"]
