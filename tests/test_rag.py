from app.rag import RAGAssistant


def test_ingest_and_retrieve():
    a = RAGAssistant()
    a.ingest("Hamburg is a major European tech hub in Germany.", source="doc1")
    result = a.answer("Where is Hamburg?")
    assert result["chunks_used"] >= 1
    assert "doc1" in result["sources"]


def test_empty_store():
    a = RAGAssistant()
    result = a.answer("anything?")
    assert result["chunks_used"] == 0
