from pathlib import Path

from app.services.knowledge_service import KnowledgeService


def test_lexical_retrieval_status_is_healthy() -> None:
    service = KnowledgeService(Path("data/knowledge"), backend="lexical")

    assert service.status == {
        "requested_backend": "lexical",
        "active_backend": "tfidf",
        "degraded": False,
        "error": None,
    }


def test_incomplete_chroma_configuration_reports_degradation() -> None:
    service = KnowledgeService(
        Path("data/knowledge"),
        backend="chroma",
        embedding_provider="azure_openai",
    )

    assert service.status["active_backend"] == "tfidf"
    assert service.status["degraded"] is True
    assert "incomplete" in str(service.status["error"])
