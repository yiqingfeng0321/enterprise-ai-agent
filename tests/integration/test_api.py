import shutil
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import container_dependency
from app.api.main import app
from app.core.config import get_settings
from app.core.container import ApplicationContainer


@pytest.fixture
def client_factory(tmp_path: Path):
    def build(**overrides) -> TestClient:
        settings = get_settings()
        database_path = tmp_path / "retention.db"
        model_path = tmp_path / "churn_model.joblib"
        metadata_path = tmp_path / "churn_model_metadata.json"
        shutil.copy2(settings.database_path, database_path)
        shutil.copy2(settings.model_path, model_path)
        if settings.model_metadata_path.exists():
            shutil.copy2(settings.model_metadata_path, metadata_path)
        test_settings = replace(
            settings,
            database_path=database_path,
            model_path=model_path,
            model_metadata_path=metadata_path,
            retrieval_backend="lexical",
            retrieval_chroma_persist_path=tmp_path / "chroma",
            **overrides,
        )
        container = ApplicationContainer(test_settings)
        app.dependency_overrides[container_dependency] = lambda: container
        return TestClient(app)

    yield build
    app.dependency_overrides.clear()


@pytest.mark.slow
def test_health_and_analysis_api(client_factory) -> None:
    client = client_factory()
    health = client.get("/health")
    assert health.status_code == 200
    customer_id = client.get("/v1/customers/high-risk?limit=1").json()[0]["customer_id"]
    response = client.post(
        "/v1/analysis",
        json={"customer_id": customer_id, "session_id": "api-test", "create_ticket": False},
    )
    assert response.status_code == 200
    assert response.json()["data"]["prediction"]["customer_id"] == customer_id


def test_operations_dashboard(client_factory) -> None:
    client = client_factory()
    response = client.get("/v1/operations/dashboard?limit=20")
    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["total_customers"] > 0
    assert len(payload["customers"]) == 20
    assert (
        payload["customers"][0]["churn_probability"]
        >= payload["customers"][-1]["churn_probability"]
    )


def test_operations_evaluation_and_traces(client_factory) -> None:
    client = client_factory()
    evaluation = client.get("/v1/operations/evaluation")
    traces = client.get("/v1/operations/traces")
    registry = client.get("/v1/operations/model-registry")
    assert evaluation.status_code == 200
    assert traces.status_code == 200
    assert registry.status_code == 200
    assert "agent_evaluation" in evaluation.json()
    assert "runtime" in traces.json()
    assert "enabled" in registry.json()


def test_deterministic_analysis_api(client_factory) -> None:
    client = client_factory(agent_provider="deterministic")
    customer_id = client.get("/v1/customers/high-risk?limit=1").json()[0]["customer_id"]
    response = client.post(
        "/v1/analysis",
        json={"customer_id": customer_id, "session_id": "deterministic-api-test"},
    )
    assert response.status_code == 200
    payload = response.json()["data"]
    assert payload["provider"] == "deterministic"
    assert payload["model"] == "rules-v1"


def test_retention_chat_ranks_customers_with_tools(client_factory) -> None:
    client = client_factory(agent_provider="deterministic")
    response = client.post(
        "/v1/chat",
        json={
            "message": "帮我选择风险最高的3个客户",
            "session_id": "chat-api-test",
            "history": [],
        },
    )
    assert response.status_code == 200
    payload = response.json()["data"]
    assert payload["provider"] == "deterministic_chat"
    assert payload["tools_used"] == ["list_high_risk_customers"]
    assert "Top 3" in payload["answer"]
    assert payload["answer"].count("CUST-") == 3


def test_retention_chat_explains_discount_with_customer_evidence(client_factory) -> None:
    client = client_factory(agent_provider="deterministic")
    response = client.post(
        "/v1/chat",
        json={
            "message": "为什么02客户不能用打折的方式？",
            "session_id": "chat-discount-test",
        },
    )
    assert response.status_code == 200
    payload = response.json()["data"]
    assert "CUST-0002" in payload["answer"]
    assert "永远不能打折" in payload["answer"]
    assert "get_customer_profile" in payload["tools_used"]
    assert "predict_churn" in payload["tools_used"]
    assert "search_retention_playbooks" in payload["tools_used"]
