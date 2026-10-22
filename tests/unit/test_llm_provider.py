from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from app.agents.providers.azure_openai import AzureOpenAIRecommendationProvider
from app.agents.providers.deterministic import DeterministicRecommendationProvider
from app.agents.providers.microsoft_ollama import MicrosoftOllamaRecommendationProvider
from app.agents.recommendation_agent import RecommendationAgent
from app.agents.reviewer_agent import ReviewerAgent
from app.core.config import get_settings
from app.core.exceptions import AgentProviderError
from app.models.domain import ActionItem, CustomerProfile, PredictionResult, Recommendation


class FakeOllamaProvider:
    def __init__(self, *, success: bool = True, error: str | None = None) -> None:
        self.name = "microsoft_ollama"
        self.model_id = "qwen2.5:7b"
        self.ollama_available = success
        self.host = "http://localhost:11434"
        self.last_latency_ms = 1840 if success else None
        self.last_successful_llm_call = "2026-07-20T00:00:00Z" if success else None
        self._error = error

    async def generate(self, *args, **kwargs):
        if not self.ollama_available:
            raise AgentProviderError(self._error or "Ollama is unavailable")
        return Recommendation(
            executive_summary="summary",
            actions=[
                ActionItem(
                    priority=1,
                    action="Call the customer",
                    owner="CSM",
                    rationale="Follow up",
                    expected_impact="Reduce risk",
                    approval_required=False,
                )
            ],
            do_not_do=["Do not discount"],
            evidence=["Model risk 83%"],
            confidence=0.8,
            generated_by="microsoft_ollama",
        )


@pytest.fixture
def sample_profile() -> CustomerProfile:
    return CustomerProfile(
        customer_id="CUST-001",
        segment="growth",
        region="us-east",
        customer_notes="ROI focused; prefers executive-ready summaries.",
        licensed_seats=120,
        active_seat_ratio_30d=0.28,
        champion_count=1,
        tenure_months=18,
        monthly_revenue=12000.0,
        logins_30d=20,
        usage_change_pct=-25.0,
        support_tickets_90d=4,
        nps=20,
        days_to_renewal=45,
        payment_failures_90d=1,
        onboarding_complete=True,
        support_ticket_rate=0.25,
        engagement_score=0.7,
        renewal_urgency=0.8,
        revenue_at_risk=8000.0,
        churned=1,
    )


@pytest.fixture
def sample_prediction() -> PredictionResult:
    return PredictionResult(
        customer_id="CUST-001",
        churn_probability=0.83,
        risk_band="high",
        top_drivers=[],
        model_version="test-v1",
    )


def test_settings_expose_ollama_runtime_options(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_PROVIDER", "microsoft_ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:7b")
    monkeypatch.setenv("ALLOW_DETERMINISTIC_FALLBACK", "false")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.2")
    monkeypatch.setenv("LLM_MAX_TOKENS", "1500")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "120")
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.agent_provider == "microsoft_ollama"
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.ollama_model == "qwen2.5:7b"
    assert settings.allow_deterministic_fallback is False
    assert settings.llm_temperature == 0.2
    assert settings.llm_max_tokens == 1500
    assert settings.llm_timeout_seconds == 120
    get_settings.cache_clear()


def test_settings_expose_azure_openai_runtime_options(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_PROVIDER", "azure_openai")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "retention-chat")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
    monkeypatch.setenv("AZURE_OPENAI_TIMEOUT_SECONDS", "90")
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.agent_provider == "azure_openai"
    assert settings.azure_openai_endpoint == "https://example.openai.azure.com"
    assert settings.azure_openai_api_key == "test-key"
    assert settings.azure_openai_deployment == "retention-chat"
    assert settings.azure_openai_api_version == "2024-06-01"
    assert settings.azure_openai_timeout_seconds == 90.0
    get_settings.cache_clear()


def test_microsoft_prompt_uses_reviewer_evidence_contract() -> None:
    provider = MicrosoftOllamaRecommendationProvider(
        "http://127.0.0.1:9",
        "qwen2.5:7b",
        None,
    )

    assert "starting with 'Model risk:'" in provider._build_prompt({})


@pytest.mark.asyncio
async def test_microsoft_provider_reports_an_actionable_timeout() -> None:
    provider = MicrosoftOllamaRecommendationProvider(
        "http://127.0.0.1:9",
        "qwen2.5:7b",
        None,
        settings=SimpleNamespace(
            llm_timeout_seconds=0.01,
            llm_temperature=0.2,
            llm_max_tokens=1500,
            skill_max_resource_bytes=100_000,
        ),
    )
    provider.ollama_available = True

    async def slow_run(*args, **kwargs):
        await asyncio.sleep(1)

    provider.agent.run = slow_run  # type: ignore[method-assign]
    provider.lean_agent.run = slow_run  # type: ignore[method-assign]

    with pytest.raises(
        AgentProviderError,
        match=r"timed out after 0\.01 seconds",
    ):
        await provider._chat("test")

    assert provider.last_error == (
        "Microsoft/Ollama agent execution timed out after 0.01 seconds"
    )


@pytest.mark.asyncio
async def test_recommendation_agent_does_not_silently_fallback_when_disabled(
    sample_profile: CustomerProfile,
    sample_prediction: PredictionResult,
) -> None:
    fallback = DeterministicRecommendationProvider(None)  # type: ignore[arg-type]
    provider = MicrosoftOllamaRecommendationProvider("http://localhost:11434", "qwen2.5:7b", None)  # type: ignore[arg-type]
    agent = RecommendationAgent(provider, fallback, allow_fallback=False)

    async def failing_generate(*args, **kwargs):
        raise AgentProviderError("ollama unavailable")

    provider.generate = failing_generate  # type: ignore[assignment]

    with pytest.raises(AgentProviderError, match="ollama unavailable"):
        await agent.run(sample_profile, sample_prediction, {}, [], [])


@pytest.mark.asyncio
async def test_recommendation_agent_reports_ollama_success_metadata(
    sample_profile: CustomerProfile,
    sample_prediction: PredictionResult,
) -> None:
    fallback = DeterministicRecommendationProvider(None)  # type: ignore[arg-type]
    provider = FakeOllamaProvider(success=True)
    agent = RecommendationAgent(provider, fallback, allow_fallback=True)

    await agent.run(sample_profile, sample_prediction, {}, [], [])

    metadata = agent.last_run_metadata
    assert metadata["provider"] == "microsoft_ollama"
    assert metadata["requested_provider"] == "microsoft_ollama"
    assert metadata["model"] == "qwen2.5:7b"
    assert metadata["generation_mode"] == "ollama"
    assert metadata["fallback_used"] is False
    assert metadata["fallback_reason"] is None
    assert metadata["llm_latency_ms"] == 1840


@pytest.mark.asyncio
async def test_recommendation_agent_reports_deterministic_fallback_metadata(
    sample_profile: CustomerProfile,
    sample_prediction: PredictionResult,
) -> None:
    fallback = DeterministicRecommendationProvider(None)  # type: ignore[arg-type]
    provider = FakeOllamaProvider(success=False, error="Ollama is unavailable at http://localhost:11434")
    agent = RecommendationAgent(provider, fallback, allow_fallback=True)

    await agent.run(sample_profile, sample_prediction, {}, [], [])

    metadata = agent.last_run_metadata
    assert metadata["provider"] == "deterministic"
    assert metadata["requested_provider"] == "microsoft_ollama"
    assert metadata["model"] == "qwen2.5:7b"
    assert metadata["generation_mode"] == "deterministic"
    assert metadata["fallback_used"] is True
    assert metadata["fallback_reason"] == "Ollama is unavailable at http://localhost:11434"
    assert metadata["llm_latency_ms"] is None


@pytest.mark.asyncio
async def test_deterministic_provider_metadata_has_string_model(
    sample_profile: CustomerProfile,
    sample_prediction: PredictionResult,
) -> None:
    provider = DeterministicRecommendationProvider(None)
    agent = RecommendationAgent(provider, provider, allow_fallback=False)

    await agent.run(sample_profile, sample_prediction, {}, [], [])

    assert agent.last_run_metadata["provider"] == "deterministic"
    assert agent.last_run_metadata["model"] == "rules-v1"
    assert agent.last_run_metadata["provider_available"] is True


@pytest.mark.asyncio
async def test_recommendation_agent_reports_azure_openai_success_metadata(
    sample_profile: CustomerProfile,
    sample_prediction: PredictionResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(AzureOpenAIRecommendationProvider, "_probe_azure_openai", lambda self: True)
    provider = AzureOpenAIRecommendationProvider(
        "https://example.openai.azure.com",
        "test-key",
        "retention-chat",
    )

    async def fake_chat(prompt: str, *, retry: bool = False) -> str:
        return (
            "{"
            '"executive_summary":"summary",'
            '"risk_assessment":"risk",'
            '"recommended_actions":['
            '{"action":"Call the customer","owner":"CSM","rationale":"Follow up","expected_impact":"Reduce risk","approval_required":false},'
            '{"action":"Escalate support","owner":"Support","rationale":"Resolve blockers","expected_impact":"Improve adoption","approval_required":true}'
            '],'
            '"rationale":"rationale",'
            '"evidence":["Model risk: churn probability 83% (high).","Evidence 2","Evidence 3"],'
            '"do_not_do":["Do not discount"],'
            '"confidence":0.82,'
            '"requires_human_approval":true'
            "}"
        )

    provider._chat = fake_chat  # type: ignore[assignment]
    fallback = DeterministicRecommendationProvider(None)  # type: ignore[arg-type]
    agent = RecommendationAgent(provider, fallback, allow_fallback=False)

    await agent.run(sample_profile, sample_prediction, {}, [], [])

    metadata = agent.last_run_metadata
    assert metadata["provider"] == "azure_openai"
    assert metadata["requested_provider"] == "azure_openai"
    assert metadata["model"] == "retention-chat"
    assert metadata["generation_mode"] == "azure_openai"
    assert metadata["provider_available"] is True
    assert metadata["fallback_used"] is False


@pytest.mark.asyncio
async def test_reviewer_rejects_unsupported_claims(
    sample_profile: CustomerProfile,
    sample_prediction: PredictionResult,
) -> None:
    reviewer = ReviewerAgent()
    recommendation = Recommendation(
        executive_summary="Summary",
        actions=[
            ActionItem(
                priority=1,
                action="Call the customer",
                owner="CSM",
                rationale="Need to engage",
                expected_impact="Reduce risk",
                approval_required=False,
            )
        ],
        do_not_do=["Do not send discounts"],
        evidence=["Customer said they are thrilled with the product"],
        confidence=0.6,
        generated_by="test",
    )
    review = await reviewer.run(sample_profile, sample_prediction, recommendation)
    assert review.passed is False
    assert any("unsupported customer claim" in issue.lower() for issue in review.issues)
