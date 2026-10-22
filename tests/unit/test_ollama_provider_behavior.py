from __future__ import annotations

import pytest

from app.agents.providers.microsoft_ollama import MicrosoftOllamaRecommendationProvider
from app.core.exceptions import AgentProviderError
from app.models.domain import ActionItem, CustomerProfile, PredictionResult, Recommendation


@pytest.fixture
def sample_profile() -> CustomerProfile:
    return CustomerProfile(
        customer_id="CUST-001",
        segment="growth",
        region="us-east",
        customer_notes="IT-led deployment; needs stronger adoption plan.",
        licensed_seats=250,
        active_seat_ratio_30d=0.22,
        champion_count=0,
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


@pytest.mark.asyncio
async def test_structured_response_parsing(sample_profile: CustomerProfile, sample_prediction: PredictionResult) -> None:
    provider = MicrosoftOllamaRecommendationProvider("http://localhost:11434", "qwen2.5:7b", None)
    provider.ollama_available = True

    async def fake_chat(prompt: str, *, retry: bool = False) -> str:
        return '{"executive_summary": "summary", "risk_assessment": "high", "recommended_actions": ["Call the customer"], "rationale": "Need follow-up", "evidence": ["Model risk 83%"], "do_not_do": ["Do not discount"], "confidence": 0.8, "requires_human_approval": true}'

    provider._chat = fake_chat  # type: ignore[assignment]
    recommendation = await provider.generate(sample_profile, sample_prediction, {}, [], [])
    assert recommendation.executive_summary == "summary"
    assert recommendation.actions[0].action == "Call the customer"
    assert recommendation.requires_human_approval is True


@pytest.mark.asyncio
async def test_invalid_json_retries_once(sample_profile: CustomerProfile, sample_prediction: PredictionResult) -> None:
    provider = MicrosoftOllamaRecommendationProvider("http://localhost:11434", "qwen2.5:7b", None)
    provider.ollama_available = True

    calls: list[str] = []

    async def fake_chat(prompt: str, *, retry: bool = False) -> str:
        calls.append(prompt)
        if retry:
            return '{"executive_summary": "ok", "risk_assessment": "high", "recommended_actions": ["Follow up"], "rationale": "Need follow-up", "evidence": ["Model risk 83%"], "do_not_do": ["Do not discount"], "confidence": 0.8, "requires_human_approval": true}'
        return 'not-json'

    provider._chat = fake_chat  # type: ignore[assignment]
    recommendation = await provider.generate(sample_profile, sample_prediction, {}, [], [])
    assert recommendation.executive_summary == "ok"
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_confidence_percent_is_normalized(sample_profile: CustomerProfile, sample_prediction: PredictionResult) -> None:
    provider = MicrosoftOllamaRecommendationProvider("http://localhost:11434", "qwen2.5:7b", None)
    provider.ollama_available = True

    async def fake_chat(prompt: str, *, retry: bool = False) -> str:
        return '{"executive_summary": "summary", "risk_assessment": "high", "recommended_actions": ["Call the customer"], "rationale": "Need follow-up", "evidence": ["Model risk 83%"], "do_not_do": ["Do not discount"], "confidence": 98, "requires_human_approval": true}'

    provider._chat = fake_chat  # type: ignore[assignment]
    recommendation = await provider.generate(sample_profile, sample_prediction, {}, [], [])
    assert recommendation.confidence == 0.98


@pytest.mark.asyncio
async def test_unavailable_ollama_raises_clear_error(sample_profile: CustomerProfile, sample_prediction: PredictionResult) -> None:
    provider = MicrosoftOllamaRecommendationProvider("http://localhost:11434", "qwen2.5:7b", None)
    provider.ollama_available = False
    with pytest.raises(AgentProviderError, match="Ollama is unavailable"):
        await provider.generate(sample_profile, sample_prediction, {}, [], [])


@pytest.mark.asyncio
async def test_reviewer_rejects_unsupported_claims(sample_profile: CustomerProfile, sample_prediction: PredictionResult) -> None:
    from app.agents.reviewer_agent import ReviewerAgent

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
        do_not_do=["Do not discount"],
        evidence=["Customer said they are thrilled with the product"],
        confidence=0.6,
        generated_by="test",
    )
    review = await ReviewerAgent().run(sample_profile, sample_prediction, recommendation)
    assert review.passed is False
    assert any("unsupported customer claim" in issue.lower() for issue in review.issues)


def test_lean_prompt_is_compact_and_keeps_authoritative_risk(
    sample_profile: CustomerProfile, sample_prediction: PredictionResult
) -> None:
    lean = MicrosoftOllamaRecommendationProvider("http://localhost:11434", "qwen2.5:7b", None)
    lean_prompt = lean._build_prompt(lean._build_context(sample_profile, sample_prediction, {}, [], []))
    assert "Model risk: 83.0% churn probability, high risk band" in lean_prompt
    assert "segment_average_not_this_customer" in lean_prompt
    assert "churned" not in lean_prompt

    class FullSettings:
        llm_prompt_mode = "full"

    full = MicrosoftOllamaRecommendationProvider(
        "http://localhost:11434", "qwen2.5:7b", None, settings=FullSettings()
    )
    full_prompt = full._build_prompt(full._build_context(sample_profile, sample_prediction, {}, [], []))
    assert full.prompt_mode == "full"
    assert len(lean_prompt) < len(full_prompt) * 0.6


@pytest.mark.parametrize(("raw", "expected"), [(0.9, 0.9), (9.5, 0.95), (85, 0.85), ("bad", 0.5)])
def test_review_score_is_normalized(raw: object, expected: float) -> None:
    assert MicrosoftOllamaRecommendationProvider._normalize_review_score(raw) == pytest.approx(expected)
