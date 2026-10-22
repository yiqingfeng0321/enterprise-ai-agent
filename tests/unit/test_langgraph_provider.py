from __future__ import annotations

import pytest

from app.agents.providers.langgraph_ollama import LangGraphOllamaRecommendationProvider
from app.core.exceptions import AgentProviderError


@pytest.fixture
def provider() -> LangGraphOllamaRecommendationProvider:
    return LangGraphOllamaRecommendationProvider("http://localhost:11434", "qwen2.5:7b")


def test_validate_payload_requires_model_risk_and_approval(provider: LangGraphOllamaRecommendationProvider) -> None:
    payload = {
        "executive_summary": "summary",
        "risk_assessment": "high",
        "recommended_actions": [
            {
                "action": "Run recovery workshop",
                "owner": "CS",
                "rationale": "usage is down",
                "expected_impact": "improve adoption",
                "approval_required": False,
            },
            {
                "action": "Escalate support",
                "owner": "Support",
                "rationale": "tickets are high",
                "expected_impact": "restore trust",
                "approval_required": False,
            },
        ],
        "rationale": "Need action",
        "evidence": ["support burden high", "usage down", "playbook says recover"],
        "do_not_do": ["Do not discount early"],
        "confidence": 0.8,
        "requires_human_approval": False,
    }
    with pytest.raises(AgentProviderError):
        provider._validate_payload(payload, "high")


def test_normalize_payload_uses_action_objects(provider: LangGraphOllamaRecommendationProvider) -> None:
    payload = {
        "executive_summary": "summary",
        "risk_assessment": "high",
        "recommended_actions": [
            {
                "action": "Launch executive value review",
                "owner": "AE + CS Leadership",
                "rationale": "renewal risk is elevated",
                "expected_impact": "align stakeholders",
                "approval_required": True,
            },
            {
                "action": "Assign AI deployment squad",
                "owner": "Solutions Engineering",
                "rationale": "activation is low",
                "expected_impact": "increase active seat ratio",
                "approval_required": False,
            },
        ],
        "rationale": "Need action",
        "evidence": ["Model risk: 83% (high)", "seat activation is low", "support tickets are elevated"],
        "do_not_do": ["Do not repeat discounts"],
        "confidence": 0.82,
        "requires_human_approval": True,
    }
    recommendation = provider._normalize_payload(payload)
    assert recommendation.actions[0].owner == "AE + CS Leadership"
    assert recommendation.actions[0].approval_required is True
    assert recommendation.recommended_actions[1] == "Assign AI deployment squad"


def test_prompt_uses_reviewer_evidence_contract(
    provider: LangGraphOllamaRecommendationProvider,
) -> None:
    prompt = provider._build_prompt("{}", "high")

    assert "must start with 'Model risk:'" in prompt
