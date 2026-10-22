from __future__ import annotations

from pathlib import Path

import pytest
from agent_framework import SkillsProvider

from app.agents.providers.microsoft_ollama import MicrosoftOllamaRecommendationProvider
from app.core.config import PROJECT_ROOT
from app.models.domain import CustomerProfile, MemoryItem, PredictionResult
from app.services.agent_skill_router import recommended_agent_skills
from app.services.agent_skill_service import AgentSkillError, AgentSkillService


def test_skill_catalog_uses_progressive_disclosure() -> None:
    service = AgentSkillService(PROJECT_ROOT / "skills")

    advertised = service.list_skills()

    assert {item["name"] for item in advertised} == {
        "account-diagnosis",
        "adoption-reactivation",
        "deployment-recovery",
        "memory-aware-strategy",
        "roi-value-story",
        "stakeholder-alignment",
    }
    assert all(set(item) == {"name", "description"} for item in advertised)

    expected_resources = {
        "account-diagnosis": ["references/evidence-rules.md"],
        "adoption-reactivation": ["references/adoption-patterns.md"],
        "deployment-recovery": ["references/rollout-checklist.md"],
        "memory-aware-strategy": ["references/intervention-outcome-rules.md"],
        "roi-value-story": ["references/value-scorecard.md"],
        "stakeholder-alignment": ["references/stakeholder-matrix.md"],
    }
    for name, resources in expected_resources.items():
        loaded = service.load_skill(name)
        assert "instructions" in loaded
        assert loaded["resources"] == resources
        assert "contract" in loaded["instructions"].lower()
        assert "approval" in loaded["instructions"].lower()

    resource = service.read_skill_resource(
        "account-diagnosis",
        "references/evidence-rules.md",
    )
    assert "ML prediction and drivers" in resource["content"]


@pytest.mark.parametrize(
    ("skill_name", "resource_path"),
    [
        ("../account-diagnosis", "references/evidence-rules.md"),
        ("account-diagnosis", "../SKILL.md"),
        ("account-diagnosis", "/etc/passwd"),
        ("account-diagnosis", "agents/openai.yaml"),
    ],
)
def test_skill_loader_rejects_unsafe_resources(skill_name: str, resource_path: str) -> None:
    service = AgentSkillService(PROJECT_ROOT / "skills")

    with pytest.raises(AgentSkillError):
        service.read_skill_resource(skill_name, resource_path)


def test_invalid_skill_metadata_is_not_advertised(tmp_path: Path) -> None:
    skill_dir = tmp_path / "wrong-name"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: another-name\ndescription: invalid mismatch\n---\nInstructions",
        encoding="utf-8",
    )

    service = AgentSkillService(tmp_path)

    assert service.list_skills() == []
    with pytest.raises(AgentSkillError, match="must match directory"):
        service.load_skill("wrong-name")


def test_microsoft_provider_attaches_native_skills_provider() -> None:
    provider = MicrosoftOllamaRecommendationProvider(
        "http://127.0.0.1:9",
        "qwen2.5:7b",
        None,
        skills_dir=PROJECT_ROOT / "skills",
    )

    assert isinstance(provider.skills_provider, SkillsProvider)
    assert provider.skills_provider in provider.agent.context_providers


def _high_risk_profile(**overrides: object) -> CustomerProfile:
    values: dict[str, object] = {
        "customer_id": "CUST-TEST",
        "segment": "Enterprise",
        "region": "NA",
        "customer_notes": "Executive team requires ROI evidence.",
        "licensed_seats": 100,
        "active_seat_ratio_30d": 0.25,
        "champion_count": 1,
        "tenure_months": 18,
        "monthly_revenue": 20_000.0,
        "logins_30d": 40,
        "usage_change_pct": -25.0,
        "support_tickets_90d": 4,
        "nps": 2,
        "days_to_renewal": 45,
        "payment_failures_90d": 0,
        "onboarding_complete": False,
        "support_ticket_rate": 0.04,
        "engagement_score": 0.2,
        "renewal_urgency": 0.75,
        "revenue_at_risk": 240_000.0,
    }
    values.update(overrides)
    return CustomerProfile.model_validate(values)


def test_skill_router_selects_all_relevant_skills_without_position_truncation() -> None:
    profile = _high_risk_profile()
    prediction = PredictionResult(
        customer_id=profile.customer_id,
        churn_probability=0.91,
        risk_band="high",
        top_drivers=[],
        model_version="test",
    )
    memories = [
        MemoryItem(
            customer_id=profile.customer_id,
            memory_type="intervention_outcome_negative",
            content="A prior generic discount was ineffective.",
        )
    ]

    selected = recommended_agent_skills(profile, prediction, memories)

    assert [item["skill"] for item in selected] == [
        "account-diagnosis",
        "deployment-recovery",
        "adoption-reactivation",
        "memory-aware-strategy",
        "stakeholder-alignment",
        "roi-value-story",
    ]


def test_support_burden_routes_to_deployment_recovery() -> None:
    profile = _high_risk_profile(
        onboarding_complete=True,
        active_seat_ratio_30d=0.7,
        usage_change_pct=5.0,
        champion_count=3,
        days_to_renewal=180,
        monthly_revenue=5_000.0,
        customer_notes="",
    )
    prediction = PredictionResult(
        customer_id=profile.customer_id,
        churn_probability=0.4,
        risk_band="medium",
        top_drivers=[],
        model_version="test",
    )

    selected = recommended_agent_skills(profile, prediction, [])

    assert [item["skill"] for item in selected] == [
        "account-diagnosis",
        "deployment-recovery",
    ]
