from __future__ import annotations

from app.models.domain import CustomerProfile, MemoryItem, PredictionResult


def recommended_agent_skills(
    profile: CustomerProfile,
    prediction: PredictionResult,
    memories: list[MemoryItem],
) -> list[dict[str, str]]:
    """Select relevant formal skills while leaving their instructions unloaded."""
    selected: list[dict[str, str]] = [
        {
            "skill": "account-diagnosis",
            "reason": "Every recommendation starts with grounded evidence and model signals.",
        }
    ]
    if (
        not profile.onboarding_complete
        or profile.active_seat_ratio_30d < 0.35
        or profile.support_tickets_90d >= 3
    ):
        selected.append(
            {
                "skill": "deployment-recovery",
                "reason": (
                    "Rollout, activation, or unresolved support weakness requires "
                    "a verified recovery sequence."
                ),
            }
        )
    if profile.usage_change_pct <= -15 or profile.active_seat_ratio_30d < 0.40:
        selected.append(
            {
                "skill": "adoption-reactivation",
                "reason": "The usage profile indicates workflow-level reactivation is needed.",
            }
        )
    if any(memory.memory_type.startswith("intervention_") for memory in memories):
        selected.append(
            {
                "skill": "memory-aware-strategy",
                "reason": "Prior intervention history should change the next plan.",
            }
        )
    if prediction.risk_band == "high" or profile.days_to_renewal <= 60 or profile.champion_count <= 1:
        selected.append(
            {
                "skill": "stakeholder-alignment",
                "reason": "Risk, timing, or champion coverage requires approval-aware coordination.",
            }
        )
    if "roi" in profile.customer_notes.lower() or profile.monthly_revenue >= 10_000:
        selected.append(
            {
                "skill": "roi-value-story",
                "reason": "Value proof and ROI framing are important in this account context.",
            }
        )
    return selected
