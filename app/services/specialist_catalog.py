from __future__ import annotations

from app.models.domain import CustomerProfile, PredictionResult


def available_specialists() -> list[dict[str, str]]:
    return [
        {
            "name": "ai_deployment_specialist",
            "summary": "Resolves tenant setup, permissions, security controls, integration blockers, and rollout sequencing.",
            "best_for": "Incomplete onboarding, low activation, deployment blockers, or stalled rollout.",
            "primary_outcome": "Restore deployment momentum and increase active usage.",
        },
        {
            "name": "adoption_coach",
            "summary": "Designs workflow-specific enablement and habit-forming adoption plans.",
            "best_for": "Usage decline, low active-seat ratio, or weak time-to-value.",
            "primary_outcome": "Increase workflow adoption and sustained usage.",
        },
        {
            "name": "executive_sponsor_manager",
            "summary": "Coordinates executive reviews, renewal alignment, and escalation governance.",
            "best_for": "Renewal pressure, strategic accounts, or executive misalignment.",
            "primary_outcome": "Align decision makers and reduce renewal uncertainty.",
        },
        {
            "name": "support_recovery_manager",
            "summary": "Owns root-cause reviews, incident coordination, closure plans, and trust recovery.",
            "best_for": "High support burden, repeated incidents, or unresolved blockers.",
            "primary_outcome": "Stabilize service experience before commercial outreach.",
        },
        {
            "name": "value_realization_lead",
            "summary": "Builds ROI narratives, success scorecards, and measurable outcome plans.",
            "best_for": "ROI-sensitive customers or renewal justification gaps.",
            "primary_outcome": "Make value tangible before pricing discussions.",
        },
        {
            "name": "champion_enablement_lead",
            "summary": "Builds internal champions, power-user networks, and local advocacy.",
            "best_for": "Too few champions, weak internal pull, or single-threaded ownership.",
            "primary_outcome": "Create durable internal momentum beyond one contact.",
        },
    ]


def recommended_specialist_focus(
    profile: CustomerProfile,
    prediction: PredictionResult,
) -> list[dict[str, str]]:
    focus: list[dict[str, str]] = []
    if not profile.onboarding_complete or profile.active_seat_ratio_30d < 0.35:
        focus.extend(
            [
                {
                    "specialist": "ai_deployment_specialist",
                    "reason": "Rollout weakness needs deployment remediation.",
                },
                {
                    "specialist": "adoption_coach",
                    "reason": "Low activation needs workflow-level coaching.",
                },
            ]
        )
    if profile.champion_count <= 1:
        focus.append(
            {
                "specialist": "champion_enablement_lead",
                "reason": "Champion coverage is too weak to sustain adoption.",
            }
        )
    if profile.support_tickets_90d >= 3:
        focus.append(
            {
                "specialist": "support_recovery_manager",
                "reason": "Support burden should be resolved before commercial outreach.",
            }
        )
    if profile.days_to_renewal <= 60 or prediction.risk_band == "high":
        focus.append(
            {
                "specialist": "executive_sponsor_manager",
                "reason": "Renewal timing or high risk needs stakeholder alignment.",
            }
        )
    if "roi" in profile.customer_notes.lower() or profile.monthly_revenue >= 10_000:
        focus.append(
            {
                "specialist": "value_realization_lead",
                "reason": "The account needs value proof and ROI framing.",
            }
        )
    seen: set[str] = set()
    deduped = []
    for item in focus:
        if item["specialist"] not in seen:
            seen.add(item["specialist"])
            deduped.append(item)
    return deduped[:4]
