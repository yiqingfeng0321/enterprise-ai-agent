from __future__ import annotations


def approved_interventions() -> list[dict[str, str | bool]]:
    return [
        {
            "name": "ai_deployment_squad",
            "summary": "Assign an AI deployment squad to fix tenant setup, permissions, rollout sequencing, and workflow enablement.",
            "best_for": "Low active-seat ratio, stalled rollout, incomplete onboarding, or blocked deployment dependencies.",
            "owner": "Solutions Engineering + Customer Success",
            "approval_required": False,
        },
        {
            "name": "workflow_adoption_workshop",
            "summary": "Run a workflow-level adoption workshop for the two most important business scenarios with weak usage.",
            "best_for": "Usage decline, weak habit formation, or missing time-to-value after launch.",
            "owner": "Adoption Consultant",
            "approval_required": False,
        },
        {
            "name": "executive_value_review",
            "summary": "Hold an executive value review with outcome metrics, realized ROI, risks, and a dated recovery plan.",
            "best_for": "Renewals within 60 days, strategic accounts, or executive misalignment.",
            "owner": "Account Executive + Customer Success Leadership",
            "approval_required": True,
        },
        {
            "name": "support_escalation",
            "summary": "Open a priority support escalation with named owner, root-cause review, and closure plan.",
            "best_for": "Repeated support issues or unresolved product blockers.",
            "owner": "Support + Customer Success",
            "approval_required": False,
        },
        {
            "name": "champion_building_plan",
            "summary": "Create a champion-building plan with named power users, manager sponsorship, and weekly enablement goals.",
            "best_for": "Too few internal champions or weak organizational pull.",
            "owner": "Customer Success Manager",
            "approval_required": False,
        },
        {
            "name": "commercial_offer_with_commitment",
            "summary": "Offer limited commercial flexibility only when tied to customer commitments, deployment milestones, or expanded adoption plans.",
            "best_for": "Commercial friction after operational recovery is underway.",
            "owner": "Account Executive",
            "approval_required": True,
        },
    ]