from __future__ import annotations

from app.models.domain import (
    ActionItem,
    CustomerProfile,
    KnowledgeHit,
    MemoryItem,
    PredictionResult,
    Recommendation,
)


class RecommendationService:
    """Deterministic business-policy engine used as the free local fallback and test oracle."""

    def generate(
        self,
        profile: CustomerProfile,
        prediction: PredictionResult,
        knowledge: list[KnowledgeHit],
        memories: list[MemoryItem],
    ) -> Recommendation:
        actions: list[ActionItem] = []
        risky_features = {driver.feature for driver in prediction.top_drivers if driver.direction == "risk"}
        negative_outcomes = [
            memory.content.lower()
            for memory in memories
            if memory.memory_type == "intervention_outcome_negative"
        ]
        discount_previously_ineffective = any(
            token in content
            for content in negative_outcomes
            for token in ("discount", "折扣")
        )

        if "support_tickets_90d" in risky_features or profile.support_tickets_90d >= 3:
            actions.append(
                ActionItem(
                    priority=1,
                    action="Open a priority support escalation and schedule a root-cause review within 48 hours.",
                    owner="Customer Success + Support",
                    rationale="Unresolved support burden is a leading churn driver for this account.",
                    expected_impact="Restore trust and remove product blockers before commercial outreach.",
                    approval_required=False,
                )
            )
        if (
            not profile.onboarding_complete
            or profile.active_seat_ratio_30d < 0.35
            or profile.champion_count == 0
        ):
            actions.append(
                ActionItem(
                    priority=len(actions) + 1,
                    action="Assign an AI deployment squad to remediate rollout blockers, permissions, enablement gaps, and low seat activation.",
                    owner="Solutions Engineering + Customer Success",
                    rationale="Low adoption, weak rollout, or lack of internal champions indicates the customer needs structured AI deployment support.",
                    expected_impact="Increase seat activation, unblock adoption, and restore confidence before renewal negotiation.",
                    approval_required=False,
                )
            )
        if "usage_change_pct" in risky_features or profile.usage_change_pct <= -20:
            actions.append(
                ActionItem(
                    priority=len(actions) + 1,
                    action="Run a usage-recovery workshop focused on the two workflows with the largest decline.",
                    owner="Solutions Engineering",
                    rationale="Product usage has materially declined relative to the customer baseline.",
                    expected_impact="Recover weekly active usage and identify adoption gaps.",
                    approval_required=False,
                )
            )
        if not profile.onboarding_complete:
            actions.append(
                ActionItem(
                    priority=len(actions) + 1,
                    action="Complete the onboarding success plan with named milestones and executive sponsorship.",
                    owner="Customer Success Manager",
                    rationale="Incomplete onboarding reduces realized value and weakens renewal intent.",
                    expected_impact="Accelerate time-to-value and increase stakeholder adoption.",
                    approval_required=False,
                )
            )
        if profile.days_to_renewal <= 60:
            actions.append(
                ActionItem(
                    priority=len(actions) + 1,
                    action="Launch a renewal risk review and present a value-realization summary before pricing discussion.",
                    owner="Account Executive",
                    rationale=f"The contract renews in {profile.days_to_renewal} days.",
                    expected_impact="Create decision clarity and preserve negotiating leverage.",
                    approval_required=True,
                )
            )
        if profile.payment_failures_90d > 0:
            actions.append(
                ActionItem(
                    priority=len(actions) + 1,
                    action="Confirm billing contacts and resolve failed-payment root causes before renewal.",
                    owner="Finance Operations",
                    rationale="Recent payment failures create operational and relationship risk.",
                    expected_impact="Reduce involuntary churn and billing friction.",
                    approval_required=False,
                )
            )
        if prediction.risk_band == "high" and not any(
            action.approval_required for action in actions[:4]
        ):
            actions = actions[:3]
            actions.append(
                ActionItem(
                    priority=len(actions) + 1,
                    action=(
                        "Hold an executive retention review before any customer-facing "
                        "commercial commitment."
                    ),
                    owner="Customer Success Leadership",
                    rationale="High-risk accounts require an explicit human approval checkpoint.",
                    expected_impact="Align accountability, escalation level, and approved commitments.",
                    approval_required=True,
                )
            )

        if not actions:
            actions.append(
                ActionItem(
                    priority=1,
                    action="Maintain proactive quarterly value reviews and monitor leading engagement indicators.",
                    owner="Customer Success Manager",
                    rationale="No single severe operational driver requires immediate escalation.",
                    expected_impact="Sustain adoption and detect deterioration early.",
                    approval_required=False,
                )
            )

        memory_warning = any("discount" in memory.content.lower() for memory in memories)
        do_not_do = ["Do not send an automated outreach without account-owner review."]
        if memory_warning:
            do_not_do.append("Do not repeat a blanket discount; prior memory indicates discounting was already used.")
        if discount_previously_ineffective:
            do_not_do.append(
                "Do not propose another generic discount; prior feedback indicates discount intervention was ineffective."
            )
            actions.append(
                ActionItem(
                    priority=len(actions) + 1,
                    action="Replace discount-led retention with a targeted adoption recovery plan tied to measurable workflow outcomes.",
                    owner="Customer Success Leadership",
                    rationale="Historical feedback indicates previous discount interventions did not improve retention outcomes.",
                    expected_impact="Improve renewal probability through value realization rather than repeated price concessions.",
                    approval_required=True,
                )
            )
        if prediction.risk_band == "high":
            do_not_do.append("Do not wait for the next quarterly business review to intervene.")

        evidence = [
            f"Model risk: {prediction.churn_probability:.1%} ({prediction.risk_band}).",
            *[
                f"{driver.feature}: {driver.customer_value} ({driver.probability_impact:+.1%} impact)"
                for driver in prediction.top_drivers[:3]
            ],
            *[
                f"{hit.document} — {hit.section} (retrieval score {hit.score:.2f})"
                for hit in knowledge[:2]
            ],
        ]
        return Recommendation(
            executive_summary=(
                f"{profile.customer_id} has {prediction.risk_band} churn risk at "
                f"{prediction.churn_probability:.1%}. Prioritize operational recovery before incentives, "
                "then align renewal stakeholders around measurable value."
            ),
            actions=actions[:4],
            do_not_do=do_not_do,
            evidence=evidence,
            confidence=min(0.95, 0.65 + len(prediction.top_drivers) * 0.04 + len(knowledge) * 0.03),
            generated_by="deterministic_policy_engine",
            requires_human_approval=any(
                action.approval_required for action in actions[:4]
            ),
            risk_assessment=f"{prediction.risk_band.title()} churn risk based on the model prediction and customer context.",
            rationale="Deterministic policy engine prioritized evidence-backed recovery actions and an explicit approval checkpoint for high-risk accounts.",
            recommended_actions=[action.action for action in actions[:4]],
        )
