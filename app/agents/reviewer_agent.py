import logging
from typing import Any

from app.core.exceptions import AgentProviderError
from app.models.domain import CustomerProfile, PredictionResult, Recommendation, ReviewResult

logger = logging.getLogger(__name__)


class ReviewerAgent:
    """Deterministic guardrail reviewer with an optional LLM-assisted review path."""

    def __init__(self, llm_provider: Any | None = None, use_llm_review: bool = False):
        self.llm_provider = llm_provider
        self.use_llm_review = use_llm_review

    async def run(
        self,
        profile: CustomerProfile,
        prediction: PredictionResult,
        recommendation: Recommendation,
    ) -> ReviewResult:
        if self.use_llm_review and self.llm_provider is not None and hasattr(
            self.llm_provider, "review_recommendation"
        ):
            try:
                return await self.llm_provider.review_recommendation(
                    profile, prediction, recommendation
                )
            except AgentProviderError as exc:
                logger.warning("LLM reviewer failed; using deterministic guardrails: %s", exc)

        issues: list[str] = []
        changes: list[str] = []
        if not recommendation.actions:
            issues.append("No actionable next step was produced")
            changes.append("Add at least one owned and prioritized action")
        priorities = [action.priority for action in recommendation.actions]
        if priorities != sorted(priorities) or len(priorities) != len(set(priorities)):
            issues.append("Action priorities are not unique and ordered")
            changes.append("Renumber actions sequentially")
        if not any("Model risk" in item for item in recommendation.evidence):
            issues.append("The model prediction is not cited in evidence")
            changes.append("Include model risk probability and band")
        if prediction.risk_band == "high" and not any(
            "approval" in action.owner.lower() or action.approval_required
            for action in recommendation.actions
        ):
            issues.append("High-risk plan has no explicit approval checkpoint")
            changes.append("Mark the commercial/renewal action as approval required")
        if profile.days_to_renewal > 120 and any(
            "discount" in action.action.lower() for action in recommendation.actions
        ):
            issues.append("Discounting is premature for a distant renewal")
            changes.append("Replace discounting with adoption and value-realization actions")
        if any("customer said" in item.lower() for item in recommendation.evidence):
            issues.append("The recommendation includes an unsupported customer claim")
            changes.append("Replace unsupported claims with evidence-backed observations")
        if recommendation.requires_human_approval is False and prediction.risk_band == "high":
            issues.append("High-risk recommendations should require human approval")
            changes.append("Mark the recommendation as requiring human approval")
        score = max(0.0, 1.0 - 0.2 * len(issues))
        return ReviewResult(
            passed=not issues,
            score=score,
            issues=issues,
            required_changes=changes,
        )
