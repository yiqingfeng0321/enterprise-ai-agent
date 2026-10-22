from app.models.domain import (
    ActionItem,
    CustomerProfile,
    KnowledgeHit,
    MemoryItem,
    PredictionResult,
    Recommendation,
)
from app.services.recommendation_service import RecommendationService


class DeterministicRecommendationProvider:
    name = "deterministic"
    model_id = "rules-v1"
    service_available = True
    ollama_available = False

    def __init__(self, service: RecommendationService | None):
        self.service = service

    async def generate(
        self,
        profile: CustomerProfile,
        prediction: PredictionResult,
        benchmark: dict,
        knowledge: list[KnowledgeHit],
        memories: list[MemoryItem],
    ) -> Recommendation:
        if self.service is not None:
            return self.service.generate(profile, prediction, knowledge, memories)

        return Recommendation(
            executive_summary=(
                f"{profile.customer_id} has {prediction.risk_band} churn risk at "
                f"{prediction.churn_probability:.1%}. Prioritize operational recovery before incentives."
            ),
            actions=[
                ActionItem(
                    priority=1,
                    action="Open a priority support escalation and schedule a root-cause review within 48 hours.",
                    owner="Customer Success + Support",
                    rationale="The account shows elevated churn risk and needs immediate attention.",
                    expected_impact="Reduce churn risk with timely intervention.",
                    approval_required=False,
                )
            ],
            do_not_do=["Do not send an automated outreach without account-owner review."],
            evidence=[f"Model risk: {prediction.churn_probability:.1%} ({prediction.risk_band})."],
            confidence=0.65,
            generated_by="deterministic_fallback",
            requires_human_approval=prediction.risk_band == "high",
            risk_assessment=f"{prediction.risk_band.title()} churn risk based on the model prediction and customer context.",
            rationale="Deterministic fallback applied because the requested provider was unavailable.",
            recommended_actions=["Open a priority support escalation and schedule a root-cause review within 48 hours."],
        )
