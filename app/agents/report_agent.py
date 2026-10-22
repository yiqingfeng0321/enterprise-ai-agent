from app.models.domain import (
    CustomerProfile,
    KnowledgeHit,
    PredictionResult,
    Recommendation,
    ReviewResult,
)


class ReportAgent:
    async def run(
        self,
        profile: CustomerProfile,
        prediction: PredictionResult,
        benchmark: dict,
        knowledge: list[KnowledgeHit],
        recommendation: Recommendation,
        review: ReviewResult,
    ) -> str:
        drivers = "\n".join(
            f"- **{driver.feature}**: `{driver.customer_value}`; impact {driver.probability_impact:+.1%}"
            for driver in prediction.top_drivers
        )
        actions = "\n".join(
            f"{action.priority}. **{action.action}**  \n"
            f"   Owner: {action.owner} | Approval: {action.approval_required}  \n"
            f"   Why: {action.rationale}"
            for action in recommendation.actions
        )
        sources = "\n".join(
            f"- `{hit.document}` — {hit.section} (score {hit.score:.2f})"
            for hit in knowledge
        ) or "- No playbook passage met the retrieval threshold."
        recommendation_evidence = "\n".join(
            f"- {item}" for item in recommendation.evidence
        ) or "- No recommendation evidence provided."
        return f"""# Retention Analysis: {profile.customer_id}

## Executive summary
{recommendation.executive_summary}

## Risk assessment
- Churn probability: **{prediction.churn_probability:.1%}**
- Risk band: **{prediction.risk_band.upper()}**
- Model version: `{prediction.model_version}`
- Annualized revenue at risk: **${profile.revenue_at_risk:,.0f}**

### Main model drivers
{drivers}

## Segment benchmark
- Segment: {benchmark.get('segment', profile.segment)}
- Observed churn rate: {float(benchmark.get('observed_churn_rate', 0)):.1%}
- Average NPS: {float(benchmark.get('avg_nps', 0)):.1f}
- Average 30-day logins: {float(benchmark.get('avg_logins_30d', 0)):.1f}

## Recommended plan
{actions}

## Guardrails
{chr(10).join(f'- {item}' for item in recommendation.do_not_do)}

## Retrieved evidence
{recommendation_evidence}
{sources}

## Reviewer
- Passed: **{review.passed}**
- Score: **{review.score:.0%}**
- Issues: {', '.join(review.issues) if review.issues else 'None'}
"""
