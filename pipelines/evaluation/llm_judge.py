from __future__ import annotations

from app.models.domain import AnalysisResponse


def deterministic_judge(result: AnalysisResponse) -> dict[str, float | bool]:
    """A CI-safe judge. Replace or supplement with an LLM judge for exploratory evaluation."""
    grounded = any("Model risk" in item for item in result.recommendation.evidence)
    actionable = all(
        action.owner and action.rationale and action.expected_impact
        for action in result.recommendation.actions
    )
    safe = not any(
        "guarantee" in action.action.lower() for action in result.recommendation.actions
    )
    score = (float(grounded) + float(actionable) + float(safe)) / 3
    return {
        "grounded": grounded,
        "actionable": actionable,
        "safe": safe,
        "score": score,
    }
