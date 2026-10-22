import pytest

from app.core.container import get_container


@pytest.mark.slow
@pytest.mark.asyncio
async def test_end_to_end_workflow() -> None:
    container = get_container()
    customer_id = container.customer_repository.list_ids(limit=1)[0]
    result = await container.workflow.analyze(customer_id, session_id="pytest", persist=False)
    assert 0 <= result.prediction.churn_probability <= 1
    assert result.recommendation.actions
    assert "Retention Analysis" in result.report_markdown
    assert result.review.score >= 0.6
