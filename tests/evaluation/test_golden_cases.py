import pytest

from app.core.container import get_container
from pipelines.evaluation.run_agent_eval import evaluate_agent


@pytest.mark.slow
@pytest.mark.asyncio
async def test_golden_dataset_regression() -> None:
    container = get_container()
    with container.database.connect() as connection:
        before = {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("predictions", "session_messages", "customer_memories")
        }
    report = await evaluate_agent(
        container.workflow, container.settings.golden_dataset_path
    )
    with container.database.connect() as connection:
        after = {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("predictions", "session_messages", "customer_memories")
        }
    metrics = report["metrics"]
    assert after == before
    assert report["provenance"]["providers"]
    assert metrics["risk_band_accuracy"] >= 0.95
    assert metrics["mean_keyword_coverage"] >= 0.65
    assert metrics["review_pass_rate"] >= 0.95
    assert metrics["review_score_threshold_rate"] == 1.0
    assert metrics["evidence_rate"] == 1.0
    assert metrics["approval_alignment_rate"] == 1.0
