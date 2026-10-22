from __future__ import annotations

import asyncio

from app.core.container import get_container


async def main() -> None:
    container = get_container()
    candidates = []
    for customer_id in container.customer_repository.list_ids(limit=200):
        profile = container.customer_repository.get(customer_id)
        prediction = container.ml_service.predict(profile, top_n=3)
        candidates.append((prediction.churn_probability, customer_id))
    customer_id = max(candidates)[1]
    result = await container.workflow.analyze(
        customer_id, session_id="local-demo", create_ticket=True
    )
    print(result.report_markdown)
    print(f"\nAnalysis ID: {result.analysis_id}")
    print(f"Ticket ID: {result.ticket_id}")
    print(f"Provider: {result.recommendation.generated_by}")


if __name__ == "__main__":
    asyncio.run(main())
